"""``clips``: each selected span cut from its session's videos, without re-encoding.

Per span of ``results.json`` and camera of the spec, the frames from the keyframe at or
before the span's start to the keyframe at or after its end are stream-copied from the
source video into a fragmented MP4 (so ffmpeg can write it to a pipe), checked (one
packet per frame, evenly spaced) and written to ``clips/<session_id>_<span>_<camera>.mp4``
with those frames' ``Time`` (repaired ``ReferenceTime``) beside it as ``.parquet``. The
thermistor over both cameras ± ``[clips] thermistor_pad_s`` is ``..._thermistor.parquet``;
``thumbnails/<session_id>_<camera>.png`` is the first frame of the session's first span.
``clips.parquet`` lists them, ``clips.toml`` is their clip list (no boxes). Resumable.
"""

import json
import subprocess
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import pyarrow as pa
import tomlkit

from . import storage
from .config import DatasetSpec, Layout
from .lists import header, subjects, validate
from .remote import thread_map
from .videos import https

BATCH = 8  # sessions per save of clips.parquet
THUMBNAIL_BYTES = 8 << 20  # read of a clip's start that holds its first frame
WINDOW_S = 4.0  # first keyframe search window either side (doubled until found)
KEY = ["session_id", "span", "camera"]
HEADER = [
    "Clips cut from the selected spans of a zephyr-dataset folder: one entry per span",
    "and camera, written by `zephyr-dataset clips` (never edit by hand: boxes come",
    "later). `group` is the subject from the session metadata.",
]


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    result = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()[-500:]
        raise RuntimeError(f"{cmd[0]} failed: {message}")
    return result.stdout


class Video:
    """A source video's stream: frame ``k`` is at pts ``k * tick`` (``time_base``)."""

    def __init__(self, url: str) -> None:
        self.url = url
        info = json.loads(
            _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=time_base,r_frame_rate,nb_frames",
                    "-of",
                    "json",
                    url,
                ]
            )
        )["streams"][0]
        self.time_base = Fraction(info["time_base"])
        self.rate = Fraction(info["r_frame_rate"])
        tick = 1 / (self.time_base * self.rate)
        if tick.denominator != 1:
            raise ValueError(f"a frame lasts {tick} ticks, not a whole number")
        self.tick = int(tick)
        self.frames = int(info["nb_frames"])

    def seconds(self, frame: int) -> float:
        return float(frame * self.tick * self.time_base)

    def keyframes(self, frame: int, window_s: float) -> list[int]:
        """Indices of the keyframes within *window_s* of frame *frame*."""
        t = self.seconds(frame)
        out = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"{max(0.0, t - window_s):.6f}%{t + window_s:.6f}",
                "-show_entries",
                "packet=pts,flags",
                "-of",
                "csv=p=0",
                self.url,
            ],
        ).decode()
        keys = set()
        for line in out.split():
            pts, _, flags = line.partition(",")
            if "K" not in flags or not pts.lstrip("-").isdigit():
                continue
            if int(pts) % self.tick:
                raise ValueError(f"keyframe pts {pts} is not on a frame")
            keys.add(int(pts) // self.tick)
        return sorted(keys)

    def keyframe_before(self, frame: int) -> int:
        """The last keyframe at or before *frame*."""
        window = WINDOW_S
        while True:
            found = [k for k in self.keyframes(frame, window) if k <= frame]
            if found:
                return found[-1]
            if self.seconds(frame) <= window:
                raise ValueError(f"no keyframe at or before frame {frame}")
            window *= 2

    def keyframe_after(self, frame: int) -> int:
        """The first keyframe at or after *frame*, or the end of the video."""
        window = WINDOW_S
        while frame < self.frames:
            found = [k for k in self.keyframes(frame, window) if k >= frame]
            if found:
                return found[0]
            if self.seconds(frame) + window >= self.seconds(self.frames):
                break
            window *= 2
        return self.frames

    def cut(self, first: int, n: int) -> bytes:
        """Frames ``first`` to ``first + n`` (*first* a keyframe), stream-copied."""
        return _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                f"{self.seconds(first):.6f}",
                "-i",
                self.url,
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-an",
                "-frames:v",
                str(n),
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                "-f",
                "mp4",
                "pipe:1",
            ]
        )


def check(data: bytes, n: int) -> None:
    """Refuse a cut that is not *n* frames, one packet each, evenly spaced."""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
            "-i",
            "pipe:0",
        ],
        data,
    ).decode()
    pts = np.sort([int(v.strip(",")) for v in out.split() if v.strip(",") != "N/A"])
    if len(pts) != n:
        raise ValueError(f"the cut has {len(pts)} frames, not {n}")
    steps = np.diff(pts)
    if n > 1 and (steps.min() <= 0 or steps.min() != steps.max()):
        raise ValueError("the cut's frames are not evenly spaced")


def thumbnail(data: bytes) -> bytes:
    """The first frame of a cut as a full-resolution PNG; the cut's start (its index
    and first GOP) is enough, so a stored clip needs only its first bytes read."""
    command = [
        "ffmpeg", "-v", "error", "-i", "pipe:0", "-frames:v", "1",
        "-f", "image2pipe", "-c:v", "png", "pipe:1",
    ]  # fmt: skip
    try:
        png = _run(command, data[:THUMBNAIL_BYTES])
    except RuntimeError:
        png = b""
    if not png and len(data) > THUMBNAIL_BYTES:  # a first GOP larger than that
        png = _run(command, data)
    if not png:
        raise ValueError("no first frame to make a thumbnail of")
    return png


def frame_range(video: Video, times: np.ndarray, start: float, end: float):
    """``(first, n)``: whole GOPs covering every frame timed in ``[start, end]``."""
    if len(times) != video.frames:
        raise ValueError(f"{len(times)} timestamps but {video.frames} video frames")
    i0 = int(np.searchsorted(times, start, "left"))
    i1 = int(np.searchsorted(times, end, "right"))
    if i0 >= i1:
        raise ValueError(f"no frames in [{start}, {end}]")
    first = video.keyframe_before(i0)
    return first, video.keyframe_after(i1) - first


def write_times(times: np.ndarray, path) -> None:
    table = pa.table({"Time": times.astype(np.float64)})
    storage.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        column_encoding={"Time": "BYTE_STREAM_SPLIT"},
    )


def relative(layout: Layout, path) -> str:
    return f"{layout.clip_dir.name}/{path.name}"


def one_session(spec, layout: Layout, s: dict, done: set, url_of) -> list[dict]:
    """Cut a session's spans; one ``clips.parquet`` row per span and camera."""
    sid, cameras = s["session_id"], spec.source.cameras
    pad = spec.clips.thermistor_pad_s
    rows = []

    def row(span, name, **fields):
        base = {"session_id": sid, "span": span, "camera": name, "error": None}
        return base | fields

    try:
        videos = {n: Video(url_of(s["cameras"][n])) for n in cameras}
        times = {
            n: storage.read_parquet(
                layout.timestamps(sid, n), columns=["ReferenceTime"]
            )["ReferenceTime"].to_numpy()
            for n in cameras
        }
        thermistor = storage.read_parquet(layout.thermistor(sid))
    except Exception as exc:  # noqa: BLE001 -- recorded; retried on the next run
        error = f"{type(exc).__name__}: {exc}"[:300]
        return [row(span, n, error=error) for span in s["clips"] for n in cameras]
    for span, window in sorted(s["clips"].items()):
        todo = [n for n in cameras if (sid, span, n) not in done]
        if not todo:
            continue
        try:
            ranges = {
                n: frame_range(videos[n], times[n], window["start"], window["end"])
                for n in cameras
            }
            lo = min(times[n][f] for n, (f, _) in ranges.items()) - pad
            hi = max(times[n][f + c - 1] for n, (f, c) in ranges.items()) + pad
            inside = thermistor["Time"].between(lo, hi)
            therm_path = layout.clip(sid, span, "thermistor", ".parquet")
            storage.write_frame(thermistor[inside].reset_index(drop=True), therm_path)
        except Exception as exc:  # noqa: BLE001 -- recorded; retried on the next run
            error = f"{type(exc).__name__}: {exc}"[:300]
            rows += [row(span, n, error=error) for n in todo]
            continue
        for name in todo:
            first, n = ranges[name]
            video_path = layout.clip(sid, span, name, ".mp4")
            times_path = layout.clip(sid, span, name, ".parquet")
            fields = {
                "video": relative(layout, video_path),
                "timestamps": relative(layout, times_path),
                "thermistor": relative(layout, therm_path),
                "span_start": float(window["start"]),
                "span_end": float(window["end"]),
                "first_frame": first,
                "frames": n,
                "start": float(times[name][first]),
                "end": float(times[name][first + n - 1]),
                "rate_hz": float(videos[name].rate),
            }
            try:
                data = videos[name].cut(first, n)
                check(data, n)
                storage.write_bytes(video_path, data)
                write_times(times[name][first : first + n], times_path)
                if span == min(s["clips"]):  # the session's thumbnail, while in memory
                    storage.write_bytes(layout.thumbnail(sid, name), thumbnail(data))
            except Exception as exc:  # noqa: BLE001 -- recorded; retried next run
                fields["error"] = f"{type(exc).__name__}: {exc}"[:300]
                rows.append(row(span, name, **fields))
                continue
            rows.append(row(span, name, **fields, bytes=len(data)))
    return rows


def merge(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """*table* with *rows* replacing any earlier row of the same clip."""
    if table.empty:
        merged = rows
    else:
        again = table.set_index(KEY).index.isin(rows.set_index(KEY).index)
        merged = pd.concat([table[~again], rows], ignore_index=True)
    return merged.sort_values(KEY).reset_index(drop=True)


def write_list(spec: DatasetSpec, layout: Layout, table: pd.DataFrame) -> int:
    """``clips.toml``: the cut clips as a zephyr clip list, grouped by subject."""
    ok = table[table["error"].isna()]
    if ok.empty:
        return 0
    subject = subjects(layout)
    doc = tomlkit.document()
    for line in HEADER:
        doc.add(tomlkit.comment(line))
    doc.add(tomlkit.nl())
    for name, value in header(spec).items():
        doc.add(name, value)
    entries = tomlkit.aot()
    for clip in ok.itertuples():
        entry = tomlkit.table()
        entry.add("video", clip.video)
        entry.add("timestamps", clip.timestamps)
        entry.add("thermistor", clip.thermistor)
        entry.add("group", subject[clip.session_id])
        entries.append(entry)
    doc.add("clip", entries)
    validate(doc, layout)
    storage.write_text(layout.clip_list, tomlkit.dumps(doc))
    return len(ok)


def add_thumbnails(layout: Layout, table: pd.DataFrame, workers: int) -> int:
    """Each session's missing thumbnails (one per camera, the first frame of its first
    span), from the start of the stored clip; returns how many were made."""
    if table.empty:
        return 0
    ok = table[table["error"].isna()]
    ok = ok[ok["span"] == ok.groupby("session_id")["span"].transform("min")]
    have = storage.names(layout.thumbnail_dir, "*.png")
    missing = [
        c
        for c in ok.itertuples()
        if layout.thumbnail(c.session_id, c.camera).name not in have
    ]

    def one(clip) -> None:
        path = layout.root / clip.video
        with path.open("rb") as handle:
            start = handle.read(THUMBNAIL_BYTES)
        try:
            png = thumbnail(start)
        except (RuntimeError, ValueError):
            png = thumbnail(path.read_bytes())
        storage.write_bytes(layout.thumbnail(clip.session_id, clip.camera), png)

    thread_map(one, missing, workers)
    return len(missing)


def run(
    spec: DatasetSpec, workers: int = 8, limit: int | None = None, url_of=None
) -> pd.DataFrame:
    url_of = url_of or (lambda folder: f"{https(folder)}/video.mp4")
    layout = Layout(spec.root)
    results = json.loads(layout.results.read_text(encoding="utf-8"))
    by_id = {s["session_id"]: s for s in results["sessions"]}
    sessions = [by_id[sid] for sid in results["selected"][:limit]]
    table = (
        storage.read_parquet(layout.clip_table)
        if layout.clip_table.exists()
        else pd.DataFrame()
    )
    done = set()
    if not table.empty:
        ok = table[table["error"].isna()]
        done = set(zip(ok["session_id"], ok["span"], ok["camera"]))
    todo = [
        s
        for s in sessions
        if any(
            (s["session_id"], span, n) not in done
            for span in s["clips"]
            for n in spec.source.cameras
        )
    ]
    start, written = time.time(), 0
    for k in range(0, len(todo), BATCH):  # saved per batch: a stop loses little
        batch = todo[k : k + BATCH]
        found = thread_map(
            lambda s: one_session(spec, layout, s, done, url_of), batch, workers
        )
        rows = pd.DataFrame([r for rs in found for r in rs])
        written += int(rows.get("bytes", pd.Series(dtype=float)).fillna(0).sum())
        table = merge(table, rows)
        storage.write_frame(table, layout.clip_table)
        elapsed = time.time() - start
        print(
            f"clips: {k + len(batch)}/{len(todo)} sessions, "
            f"{written / 1e9:.1f} GB in {elapsed / 60:.0f} min "
            f"({written / 1e6 / max(elapsed, 1e-9):.0f} MB/s)",
            flush=True,
        )
    backfilled = add_thumbnails(layout, table, workers)
    listed = write_list(spec, layout, table)
    failed = 0 if table.empty else int(table["error"].notna().sum())
    print(
        f"clips: {listed} clips in {layout.clip_list.name}, {failed} failed, "
        f"{backfilled} thumbnails added from stored clips"
    )
    return table
