"""``clips``: each selected span cut from its session's videos, without re-encoding.

Per span of a session ``raw/folds.parquet`` assigns (once it exists) and camera, the frames
from the keyframe at or before the span's start to the keyframe at or after its end are
stream-copied into a fragmented MP4 (so ffmpeg can write it to a pipe), checked (one
packet per frame, evenly spaced) and written to ``clips/<session_id>_<span>_<camera>.mp4``
with those frames' ``Time`` beside it as ``.parquet``. ``..._thermistor.parquet`` spans
the required cameras' clips ± ``[clips] thermistor_pad_s`` (a clean optional camera is cut
inside it); ``thumbnails/<session_id>_<camera>.png`` is the session's first frame.
``raw/clips.parquet`` lists them; ``clips.toml`` (group = session, boxes kept). Resumable.
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
from .lists import header, validate
from .remote import thread_map
from .videos import https

BATCH = 8  # sessions per save of clips.parquet
THUMBNAIL_BYTES = 8 << 20  # read of a clip's start that holds its first frame
WINDOW_S = 4.0  # first keyframe search window either side (doubled until found)
KEY = ["session_id", "span", "camera"]
HEADER = [
    "Clips cut from the selected spans of a zephyr-dataset folder: one entry per span",
    "and camera, written by `zephyr-dataset clips` (never edit by hand). `box` is each",
    "camera's crop, put there by `python -m zephyr.corpus.boxes publish`, kept by",
    '`clips`; cameras other than the face are `selection = "exclude"`, so zephyr uses',
    "the face only unless a fold selects them. `group` is the session (zephyr's",
    "recording); each session's subject and its set in every fold are in sessions.parquet.",
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


def one_session(spec, layout: Layout, s: dict, done: set, url_of, optional) -> list:
    """Cut a session's spans; one ``clips.parquet`` row per span and camera. *optional*
    holds the folders of its clean optional cameras, which are cut too but never set the
    clip thermistor's range (they must lie inside it)."""
    sid, required = s["session_id"], spec.source.cameras
    folders = {**s["cameras"], **optional}
    cameras = [*required, *optional]
    pad = spec.clips.thermistor_pad_s
    rows = []
    videos, times, thermistor = {}, {}, []

    def row(span, name, **fields):
        base = {"session_id": sid, "span": span, "camera": name, "error": None}
        return base | fields

    def video(name) -> Video:  # opened (ffprobe) only for what a span needs
        if name not in videos:
            videos[name] = Video(url_of(folders[name]))
        return videos[name]

    def stamps(name) -> np.ndarray:
        if name not in times:
            frame = storage.read_parquet(
                layout.timestamps(sid, name), columns=["ReferenceTime"]
            )
            times[name] = frame["ReferenceTime"].to_numpy()
        return times[name]

    for span, window in sorted(s["clips"].items()):
        todo = [n for n in cameras if (sid, span, n) not in done]
        if not todo:
            continue
        therm_path = layout.clip(sid, span, "thermistor", ".parquet")
        # Written with the required cameras' clips; an optional camera alone never
        # rewrites it, so its time range (and every target made from it) stays put.
        fresh = any(n in required for n in todo) or not therm_path.exists()
        try:
            need = [*required, *todo] if fresh else todo
            ranges = {
                n: frame_range(video(n), stamps(n), window["start"], window["end"])
                for n in dict.fromkeys(need)
            }
            if fresh:
                if not thermistor:
                    thermistor.append(storage.read_parquet(layout.thermistor(sid)))
                lo = min(stamps(n)[ranges[n][0]] for n in required) - pad
                hi = max(stamps(n)[sum(ranges[n]) - 1] for n in required) + pad
                inside = thermistor[0]["Time"].between(lo, hi)
                kept = thermistor[0][inside].reset_index(drop=True)
                storage.write_frame(kept, therm_path)
                covered = kept["Time"]
            else:
                covered = storage.read_parquet(therm_path, columns=["Time"])["Time"]
            covered = (float(covered.min()), float(covered.max()))
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
                "start": float(stamps(name)[first]),
                "end": float(stamps(name)[first + n - 1]),
                "rate_hz": float(videos[name].rate),
            }
            if not covered[0] <= fields["start"] <= fields["end"] <= covered[1]:
                fields["error"] = "the clip runs past its thermistor"
                rows.append(row(span, name, **fields))
                continue
            try:
                data = videos[name].cut(first, n)
                check(data, n)
                storage.write_bytes(video_path, data)
                write_times(stamps(name)[first : first + n], times_path)
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


def existing_boxes(layout: Layout) -> dict[str, list[int]]:
    """The ``box`` of each entry of the current ``clips.toml``, by its ``video``."""
    if not layout.clip_list.exists():
        return {}
    doc = tomlkit.parse(layout.clip_list.read_text(encoding="utf-8"))
    return {e["video"]: [int(v) for v in e["box"]] for e in doc["clip"] if "box" in e}


def write_list(spec: DatasetSpec, layout: Layout, table: pd.DataFrame) -> int:
    """``clips.toml``: the cut clips as a zephyr clip list, one group per session (the
    unit zephyr judges leakage on: test B holds out sessions of training animals)."""
    ok = table if table.empty else table[table["error"].isna()]
    if not ok.empty:  # cameras a review drops, until --prune deletes them
        ok = ok[~reviewed_out(spec, ok)]
    if ok.empty:
        if layout.clip_list.exists():  # every clip pruned: no list naming them
            storage.delete([layout.clip_list])
        return 0
    boxes = existing_boxes(layout)
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
        entry.add("group", clip.session_id)
        if clip.camera in spec.clips.exclude_cameras:
            entry.add("selection", "exclude")
        if clip.video in boxes:
            entry.add("box", boxes[clip.video])
        entries.append(entry)
    doc.add("clip", entries)
    validate(doc, layout.clip_list)
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


def reviewed_out(spec: DatasetSpec, table: pd.DataFrame) -> pd.Series:
    """Which rows are of a camera a ``[[review]]`` drops from its session."""
    dropped = spec.dropped_cameras()
    return pd.Series(
        [(r.session_id, r.camera) in dropped for r in table.itertuples()],
        index=table.index,
        dtype=bool,
    )


def prunable(spec: DatasetSpec, layout: Layout, assigned: set[str]) -> list:
    """The clips and thumbnails of sessions no set uses (also files a stopped batch
    left unrecorded) and of cameras a review drops."""
    dropped = spec.dropped_cameras()

    def gone(session: str, stem: str) -> bool:
        return session not in assigned or (session, stem.rsplit("_", 1)[-1]) in dropped

    return [
        layout.clip_dir / n
        for n in sorted(storage.names(layout.clip_dir))
        if gone(n.rsplit("_span", 1)[0], n.rsplit(".", 1)[0])
    ] + [
        layout.thumbnail_dir / n
        for n in sorted(storage.names(layout.thumbnail_dir, "*.png"))
        if gone(n.rsplit("_", 1)[0], n.rsplit(".", 1)[0])
    ]


def prune(
    spec: DatasetSpec,
    layout: Layout,
    table: pd.DataFrame,
    assigned: set[str],
    dry_run: bool = False,
) -> pd.DataFrame:
    """Delete :func:`prunable`'s files and drop their rows; needs delete rights. A dry
    run prints the files and deletes nothing."""
    gone = prunable(spec, layout, assigned)
    if dry_run:
        for path in gone:
            print(path)
        print(f"clips: --prune would delete {len(gone)} files (dry run)")
        return table
    storage.delete(gone)
    kept = (
        table
        if table.empty
        else table[table["session_id"].isin(assigned) & ~reviewed_out(spec, table)]
    )
    sessions = {
        p.name.rsplit("_span", 1)[0] for p in gone if p.parent == layout.clip_dir
    }
    print(f"clips: pruned {len(gone)} files of {len(sessions)} sessions")
    return kept.reset_index(drop=True)


def optional_folders(spec: DatasetSpec, layout: Layout) -> dict[str, dict[str, str]]:
    """Each session's optional cameras that ``videos`` found clean and no review drops,
    with their folders."""
    if not spec.source.optional_cameras or not layout.cameras.exists():
        return {}
    table = storage.read_parquet(layout.cameras)
    table = table[
        table["camera"].isin(spec.source.optional_cameras)
        & table["error"].isna()
        & table["clean"].fillna(False).astype(bool)
    ]
    out: dict[str, dict[str, str]] = {}
    dropped = spec.dropped_cameras()
    for r in table.itertuples():
        if (r.session_id, r.camera) not in dropped:
            out.setdefault(r.session_id, {})[r.camera] = r.folder
    return out


def run(
    spec: DatasetSpec,
    workers: int = 8,
    limit: int | None = None,
    url_of=None,
    prune_unassigned: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    url_of = url_of or (lambda folder: f"{https(folder)}/video.mp4")
    layout = Layout(spec.root)
    results = json.loads(layout.selection.read_text(encoding="utf-8"))
    by_id = {s["session_id"]: s for s in results["sessions"]}
    wanted = results["selected"]
    assigned = None
    if layout.folds.exists():  # only sessions some set uses
        folds = storage.read_parquet(layout.folds, columns=["session_id", "set"])
        assigned = set(folds.loc[folds["set"].notna(), "session_id"])
        wanted = [sid for sid in wanted if sid in assigned]
    sessions = [by_id[sid] for sid in wanted[:limit]]
    table = (
        storage.read_parquet(layout.clip_table)
        if layout.clip_table.exists()
        else pd.DataFrame()
    )
    if prune_unassigned:
        if assigned is None:
            raise SystemExit("clips: --prune needs raw/folds.parquet (run folds first)")
        if dry_run:
            return prune(spec, layout, table, assigned, dry_run=True)
        table = prune(spec, layout, table, assigned)
        storage.write_frame(table, layout.clip_table)
        write_list(spec, layout, table)
    done = set()
    if not table.empty:
        ok = table[table["error"].isna()]
        done = set(zip(ok["session_id"], ok["span"], ok["camera"]))
    optional = optional_folders(spec, layout)
    todo = [
        s
        for s in sessions
        if any(
            (s["session_id"], span, n) not in done
            for span in s["clips"]
            for n in [*spec.source.cameras, *optional.get(s["session_id"], {})]
        )
    ]
    start, written = time.time(), 0
    size = max(
        BATCH, workers
    )  # every worker busy; saved per batch: a stop loses little
    for k in range(0, len(todo), size):
        batch = todo[k : k + size]
        found = thread_map(
            lambda s: one_session(
                spec, layout, s, done, url_of, optional.get(s["session_id"], {})
            ),
            batch,
            workers,
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
