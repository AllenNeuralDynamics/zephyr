"""Writing and checking clip lists: ``zephyr clips scan``, ``zephyr clips check`` and
the box write-back annotate uses.

A clip list is a human-edited TOML file (see :mod:`zephyr.config`), written with
``tomlkit``, which keeps comments, ordering and layout across rewrites.  Scanning writes
the minimum (the videos, and ``[derive]`` for names other than the default); ``check``
reads every file a list names and reports, per clip, what would be refused.

CLI
---
    zephyr clips scan data/train --glob 'video_face_*.mp4' -o examples/clips.toml
    zephyr clips scan data --glob '*.mp4' -o list.toml --pattern '(?P<group>.+)'
        --timestamps '{stem}_times.parquet' --thermistor '{stem}_breath.parquet'
    zephyr clips check examples/clips.toml [--json report.json]
"""

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit

from .config import ClipList, Derive, ResolvedClip, load
from .signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from .video import Box, probe_size

SENSOR_MIN_HZ = 80.0
"""The breathing signal is low-passed at 40 Hz at its own rate, which needs more than
twice that (:func:`zephyr.signal.filter_sniff_signal`)."""


def natural_key(path: Path) -> list[int | str]:
    """Sort key putting ``part_2`` before ``part_10``."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def scan(
    directory: Path,
    patterns: list[str],
    out: Path,
    *,
    target_size: tuple[int, int] | None = None,
    unlabelled: bool = False,
    derive: Derive | None = None,
) -> None:
    """Write a clip list of the videos in *directory* matching any of *patterns*.

    *target_size* defaults to the first video's native size; *derive*, unless it is the
    default naming, is written as the list's ``[derive]``.  The list is
    validated before it is written, so a scan that would produce an unloadable
    file (say, thermistors that do not exist) fails instead of leaving it behind.
    """
    out = out.resolve()
    found = {
        video
        for pattern in patterns
        for video in Path(directory).resolve().glob(pattern)
    }
    videos = sorted(found, key=natural_key)
    if not videos:
        raise SystemExit(f"no videos match {patterns} in {directory}")
    if target_size is None:
        target_size = probe_size(videos[0])

    doc = tomlkit.document()
    doc.add(tomlkit.comment("Clip list written by `zephyr clips scan`."))
    doc.add(
        tomlkit.comment(
            "Only the videos are listed; timestamps, thermistor and group are "
            "derived from each file name (see [derive] in zephyr.config)."
        )
    )
    preprocess = tomlkit.table()
    preprocess.add("target_size", list(target_size))
    doc.add("preprocess", preprocess)
    if derive is not None and derive != Derive():
        table = tomlkit.table()
        for name, value in derive.model_dump(exclude_defaults=True).items():
            table.add(name, value)
        doc.add("derive", table)
    clips = tomlkit.aot()
    for video in videos:
        entry = tomlkit.table()
        entry.add("video", _relative(video, out.parent))
        if unlabelled:
            entry.add("thermistor", False)
        clips.append(entry)
    doc.add("clip", clips)

    try:
        ClipList.model_validate(doc.unwrap(), context={"base": out.parent})
    except ValueError as exc:
        hint = "" if unlabelled else "  (clips with no thermistor need --unlabelled)"
        raise SystemExit(f"the scanned list does not validate{hint}:\n{exc}") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tomlkit.dumps(doc))
    print(f"wrote {out}: {len(videos)} clips, target {target_size[0]}x{target_size[1]}")


def write_boxes(
    path: Path,
    boxes: dict[Path, Box],
    target_size: tuple[int, int] | None = None,
    *,
    clear: set[Path] = frozenset(),
) -> None:
    """Write ``box =`` entries (and optionally ``target_size``) back into *path*, and
    remove the box of every *clear* video.

    Only those keys are touched, so comments, ordering and every other field of
    the hand-edited file survive.  Videos are keyed by resolved path.
    """
    path = path.resolve()
    doc = tomlkit.parse(path.read_text())
    if target_size is not None:
        doc["preprocess"]["target_size"] = list(target_size)
    for entry in doc["clip"]:
        if "video" not in entry:  # thermistor only: never boxed
            continue
        video = (path.parent / str(entry["video"])).resolve()
        if video in boxes:
            entry["box"] = list(boxes[video])
        elif video in clear and "box" in entry:
            del entry["box"]
    path.write_text(tomlkit.dumps(doc))


def update_clip_list(
    path: Path,
    *,
    events: dict[Path, tuple[str, dict[str, float]]] | None = None,
    excluded: dict[Path, list[tuple[float, float]]] | None = None,
    included: dict[Path, list[tuple[float, float]]] | None = None,
    selection: dict[Path, str | None] | None = None,
    reviewed: dict[Path, bool] | None = None,
    add: list[Path] | None = None,
    remove: set[Path] | None = None,
) -> None:
    """Edit *path* in place: set each clip's ``events = {method, params}`` (keyed
    by resolved source path: the video, or a thermistor-only entry's thermistor), set each clip's ``excluded`` and ``included`` spans and ``selection`` mark (removing the key when
    there are none, or the clip is accepted), append the *add* videos and delete the *remove* ones.

    Like :func:`write_boxes` this keeps every comment and other field.  The result
    is validated before it is written.
    """
    path = path.resolve()
    doc = tomlkit.parse(path.read_text())
    clips = doc["clip"]
    gone = {Path(v).resolve() for v in (remove or ())}

    def resolved(entry) -> Path:
        """The entry's source: its video, else (thermistor only) its thermistor."""
        source = entry["video"] if "video" in entry else entry["thermistor"]
        return (path.parent / str(source)).resolve()

    for index in reversed(range(len(clips))):
        if resolved(clips[index]) in gone:
            del clips[index]
    present = {resolved(entry) for entry in clips}
    for video in add or []:
        video = Path(video).resolve()
        if video not in present:
            entry = tomlkit.table()
            entry.add("video", _relative(video, path.parent))
            clips.append(entry)
            present.add(video)
    for entry in clips:
        good = (included or {}).get(resolved(entry))
        if good is not None:
            if good:
                entry["included"] = [[float(a), float(b)] for a, b in good]
            elif "included" in entry:
                del entry["included"]
        seen = (reviewed or {}).get(resolved(entry))
        if seen:
            entry["reviewed"] = True
        elif seen is not None and "reviewed" in entry:
            del entry["reviewed"]
        chosen = (selection or {}).get(resolved(entry), "unchanged")
        if chosen is None and "selection" in entry:
            del entry["selection"]
        elif chosen in ("include", "exclude"):
            entry["selection"] = chosen
        spans = (excluded or {}).get(resolved(entry))
        if spans is not None:
            if spans:
                entry["excluded"] = [[float(a), float(b)] for a, b in spans]
            elif "excluded" in entry:
                del entry["excluded"]
        chosen = (events or {}).get(resolved(entry))
        if chosen is None:
            continue
        method, params = chosen
        inline = tomlkit.inline_table()
        inline["method"] = method
        if params:
            inline["params"] = tomlkit.inline_table()
            inline["params"].update({k: float(v) for k, v in params.items()})
        entry["events"] = inline
    ClipList.model_validate(doc.unwrap(), context={"base": path.parent})
    path.write_text(tomlkit.dumps(doc))


# ---------------------------------------------------------------------------
# Checking a list's files
# ---------------------------------------------------------------------------


@dataclass
class ClipCheck:
    """What ``check`` found for one clip: *problems* would stop preprocessing or
    training; *facts* are what was measured (rates, frames, overlap)."""

    name: str
    group: str
    problems: list[str] = field(default_factory=list)
    facts: dict[str, float | int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def count_frames(video: Path) -> int:
    """Frames in *video*'s first stream, counted by demuxing (no decode)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return int(out.strip().strip(","))


def _rate(times: np.ndarray) -> float:
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return (len(times) - 1) / span if span > 0 else 0.0


def _read_times(path: Path, columns: tuple[str, ...], found: "ClipCheck", what: str):
    """The ``Time`` column of parquet *path*, or ``None`` after noting why not."""
    try:
        table = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - reported per clip
        found.problems.append(f"{what} {path.name} cannot be read: {exc}")
        return None
    absent = [c for c in columns if c not in table.columns]
    if absent:
        found.problems.append(
            f"{what} {path.name} has no column {absent} (it has {list(table.columns)})"
        )
        return None
    times = table[TIME_COLUMN].to_numpy(np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        found.problems.append(f"{what} {path.name}: Time must increase, row by row")
        return None
    return times


def check_clip(clip: ResolvedClip, select_fs: float) -> ClipCheck:
    """Read *clip*'s files and say what preprocessing or training would refuse."""
    found = ClipCheck(clip.name, clip.group)
    files = (
        ("video", clip.video),
        ("timestamps", clip.timestamps),
        ("thermistor", clip.thermistor),
    )
    found.problems += [
        f"{what} {path} does not exist"
        for what, path in files
        if path is not None and not path.is_file()
    ]
    video_times = None
    if clip.timestamps is not None and clip.timestamps.is_file():
        video_times = _read_times(clip.timestamps, (TIME_COLUMN,), found, "timestamps")
    if video_times is not None:
        camera_hz = _rate(video_times)
        found.facts |= {
            "timestamps": len(video_times),
            "camera_hz": round(camera_hz, 2),
            "duration_s": round(float(video_times[-1] - video_times[0]), 2),
        }
        if camera_hz < select_fs:
            found.problems.append(
                f"the camera runs at {camera_hz:.1f} Hz, below select_fs "
                f"{select_fs:g} Hz: lower select_fs in [preprocess]"
            )
        if clip.video is not None and clip.video.is_file():
            try:
                frames = count_frames(clip.video)
            except (subprocess.CalledProcessError, ValueError):
                found.problems.append(f"video {clip.video.name} cannot be read")
            else:
                found.facts["frames"] = frames
                if frames != len(video_times):
                    found.problems.append(
                        f"the video has {frames} frames but its timestamps "
                        f"{len(video_times)} rows: one row per frame is needed"
                    )
    if clip.thermistor is not None and clip.thermistor.is_file():
        sensor = _read_times(
            clip.thermistor, (TIME_COLUMN, BREATHING_SIGNAL_COLUMN), found, "thermistor"
        )
        if sensor is not None:
            sensor_hz = _rate(sensor)
            found.facts["sensor_hz"] = round(sensor_hz, 1)
            if sensor_hz <= SENSOR_MIN_HZ:
                found.problems.append(
                    f"the breathing sensor is sampled at {sensor_hz:.1f} Hz; it "
                    f"needs more than {SENSOR_MIN_HZ:g} Hz"
                )
            if video_times is not None:
                overlap = min(sensor[-1], video_times[-1]) - max(
                    sensor[0], video_times[0]
                )
                found.facts["overlap_s"] = round(float(max(overlap, 0.0)), 2)
                if overlap <= 0:
                    found.problems.append(
                        "the sensor and the video share no time: are both Time "
                        "columns on the same clock?"
                    )
    if clip.video is not None and clip.box is None:
        found.problems.append("no crop box: place one with `zephyr annotate`")
    return found


def check(path: Path) -> list[ClipCheck]:
    """Check every clip in use of the list at *path*.  The list itself must load; its
    files need not exist (a missing one is reported on its clip)."""
    clip_list = load(ClipList, path, check_files=False)
    select_fs = clip_list.preprocess.select_fs
    return [check_clip(clip, select_fs) for clip in clip_list.resolve()]


def print_checks(checks: list[ClipCheck]) -> None:
    for c in checks:
        facts = "  ".join(f"{k} {v}" for k, v in c.facts.items())
        print(f"{'ok ' if c.ok else 'BAD'} {c.name}  [{c.group}]  {facts}")
        for problem in c.problems:
            print(f"      - {problem}")
    bad = sum(not c.ok for c in checks)
    print(f"{len(checks) - bad} of {len(checks)} clip(s) ok")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create and maintain clip lists.")
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan", help="List the videos in a directory.")
    scan_parser.add_argument("directory", type=Path)
    scan_parser.add_argument(
        "--glob",
        action="append",
        required=True,
        help="Video file pattern, e.g. 'video_face_*.mp4'; repeat to combine.",
    )
    scan_parser.add_argument("-o", "--out", type=Path, required=True)
    scan_parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Working frame size; defaults to the first video's native size.",
    )
    scan_parser.add_argument(
        "--unlabelled",
        action="store_true",
        help="Mark every clip `thermistor = false` (inference only).",
    )
    scan_parser.add_argument(
        "--pattern", help="[derive] pattern: a regex over the video stem with a group."
    )
    scan_parser.add_argument("--timestamps", help="[derive] timestamps template.")
    scan_parser.add_argument("--thermistor", help="[derive] thermistor template.")
    check_parser = sub.add_parser(
        "check", help="Read every file of a list and report what would be refused."
    )
    check_parser.add_argument("clips", type=Path)
    check_parser.add_argument("--json", type=Path, help="Also write the report here.")
    sub.add_parser(
        "select",
        help="Suggest clips and spans to use (see zephyr.session_selection).",
        add_help=False,
    )
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "select":
        from .session_selection import main as select

        select(argv[1:])
        return
    args = parser.parse_args(argv)
    if args.command == "check":
        try:
            checks = check(args.clips)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print_checks(checks)
        if args.json:
            args.json.write_text(json.dumps([asdict(c) for c in checks], indent=2))
        if any(not c.ok for c in checks):
            raise SystemExit(1)
        return
    derive = {
        name: value
        for name in ("pattern", "timestamps", "thermistor")
        if (value := getattr(args, name)) is not None
    }
    try:
        rule = Derive(**derive) if derive else None
    except ValueError as exc:
        raise SystemExit(f"[derive]: {exc}") from None
    scan(
        args.directory,
        args.glob,
        args.out,
        target_size=tuple(args.target_size) if args.target_size else None,
        unlabelled=args.unlabelled,
        derive=rule,
    )
