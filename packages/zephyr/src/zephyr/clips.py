"""Writing clip lists: ``zephyr clips scan`` and the box write-back annotate uses.

A clip list is a human-edited TOML file (see :mod:`zephyr.config`), so it is
written with ``tomlkit``, which keeps comments, ordering and layout across
rewrites.  Scanning writes the minimum -- the videos -- and leaves everything
else to be derived from their names.

CLI
---
    zephyr clips scan data/train --glob 'video_face_*.mp4' -o examples/clips.toml
"""

import argparse
import os
import re
from pathlib import Path

import tomlkit

from .config import ClipList
from .video import Box, probe_size


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
) -> None:
    """Write a clip list of the videos in *directory* matching any of *patterns*.

    *target_size* defaults to the first video's native size.  The list is
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
    path: Path, boxes: dict[Path, Box], target_size: tuple[int, int] | None = None
) -> None:
    """Write ``box =`` entries (and optionally ``target_size``) back into *path*.

    Only those keys are touched, so comments, ordering and every other field of
    the hand-edited file survive.  *boxes* is keyed by resolved video path.
    """
    path = path.resolve()
    doc = tomlkit.parse(path.read_text())
    if target_size is not None:
        doc["preprocess"]["target_size"] = list(target_size)
    for entry in doc["clip"]:
        video = (path.parent / str(entry["video"])).resolve()
        if video in boxes:
            entry["box"] = list(boxes[video])
    path.write_text(tomlkit.dumps(doc))


def update_clip_list(
    path: Path,
    *,
    events: dict[Path, tuple[str, dict[str, float]]] | None = None,
    excluded: dict[Path, list[tuple[float, float]]] | None = None,
    add: list[Path] | None = None,
    remove: set[Path] | None = None,
) -> None:
    """Edit *path* in place: set each clip's ``events = {method, params}`` (keyed
    by resolved video path), set each clip's ``excluded`` spans (removing the key when
    there are none), append the *add* videos and delete the *remove* ones.

    Like :func:`write_boxes` this keeps every comment and other field.  The result
    is validated before it is written.
    """
    path = path.resolve()
    doc = tomlkit.parse(path.read_text())
    clips = doc["clip"]
    gone = {Path(v).resolve() for v in (remove or ())}

    def resolved(entry) -> Path:
        return (path.parent / str(entry["video"])).resolve()

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
    args = parser.parse_args(argv)
    scan(
        args.directory,
        args.glob,
        args.out,
        target_size=tuple(args.target_size) if args.target_size else None,
        unlabelled=args.unlabelled,
    )
