"""Predict breathing from videos with a trained checkpoint; no sensor is needed.

Each clip (of the lists, or the one ``--video``) is preprocessed, into ``--cache`` or a
temporary folder, and predicted end to end into ``<out>/<video stem>.prediction.parquet``:
per output sample ``time``, ``frame`` (the video frame there), the z-scored ``signal``,
the head's ``onset`` probability and ``inhale`` (at the checkpoint's onset threshold
unless ``--head-threshold``). Lists must be on the checkpoint's recipe and crop size
(``clips scan --checkpoint``); ``--video`` writes its own, ``<stem>.clip.toml``.

CLI
---
    zephyr predict --checkpoint best.pt --clips <list.toml>... --out predictions/
        [--cache <dir>] [--groups <g>...] [--restart] [--device cpu] [--amp off]
        [--head-threshold 0.6]
    zephyr predict --checkpoint best.pt --video session.mp4 --frame-rate 240
        --box-gui | --box X Y | --box-from <stem>.clip.toml  [--box-size S] --out dir
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit
import torch

from .config import ClipList, PreprocessParams, ResolvedClip, canonical, dump, load
from .evaluate import (
    HEAD_THRESHOLD,
    has_onset_head,
    head_threshold,
    load_checkpoint,
    zscore,
)
from .infer import predict_clip
from .preprocess import frame_at, preprocess_clips
from .progress import Progress
from .quality_control.metrics import find_events
from .quality_control.reference import checkpoint_hash
from .video import clamp_box, probe_size

SUFFIX = ".prediction.parquet"
RECORD = ".clip.toml"
"""Suffix of the one-clip list ``--video`` writes into ``--out``: what was predicted."""
SETTINGS = "predictions.json"
"""The checkpoint a folder's predictions came from, written once into it."""
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}


def recipe_mismatch(state: dict, recipe: PreprocessParams) -> list[str]:
    """The ``[preprocess]`` settings in which *recipe* differs from the checkpoint's
    (none for a checkpoint that does not record its recipe)."""
    trained = state.get("feature_config") or {}
    return [
        f"{key} {value} (trained with {trained[key]})"
        for key, value in dump(recipe).items()
        if key in trained and _plain(trained[key]) != _plain(value)
    ]


def _plain(value):
    return list(value) if isinstance(value, list | tuple) else value


def crop_size(state: dict) -> tuple[int, int] | None:
    """The ``(width, height)`` of the crops a checkpoint's network trained on: the
    one box size of the clips its run lists as training (None when they differ, or
    the checkpoint does not say)."""
    lists = (state.get("config") or {}).get("train_clip_lists") or []
    sizes = {
        (int(clip["box"][2]), int(clip["box"][3]))
        for clip_list in lists
        for clip in clip_list.get("clip", [])
        if clip.get("box") and clip.get("selection") != "exclude"
    }
    return sizes.pop() if len(sizes) == 1 else None


def read_checkpoint(path: Path) -> dict:
    """A checkpoint's saved state, on the CPU (its recipe, crop and lineage)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def select_clips(lists: list[ClipList], groups: list[str] | None) -> list[ResolvedClip]:
    """The clips to predict: every clip in use, or those of *groups*; a thermistor
    is never needed, so one that is not at hand is dropped from the clip."""
    clips = [c for lst in lists for c in lst.resolve()]
    if groups is not None:
        unknown = sorted(set(groups) - {c.group for c in clips})
        if unknown:
            raise SystemExit(f"--groups {unknown} are in none of the clip lists")
        clips = [c for c in clips if c.group in groups]
    missing = [
        str(path)
        for c in clips
        for path in (c.video, c.timestamps if c.frame_rate is None else c.video)
        if path is None or not path.is_file()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} video or timestamp file(s) missing: {missing}"
        )
    unboxed = [c.name for c in clips if c.box is None]
    if unboxed:
        raise SystemExit(f"{len(unboxed)} clip(s) have no box: {unboxed}")
    stems = [c.video.stem for c in clips]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    if repeated:
        raise SystemExit(
            f"videos share a file name, so their outputs would: {repeated}"
        )
    return [
        c
        if c.thermistor is None or c.thermistor.is_file()
        else c.model_copy(update={"thermistor": None})
        for c in clips
    ]


def _settings(out: Path, checkpoint: Path, sha: str, threshold: dict | None) -> None:
    """Record the checkpoint and the onset threshold its inhales were found at; refuse
    a folder of another checkpoint's predictions, or of another threshold's."""
    settings = {
        "checkpoint": canonical(checkpoint),
        "checkpoint_sha256": sha,
        "head_threshold": threshold,
    }
    path = out / SETTINGS
    if path.is_file():
        before = json.loads(path.read_text())
        if before["checkpoint_sha256"] != sha:
            raise SystemExit(
                f"{out} holds predictions of another checkpoint ({path}); use a new "
                "--out"
            )
        if before.get("head_threshold") != threshold:
            raise SystemExit(
                f"{out} holds inhales found at another onset threshold ({path}); use "
                "a new --out, or --restart"
            )
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


def predict(
    checkpoint: Path,
    clips: list[ResolvedClip],
    recipe: PreprocessParams,
    out: Path,
    *,
    cache: Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp: str = "bf16",
    window: int = 1024,
    frame_chunk: int = 256,
    workers: int = 8,
    restart: bool = False,
    threshold: float | None = None,
) -> list[Path]:
    """Write one prediction per clip that has none yet into *out* (with *restart*,
    first removing the ones there and the checkpoint they came from, so new crops,
    a new list or another checkpoint are predicted again); return them all. The
    head's inhales are found at *threshold*, else the checkpoint's own."""
    torch_device = torch.device(device)
    model, mean, std, state = load_checkpoint(checkpoint, torch_device)
    differs = recipe_mismatch(state, recipe)
    if differs:
        raise SystemExit(
            f"the clip lists' [preprocess] is not {checkpoint.name}'s: {differs}"
        )
    crop = crop_size(state)
    wrong = [c.name for c in clips if crop is not None and tuple(c.box[2:]) != crop]
    if wrong:
        raise SystemExit(
            f"{checkpoint.name} trained on {crop[0]}x{crop[1]} crops; these clips' "
            f"boxes are another size: {wrong}"
        )
    head = has_onset_head([state])
    chosen = head_threshold(state, threshold) if head else None
    if restart:  # the folder's predictions, and whose they were
        for old in [*out.glob(f"*{SUFFIX}"), out / SETTINGS]:
            old.unlink(missing_ok=True)
        print(f"{out}: earlier predictions removed", flush=True)
    _settings(out, checkpoint, checkpoint_hash(checkpoint), chosen)
    todo = [c for c in clips if not (out / f"{c.video.stem}{SUFFIX}").is_file()]
    print(f"{len(clips)} clip(s), {len(clips) - len(todo)} already predicted")
    with tempfile.TemporaryDirectory(prefix="zephyr-predict-") as scratch:
        folder = cache or Path(scratch)
        entries = (
            preprocess_clips(todo, recipe, folder, workers=workers) if todo else []
        )
        for k, (clip, entry) in enumerate(zip(todo, entries, strict=True)):
            shown = Progress(f"predicting {entry.clip_id}", 1, "windows")

            def counted(done: int, total: int, shown=shown) -> None:
                shown.total = total
                shown.add(1)

            signal, onset = predict_clip(
                model,
                entry,
                mean,
                std,
                window=window,
                device=torch_device,
                frame_chunk=frame_chunk,
                amp_dtype=AMP[amp],
                on_window=counted,
            )
            shown.close()
            signal, times = zscore(signal), np.load(entry.times)
            frame = frame_at(clip, times)  # the video's own frame
            found = find_events(
                times,
                signal,
                onset if head else None,
                chosen["value"] if chosen else HEAD_THRESHOLD,
            )
            inhale = np.zeros(len(times), bool)
            inhale[found.inhale] = True
            path = out / f"{clip.video.stem}{SUFFIX}"
            part = path.with_name(path.name + ".part")
            pd.DataFrame(
                {
                    "time": times,
                    "frame": frame,
                    "signal": signal,
                    "onset": onset,
                    "inhale": inhale,
                }
            ).to_parquet(part, index=False)
            part.replace(path)
            print(f"[{k + 1}/{len(todo)}] {entry.clip_id} -> {path.name}", flush=True)
    return [out / f"{c.video.stem}{SUFFIX}" for c in clips]


def video_frame(
    native: tuple[int, int], target: tuple[int, int], crop: int, box_size: int | None
) -> tuple[tuple[int, int], int]:
    """The working frame of a video of *native* size whose box is *box_size* native
    pixels (default: the share of the frame a *crop* on the checkpoint's *target*
    covers), scaled so that box is the *crop*; and that box size, in native pixels."""
    if box_size is None:
        box_size = round(crop * native[0] / target[0])
    if not 0 < box_size <= min(native):
        raise SystemExit(
            f"--box-size {box_size} must fit the {native[0]}x{native[1]} video"
        )
    scale = crop / box_size
    return (round(native[0] * scale), round(native[1] * scale)), box_size


def _write_list(doc, path: Path, video: Path) -> None:
    """Validate the one-clip list *doc*, then write it to *path*."""
    try:
        ClipList.model_validate(doc.unwrap(), context={"base": path.parent})
    except ValueError as exc:
        raise SystemExit(f"{video}: {exc}") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc))


def write_video_list(
    path: Path,
    video: Path,
    checkpoint: Path,
    *,
    frame_rate: float | None,
    corner: tuple[int, int] | None,
    box_size: int | None,
) -> None:
    """Write the one-clip list *path* that predicts *video* with *checkpoint*: its
    recipe, the video on *frame_rate* (else its ``{stem}.parquet`` timestamps), and a
    box at native-pixel *corner* (default: centred) of *box_size* native pixels,
    which becomes the checkpoint's crop (:func:`video_frame`)."""
    from .clips import EACH_VIDEO, checkpoint_recipe

    recipe, crop = checkpoint_recipe(checkpoint)
    if crop is None or crop[0] != crop[1]:
        raise SystemExit(f"{checkpoint} does not say one square crop size")
    native = probe_size(video)
    target = tuple(recipe["target_size"])
    frame, box_size = video_frame(native, target, crop[0], box_size)
    scale = frame[0] / native[0]
    if corner is None:
        corner = ((native[0] - box_size) // 2, (native[1] - box_size) // 2)
    box = clamp_box((round(corner[0] * scale), round(corner[1] * scale), *crop), *frame)
    doc = tomlkit.document()
    doc.add(tomlkit.comment(f"zephyr predict --video {video.name} with {checkpoint}"))
    doc.add(
        tomlkit.comment(
            f"Its box is {box_size} px of the {native[0]}x{native[1]} video, scaled to "
            f"{frame[0]}x{frame[1]} so it is the network's {crop[0]}x{crop[1]}."
        )
    )
    preprocess = tomlkit.table()
    for key, value in recipe.items():
        preprocess.add(key, list(value) if isinstance(value, list | tuple) else value)
    doc.add("preprocess", preprocess)
    derive = tomlkit.table()
    derive.add("pattern", EACH_VIDEO)
    derive.add("thermistor", "{stem}.thermistor.parquet")  # never read: unlabelled
    doc.add("derive", derive)
    entry = tomlkit.table()
    entry.add("video", Path(video).resolve().as_posix())
    if frame_rate is not None:
        entry.add("frame_rate", frame_rate)
    entry.add("thermistor", False)
    if frame != target:
        entry.add("frame", list(frame))
    entry.add("box", list(box))
    clips = tomlkit.aot()
    clips.append(entry)
    doc.add("clip", clips)
    _write_list(doc, path, video)
    print(
        f"{video.name}: box {box_size} px at {list(corner)} of {native[0]}x"
        f"{native[1]} -> {crop[0]}x{crop[1]} at {list(box[:2])} of "
        f"{frame[0]}x{frame[1]} ({path.name})",
        flush=True,
    )


def reuse_video_list(source: Path, path: Path, video: Path, frame_rate) -> None:
    """Write *path* for *video* with the frame, box and (unless *frame_rate*) frame
    rate of the one clip in use of *source* (an earlier ``<stem>.clip.toml``)."""
    clips = load(ClipList, source, check_files=False).resolve()
    if len(clips) != 1 or clips[0].box is None:
        raise SystemExit(f"--box-from {source}: needs a list of one boxed clip")
    doc = tomlkit.parse(source.read_text())
    (entry,) = [e for e in doc["clip"] if e.get("selection") != "exclude"]
    entry["video"] = Path(video).resolve().as_posix()
    for key in ("timestamps", "group"):  # the source video's own
        if key in entry:
            del entry[key]
    if frame_rate is not None:  # else the source's, if it has one
        entry["frame_rate"] = frame_rate
    _write_list(doc, path, video)
    print(f"{video.name}: the crop of {source} ({path.name})", flush=True)


def video_list(args) -> Path:
    """The list ``--video`` is predicted from, written into ``--out`` from the crop
    flags. A video predicted there already is refused, or with ``--restart`` its
    prediction (only its own: the folder may hold others) is removed first."""
    video = args.video
    if not video.is_file():
        raise SystemExit(f"no video {video}")
    if args.box_from is not None and args.box_size is not None:
        raise SystemExit("--box-size goes with --box or --box-gui; --box-from has one")
    predicted = args.out / f"{video.stem}{SUFFIX}"
    if predicted.is_file():
        if not args.restart:
            raise SystemExit(
                f"{predicted} exists: add --restart to predict {video.name} again"
            )
        predicted.unlink()
        args.restart = False  # the folder's other predictions stay
    path = args.out / f"{video.stem}{RECORD}"
    before = path.read_text() if path.is_file() else None
    frame_rate = args.frame_rate
    if frame_rate is None and before is not None:  # the one it was predicted on
        frame_rate = load(ClipList, path, check_files=False).resolve()[0].frame_rate
    source = args.box_from
    if source is None and args.box is None and args.box_size is None:
        source = path if before is not None else None  # its crop here, again
    if source is None and args.box is None and not args.box_gui:
        raise SystemExit(
            "--video needs a crop: --box X Y (its own pixels), --box-gui to place "
            "it, or --box-from an earlier <stem>.clip.toml"
        )
    if source is not None:
        with tempfile.TemporaryDirectory() as scratch:  # *path* may be the source
            copy = Path(scratch) / source.name
            copy.write_text(source.read_text())
            reuse_video_list(copy, path, video, frame_rate)
    else:
        write_video_list(
            path,
            video,
            args.checkpoint,
            frame_rate=frame_rate,
            corner=tuple(args.box) if args.box else None,
            box_size=args.box_size,
        )
    if args.box_gui:
        from .gui import need_ui

        need_ui()
        from .gui.boxes import place_crop

        place_crop(path, args.checkpoint)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    given = parser.add_mutually_exclusive_group(required=True)
    given.add_argument("--clips", type=Path, nargs="+")
    given.add_argument(
        "--video",
        type=Path,
        help="One video instead of lists: its list (<stem>.clip.toml in --out) is "
        "made here, the crop from --box, --box-from or --box-gui.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frame-rate",
        type=float,
        metavar="HZ",
        help="With --video: it has no timestamps, frame i is at i / HZ s (assumes "
        "no dropped frames); default: its {stem}.parquet timestamps.",
    )
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument(
        "--box",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="With --video: the crop's top-left corner, in the video's own pixels.",
    )
    crop.add_argument(
        "--box-from",
        type=Path,
        metavar="LIST",
        help="With --video: the crop (and frame) of this one-clip list, e.g. an "
        "earlier --out's <stem>.clip.toml.",
    )
    crop.add_argument(
        "--box-gui",
        action="store_true",
        help="With --video: place the crop in the app's Boxes window (needs the ui "
        "extra); prediction starts when it is closed.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        metavar="S",
        help="With --box or --box-gui: the crop's side in the video's own pixels; "
        "the video is scaled so it becomes the network's crop (default: the share "
        "of the frame the network's crops covered).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Keep the preprocessed frames here (and reuse them); default: a "
        "temporary folder, removed afterwards.",
    )
    parser.add_argument("--groups", nargs="+", help="Keep only these groups.")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Remove the predictions already in --out first, and predict every clip.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=list(AMP))
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--head-threshold",
        type=float,
        help="Onset threshold for the head's inhales (default: the checkpoint's "
        f"own, else {HEAD_THRESHOLD}); predictions.json records the value used.",
    )
    args = parser.parse_args(argv)

    if args.video is not None:
        args.clips = [video_list(args)]
    else:
        for flag in ("frame_rate", "box", "box_from", "box_size"):
            if getattr(args, flag) is not None:
                raise SystemExit(f"--{flag.replace('_', '-')} needs --video")
        if args.box_gui:
            raise SystemExit("--box-gui needs --video")
    lists = [load(ClipList, path, check_files=False) for path in args.clips]
    recipes = {lst.preprocess for lst in lists}
    if len(recipes) != 1:
        raise SystemExit("the clip lists disagree on [preprocess]")
    written = predict(
        args.checkpoint,
        select_clips(lists, args.groups),
        recipes.pop(),
        args.out,
        cache=args.cache,
        device=args.device,
        amp=args.amp,
        window=args.infer_window,
        frame_chunk=args.frame_chunk,
        workers=args.workers,
        restart=args.restart,
        threshold=args.head_threshold,
    )
    print(f"{len(written)} prediction(s) in {args.out}")


if __name__ == "__main__":
    main()
