"""Predict breathing from videos with a trained checkpoint; no sensor is needed.

Each clip of the lists (boxed like any other) is preprocessed, into ``--cache`` or else a
temporary folder removed afterwards, and predicted end to end. ``<out>/<video
stem>.prediction.parquet`` holds one row per output sample: ``time``, the z-scored
``signal``, the onset head's ``onset`` probability and ``inhale`` (an inhale found
there, by the head). The lists' ``[preprocess]`` must be the checkpoint's recipe, and
every box the size it trained on (``clips scan --checkpoint`` writes such a list).

CLI
---
    zephyr predict --checkpoint best.pt --clips <list.toml>... --out predictions/
        [--cache <dir>] [--groups <g>...] [--device cpu] [--amp off] [--workers 8]
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ClipList, PreprocessParams, ResolvedClip, canonical, dump, load
from .evaluate import has_onset_head, load_checkpoint, zscore
from .infer import predict_clip
from .preprocess import preprocess_clips
from .quality_control.metrics import find_events
from .quality_control.reference import checkpoint_hash

SUFFIX = ".prediction.parquet"
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
        for path in (c.video, c.timestamps)
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


def _settings(out: Path, checkpoint: Path, sha: str) -> None:
    """Record the checkpoint; refuse a folder of another checkpoint's predictions."""
    settings = {"checkpoint": canonical(checkpoint), "checkpoint_sha256": sha}
    path = out / SETTINGS
    if path.is_file():
        if json.loads(path.read_text())["checkpoint_sha256"] != sha:
            raise SystemExit(
                f"{out} holds predictions of another checkpoint ({path}); use a new "
                "--out"
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
) -> list[Path]:
    """Write one prediction per clip that has none yet into *out*; return them all."""
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
    _settings(out, checkpoint, checkpoint_hash(checkpoint))
    todo = [c for c in clips if not (out / f"{c.video.stem}{SUFFIX}").is_file()]
    print(f"{len(clips)} clip(s), {len(clips) - len(todo)} already predicted")
    with tempfile.TemporaryDirectory(prefix="zephyr-predict-") as scratch:
        folder = cache or Path(scratch)
        entries = (
            preprocess_clips(todo, recipe, folder, workers=workers) if todo else []
        )
        for k, (clip, entry) in enumerate(zip(todo, entries, strict=True)):
            signal, onset = predict_clip(
                model,
                entry,
                mean,
                std,
                window=window,
                device=torch_device,
                frame_chunk=frame_chunk,
                amp_dtype=AMP[amp],
            )
            signal, times = zscore(signal), np.load(entry.times)
            found = find_events(times, signal, onset if head else None)
            inhale = np.zeros(len(times), bool)
            inhale[found.inhale] = True
            path = out / f"{clip.video.stem}{SUFFIX}"
            part = path.with_name(path.name + ".part")
            pd.DataFrame(
                {"time": times, "signal": signal, "onset": onset, "inhale": inhale}
            ).to_parquet(part, index=False)
            part.replace(path)
            print(f"[{k + 1}/{len(todo)}] {entry.clip_id} -> {path.name}", flush=True)
    return [out / f"{c.video.stem}{SUFFIX}" for c in clips]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Keep the preprocessed frames here (and reuse them); default: a "
        "temporary folder, removed afterwards.",
    )
    parser.add_argument("--groups", nargs="+", help="Keep only these groups.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=list(AMP))
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

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
    )
    print(f"{len(written)} prediction(s) in {args.out}")


if __name__ == "__main__":
    main()
