"""Input occlusion importance: where in the crop does each channel matter?

One patch of one channel is hidden for the whole segment by replacing it with the
training mean (zero after normalisation, i.e. "nothing here"), the network is re-run,
and the drop in event F1 (mean of inhale and exhale) against the unoccluded baseline
is recorded. A pixel's value is the mean drop over all patches covering it. Inference
only, through a wrapper around the network: one ``(96, 96)`` map per channel per clip,
written to one ``.npz``.

CLI
---
    zephyr-benchmarks occlusion --checkpoint <best.pt> --clips <list.toml>...
        --cache <dir> --out <file.npz> [--patch 16 --stride 8 --window-s 60]
"""

import argparse
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from zephyr import features
from zephyr.config import ClipList, load
from zephyr.evaluate import load_checkpoint, refuse_leaked, score_entry
from zephyr.features import ClipEntry
from zephyr.infer import INPUT_MARGIN, predict_clip
from zephyr.signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

Hidden = tuple[int, int, int]
"""``(channel, y, x)`` of the patch to hide."""
Score = Callable[[Hidden | None], float]
"""Event F1 of the network with a patch hidden (``None``: nothing hidden)."""


class Occluded(nn.Module):
    """A network that sees its input with one patch of one channel set to the mean.

    The input is already normalised, so the training mean is zero.  Everything
    :func:`zephyr.infer.predict_clip` reads from a model is passed through.
    """

    def __init__(self, model: nn.Module, patch: int) -> None:
        super().__init__()
        self.model = model
        self.patch = patch
        self.hidden: Hidden | None = None
        self.channels = model.channels
        self.receptive_field = model.receptive_field

    def forward(self, features: torch.Tensor, *args, **kwargs):
        """*features* is ``(batch, time, channel, height, width)``."""
        if self.hidden is not None:
            channel, y, x = self.hidden
            features = features.clone()
            features[:, :, channel, y : y + self.patch, x : x + self.patch] = 0
        return self.model(features, *args, **kwargs)


def patch_origins(size: int, patch: int, stride: int) -> list[int]:
    """Top-left coordinates along one axis; the last patch is flush with the edge."""
    origins = list(range(0, size - patch + 1, stride))
    if origins[-1] != size - patch:
        origins.append(size - patch)
    return origins


def occlusion_maps(
    score: Score, channels: list[str], *, patch: int, stride: int, size: int = 96
) -> tuple[float, dict[str, np.ndarray]]:
    """Baseline score and, per channel, the mean score drop of patches covering a pixel."""
    base = score(None)
    origins = patch_origins(size, patch, stride)
    maps = {}
    for channel, name in enumerate(channels):
        total = np.zeros((size, size))
        count = np.zeros((size, size))
        for y in origins:
            for x in origins:
                drop = base - score((channel, y, x))
                total[y : y + patch, x : x + patch] += drop
                count[y : y + patch, x : x + patch] += 1
        maps[name] = total / count
    return base, maps


def head_entry(entry: ClipEntry, seconds: float, directory: Path) -> ClipEntry:
    """A copy of *entry* holding only its first *seconds*, written under *directory*."""
    times = np.load(entry.times)
    n_out = int(np.searchsorted(times, times[0] + seconds))
    frame_times = np.load(entry.frame_times)
    n_in = min(
        len(frame_times),
        int(np.searchsorted(frame_times, times[n_out - 1])) + 1 + INPUT_MARGIN,
    )
    files = {}
    for name, array in {
        "features": np.load(entry.features, mmap_mode="r")[:n_in],
        "frame_times": frame_times[:n_in],
        "times": times[:n_out],
    }.items():
        files[name] = directory / f"{name}.npy"
        np.save(files[name], array)
    return replace(entry, n_frames=n_in, n_output=n_out, **files)


def clip_scorer(
    model: Occluded, mean, std, entry: ClipEntry, device, amp_dtype, *, frame_chunk=256
) -> Score:
    """Event F1 (mean of inhale and exhale F1) of the network on *entry*."""
    times = np.load(entry.times)

    def score(hidden: Hidden | None) -> float:
        model.hidden = hidden
        signal, _ = predict_clip(
            model,
            entry,
            mean,
            std,
            device=device,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        predicted = pd.DataFrame(
            {
                TIME_COLUMN: times.astype(np.float64),
                BREATHING_SIGNAL_COLUMN: signal.astype(np.float64),
            }
        )
        result = score_entry(entry, predicted)
        return (result.inhale_f1 + result.exhale_f1) / 2

    return score


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Input occlusion importance maps.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--clips",
        type=Path,
        nargs="+",
        required=True,
        help="Clip list(s); each is a group named after its file stem.",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="The .npz to write.")
    parser.add_argument("--patch", type=int, default=16, help="Patch side, pixels.")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride, pixels.")
    parser.add_argument(
        "--window-s", type=float, default=60.0, help="Seconds from each clip's start."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    network, mean, std, state = load_checkpoint(args.checkpoint, device)
    model = Occluded(network, args.patch)
    channels = list(model.channels.names)

    lists = {path.stem: load(ClipList, path) for path in args.clips}
    recipes = {lst.preprocess for lst in lists.values()}
    if len(recipes) != 1:
        raise SystemExit("the clip lists disagree on [preprocess]")
    recipe = recipes.pop()
    clips = {n: [c for c in lst.resolve() if c.labelled] for n, lst in lists.items()}
    refuse_leaked([state], [c for group in clips.values() for c in group])

    names, groups, bases = [], [], []
    channel_maps: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    for group, group_clips in clips.items():
        for entry in features.require(group_clips, recipe, args.cache):
            started = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                head = head_entry(entry, args.window_s, Path(tmp))
                score = clip_scorer(model, mean, std, head, device, amp_dtype)
                base, maps = occlusion_maps(
                    score, channels, patch=args.patch, stride=args.stride
                )
            names.append(entry.clip_id)
            groups.append(group)
            bases.append(base)
            for name, map_ in maps.items():
                channel_maps[name].append(map_)
            peaks = "; ".join(f"{k} max drop {v.max():.3f}" for k, v in maps.items())
            print(
                f"[{time.time() - started:.0f}s] {entry.clip_id}: baseline event-F1 "
                f"{base:.3f}; {peaks}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        bases=np.array(bases),
        clips=np.array(names),
        groups=np.array(groups),
        **{name: np.stack(maps) for name, maps in channel_maps.items()},
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
