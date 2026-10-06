"""Inference time per clip, for every method in the benchmark.

Timed: wall-clock seconds from the preprocessed crops on disk to the finished 60 Hz
trace. Not timed: decoding, preprocessing (optical-flow cost is estimated by
:func:`estimate_flow_preprocessing`) and scoring.

Fair: the same clips for every method, one untimed warm-up clip outside them, every file
read into the OS cache first. Networks run on GPU (bf16) and CPU (fp32); pixel and
Facemap-style methods are numpy. Facemap weights are random of the right shape: cost
depends on shape, not values.
"""

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import baselines, common, facemap_ridge

CLIP_FS = 60.0
"""Output-grid rate used to turn a clip's sample count into seconds of recording."""


def summarise(times: list[float]) -> dict:
    values = np.asarray(times, dtype=float)
    return {
        "n": len(values),
        "mean_s": float(values.mean()),
        "sd_s": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median_s": float(np.median(values)),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
    }


def pick_clips(entries: list, n: int) -> list:
    """*n* evenly spaced clips from *entries*, in list order."""
    ordered = list(entries)
    if n >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(positions)]


def warm_file_cache(entries: list, block: int = 64 << 20) -> int:
    """Read each distinct ``features`` file once; returns the bytes read."""
    total = 0
    for path in dict.fromkeys(Path(e.features) for e in entries):
        with open(path, "rb") as handle:
            while chunk := handle.read(block):
                total += len(chunk)
    return total


def pick_warmup(entries: list, timed: list):
    """A clip that is *not* in *timed*, for the warm-up call.

    Warming up on a timed clip would leave its file in the operating system's cache
    and make that one clip look faster than the rest (3.4 s against 6.4 s for the
    pixel flow method, whose cost is partly disk reads).
    """
    ordered = list(entries)
    return next((e for e in ordered if e not in timed), ordered[0])


def time_method(fn: Callable, entries: list, warmup=None) -> dict:
    """Time ``fn(entry)`` per clip after one untimed warm-up call on *warmup*."""
    fn(entries[0] if warmup is None else warmup)
    clips = []
    for entry in entries:
        started = time.perf_counter()
        fn(entry)
        clips.append(
            {"clip_id": entry.clip_id, "seconds": time.perf_counter() - started}
        )
    return {"clips": clips, "summary": summarise([c["seconds"] for c in clips])}


def pixel_method(method: str, fs: float, bin_factor: int) -> Callable:
    def infer(entry):
        trace = baselines.pixel_trace(method, entry, fs, bin_factor)
        return common.blind_polarity(common.to_output_grid(entry, trace))

    return infer


def facemap_method(
    variant: str, bases_path: Path, *, n_components: int, lags: np.ndarray, seed: int
) -> Callable:
    stored = np.load(bases_path)
    bases = {
        k: facemap_ridge.SvdBasis(stored[f"{k}_mean"], stored[f"{k}_components"])
        for k in ("motion", "movie")
    }
    used = ("motion", "movie") if variant == "both" else (variant,)
    n_features = n_components * len(lags) * len(used)
    weights = np.random.default_rng(seed).normal(size=n_features)
    builders = {
        "motion": facemap_ridge.motion_matrix,
        "movie": facemap_ridge.movie_matrix,
    }

    def infer(entry):
        frames = common.load_channel(entry, "gray", bin_factor=2)
        blocks = [
            facemap_ridge.zscore_columns(
                common.to_output_grid(
                    entry, facemap_ridge.project(builders[k](frames), bases[k])
                )
            ).astype(np.float32)
            for k in used
        ]
        design = facemap_ridge.lagged_design(np.concatenate(blocks, axis=1), lags)
        return design @ weights

    return infer


def network_method(checkpoint: Path, device: torch.device) -> tuple[Callable, float]:
    """``(infer, parameters in millions)`` for a trained network checkpoint."""
    from zephyr.infer import predict_clip

    from .runner import load_checkpoint

    model, mean, std, _ = load_checkpoint(checkpoint, device)
    amp = torch.bfloat16 if device.type == "cuda" else None

    def infer(entry):
        return predict_clip(
            model,
            entry,
            mean,
            std,
            window=1024,
            device=device,
            frame_chunk=256,
            amp_dtype=amp,
        )[0]

    params = sum(p.numel() for p in model.parameters()) / 1e6
    return infer, params


def estimate_flow_preprocessing(entry, n_frames: int = 300) -> dict:
    """Seconds of DIS optical flow per clip, extrapolated from *n_frames* frames.

    This is the preprocessing cost a ``flow`` channel adds, which the per-method
    timings above do not include.
    """
    import cv2

    from zephyr.channels import make_flow_estimator

    array = np.load(entry.features, mmap_mode="r")
    gray = np.ascontiguousarray(array[: n_frames + 1, 0])
    estimator = make_flow_estimator()
    estimator.calc(gray[0], gray[1], None)  # warm-up
    started = time.perf_counter()
    for i in range(1, len(gray)):
        estimator.calc(gray[i - 1], gray[i], None)
    per_frame = (time.perf_counter() - started) / (len(gray) - 1)
    return {
        "per_frame_ms": per_frame * 1000,
        "frames_per_clip": int(array.shape[0]),
        "seconds_per_clip": per_frame * int(array.shape[0]),
        "opencv": cv2.__version__,
    }


def hardware() -> dict:
    return {
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
