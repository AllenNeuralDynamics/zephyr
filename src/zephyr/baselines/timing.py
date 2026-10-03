"""Inference time per clip, for every method in the benchmark.

What is timed
-------------
Wall-clock seconds from the *preprocessed crops on disk* (``feat_*.npy``, the output
of ``zephyr preprocess``) to the finished 60 Hz breathing trace, one clip at a time.
That covers reading the crop array, the method itself, and any trace reconstruction
(TS-CAN's integrate-and-filter, the pixel methods' polarity rule).

What is not timed
-----------------
- Video decoding and cropping, which every method shares.
- The preprocessing channels a method needs: ``flow`` (optical flow) and ``diff``
  are computed once in ``zephyr preprocess``.  Optical flow is not free, so its cost
  is estimated separately (:func:`estimate_flow_preprocessing`) and should be added
  for any method that reads ``flow``.
- Scoring.

Fairness
--------
- Same clips for every method; the first clip is run once beforehand and discarded,
  so one-off costs (cuDNN autotuning, lazy imports, memmap page-in) are not counted.
- Neural networks run on the GPU (bf16, as ``zephyr evaluate`` does) *and* on the CPU
  (fp32); the pixel and Facemap-style methods are numpy and run on the CPU only.
- The Facemap-style readout weights are not saved by the benchmark run, so timing
  uses weights of the right shape drawn at random.  Cost depends on the shape, not
  the values.
"""

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import common, facemap_ridge, run

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
    """*n* evenly spaced clips from *entries*, in (session, part) order."""
    ordered = sorted(entries, key=lambda e: (e.session_idx, e.part))
    if n >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(positions)]


def pick_warmup(entries: list, timed: list):
    """A clip that is *not* in *timed*, for the warm-up call.

    Warming up on a timed clip would leave its file in the operating system's cache
    and make that one clip look faster than the rest (3.4 s against 6.4 s for the
    pixel flow method, whose cost is partly disk reads).
    """
    ordered = sorted(entries, key=lambda e: (e.session_idx, e.part))
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
        trace = run.pixel_trace(method, entry, fs, bin_factor)
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
    from ..evaluate import load_checkpoint
    from ..infer import predict_clip

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

    from ..channels import make_flow_estimator

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
