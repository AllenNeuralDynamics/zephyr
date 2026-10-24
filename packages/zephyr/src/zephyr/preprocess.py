"""Decode, crop and gray every clip into a uint8 array on disk.

Done once up front because decoding in the training loop would cap speed at the codec.
Three grids: *native* (camera timestamps), *selection* (``select_fs``, the frames the
CNN sees) and *output* (fixed 60 Hz, where prediction and target live; reconciled after
the CNN by :meth:`zephyr.model.BreathingNet.forward`). Frames decode from the file start
so frame ``i`` is row ``i`` of the timestamps.

Every clip needs a hand-placed box (a guessed one fails silently). Results are cached
per clip; see :mod:`.features`.

CLI
---
    zephyr preprocess examples/clips.toml --cache data/features-example
"""

import argparse
import json
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import features
from .config import (
    OUTPUT_FS,
    ClipList,
    PreprocessParams,
    ResolvedClip,
    canonical,
    load,
    require_videos,
)
from .targets import load_target
from .video import iter_frames, probe_size


def _nearest_index(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Index of the entry of sorted *times* closest to each of *wanted*.

    Clamped at both ends.
    """
    right = np.clip(np.searchsorted(times, wanted), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    nearer_left = np.abs(times[left] - wanted) <= np.abs(times[right] - wanted)
    return np.where(nearer_left, left, right)


def select_anchor_indices(frame_times: np.ndarray, select_fs: float) -> np.ndarray:
    """Native frame indices nearest a uniform *select_fs* grid over the clip."""
    n = len(frame_times)
    duration = float(frame_times[-1] - frame_times[0])
    if n < 2 or duration <= 0:
        raise ValueError(f"clip has no usable time span ({n} frames, {duration:.3f}s)")

    native_fs = (n - 1) / duration
    if select_fs > native_fs:
        raise ValueError(
            f"select_fs {select_fs:.1f} Hz exceeds the clip's native rate "
            f"{native_fs:.1f} Hz; there are not enough frames to select from"
        )

    n_anchors = int(np.floor(duration * select_fs)) + 1
    wanted = frame_times[0] + np.arange(n_anchors) / select_fs
    indices = _nearest_index(frame_times, wanted)
    if np.any(np.diff(indices) <= 0):
        raise ValueError(
            f"select_fs {select_fs:.1f} Hz is too close to the source rate "
            f"{native_fs:.1f} Hz: two anchors landed on the same (or an "
            "out-of-order) source frame"
        )
    return indices


def output_times(anchor_times: np.ndarray, output_fs: float = OUTPUT_FS) -> np.ndarray:
    """The uniform grid the prediction and target live on, in clip time."""
    duration = float(anchor_times[-1] - anchor_times[0])
    n_out = int(np.floor(duration * output_fs)) + 1
    return anchor_times[0] + np.arange(n_out) / output_fs


TARGET_STATS = (
    "n_onsets",
    "breathing_rate_hz",
    "thermistor_fs_hz",
    "target_scale_adc",
    "target_offset_adc",
)
"""Sidecar stats that come from the thermistor rather than the video."""


def _write_target(clip: ResolvedClip, files: dict[str, Path], out_times):
    """Target and event set of a labelled clip; returns their stats.

    The events are the clip's ``[events]`` set (by default ``peak_detection``),
    at the thermistor's own time resolution, never snapped to *out_times*. The
    onset head's target is not stored: training builds it per window from the
    event times (``targets.onset_heatmap``).
    """
    target = load_target(clip.thermistor, out_times)
    found = clip.events.for_clip(clip, (out_times[0], out_times[-1]))
    pd.DataFrame(
        {"time": out_times, "signal": target.signal.astype(np.float32)}
    ).to_parquet(files["target"], index=False)
    np.savez(
        files["events"],
        onset_times=found.inhale,
        offset_times=found.exhale,
        excluded=found.excluded,
    )
    return dict(
        zip(
            TARGET_STATS,
            (
                len(found.inhale),
                float(len(found.inhale) / (out_times[-1] - out_times[0])),
                target.native_fs,
                target.scale,
                target.offset,
            ),
        )
    )


def preprocess_clip(
    clip: ResolvedClip, params: PreprocessParams, cache_dir: Path
) -> features.ClipEntry:
    """Preprocess one clip into *cache_dir*; return its entry.

    Does nothing for a clip already cached; rebuilds only the target when the frames
    are cached but the clip's thermistor or event set changed.
    """
    started = time_module.perf_counter()
    key = features.cache_key(clip, params)
    files = features.paths(cache_dir, features.prefix(clip, params))
    thermistor = canonical(clip.thermistor) if clip.thermistor else None
    labels = features.events_identity(clip) if thermistor else None

    if files["sidecar"].is_file() and files["frames"].is_file():
        sidecar = json.loads(files["sidecar"].read_text())
        if sidecar["key"] == key:
            if (sidecar["thermistor"], sidecar.get("events")) != (thermistor, labels):
                # Frames are reusable; only the thermistor-derived target is not.
                stats = {
                    k: v for k, v in sidecar["stats"].items() if k not in TARGET_STATS
                }
                if thermistor:
                    stats |= _write_target(clip, files, np.load(files["times"]))
                sidecar |= {"stats": stats, "thermistor": thermistor, "events": labels}
                files["sidecar"].write_text(json.dumps(sidecar, indent=2))
            return features.lookup(clip, params, cache_dir)

    frame_times = pd.read_parquet(clip.timestamps)["Time"].to_numpy()
    n_expected = len(frame_times)
    anchor_indices = select_anchor_indices(frame_times, params.select_fs)
    n_anchors = len(anchor_indices)
    anchor_times = frame_times[anchor_indices]

    box_w, box_h = clip.box[2], clip.box[3]
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed once complete.
    partial = files["frames"].with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.uint8, shape=(n_anchors, 1, box_h, box_w)
    )

    n_decoded = 0
    n_written = 0
    frame = params.for_clip(clip).target_size
    for block in iter_frames(clip.video, scale_to=frame, crop=clip.box):
        for frame in block:
            index = n_decoded
            n_decoded += 1
            if n_written >= n_anchors or index != anchor_indices[n_written]:
                continue
            out[n_written, 0] = frame
            n_written += 1

    out.flush()
    del out  # closes the memmap: Windows cannot rename an open file

    if n_decoded != n_expected:
        # The timestamp parquet is the authoritative frame record, so a mismatch
        # means the stored array and the time base have diverged.
        raise ValueError(
            f"{clip.name}: decoded {n_decoded} frames but the timestamp "
            f"parquet has {n_expected} rows"
        )
    if n_written != n_anchors:
        raise ValueError(
            f"{clip.name}: expected {n_anchors} anchors but only "
            f"{n_written} were reached before decoding ended"
        )

    partial.replace(files["frames"])
    out_times = output_times(anchor_times, params.output_fs)
    np.save(files["frame_times"], anchor_times)
    np.save(files["times"], out_times)

    stats = {
        "native_fs_hz": float((n_expected - 1) / (frame_times[-1] - frame_times[0])),
        "select_fs_hz": float(1.0 / np.median(np.diff(anchor_times))),
        "elapsed_s": round(time_module.perf_counter() - started, 1),
    }
    if thermistor:
        stats |= _write_target(clip, files, out_times)

    files["sidecar"].write_text(
        json.dumps(
            {
                "key": key,
                "thermistor": thermistor,
                "events": labels,
                "n_frames": n_written,
                "n_output": len(out_times),
                "stats": stats,
            },
            indent=2,
        )
    )
    return features.lookup(clip, params, cache_dir)


def _describe(
    clip: ResolvedClip,
    entry: features.ClipEntry,
    params: PreprocessParams,
    cache_dir: Path,
) -> str:
    sidecar = features.paths(cache_dir, features.prefix(clip, params))["sidecar"]
    stats = json.loads(sidecar.read_text())["stats"]
    return (
        f"{clip.name:34s} T={entry.n_frames}->{entry.n_output}  "
        f"native {stats['native_fs_hz']:.1f} Hz  "
        f"rate={stats.get('breathing_rate_hz', float('nan')):.2f} Hz  "
        f"elapsed {stats['elapsed_s']:.0f} s"
    )


def preprocess_clips(
    clips: list[ResolvedClip],
    params: PreprocessParams,
    cache_dir: Path,
    *,
    workers: int = 8,
) -> list[features.ClipEntry]:
    """Preprocess *clips* (in parallel) and return their entries in order."""
    unique: dict[str, ResolvedClip] = {}
    for clip in clips:
        unique.setdefault(features.prefix(clip, params), clip)

    pending = [
        c for c in unique.values() if features.lookup(c, params, cache_dir) is None
    ]
    print(
        f"{len(unique)} clip(s), {len(unique) - len(pending)} already cached | "
        f"target {params.target_size[0]}x{params.target_size[1]} | select "
        f"{params.select_fs:.1f} Hz -> output {params.output_fs:g} Hz | "
        f"{workers} worker(s)",
        flush=True,
    )

    # cv2 multithreads its own ops by default, which would oversubscribe cores on
    # top of our thread pool. Force it single-threaded per call so all the
    # parallelism comes from the worker count.
    cv2.setNumThreads(1)

    completed = 0
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(preprocess_clip, clip, params, cache_dir): clip
            for clip in pending
        }
        try:
            for future in as_completed(futures):
                entry = future.result()
                with print_lock:
                    completed += 1
                    print(
                        f"  [{completed}/{len(pending)}] "
                        f"{_describe(futures[future], entry, params, cache_dir)}",
                        flush=True,
                    )
        except BaseException:
            # Don't wait out every already-submitted clip before reporting a
            # failure -- drop whatever hasn't started yet and re-raise.
            for waiting in futures:
                waiting.cancel()
            raise

    return features.require(clips, params, cache_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Decode, crop and gray clips into uint8 feature arrays."
    )
    parser.add_argument("clips", type=Path, nargs="+", help="Clip list TOML file(s).")
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Feature cache directory (created if missing).",
    )
    parser.add_argument(
        "--limit", type=int, help="Process only the first N clips (for smoke tests)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Clips to process concurrently. Each clip's decode already runs "
        "in its own ffmpeg subprocess, so threads -- not just processes -- give "
        "real parallelism here without Windows' expensive per-worker interpreter relaunch. "
        "1 disables concurrency.",
    )
    args = parser.parse_args(argv)

    for path in args.clips:
        clip_list = load(ClipList, path)
        clips = clip_list.resolve()
        if args.limit:
            clips = clips[: args.limit]
        try:
            require_videos(clips, "preprocessing")
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from None
        unboxed = [c.name for c in clips if c.box is None]
        if unboxed:
            raise SystemExit(
                f"{len(unboxed)} clip(s) in {path} have no hand-placed box: "
                f"{unboxed}\nPlace them in `zephyr gui` (Boxes)."
            )
        native = probe_size(clips[0].video)
        print(f"{path.name}: native {native[0]}x{native[1]}", flush=True)
        preprocess_clips(
            clips,
            clip_list.preprocess,
            args.cache,
            workers=args.workers,
        )
        print(f"{path.name}: done", flush=True)
