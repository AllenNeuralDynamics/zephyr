"""Decode, crop, and channelise every clip into a uint8 array on disk.

Decoding is separated from training on purpose: decoding inside the training
loop would cap iteration speed at the video codec rather than the GPU. Done
once up front, a whole clip becomes a memory-mappable array that many epochs
can stream cheaply.

Rates
-----
Source video and target signal are decimated/resampled to a shared output rate
(see ``--stride``).  Motion channels are computed across that same stride, so
no motion falls between stored frames.

Anchor ``i`` of the stored array corresponds to source frame ``i * stride``,
and therefore to row ``i * stride`` of the clip's timestamp file -- which is
why frames are decoded from the start of the file with no seeking.

Layout
------
For each clip, in ``--out-dir``::

    feat_{camera}_{clip_id}.npy      uint8 (T, 4, size, size)
    time_{camera}_{clip_id}.npy      float64 (T,)  anchor timestamps
    target_{clip_id}.parquet         time, signal, onset_heatmap   [public only]
    events_{clip_id}.npz             onset_times, offset_times     [public only]
    manifest.json                    one entry per clip, plus shared config

Crop boxes
----------
``--boxes-json`` is required, and takes the hand-placed boxes from
:mod:`.annotate` (session-keyed or clip-keyed, both accepted).  There is no
automatic fallback, deliberately: a guessed box would put a clip's crop region
anywhere, and the failure would be silent -- the stored array would look
perfectly normal and the model would simply never learn.  Every box must share
one size, since the model needs a single input shape.

CLI
---
    python -m breathing_cnn_tcn.preprocess \\
        --boxes-json baseline-cnn-tcn/artifacts/session_boxes_face.json \\
        --drop-sessions 14
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

from .annotate import clip_boxes
from .channels import (
    CHANNEL_NAMES,
    FLOW_CLIP_PX,
    FLOW_SCALE_PX,
    N_CHANNELS,
    channel_encoding,
    encode_stack,
    make_flow_estimator,
)
from .clips import PUBLIC_SPLIT, ClipRef, discover_clips
from .targets import load_target, onset_heatmap
from .video import Box, iter_frames, probe_size


def preprocess_clip(
    clip: ClipRef,
    camera: str,
    target_size: tuple[int, int],
    box: Box,
    out_dir: Path,
    *,
    stride: int = 4,
    flow_scale_px: float = FLOW_SCALE_PX,
    flow_clip_px: float = FLOW_CLIP_PX,
    onset_sigma_s: float = 0.020,
) -> dict:
    """Preprocess one clip and return its manifest entry."""
    started = time_module.perf_counter()

    frame_times = pd.read_parquet(clip.frame_times_path(camera))["Time"].to_numpy()
    n_expected = len(frame_times)
    n_anchors = (n_expected + stride - 1) // stride

    box_w, box_h = box[2], box[3]
    feat_path = out_dir / f"feat_{camera}_{clip.clip_id}.npy"
    features = np.lib.format.open_memmap(
        feat_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_anchors, N_CHANNELS, box_h, box_w),
    )

    flow_estimator = make_flow_estimator()
    history: dict[int, np.ndarray] = {}
    n_decoded = 0
    n_written = 0
    diff_saturated = 0
    flow_saturated = 0

    for block in iter_frames(clip.video(camera), scale_to=target_size, crop=box):
        for frame in block:
            index = n_decoded
            n_decoded += 1
            if index % stride:
                continue
            reference = history.pop(index - stride, None)
            if n_written < n_anchors:
                stack, n_diff, n_flow = encode_stack(
                    frame,
                    reference,
                    flow_estimator,
                    flow_scale_px=flow_scale_px,
                    flow_clip_px=flow_clip_px,
                )
                features[n_written] = stack
                diff_saturated += n_diff
                flow_saturated += n_flow
                n_written += 1
            # Retain this anchor as the reference for the next one.
            history[index] = frame.copy()

    features.flush()
    del features

    if n_decoded != n_expected:
        # The timestamp parquet is the authoritative frame record, so a mismatch
        # means the stored array and the time base have diverged.
        raise ValueError(
            f"{clip.clip_id}/{camera}: decoded {n_decoded} frames but the "
            f"timestamp parquet has {n_expected} rows"
        )

    anchor_times = frame_times[::stride][:n_written]
    np.save(out_dir / f"time_{camera}_{clip.clip_id}.npy", anchor_times)

    n_pixels = n_written * box_w * box_h
    entry = {
        "clip_id": clip.clip_id,
        "split": clip.split,
        "session_idx": clip.session_idx,
        "part": clip.part,
        "camera": camera,
        "target_size": list(target_size),
        "crop_box": list(box),
        "n_frames": n_written,
        "features": feat_path.name,
        "times": f"time_{camera}_{clip.clip_id}.npy",
        "output_fs_hz": float(1.0 / np.median(np.diff(anchor_times))),
        "diff_saturated_frac": diff_saturated / n_pixels,
        "flow_saturated_frac": flow_saturated / (2 * n_pixels),
        "elapsed_s": round(time_module.perf_counter() - started, 1),
    }

    if clip.split == PUBLIC_SPLIT:
        target = load_target(clip, anchor_times)
        heatmap = onset_heatmap(anchor_times, target.onset_times, sigma_s=onset_sigma_s)
        pd.DataFrame(
            {
                "time": anchor_times,
                "signal": target.signal.astype(np.float32),
                "onset_heatmap": heatmap,
            }
        ).to_parquet(out_dir / f"target_{clip.clip_id}.parquet", index=False)
        np.savez(
            out_dir / f"events_{clip.clip_id}.npz",
            onset_times=target.onset_times,
            offset_times=target.offset_times,
        )
        entry |= {
            "target": f"target_{clip.clip_id}.parquet",
            "n_onsets": len(target.onset_times),
            "breathing_rate_hz": float(
                len(target.onset_times) / (anchor_times[-1] - anchor_times[0])
            ),
            "thermistor_fs_hz": target.native_fs,
            "target_scale_adc": target.scale,
            "target_offset_adc": target.offset,
        }

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode and channelise clips into uint8 feature arrays."
    )
    parser.add_argument("--packaged-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default=PUBLIC_SPLIT)
    parser.add_argument("--camera", default="face", choices=["face", "side"])
    parser.add_argument(
        "--boxes-json",
        type=Path,
        required=True,
        help="Hand-placed boxes from `python -m breathing_cnn_tcn.annotate`, "
        "session-keyed or clip-keyed.",
    )
    parser.add_argument(
        "--drop-sessions",
        type=int,
        nargs="*",
        default=[],
        metavar="SESSION_IDX",
        help="Session indices to exclude entirely. Empty by default -- every "
        "session is included.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Source frames per stored frame.  4 maps 240 fps onto the 60 Hz "
        "scoring grid.",
    )
    parser.add_argument("--flow-scale-px", type=float, default=FLOW_SCALE_PX)
    parser.add_argument("--flow-clip-px", type=float, default=FLOW_CLIP_PX)
    parser.add_argument("--onset-sigma-s", type=float, default=0.020)
    parser.add_argument(
        "--limit", type=int, help="Process only the first N clips (for smoke tests)."
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        metavar="IDX",
        help="Reprocess only these session indices.  Entries for these clips "
        "are merged into the existing manifest and the rest are left alone, so "
        "re-annotating one session costs one clip's decode instead of 32.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Clips to process concurrently. Each clip's decode already runs "
        "in its own ffmpeg subprocess, and DIS flow releases the GIL while it "
        "computes, so threads -- not just processes -- give real parallelism "
        "here without Windows' expensive per-worker interpreter relaunch. "
        "1 disables concurrency.",
    )
    args = parser.parse_args()

    data = json.loads(args.boxes_json.read_text())
    if data.get("camera") not in (None, args.camera):
        raise SystemExit(
            f"--boxes-json is for the {data['camera']} camera, not {args.camera}"
        )
    target_size, boxes = clip_boxes(args.boxes_json)
    if not boxes:
        raise SystemExit(f"No boxes found in {args.boxes_json}")
    sizes = {(b[2], b[3]) for b in boxes.values()}
    if len(sizes) != 1:
        raise SystemExit(
            f"--boxes-json holds mixed box sizes {sorted(sizes)}; every clip "
            "must crop to the same shape."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(args.packaged_root, args.split, camera=args.camera)
    clips = [c for c in clips if c.exists(args.camera)]

    dropped = [c.clip_id for c in clips if c.session_idx in set(args.drop_sessions)]
    if dropped:
        clips = [c for c in clips if c.session_idx not in set(args.drop_sessions)]
        print(
            f"dropping {len(dropped)} clips for sessions {args.drop_sessions}: {dropped}"
        )
    if args.sessions:
        wanted = set(args.sessions)
        clips = [c for c in clips if c.session_idx in wanted]
        missing = wanted - {c.session_idx for c in clips}
        if missing:
            raise SystemExit(f"no clips found for session(s) {sorted(missing)}")
    if args.limit:
        clips = clips[: args.limit]

    unannotated = [c.clip_id for c in clips if c.clip_id not in boxes]
    if unannotated:
        raise SystemExit(
            f"{len(unannotated)} clips have no hand-placed box in "
            f"{args.boxes_json}: {unannotated}\n"
            "Annotate them with `python -m breathing_cnn_tcn.annotate`."
        )

    frame_w, frame_h = probe_size(clips[0].video(args.camera))
    box_w, box_h = sizes.pop()
    n_distinct = len({boxes[c.clip_id] for c in clips})
    print(
        f"{len(clips)} clips | native {frame_w}x{frame_h} -> "
        f"downsample {target_size[0]}x{target_size[1]} "
        f"({frame_w / target_size[0]:.2f}x) -> crop {box_w}x{box_h} | "
        f"{n_distinct} distinct box(es) | stride {args.stride} | "
        f"{args.workers} worker(s)"
    )

    # Each worker's flow_estimator is created fresh inside preprocess_clip, so
    # there is no shared OpenCV state across threads -- but cv2 also
    # multithreads its own ops by default, which would oversubscribe cores on
    # top of our thread pool. Force it single-threaded per call so all the
    # parallelism comes from --workers.
    cv2.setNumThreads(1)

    entries = []
    completed = 0
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                preprocess_clip,
                clip,
                args.camera,
                target_size,
                boxes[clip.clip_id],
                args.out_dir,
                stride=args.stride,
                flow_scale_px=args.flow_scale_px,
                flow_clip_px=args.flow_clip_px,
                onset_sigma_s=args.onset_sigma_s,
            ): clip
            for clip in clips
        }
        try:
            for future in as_completed(futures):
                entry = future.result()
                entries.append(entry)
                with print_lock:
                    completed += 1
                    print(
                        f"  [{completed}/{len(clips)}] {entry['clip_id']:16s} session {entry['session_idx']:>2d}  "
                        f"T={entry['n_frames']}  {entry['output_fs_hz']:.2f} Hz  "
                        f"rate={entry.get('breathing_rate_hz', float('nan')):.2f} Hz  "
                        f"sat diff={entry['diff_saturated_frac']:.2e} "
                        f"flow={entry['flow_saturated_frac']:.2e}  "
                        f"{entry['elapsed_s']:.0f}s",
                        flush=True,
                    )
        except BaseException:
            # Don't wait out every already-submitted clip before reporting a
            # failure -- drop whatever hasn't started yet and re-raise.
            for pending in futures:
                pending.cancel()
            raise

    # Completion order follows whichever worker finishes first, not clip
    # order -- restore it so the manifest is identical regardless of
    # --workers or scheduling.
    entries.sort(key=lambda e: (e["session_idx"], e["part"]))

    manifest = {
        "config": {
            "split": args.split,
            "camera": args.camera,
            "boxes_json": str(args.boxes_json),
            "dropped_sessions": list(args.drop_sessions),
            "native_size": [frame_w, frame_h],
            "target_size": list(target_size),
            "box_size": [box_w, box_h],
            "stride": args.stride,
            "channel_names": list(CHANNEL_NAMES),
            "channel_encoding": channel_encoding(
                flow_scale_px=args.flow_scale_px, flow_clip_px=args.flow_clip_px
            ),
            "onset_sigma_s": args.onset_sigma_s,
        },
        "clips": entries,
    }
    manifest_path = args.out_dir / f"manifest_{args.split}_{args.camera}.json"

    if args.sessions and manifest_path.exists():
        # Merge rather than replace: a partial run must not silently shrink
        # the manifest to the clips it happened to touch, which would quietly
        # drop the rest of the split from every downstream consumer.
        previous = json.loads(manifest_path.read_text())
        for key in ("target_size", "box_size", "stride"):
            if previous["config"].get(key) != manifest["config"].get(key):
                raise SystemExit(
                    f"refusing to merge: existing manifest has {key}="
                    f"{previous['config'].get(key)} but this run used "
                    f"{manifest['config'].get(key)}.  Reprocess the whole split."
                )
        replaced = {e["clip_id"] for e in entries}
        kept = [e for e in previous["clips"] if e["clip_id"] not in replaced]
        manifest["clips"] = sorted(
            kept + entries, key=lambda e: (e["session_idx"], e["part"])
        )
        print(f"merged {len(entries)} reprocessed clip(s) into {len(kept)} kept")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
