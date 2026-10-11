"""Camera timestamps (a video's ``metadata.csv``): measurements and the rollback repair.

Columns: ``ReferenceTime`` (Harp clock, s), ``CameraFrameNumber`` (frame counter) and
``CameraFrameTime`` (camera clock, ns). After :func:`fix_rollbacks` a clean camera has
no dropped frames, no backward or oversized step in either clock, and no clock jitter
above :data:`CLOCK_JITTER_S`. Adapted from the Breathing CodaBench dataset generation
(``dataset_gen/qc/camera.py``).
"""

import subprocess

import numpy as np
import pandas as pd

CLOCK_JITTER_S = 3e-3
"""Largest allowed per-frame difference between the two clocks' steps (s)."""
JUMP = 1.5
"""A step longer than this many median frame intervals is a jump (a gap)."""
SPIKE = 100
"""A step this many median intervals long before a rollback is its spike half."""


def video_frames(url: str) -> int:
    """The video's frame count from its container (ffprobe; a URL works)."""
    done = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            url,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    raw = done.stdout.strip()
    if done.returncode != 0 or not raw.isdigit():
        raise RuntimeError(f"ffprobe found no frame count for {url}: {done.stderr}")
    return int(raw)


def measure(frame: pd.DataFrame) -> dict:
    """Every check of one camera's timestamps, as counts and values."""
    ref = frame["ReferenceTime"].to_numpy(float)
    cam = frame["CameraFrameTime"].to_numpy(np.int64)
    number = frame["CameraFrameNumber"].to_numpy(np.int64)
    ref_dt, cam_dt, step = np.diff(ref), np.diff(cam) * 1e-9, np.diff(number)
    ref_med = float(np.median(ref_dt)) if len(ref_dt) else float("nan")
    cam_med = float(np.median(cam_dt)) if len(cam_dt) else float("nan")
    return {
        "frames": len(frame),
        "rate_hz": 1.0 / ref_med if ref_med > 0 else float("nan"),
        "dropped_frames": int((step[step > 1] - 1).sum()),
        "frame_number_backwards": int((step < 1).sum()),
        "ref_backwards": int((ref_dt < 0).sum()),
        "ref_jumps": int((ref_dt > JUMP * ref_med).sum()),
        "cam_backwards": int((cam_dt < 0).sum()),
        "cam_jumps": int((cam_dt > JUMP * cam_med).sum()),
        "max_jitter_s": float(np.abs(cam_dt - ref_dt).max()) if len(ref_dt) else 0.0,
    }


def clean(measured: dict, video_frames: int | None = None) -> bool:
    """Whether a camera's (repaired) timestamps pass every check."""
    ok = (
        measured["dropped_frames"] == 0
        and measured["frame_number_backwards"] == 0
        and measured["ref_backwards"] == 0
        and measured["ref_jumps"] == 0
        and measured["cam_backwards"] == 0
        and measured["cam_jumps"] == 0
        and measured["max_jitter_s"] <= CLOCK_JITTER_S
    )
    return ok and (video_frames is None or measured["frames"] == video_frames)


def problems(row) -> list[str]:
    """Why a camera's repaired timestamps are not clean, in words (a row of
    ``cameras.parquet``, after the repair)."""
    found = []
    if row["after_frames"] != row["video_frames"]:
        rows, frames = int(row["after_frames"]), int(row["video_frames"])
        found.append(f"{rows} timestamps but {frames} video frames")
    if row["after_ref_backwards"] or row["after_ref_jumps"]:
        found.append("ReferenceTime steps back or jumps (a lasting Harp clock step)")
    if row["after_dropped_frames"] or row["after_frame_number_backwards"]:
        found.append(f"{int(row['after_dropped_frames'])} dropped frames")
    if row["after_cam_backwards"] or row["after_cam_jumps"]:
        found.append("CameraFrameTime steps back or jumps")
    if not found and row["after_max_jitter_s"] > CLOCK_JITTER_S:
        found.append(
            f"one ReferenceTime row off by {row['after_max_jitter_s'] * 1e3:.1f} ms "
            "(a Harp glitch the rollback repair does not catch)"
        )
    return found


def fix_rollbacks(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair ``ReferenceTime`` rollbacks (the Harp clock bug) by linear interpolation.

    Every row whose ``ReferenceTime`` steps back is blanked, and so is the row before
    it when that row's step is a spike (over :data:`SPIKE` median steps: the spike half
    of a spike + rollback pair); the blanks are interpolated between their good
    neighbours. Returns the repaired copy and the number of rows repaired.
    """
    frame = frame.reset_index(drop=True).copy()
    diffs = frame["ReferenceTime"].diff()
    bad = diffs < 0
    if not bad.any():
        return frame, 0
    positive = diffs[diffs > 0]
    spike = float(positive.median()) * SPIKE if len(positive) else 0.0
    for row in np.flatnonzero(bad.to_numpy()):
        if row >= 1 and spike > 0 and diffs.iloc[row - 1] > spike:
            bad.iloc[row - 1] = True
    frame.loc[bad, "ReferenceTime"] = np.nan
    frame["ReferenceTime"] = frame["ReferenceTime"].interpolate(method="linear")
    return frame, int(bad.sum())
