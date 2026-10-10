"""Ground-truth signal → model training target.

The raw signal is filtered and resampled onto the model's output time grid,
then z-scored per clip since its amplitude is in arbitrary units with a
per-recording offset and gain.  Filtering must happen at the signal's native
rate, before resampling, since a low-pass filter's corner can sit above the
resampled grid's Nyquist frequency.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .signal import (
    BREATHING_SIGNAL_COLUMN,
    TIME_COLUMN,
    detect_inhalation_events,
    filter_sniff_signal,
)


@dataclass
class Target:
    """Training target for one clip, on the model's output time grid.

    Attributes
    ----------
    times:
        Output-grid timestamps (seconds, the clip's shared time base).
    signal:
        Filtered, z-scored breathing trace sampled at *times*.
    onset_times:
        Inhalation-onset times (positive temperature peaks) detected on
        *signal*, for the auxiliary event head and event-level diagnostics.
    offset_times:
        Exhalation-onset times (negative temperature troughs). These are also
        the preceding inhalations' offsets.
    native_fs:
        Thermistor sampling rate inferred from its own timestamps.
    scale:
        Standard deviation removed by z-scoring, in ADC units.  Kept so a
        prediction can be mapped back onto the raw amplitude if ever needed.
    offset:
        Mean removed by z-scoring, in ADC units.
    """

    times: np.ndarray
    signal: np.ndarray
    onset_times: np.ndarray
    offset_times: np.ndarray
    native_fs: float
    scale: float
    offset: float


def load_target(thermistor: Path, times: np.ndarray) -> Target:
    """Build the training target from a thermistor parquet, sampled at *times*.

    Its ``onset_times``/``offset_times`` are detected on *times*, for
    diagnostics; the events training and scoring use come from
    :mod:`zephyr.events`, at the thermistor's own resolution.
    """
    frame = pd.read_parquet(thermistor)
    t = frame[TIME_COLUMN].to_numpy(dtype=float)
    v = frame[BREATHING_SIGNAL_COLUMN].to_numpy(dtype=float)
    try:
        signal, native_fs, scale, offset = filtered_on_grid(t, v, times)
    except ValueError as exc:
        raise ValueError(f"{thermistor}: {exc}") from exc

    out_fs = 1.0 / float(np.median(np.diff(times)))
    onsets, offsets = detect_inhalation_events(signal, out_fs)

    return Target(
        times=times,
        signal=signal,
        onset_times=times[onsets],
        offset_times=times[offsets],
        native_fs=native_fs,
        scale=scale,
        offset=offset,
    )


def filtered_on_grid(
    t: np.ndarray, v: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """The thermistor filtered at its own rate and z-scored on *times*.

    Returns ``(signal, native_fs, scale, offset)``: the trace on *times*, the
    thermistor's rate, and the standard deviation and mean removed.
    """
    native_fs = 1.0 / float(np.median(np.diff(t)))

    # filter_sniff_signal assumes uniform sampling; the source clock is close
    # to but not exactly uniform, so put the trace on a regular grid at its
    # own rate.
    t_uniform = np.arange(t[0], t[-1], 1.0 / native_fs)
    v_uniform = np.interp(t_uniform, t, v)
    filtered = filter_sniff_signal(v_uniform, native_fs)

    # Outside the thermistor's span the edge value is held, never extrapolated:
    # a video can start ~1 s before its thermistor, and a line extended that far
    # from the first two samples is not breathing.  The z-score statistics come
    # from the covered span only, so the held stretch cannot skew them.
    resampled = np.interp(times, t_uniform, filtered)
    covered = (times >= t_uniform[0]) & (times <= t_uniform[-1])
    if not covered.any():
        raise ValueError("the thermistor does not overlap the clip's output grid")

    offset = float(resampled[covered].mean())
    scale = float(resampled[covered].std())
    signal = (resampled - offset) / (scale if scale > 0 else 1.0)
    return signal, native_fs, scale, offset


def onset_heatmap(
    times: np.ndarray, onset_times: np.ndarray, *, sigma_s: float = 0.020
) -> np.ndarray:
    """Gaussian-smoothed inhalation-onset target for the auxiliary event head.

    A hard one-hot target is nearly impossible to optimise at 60 Hz — one
    positive sample per ~11 — so onsets are blurred into soft bumps of width
    *sigma_s*. The heatmap is centred exactly on each event; its width makes
    the sparse target trainable without changing the desired peak location.
    """
    heatmap = np.zeros_like(times, dtype=np.float32)
    if len(onset_times) == 0:
        return heatmap
    fs = 1.0 / float(np.median(np.diff(times)))
    sigma_samples = max(sigma_s * fs, 1e-6)
    radius = int(np.ceil(3 * sigma_samples))
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma_samples) ** 2)

    indices = np.searchsorted(times, onset_times).clip(0, len(times) - 1)
    for i in indices:
        lo, hi = max(0, i - radius), min(len(times), i + radius + 1)
        heatmap[lo:hi] = np.maximum(
            heatmap[lo:hi], kernel[lo - i + radius : hi - i + radius]
        )
    return heatmap
