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
    *sigma_s*. Each bump is the Gaussian at every sample's distance from the
    event's own time, so it is centred exactly on the event, between samples
    where it falls there (events are at the thermistor's resolution, never
    snapped to *times*). An event outside *times* adds only the tail that
    reaches in, never a peak at the edge.
    """
    heatmap = np.zeros_like(times, dtype=np.float32)
    onset_times = np.asarray(onset_times, dtype=np.float64)
    if len(onset_times) == 0 or len(times) == 0:
        return heatmap
    reach = 3.0 * sigma_s
    first = np.searchsorted(times, onset_times - reach, side="left")
    last = np.searchsorted(times, onset_times + reach, side="right")
    for t, lo, hi in zip(onset_times, first, last, strict=True):
        if hi > lo:
            bump = np.exp(-0.5 * ((times[lo:hi] - t) / sigma_s) ** 2)
            heatmap[lo:hi] = np.maximum(heatmap[lo:hi], bump)
    return heatmap


def breath_phase(
    times: np.ndarray, onset_times: np.ndarray, excluded: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Breath phase on *times* and the mask of samples it is defined on.

    The phase runs linearly from 0 to 2π between consecutive inhalations (the
    thermistor's, as the onset target uses them), so every sample of a cycle is
    supervised whatever the rate. It is undefined, and masked, before the first
    inhalation, from the last one on, and over a whole cycle that overlaps an
    *excluded* span (``(n, 2)`` start/stop seconds): its ends are not both real.
    """
    times = np.asarray(times, np.float64)
    onset_times = np.sort(np.asarray(onset_times, np.float64))
    phase = np.zeros(len(times), np.float32)
    mask = np.zeros(len(times), bool)
    if len(onset_times) < 2 or len(times) == 0:
        return phase, mask
    cycle = np.searchsorted(onset_times, times, side="right") - 1
    mask = (cycle >= 0) & (cycle < len(onset_times) - 1)
    spans = np.empty((0, 2)) if excluded is None else np.reshape(excluded, (-1, 2))
    if len(spans):
        starts, stops = onset_times[:-1], onset_times[1:]
        bad = ((spans[:, None, 0] <= stops) & (spans[:, None, 1] >= starts)).any(0)
        mask &= ~bad[np.clip(cycle, 0, len(bad) - 1)]
    k = cycle[mask]
    start, stop = onset_times[k], onset_times[k + 1]
    phase[mask] = 2.0 * np.pi * (times[mask] - start) / (stop - start)
    return phase, mask
