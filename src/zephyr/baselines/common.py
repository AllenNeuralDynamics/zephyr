"""Pieces every baseline shares: the breathing band, frame loading, polarity, lag.

Polarity
--------
The scorer counts inhale onsets as positive peaks, so a trace with the wrong sign
scores badly however well it tracks the rhythm, and an unsupervised component's
sign is arbitrary.  :func:`blind_polarity` fixes it from the waveform alone:
inhalation (temperature falling) is shorter than exhalation, so a correctly signed
trace falls faster than it rises and its first difference has negative skew.  All
32 training targets satisfy this (median skew -1.45).

Lag
---
The scorer measures correlation at zero lag and matches events within 17 ms, so a
method that is right but late scores as wrong.  :func:`fit_sign_lag` fits one global
sign and lag on the training split -- the ``calibrated`` variant.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew

from ..channels import CHANNEL_NAMES, decode_flow

BREATH_BAND_HZ = (1.0, 15.0)
"""Pass band for every baseline filter: resting breathing through fast sniffing."""

FS = 60.0
"""Selection and output grid rate, Hz (from the preprocessing manifest)."""

MAX_LAG_S = 0.5
"""Largest lag the calibrated variant may apply."""


def bandpass(
    x: np.ndarray,
    fs: float = FS,
    band: tuple[float, float] = BREATH_BAND_HZ,
    order: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along *axis*."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def load_channel(
    entry, name: str, *, bin_factor: int = 1, chunk: int = 2048
) -> np.ndarray:
    """One stored channel as float32 ``(T, H // bin, W // bin)`` in physical units.

    ``gray`` in counts, ``diff`` in counts per tau, flow in pixels per tau.  Read
    in chunks from the memmap so a whole 4-channel clip never sits in memory.
    """
    index = CHANNEL_NAMES.index(name)
    array = np.load(entry.features, mmap_mode="r")
    t, _, h, w = array.shape
    hb, wb = h // bin_factor, w // bin_factor
    out = np.empty((t, hb, wb), np.float32)
    for start in range(0, t, chunk):
        block = np.asarray(array[start : start + chunk, index])
        if name in ("flow_x", "flow_y"):
            block = decode_flow(block)
        elif name == "diff":
            block = block.astype(np.float32) - 128.0
        else:
            block = block.astype(np.float32)
        if bin_factor > 1:
            block = (
                block[:, : hb * bin_factor, : wb * bin_factor]
                .reshape(len(block), hb, bin_factor, wb, bin_factor)
                .mean(axis=(2, 4))
            )
        out[start : start + len(block)] = block
    return out


def to_output_grid(entry, values: np.ndarray) -> np.ndarray:
    """Linearly resample selection-grid *values* (``(T,)`` or ``(T, K)``) to the output grid."""
    frame_times = np.load(entry.frame_times)
    grid = np.load(entry.times)
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(grid, frame_times, values)
    return np.stack(
        [np.interp(grid, frame_times, values[:, k]) for k in range(values.shape[1])],
        axis=1,
    )


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Delay *x* by *lag* samples (negative advances it), holding the edge value."""
    if lag == 0:
        return x.copy()
    if lag > 0:
        return np.concatenate([np.full(lag, x[0]), x[:-lag]])
    return np.concatenate([x[-lag:], np.full(-lag, x[-1])])


def blind_polarity(x: np.ndarray) -> np.ndarray:
    """Sign *x* so it falls faster than it rises; see the module docstring."""
    return -x if skew(np.diff(x)) > 0 else x


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def fit_sign_lag(
    pairs: list[tuple[np.ndarray, np.ndarray]], max_lag: int
) -> tuple[int, int, float]:
    """One global ``(sign, lag, mean r)`` maximising mean correlation over *pairs*.

    Apply as ``sign * shift(pred, lag)``.
    """
    best = (1, 0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        r = float(np.mean([_corr(shift(p, lag), t) for p, t in pairs]))
        for sign in (1, -1):
            if sign * r > best[2]:
                best = (sign, lag, sign * r)
    return best
