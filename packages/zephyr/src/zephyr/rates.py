"""Breathing rate from inhale onsets, and window weights that even out its distribution.

A breath's rate is ``1 / (time to the next onset)``; a sample's rate is that of the
breath it falls in. :func:`balance_weights` reweights candidate training windows by
the share of windows in their rate bin, so rare rates are drawn more often.
"""

from collections.abc import Sequence

import numpy as np

DEFAULT_RATE_BINS_HZ: tuple[float, ...] = (2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15)
"""Rate bin edges (Hz). Rates outside them count in the first or last bin."""

MIN_BIN_SHARE: float = 0.01
"""A bin holding fewer windows is weighted as if it held this share, so a handful of
windows (e.g. straddling a rate change) is never drawn over and over."""


def breath_rates(onsets_s: np.ndarray) -> np.ndarray:
    """Rate (Hz) of each breath but the last, which has no successor."""
    return 1.0 / np.diff(np.asarray(onsets_s, np.float64))


def rate_track(times_s: np.ndarray, onsets_s: np.ndarray) -> np.ndarray:
    """Rate of the breath each sample falls in; NaN before the first and after the
    last onset."""
    onsets_s = np.asarray(onsets_s, np.float64)
    track = np.full(len(times_s), np.nan)
    if len(onsets_s) < 2:
        return track
    breath = np.searchsorted(onsets_s, times_s, side="right") - 1
    inside = (breath >= 0) & (breath < len(onsets_s) - 1)
    track[inside] = breath_rates(onsets_s)[breath[inside]]
    return track


def window_rates(track: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """Median rate of each window ``track[s : s + window]``; 0 when it holds no
    complete breath (a pause longer than the window)."""
    out = np.zeros(len(starts))
    for k, s in enumerate(np.asarray(starts, int)):
        piece = track[s : s + window]
        piece = piece[np.isfinite(piece)]
        if len(piece):
            out[k] = float(np.median(piece))
    return out


def rate_bin(rates: np.ndarray, bins_hz: Sequence[float]) -> np.ndarray:
    """Index of each rate's bin, out-of-range rates clamped to the end bins."""
    edges = np.asarray(bins_hz, np.float64)
    return np.clip(np.searchsorted(edges, rates, side="right") - 1, 0, len(edges) - 2)


def balance_weights(
    rates: np.ndarray, bins_hz: Sequence[float], power: float
) -> np.ndarray:
    """Draw probabilities ``share(bin) ** -power``, normalised to sum to 1.

    *power* 0 is uniform over windows; 1 gives every occupied bin the same mass,
    except that shares below :data:`MIN_BIN_SHARE` are raised to it.
    """
    index = rate_bin(rates, bins_hz)
    share = np.bincount(index, minlength=len(bins_hz) - 1) / len(index)
    weights = np.maximum(share[index], MIN_BIN_SHARE) ** -power
    return weights / weights.sum()
