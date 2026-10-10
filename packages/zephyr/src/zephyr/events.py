"""Thermistor breathing events: the detectors that find them.

An *event set* is a clip's inhale peaks and exhale troughs (and excluded spans, kept
for the cache format), from a detector in :data:`DETECTORS` run on the thermistor.
Training (the onset head, rate balancing) and scoring both read the set a clip's
``events`` (or its list's ``[events]``) chooses (:class:`zephyr.config.EventsSource`);
without either, ``peak_detection`` with defaults.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import find_peaks

from .signal import (
    BREATHING_SIGNAL_COLUMN,
    TIME_COLUMN,
    detect_inhalation_events,
    filter_sniff_signal,
)

DEFAULT_METHOD = "peak_detection"


@dataclass(frozen=True)
class Events:
    """One clip's event set; all times in seconds on the thermistor's clock."""

    inhale: np.ndarray
    exhale: np.ndarray
    excluded: np.ndarray
    """Excluded spans, shape ``(n, 2)``: start and end."""

    @staticmethod
    def of(inhale, exhale, excluded=()) -> "Events":
        spans = np.asarray(excluded, np.float64).reshape(-1, 2)
        return Events(
            np.sort(np.asarray(inhale, np.float64)),
            np.sort(np.asarray(exhale, np.float64)),
            spans[np.argsort(spans[:, 0])],
        )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class DetectorParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PeakDetectionParams(DetectorParams):
    """``peak_detection`` (the default): ``find_peaks`` on the filtered, z-scored
    thermistor at its own rate, with one prominence threshold for the whole clip
    (the rule the target and scorer used before event sets, off any grid)."""

    distance_s: float = Field(0.05, gt=0)
    """Peaks closer than this are suppressed."""
    prominence_frac: float = Field(0.1, gt=0, lt=1)
    """Prominence threshold, as a fraction of the whole clip's range."""


class AdaptiveProminenceParams(DetectorParams):
    """``adaptive_prominence``: the same peaks on the thermistor at its own rate,
    each standing a fraction of the *local* range above its surroundings, so a
    shallow breath next to deep ones is kept."""

    distance_s: float = Field(0.05, gt=0)
    prominence_frac: float = Field(0.25, gt=0, lt=1)
    """Prominence threshold, as a fraction of the range around each peak."""
    window_s: float = Field(2.0, gt=0)
    """Width of the rolling window the local range is measured over."""


class HysteresisCyclesParams(DetectorParams):
    """``hysteresis_cycles``: the thermistor at its own rate minus a running
    median, cut into cycles where it crosses a local hysteresis band; one peak
    and one trough per cycle, so inhales and exhales always alternate."""

    baseline_s: float = Field(1.0, gt=0)
    """Width of the running median removed before segmenting."""
    hysteresis_frac: float = Field(0.15, gt=0, lt=1)
    """Half-width of the hysteresis band, as a fraction of the local range."""
    window_s: float = Field(2.0, gt=0)
    """Width of the rolling window the local range is measured over."""


Detector = Callable[[np.ndarray, np.ndarray, DetectorParams], tuple]
"""``(thermistor times, values, params) -> (inhale times, exhale times)``; every
detector works on the thermistor at its own rate, so its times are not snapped
to any output grid."""


def native_trace(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The thermistor on a uniform grid at its own rate, filtered and z-scored."""
    fs = 1.0 / float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    x = filter_sniff_signal(np.interp(tu, t, v), fs)
    return tu, (x - x.mean()) / (x.std() or 1.0), fs


def _rolling_range(x: np.ndarray, n: int) -> np.ndarray:
    """5th-95th percentile range over a centred window of *n* samples."""
    rolling = pd.Series(x).rolling(max(3, n), center=True, min_periods=1)
    return (rolling.quantile(0.95) - rolling.quantile(0.05)).to_numpy()


def _peak_detection(t, v, params: PeakDetectionParams):
    tu, x, fs = native_trace(t, v)
    on, off = detect_inhalation_events(
        x, fs, distance_s=params.distance_s, prominence_frac=params.prominence_frac
    )
    return tu[on], tu[off]


def _adaptive_prominence(t, v, params: AdaptiveProminenceParams):
    tu, x, fs = native_trace(t, v)
    local = _rolling_range(x, round(params.window_s * fs))
    distance = max(1, round(params.distance_s * fs))
    times = []
    for sign in (1.0, -1.0):
        peaks, props = find_peaks(sign * x, distance=distance, prominence=0.0)
        keep = props["prominences"] >= params.prominence_frac * local[peaks]
        times.append(tu[peaks[keep]])
    return tuple(times)


def _hysteresis_cycles(t, v, params: HysteresisCyclesParams):
    tu, x, fs = native_trace(t, v)
    baseline = (
        pd.Series(x)
        .rolling(max(3, round(params.baseline_s * fs)), center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    y = x - baseline
    band = params.hysteresis_frac * _rolling_range(y, round(params.window_s * fs)) / 2
    # Crossings alternate up (above +band) and down (below -band); a wiggle
    # inside the band never changes state.
    crossings, state = [], 0
    above, below = y > band, y < -band
    for i in np.flatnonzero(above | below):
        if above[i] and state <= 0:
            crossings.append((i, 1))
            state = 1
        elif below[i] and state >= 0:
            crossings.append((i, -1))
            state = -1
    inhale, exhale = [], []
    for (i, direction), (j, _) in itertools.pairwise(crossings):
        if direction > 0:
            inhale.append(i + int(np.argmax(y[i:j])))
        else:
            exhale.append(i + int(np.argmin(y[i:j])))
    return tu[np.asarray(inhale, int)], tu[np.asarray(exhale, int)]


DETECTORS: dict[str, tuple[type[DetectorParams], Detector]] = {
    "peak_detection": (PeakDetectionParams, _peak_detection),
    "adaptive_prominence": (AdaptiveProminenceParams, _adaptive_prominence),
    "hysteresis_cycles": (HysteresisCyclesParams, _hysteresis_cycles),
}


def detector_params(method: str, params: dict | None = None) -> DetectorParams:
    """Validated parameters of detector *method*; unknown names or keys raise."""
    if method not in DETECTORS:
        raise ValueError(f"unknown detector {method!r}; known: {sorted(DETECTORS)}")
    return DETECTORS[method][0].model_validate(params or {})


def detect(
    method: str,
    thermistor: pd.DataFrame,
    span: tuple[float, float] | None = None,
    params: dict | DetectorParams | None = None,
) -> Events:
    """Run detector *method* on a thermistor frame; events inside *span* (the
    clip's ``(start, end)`` in seconds) when given."""
    if not isinstance(params, DetectorParams):
        params = detector_params(method, params)
    t = thermistor[TIME_COLUMN].to_numpy(float)
    v = thermistor[BREATHING_SIGNAL_COLUMN].to_numpy(float)
    inhale, exhale = DETECTORS[method][1](t, v, params)
    lo, hi = span if span is not None else (-np.inf, np.inf)

    def inside(x):
        return x[(x >= lo) & (x <= hi)]

    return Events.of(inside(inhale), inside(exhale))


def for_clip(
    method: str,
    params: dict | None,
    thermistor: Path,
    span: tuple[float, float] | None = None,
) -> Events:
    """A clip's event set: detector *method* run with *params* on its thermistor."""
    return detect(method, pd.read_parquet(thermistor), span, params)


def identity(method: str, params: dict | None) -> dict:
    """What a clip's event set depends on, for the feature cache's staleness
    check: the method plus its validated parameters."""
    return {"method": method, "params": detector_params(method, params).model_dump()}


def in_spans(times: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Which *times* fall inside any of *spans* (``(n, 2)`` start, end)."""
    times = np.asarray(times, np.float64)
    inside = np.zeros(times.shape, bool)
    for start, end in np.asarray(spans, np.float64).reshape(-1, 2):
        inside |= (times >= start) & (times <= end)
    return inside
