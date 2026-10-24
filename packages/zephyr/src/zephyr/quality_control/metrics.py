"""Label-free measures of one video's prediction, over any span of it.

Pure numpy on the arrays of :class:`Video`, the network's output (its trace and onset
probability): plausibility of the trace, and consistency of the onset head with it
(calibrated by a :class:`~.reference.Reference`). None reads a thermistor.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..evaluate import HEAD_MIN_DISTANCE_S, HEAD_THRESHOLD, head_event_indices
from ..evaluation import event_f1, frame_tolerance_s
from ..rates import breath_rates
from ..signal import detect_inhalation_events
from .reference import Reference

MIN_SPECTRUM_S = 2.0
"""Shortest span whose spectrum is measured."""
BACKGROUND_S = 0.1
"""Onset probability further than this from any inhale is background."""
NAN = float("nan")

MEASURES: dict[str, tuple[str, str, str | None]] = {
    "n_inhales": ("rhythm", "Inhales", None),
    "rate_hz": ("rhythm", "Breathing rate (Hz)", None),
    "too_fast_frac": ("rhythm", "Breaths faster than the ceiling", "lower"),
    "alternation_break_frac": ("rhythm", "Alternation breaks", "lower"),
    "flat_breath_frac": ("rhythm", "Flat breaths", "lower"),
    "head_trace_f1": ("head", "Head vs trace F1", "higher"),
    "peak_prob": ("head", "Onset peak height", "higher"),
    "ambiguous_peak_frac": ("head", "Ambiguous onset peaks", "lower"),
    "onset_background": ("head", "Onset background", "lower"),
    "expected_precision": ("head", "Expected precision", "higher"),
    "inhale_phase_spread": ("head", "Inhale phase spread", "lower"),
}
"""Every measure: its family, a label, and whether a good prediction scores
``"higher"`` or ``"lower"`` (``None``: a description, not a judgement)."""


@dataclass(frozen=True)
class Events:
    """One trace's events, as sample indices of its grid.

    *inhale* comes from the onset head when there is one (its maxima above
    *threshold*), else DSP peaks on the trace (*peaks*); *exhale* is always DSP troughs
    on the trace.
    """

    inhale: np.ndarray
    exhale: np.ndarray
    peaks: np.ndarray
    threshold: float = HEAD_THRESHOLD


def find_events(
    times: np.ndarray,
    signal: np.ndarray,
    onset: np.ndarray | None,
    threshold: float = HEAD_THRESHOLD,
):
    """:class:`Events` of one trace (and its onset probability, if the head exists,
    read at the checkpoint's onset *threshold*)."""
    fs = 1.0 / float(np.median(np.diff(times)))
    peaks, troughs = detect_inhalation_events(signal, fs)
    inhale = peaks if onset is None else head_event_indices(onset, times, threshold)
    return Events(inhale=inhale, exhale=troughs, peaks=peaks, threshold=threshold)


@dataclass(frozen=True)
class Video:
    """The network's output for one video, on its output grid *times*."""

    times: np.ndarray
    signal: np.ndarray
    onset: np.ndarray | None
    events: Events


def spans(times: np.ndarray, window_s: float) -> list[tuple[float, float]]:
    """Equal windows about *window_s* wide covering *times* (the last one closed)."""
    start, stop = float(times[0]), float(times[-1])
    n = max(1, round((stop - start) / window_s))
    edges = np.linspace(start, stop, n + 1)
    edges[-1] = np.nextafter(stop, np.inf)
    return [(float(a), float(b)) for a, b in itertools.pairwise(edges)]


def _inside(times: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    return (times >= span[0]) & (times < span[1])


def _at(times: np.ndarray, index: np.ndarray, span) -> np.ndarray:
    """The event times of *index* inside *span*."""
    at = times[index]
    return at[_inside(at, span)]


def _median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else NAN


def _mean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else NAN


def _nearest(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Distance from each of *times* to the nearest of the sorted *events* (inf if
    there are none)."""
    if not len(events):
        return np.full(len(times), np.inf)
    right = np.clip(np.searchsorted(events, times), 0, len(events) - 1)
    left = np.clip(right - 1, 0, len(events) - 1)
    return np.minimum(np.abs(times - events[left]), np.abs(times - events[right]))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else NAN


# --- plausibility of the trace -------------------------------------------------------


def rhythm(onsets: np.ndarray, ceiling_hz: float) -> dict:
    """Median rate (a description), and the share of breaths faster than a mouse
    breathes (*ceiling_hz*). Nothing is assumed about which slower rates are usual:
    they change with running, odours and task."""
    rates = breath_rates(onsets) if len(onsets) > 1 else np.empty(0)
    out = {"rate_hz": NAN, "too_fast_frac": NAN}
    if len(rates):
        out["rate_hz"] = float(np.median(rates))
        out["too_fast_frac"] = float(np.mean(rates > ceiling_hz))
    return out


def alternation_breaks(inhales: np.ndarray, exhales: np.ndarray) -> float:
    """Share of events (inhales and exhales in time order) repeating the last kind."""
    kinds = np.concatenate([np.zeros(len(inhales)), np.ones(len(exhales))])
    if len(kinds) < 2:
        return NAN
    kinds = kinds[np.argsort(np.concatenate([inhales, exhales]), kind="stable")]
    return float(np.mean(kinds[1:] == kinds[:-1]))


def head(
    video: Video, span, ambiguous_floor: float, reference: Reference | None
) -> dict:
    """The onset head against the trace, and how decided its probability is."""
    out = {
        "head_trace_f1": NAN,
        "peak_prob": NAN,
        "ambiguous_peak_frac": NAN,
        "onset_background": NAN,
        "expected_precision": NAN,
    }
    if video.onset is None:
        return out
    times, onset, ev = video.times, video.onset, video.events
    tolerance = frame_tolerance_s(times)
    out["head_trace_f1"] = event_f1(
        _at(times, ev.peaks, span), _at(times, ev.inhale, span), tolerance
    )
    inside = ev.inhale[_inside(times[ev.inhale], span)]
    heights = onset[inside]
    out["peak_prob"] = _median(heights)
    fs = 1.0 / float(np.median(np.diff(times)))
    mask = _inside(times, span)
    maxima, props = find_peaks(
        np.where(mask, onset, 0.0),
        height=ambiguous_floor,
        distance=max(1, round(HEAD_MIN_DISTANCE_S * fs)),
    )
    if len(maxima):
        out["ambiguous_peak_frac"] = float(
            np.mean(props["peak_heights"] < ev.threshold)
        )
    far = _nearest(times[mask], times[ev.inhale]) > BACKGROUND_S
    out["onset_background"] = _mean(onset[mask][far])
    if reference is not None and reference.onset_calibration is not None:
        out["expected_precision"] = _mean(reference.calibrate(heights))
    return out


def inhale_phase_spread(times, inhale: np.ndarray, troughs: np.ndarray, span) -> float:
    """How much the point of the trace's cycle (trough to trough) where the head puts
    each inhale wanders: 1 minus the resultant length of those phases (0: always the
    same point; 1: anywhere)."""
    t_in = _at(times, inhale, span)
    t_tr = np.sort(times[troughs])
    after = np.searchsorted(t_tr, t_in, side="right")
    ok = (after > 0) & (after < len(t_tr))
    if ok.sum() < 5:
        return NAN
    before, after = t_tr[after[ok] - 1], t_tr[after[ok]]
    phase = (t_in[ok] - before) / (after - before)
    return float(1 - np.abs(np.mean(np.exp(2j * np.pi * phase))))


def flat_breaths(signal: np.ndarray, times, inhale: np.ndarray, span, ratio: float):
    """Share of the span's breaths (inhale to inhale) whose trace swings less than
    *ratio* of the clip's median breath: stretches where the trace sees nothing.
    Breaths are bounded by the inhales, not the trace's own troughs, which a flat
    stretch would not have."""
    inhale = np.sort(inhale)
    if len(inhale) < 3:
        return NAN
    swing = np.array([np.ptp(signal[a : b + 1]) for a, b in itertools.pairwise(inhale)])
    inside = _inside(times[inhale[:-1]], span)
    if not inside.any():
        return NAN
    return float(np.mean(swing[inside] < ratio * np.median(swing)))


def measure(
    video: Video,
    span: tuple[float, float],
    *,
    ceiling_hz: float,
    ambiguous_floor: float,
    reference: Reference | None,
    flat_ratio: float = 0.25,
) -> dict[str, float]:
    """Every measure of *video* over *span* (``[start, end)`` on its time axis)."""
    times, ev = video.times, video.events
    inside = ev.inhale[_inside(times[ev.inhale], span)]
    onsets_s = times[inside]
    out: dict[str, float] = {"n_inhales": float(len(inside))}
    out |= rhythm(onsets_s, ceiling_hz)
    out["alternation_break_frac"] = alternation_breaks(
        onsets_s, _at(times, ev.exhale, span)
    )
    out["flat_breath_frac"] = flat_breaths(
        video.signal, times, ev.inhale, span, flat_ratio
    )
    out |= head(video, span, ambiguous_floor, reference)
    if video.onset is not None:  # DSP peaks against the same trace say nothing
        out["inhale_phase_spread"] = inhale_phase_spread(
            times, ev.inhale, ev.exhale, span
        )
    else:
        out["inhale_phase_spread"] = NAN
    return out


def check(
    values: dict[str, float], thresholds: dict[str, float | None]
) -> dict[str, bool | None]:
    """Whether each measure is on the good side of its threshold (``"higher"`` measures
    at or above it, ``"lower"`` ones at or below). A measure without a threshold
    (``None`` or NaN) is not checked; one without a value (NaN) is ``None``."""
    checks: dict[str, bool | None] = {}
    for name, bound in thresholds.items():
        if bound is None or not np.isfinite(bound):  # TOML says "unchecked" with nan
            continue
        value = values.get(name)
        if value is None or not np.isfinite(value):
            checks[name] = None
        elif MEASURES[name][2] == "higher":
            checks[name] = bool(value >= bound)
        else:
            checks[name] = bool(value <= bound)
    return checks
