"""Label-free measures of one video's prediction, over any span of it.

Pure numpy on the arrays of :class:`Video`: plausibility of the trace, consistency of
the onset head with it, agreement with dropout draws (or other checkpoints), and how
familiar the input is (with a :class:`~.reference.Reference`). None reads a thermistor.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, welch

from ..evaluate import HEAD_MIN_DISTANCE_S, HEAD_THRESHOLD, head_event_indices
from ..evaluation import event_f1, frame_tolerance_s, match_events
from ..rates import breath_rates
from ..signal import detect_inhalation_events
from .reference import Reference

CYCLE_SAMPLES = 32
"""Points each breath is resampled to before its shape is compared."""
MIN_SPECTRUM_S = 2.0
"""Shortest span whose spectrum is measured."""
BACKGROUND_S = 0.1
"""Onset probability further than this from any inhale is background."""
NAN = float("nan")

MEASURES: dict[str, tuple[str, str, str | None]] = {
    "n_inhales": ("rhythm", "Inhales", None),
    "rate_hz": ("rhythm", "Breathing rate (Hz)", None),
    "too_fast_frac": ("rhythm", "Breaths faster than the ceiling", "lower"),
    "high_freq_power_frac": ("rhythm", "Trace power above the ceiling", "lower"),
    "alternation_break_frac": ("rhythm", "Alternation breaks", "lower"),
    "cycle_similarity": ("rhythm", "Breath shape similarity", "higher"),
    "head_trace_f1": ("head", "Head vs trace F1", "higher"),
    "peak_prob": ("head", "Onset peak height", "higher"),
    "ambiguous_peak_frac": ("head", "Ambiguous onset peaks", "lower"),
    "onset_background": ("head", "Onset background", "lower"),
    "expected_precision": ("head", "Expected precision", "higher"),
    "agreement_f1": ("agreement", "Agreement F1", "higher"),
    "event_support": ("agreement", "Event support", "higher"),
    "agreement_corr": ("agreement", "Agreement correlation", "higher"),
    "trace_spread": ("agreement", "Trace spread", "lower"),
    "embedding_distance": ("input", "Embedding distance", "lower"),
    "embedding_percentile": ("input", "Embedding percentile", "lower"),
    "unfamiliar_frac": ("input", "Unfamiliar frames", "lower"),
    "brightness": ("input", "Brightness", None),
    "contrast": ("input", "Contrast", None),
    "sharpness": ("input", "Sharpness", None),
    "motion": ("input", "Motion", None),
    "image_outlier_frac": ("input", "Image outlier frames", "lower"),
}
"""Every measure: its family, a label, and whether a good prediction scores
``"higher"`` or ``"lower"`` (``None``: a description, not a judgement)."""


@dataclass(frozen=True)
class Events:
    """One trace's events, as sample indices of its grid.

    *inhale* comes from the onset head when there is one, else DSP peaks on the trace
    (*peaks*); *exhale* is always DSP troughs on the trace.
    """

    inhale: np.ndarray
    exhale: np.ndarray
    peaks: np.ndarray


def find_events(times: np.ndarray, signal: np.ndarray, onset: np.ndarray | None):
    """:class:`Events` of one trace (and its onset probability, if the head exists)."""
    fs = 1.0 / float(np.median(np.diff(times)))
    peaks, troughs = detect_inhalation_events(signal, fs)
    inhale = peaks if onset is None else head_event_indices(onset, times)
    return Events(inhale=inhale, exhale=troughs, peaks=peaks)


@dataclass(frozen=True)
class Video:
    """Everything measured from one video, on its output grid *times*.

    *draws* are the traces compared with the prediction (dropout draws or other
    checkpoints), *draw_events* their events. *frame_times* index the per-frame
    *distances* (embedding distance, NaN where unknown) and *image* statistics.
    """

    times: np.ndarray
    signal: np.ndarray
    onset: np.ndarray | None
    events: Events
    draws: np.ndarray
    draw_events: list[Events]
    frame_times: np.ndarray
    distances: np.ndarray | None = None
    image: dict[str, np.ndarray] | None = None


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


def high_frequency_power(x: np.ndarray, fs: float, ceiling_hz: float) -> float:
    """Share of the trace's power above *ceiling_hz*, faster than any breath: jitter
    the network adds, not breathing (NaN when the grid cannot resolve the ceiling)."""
    if len(x) < MIN_SPECTRUM_S * fs or ceiling_hz >= fs / 2:
        return NAN
    freq, power = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    total = power[freq > 0].sum()
    return float(power[freq > ceiling_hz].sum() / total) if total > 0 else NAN


def cycle_similarity(signal: np.ndarray, onsets: np.ndarray) -> float:
    """Median correlation of each breath (onset to onset) with the median breath."""
    cycles = []
    grid = np.linspace(0, 1, CYCLE_SAMPLES)
    for a, b in itertools.pairwise(onsets):
        if b - a < 3:
            continue
        piece = signal[a : b + 1]
        cycle = np.interp(grid, np.linspace(0, 1, len(piece)), piece)
        if cycle.std() > 0:
            cycles.append((cycle - cycle.mean()) / cycle.std())
    if len(cycles) < 3:
        return NAN
    template = np.median(cycles, axis=0)
    return _median([_correlation(c, template) for c in cycles])


# --- the onset head --------------------------------------------------------------------


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
            np.mean(props["peak_heights"] < HEAD_THRESHOLD)
        )
    far = _nearest(times[mask], times[ev.inhale]) > BACKGROUND_S
    out["onset_background"] = _mean(onset[mask][far])
    if reference is not None and reference.onset_calibration is not None:
        out["expected_precision"] = _mean(reference.calibrate(heights))
    return out


# --- agreement with the draws -----------------------------------------------------------


def spread(video: Video) -> np.ndarray:
    """Per sample: RMS deviation of the draws from the prediction (empty without
    draws); ``trace_spread`` is its RMS over a span."""
    if not len(video.draws):
        return np.empty(0)
    return np.sqrt(np.mean((video.draws - video.signal) ** 2, axis=0))


def agreement(video: Video, span) -> dict:
    """How far the draws reproduce the prediction's inhales and trace."""
    out = {
        "agreement_f1": NAN,
        "event_support": NAN,
        "agreement_corr": NAN,
        "trace_spread": NAN,
    }
    if not len(video.draws):
        return out
    times = video.times
    tolerance = frame_tolerance_s(times)
    mine = _at(times, video.events.inhale, span)
    found = np.zeros(len(mine))
    f1 = []
    for events in video.draw_events:
        theirs = _at(times, events.inhale, span)
        f1.append(event_f1(mine, theirs, tolerance))
        for i, _ in match_events(mine, theirs, tolerance):
            found[i] += 1
    mask = _inside(times, span)
    out["agreement_f1"] = _mean(f1)
    if len(mine):
        out["event_support"] = float(np.mean(found / len(video.draw_events)))
    out["agreement_corr"] = _mean(
        [_correlation(video.signal[mask], d[mask]) for d in video.draws]
    )
    deviation = video.draws[:, mask] - video.signal[mask]
    out["trace_spread"] = float(np.sqrt(np.mean(deviation**2)))
    return out


# --- how familiar the input is ---------------------------------------------------------


def familiarity(video: Video, span, reference: Reference | None) -> dict:
    """Embedding distance from what the network knows, and the image statistics."""
    out: dict[str, float] = {}
    frames = _inside(video.frame_times, span)
    if video.distances is not None and reference is not None:
        d = video.distances[frames]
        d = d[np.isfinite(d)]
        out["embedding_distance"] = _median(d)
        out["embedding_percentile"] = (
            _median(reference.distance_percentile(d)) if len(d) else NAN
        )
        out["unfamiliar_frac"] = (
            float(np.mean(d > reference.distance_quantiles[99])) if len(d) else NAN
        )
    if video.image is not None:
        for name, values in video.image.items():
            out[name] = _median(values[frames])
        if reference is not None and reference.image_quantiles is not None:
            outside = np.zeros(int(frames.sum()), bool)
            for name, values in video.image.items():
                q = reference.image_quantiles[name]
                v = values[frames]
                outside |= (v < q[1]) | (v > q[99])
            out["image_outlier_frac"] = float(outside.mean()) if len(outside) else NAN
    return out


def measure(
    video: Video,
    span: tuple[float, float],
    *,
    ceiling_hz: float,
    ambiguous_floor: float,
    reference: Reference | None,
) -> dict[str, float]:
    """Every measure of *video* over *span* (``[start, end)`` on its time axis)."""
    times, ev = video.times, video.events
    inside = ev.inhale[_inside(times[ev.inhale], span)]
    onsets_s = times[inside]
    fs = 1.0 / float(np.median(np.diff(times)))
    mask = _inside(times, span)
    out: dict[str, float] = {"n_inhales": float(len(inside))}
    out |= rhythm(onsets_s, ceiling_hz)
    out["high_freq_power_frac"] = high_frequency_power(
        video.signal[mask], fs, ceiling_hz
    )
    out["alternation_break_frac"] = alternation_breaks(
        onsets_s, _at(times, ev.exhale, span)
    )
    out["cycle_similarity"] = cycle_similarity(video.signal, inside)
    out |= head(video, span, ambiguous_floor, reference)
    out |= agreement(video, span)
    out |= familiarity(video, span, reference)
    return out
