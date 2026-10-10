"""Local evaluation metrics for breathing predictions.

:class:`Score` is the per-clip result; :meth:`Score.to_dict` / :meth:`Score.to_json`
serialise it. A field is NaN when the metric cannot be computed (e.g. no inhalations
detected) and becomes ``None`` in the dict, so output is always valid JSON.
"""

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .events import in_spans
from .signal import (
    BREATHING_SIGNAL_COLUMN,
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    detect_inhalation_events,
    resample_uniform,
)

# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

EVENT_TOLERANCE_S: float = 1.0 / CANONICAL_BREATHING_SAMPLING_RATE
"""Default tolerance (seconds) for matching predicted events to GT events: one
60 Hz frame (16.7 ms) either side.

GT events are at the thermistor's own resolution while predictions sit on 60 Hz
frames, so this accepts exactly the two frames that bracket the true event,
wherever it falls between them; the next frame out is always more than a frame
away.
"""

MATCH_SLACK_S: float = 1e-9
"""Added to the tolerance when matching, so an exact tie is always a match."""


@dataclass
class Score:
    """Per-clip scoring result for the breathing-from-video benchmark.

    Construct via the individual metric functions below, or via
    ``score_clip()`` in the local evaluation workflow.  Serialise with
    :meth:`to_dict` / :meth:`to_json`.
    """

    # ------------------------------------------------------------------
    # Signal-level similarity
    # ------------------------------------------------------------------

    correlation: float
    """Normalised correlation between GT and prediction at zero lag.

    Range [-1, 1].  Computed as
    ``sum(gt * pred) / sqrt(energy_gt * energy_pred)`` with both signals
    mean-subtracted.  Always evaluated at zero lag -- no lag search is
    performed.  Higher is better; 1.0 is a perfect match.
    """

    # ------------------------------------------------------------------
    # Event-detection quality
    # ------------------------------------------------------------------

    inhale_f1: float
    """F1 score for inhalation-onset events.

    A predicted event is a true positive if it falls within
    ``EVENT_TOLERANCE_S`` (one 60 Hz frame, so one of the two frames bracketing
    it) of a GT inhalation onset.
    Matching is an optimal one-to-one assignment (Hungarian algorithm);
    each GT event may be matched at most once.
    Range [0, 1].  Higher is better.
    """

    exhale_f1: float
    """F1 score for exhalation-onset events.

    Same matching strategy as ``inhale_f1`` but applied to exhalation-onset
    events.
    Range [0, 1].  Higher is better.
    """

    # ------------------------------------------------------------------
    # Rhythm / inter-event statistics
    # ------------------------------------------------------------------

    kl_ibi: float
    """KL divergence D_KL(GT ‖ pred) of inter-inhalation-interval (IBI)
    histograms.

    Measures whether the predicted breathing rhythm matches the GT rhythm
    independent of absolute timing.  Lower is better; NaN if fewer than
    two inhalation events are detected in either signal.
    """

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float | None]:
        """Return a JSON-safe dict; NaN values become ``None``."""
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
        }

    def to_json(self, **kwargs) -> str:
        """Serialise to a JSON string.  NaN → null."""
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------


def match_events(
    truth_times_s: np.ndarray,
    predicted_times_s: np.ndarray,
    tolerance_s: float = EVENT_TOLERANCE_S,
) -> list[tuple[int, int]]:
    """Match predicted events to ground-truth events within a tolerance window.

    Optimal one-to-one assignment via the Hungarian algorithm
    (:func:`scipy.optimize.linear_sum_assignment`) on the |Δt| cost matrix,
    with out-of-tolerance pairs masked out.  This simultaneously maximises
    the number of matches and minimises the total timing error — unlike
    greedy nearest-neighbour, which can consume events needed by better
    global assignments.

    Parameters
    ----------
    truth_times_s:
        Ground-truth event times (seconds).
    predicted_times_s:
        Predicted event times (seconds).
    tolerance_s:
        Maximum |t_truth - t_predicted| for a pair to be matchable.

    Returns
    -------
    list of (truth_index, predicted_index)
        Matched index pairs into the two input arrays.
    """
    n_truth = len(truth_times_s)
    n_predicted = len(predicted_times_s)
    if n_truth == 0 or n_predicted == 0:
        return []

    delta = np.abs(
        np.asarray(truth_times_s)[:, None] - np.asarray(predicted_times_s)[None, :]
    )
    # Times on two grids of the same rate sit whole frames apart, so with a
    # one-frame tolerance a distance of exactly one frame must not be left to
    # floating-point rounding.
    feasible = delta <= tolerance_s + MATCH_SLACK_S
    if not feasible.any():
        return []

    # Out-of-tolerance pairs get a cost so large the solver only picks one
    # when no feasible pair is available for that row/column; those
    # assignments are filtered out afterwards.
    penalty = tolerance_s * (n_truth + n_predicted + 1)
    cost = np.where(feasible, delta, penalty)

    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if feasible[i, j]]


def event_f1(
    truth_times_s: np.ndarray,
    predicted_times_s: np.ndarray,
    tolerance_s: float = EVENT_TOLERANCE_S,
) -> float:
    """F1 score for event detection.

    Matched pairs (via :func:`match_events`) are true positives; unmatched
    predicted events are false positives; unmatched truth events are false
    negatives.

    Returns NaN when there are no truth events and no predicted events
    (F1 is undefined), and 0.0 when one side has events but nothing matches.
    """
    n_truth = len(truth_times_s)
    n_predicted = len(predicted_times_s)
    if n_truth == 0 and n_predicted == 0:
        return float("nan")

    tp = len(match_events(truth_times_s, predicted_times_s, tolerance_s))
    denominator = 2 * tp + (n_predicted - tp) + (n_truth - tp)
    if denominator == 0:
        return 0.0
    return float(2 * tp / denominator)


# ---------------------------------------------------------------------------
# Signal-level metrics
# ---------------------------------------------------------------------------


def zero_lag_correlation(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Normalised correlation between GT and prediction at zero lag.

    Both signals are mean-subtracted, and the correlation is normalised by
    ``sqrt(energy_truth * energy_predicted)`` so it lies in [-1, 1] and is
    independent of signal amplitude.  No lag search is performed -- the two
    signals are assumed to already be aligned in time.

    Parameters
    ----------
    truth, predicted:
        1-D signals of equal length.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1].  NaN if either signal has zero
        energy (e.g. constant).
    """
    truth = truth - truth.mean()
    predicted = predicted - predicted.mean()

    energy = np.sqrt(np.sum(truth**2) * np.sum(predicted**2))
    if energy == 0:
        return float("nan")

    return float(np.sum(truth * predicted) / energy)


# ---------------------------------------------------------------------------
# Rhythm metrics
# ---------------------------------------------------------------------------


def kl_ibi(
    truth_event_times_s: np.ndarray,
    predicted_event_times_s: np.ndarray,
    bins: int = 20,
    *,
    excluded_s: np.ndarray | None = None,
) -> float:
    """KL divergence D_KL(truth ‖ predicted) between inter-event-interval
    distributions.

    Intervals are the first differences of the event times; one that overlaps
    an *excluded_s* span is dropped, since the events of the gap are unknown.
    Both interval sets are binned on a common grid spanning their joint range;
    histograms are smoothed with a small epsilon so the divergence is always
    finite.

    Returns NaN when either side has no interval left.
    """

    def intervals(times):
        times = np.asarray(times, float)
        if len(times) < 2:
            return np.empty(0)
        start, stop = times[:-1], times[1:]
        keep = np.ones(len(start), bool)
        for lo, hi in np.asarray(
            excluded_s if excluded_s is not None else [], float
        ).reshape(-1, 2):
            keep &= ~((start <= hi) & (stop >= lo))
        return (stop - start)[keep]

    truth_ibi = intervals(truth_event_times_s)
    predicted_ibi = intervals(predicted_event_times_s)
    if len(truth_ibi) == 0 or len(predicted_ibi) == 0:
        return float("nan")

    lo = min(truth_ibi.min(), predicted_ibi.min())
    hi = max(truth_ibi.max(), predicted_ibi.max())
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(truth_ibi, bins=edges)
    q, _ = np.histogram(predicted_ibi, bins=edges)

    eps = 1e-10
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------


def score_clip(
    truth_thermistor: pd.DataFrame,
    predicted_thermistor: pd.DataFrame,
    *,
    truth_onset_times_s: np.ndarray | None = None,
    truth_offset_times_s: np.ndarray | None = None,
    predicted_onset_times_s: np.ndarray | None = None,
    predicted_offset_times_s: np.ndarray | None = None,
    tolerance_s: float = EVENT_TOLERANCE_S,
    excluded_s: np.ndarray | None = None,
) -> Score:
    """Compute all metrics for one clip and assemble a :class:`Score`.

    Both signals are resampled onto the canonical scoring grid
    (``CANONICAL_BREATHING_SAMPLING_RATE``) before any metric is computed, and
    only the time span both cover is scored.  Correlation compares the two at the
    prediction's own sample times, so a clip whose video starts before or after
    its thermistor is still compared instant by instant.

    Parameters
    ----------
    truth_thermistor, predicted_thermistor:
        Clip dataframes with columns ``Time`` and ``Signal``.
    truth_onset_times_s, predicted_onset_times_s:
        Optional inhale-onset times (positive temperature peaks).
    truth_offset_times_s, predicted_offset_times_s:
        Optional exhale-onset times (negative temperature troughs). The
        ``offset`` name is retained for API compatibility: an exhale onset is
        also the preceding inhalation's offset.
    All event-time overrides:
        Overrides, independent of each other; any omitted here are detected
        from the resampled signal via
        :func:`~zephyr.signal.detect_inhalation_events`. Inhale and
        exhale onsets need not pair up or match in count. Truth events
        are auto-detected in practice; predicted events may be supplied
        explicitly, and are otherwise detected from the predicted signal.
    tolerance_s:
        Event-matching tolerance passed to the event metrics.
    excluded_s:
        Spans ``(n, 2)`` (start, end in seconds) left out of every metric:
        their samples are not correlated, events inside them on either side
        are dropped, and inter-event intervals reaching into them are not
        compared.  From a clip's event set (see :mod:`zephyr.events`).

    Returns
    -------
    Score
        Per-clip scoring result.  Fields that cannot be computed are NaN.
    """
    fs = CANONICAL_BREATHING_SAMPLING_RATE

    # Each signal on the canonical grid from its own first sample.  Events are
    # detected there and matched as times, so the two grids need not coincide.
    truth_resampled = resample_uniform(truth_thermistor)
    truth = truth_resampled[BREATHING_SIGNAL_COLUMN].to_numpy()
    truth_time = truth_resampled[TIME_COLUMN].to_numpy()
    predicted_resampled = resample_uniform(predicted_thermistor)
    predicted = predicted_resampled[BREATHING_SIGNAL_COLUMN].to_numpy()
    predicted_time = predicted_resampled[TIME_COLUMN].to_numpy()

    # Score only the span both signals cover.  Video and thermistor need not
    # start together (side-camera clips can start ~1 s apart), so this is cut by
    # time, never by sample count.
    start = max(truth_time[0], predicted_time[0])
    stop = min(truth_time[-1], predicted_time[-1])
    keep = (truth_time >= start) & (truth_time <= stop)
    truth, truth_time = truth[keep], truth_time[keep]
    keep = (predicted_time >= start) & (predicted_time <= stop)
    predicted, predicted_time = predicted[keep], predicted_time[keep]

    # ── Signal-level ─────────────────────────────────────────────────────
    # Correlation pairs samples, so both must be at the same instants: the
    # thermistor is interpolated onto the prediction's own grid.
    truth_on_predicted = np.interp(
        predicted_time,
        truth_thermistor[TIME_COLUMN].to_numpy(dtype=float),
        truth_thermistor[BREATHING_SIGNAL_COLUMN].to_numpy(dtype=float),
    )
    spans = np.empty((0, 2)) if excluded_s is None else np.asarray(excluded_s, float)
    spans = spans.reshape(-1, 2)
    scored = ~in_spans(predicted_time, spans)
    corr = zero_lag_correlation(truth_on_predicted[scored], predicted[scored])

    # ── Events ───────────────────────────────────────────────────────────
    # Defaults for whichever side/type isn't overridden below. Indices are
    # converted through the resampled Time column, not `index / fs` -- a
    # clip's Time axis need not start at 0 (side-camera clips can start
    # around Time = -1s), so that would silently misalign against an
    # explicitly-submitted (real Time-based) side.
    truth_on_default, truth_off_default = detect_inhalation_events(truth, fs)
    predicted_on_default, predicted_off_default = detect_inhalation_events(
        predicted, fs
    )

    truth_on_s = (
        truth_onset_times_s
        if truth_onset_times_s is not None
        else truth_time[truth_on_default]
    )
    truth_off_s = (
        truth_offset_times_s
        if truth_offset_times_s is not None
        else truth_time[truth_off_default]
    )
    predicted_on_s = (
        predicted_onset_times_s
        if predicted_onset_times_s is not None
        else predicted_time[predicted_on_default]
    )
    predicted_off_s = (
        predicted_offset_times_s
        if predicted_offset_times_s is not None
        else predicted_time[predicted_off_default]
    )

    def kept(times):
        times = np.asarray(times, float)
        return times[~in_spans(times, spans)]

    truth_on_s, truth_off_s = kept(truth_on_s), kept(truth_off_s)
    predicted_on_s, predicted_off_s = kept(predicted_on_s), kept(predicted_off_s)
    return Score(
        correlation=corr,
        inhale_f1=event_f1(truth_on_s, predicted_on_s, tolerance_s),
        exhale_f1=event_f1(truth_off_s, predicted_off_s, tolerance_s),
        kl_ibi=kl_ibi(truth_on_s, predicted_on_s, excluded_s=spans),
    )
