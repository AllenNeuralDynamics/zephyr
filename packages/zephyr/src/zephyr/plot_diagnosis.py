"""Diagnostic plots over held-out predictions: rate breakdown and reserved grid.

Library only; ``evaluate --plot`` calls it with predictions it already computed.
:func:`rate_breakdown` splits performance by the ground truth's instantaneous frequency,
since one event-F1 per clip cannot say whether a model is poor in one rate regime, and
recordings differ in time spent per rate (the standardised-recall panel reweights each
onto the pooled distribution).

:func:`reserved_grid` shows worst / typical / best windows per recording for one model's
onset head.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from zephyr.model import BreathingNet

from .evaluation import EVENT_TOLERANCE_S, match_events
from .features import ClipEntry
from .infer import predict_clip
from .signal import (
    BREATHING_SIGNAL_COLUMN,
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    detect_inhalation_events,
    filter_sniff_signal,
    resample_uniform,
)

FS = CANONICAL_BREATHING_SAMPLING_RATE

# ---------------------------------------------------------------------------
# Rate breakdown
# ---------------------------------------------------------------------------

# Bin edges in Hz.  Coarse above 6 Hz on purpose: an interval quantised to
# 1/60 s quantises the frequency axis with it, so around 10 Hz the achievable
# values are already 8.6, 10, 12 -- narrower bins there would be empty or
# single-valued rather than more informative.
FREQ_EDGES = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.5])

MIN_BIN_BREATHS = 25
"""Per-session bins thinner than this are dropped from the per-session curves.

A recall estimated on a handful of breaths swings between 0 and 1 on noise and
would read as a frequency effect that is not there.
"""

BIN_CENTRES = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
"""Geometric bin centres -- the bins widen with frequency, so linear midpoints
would sit visually off-centre on a log axis."""

SESSION_COLOURS = ("#1f6aa5", "#e8944a", "#c0392b", "#6a3d9a", "#2e8b57")
POOLED_COLOUR = "black"
XTICKS = [2, 3, 4, 6, 8, 12]


def short_label(recording: str) -> str:
    """Folder and group of a recording id (``.../test#7`` -> ``test/7``), as a plot label.

    A "session" in this module is a recording; the full id is a path and would
    swamp every legend and axis.
    """
    folder, _, group = recording.rpartition("#")
    return f"{Path(folder).name}/{group}"


@dataclass(frozen=True)
class ClipPrediction:
    """One clip's whole-clip prediction, already computed by the caller."""

    entry: ClipEntry
    signal: np.ndarray
    times: np.ndarray
    truth: pd.DataFrame


def canonical_clip(
    prediction: np.ndarray, pred_times: np.ndarray, truth: pd.DataFrame
) -> dict:
    """Put truth (raw and filtered) and prediction on the scorer's 60 Hz grid.

    ``score_clip`` resamples each side independently and truncates; the grids
    start ~1 ms apart, so both are put on the *truth's* grid instead -- the same
    operation to well under a sample, and one time base for everything below.
    """
    grid = resample_uniform(truth)
    times = grid[TIME_COLUMN].to_numpy()
    raw = grid[BREATHING_SIGNAL_COLUMN].to_numpy()

    # filter_sniff_signal's 40 Hz corner is above the 60 Hz grid's Nyquist, so
    # it has to be applied at the thermistor's native rate and resampled after.
    t_native = truth[TIME_COLUMN].to_numpy(dtype=float)
    v_native = truth[BREATHING_SIGNAL_COLUMN].to_numpy(dtype=float)
    native_fs = 1.0 / float(np.median(np.diff(t_native)))
    t_uniform = np.arange(t_native[0], t_native[-1], 1.0 / native_fs)
    filtered = filter_sniff_signal(np.interp(t_uniform, t_native, v_native), native_fs)
    filtered = interp1d(
        t_uniform, filtered, kind="linear", bounds_error=False, fill_value="extrapolate"
    )(times)

    predicted = interp1d(
        pred_times.astype(np.float64),
        prediction.astype(np.float64),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(times)

    return {
        "times": times,
        "truth_raw": raw,
        "truth_filtered": filtered,
        "prediction": predicted,
    }


def chance_hits(
    truth_s: np.ndarray,
    pred_s: np.ndarray,
    span: tuple[float, float],
    tolerance_s: float,
    n_shifts: int,
) -> np.ndarray:
    """Per-breath probability of a match under a rate-preserving null.

    The 50 ms tolerance is a fixed window against a shrinking cycle, so a
    prediction that merely ticks at the right rate matches more and more onsets
    as frequency rises without tracking any of them.  That has to be measured
    before it can be discounted.

    The null circularly shifts the model's own onset train: alignment is
    destroyed, rate and burstiness and count are kept -- the three things that
    set how often a blind prediction lands inside the tolerance.  Shifts are
    evenly spaced rather than random, so the estimate is reproducible.
    """
    lo, hi = span
    duration = hi - lo
    counts = np.zeros(len(truth_s))
    if len(truth_s) == 0 or len(pred_s) == 0 or n_shifts <= 0 or duration <= 0:
        return counts
    for k in range(1, n_shifts + 1):
        shifted = np.sort((pred_s - lo + duration * k / (n_shifts + 1)) % duration + lo)
        for i, _ in match_events(truth_s, shifted, tolerance_s):
            counts[i] += 1
    return counts / n_shifts


def breath_table(
    clip: dict, tolerance_s: float, n_shifts: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per ground-truth breath, and one per predicted onset.

    A breath is the interval between consecutive onsets, so the last onset has
    no frequency and is dropped.  Predicted onsets inherit the frequency of the
    truth breath they land inside, which is what lets a false positive be binned
    on the same axis as the breaths it competes with.
    """
    truth_on, _ = detect_inhalation_events(clip["truth_raw"], FS)
    pred_on, _ = detect_inhalation_events(clip["prediction"], FS)
    truth_s = clip["times"][truth_on]
    pred_s = clip["times"][pred_on]

    matches = match_events(truth_s, pred_s, tolerance_s)
    matched_truth = {i: j for i, j in matches}
    matched_pred = {j for _, j in matches}

    chance = chance_hits(
        truth_s,
        pred_s,
        (float(clip["times"][0]), float(clip["times"][-1])),
        tolerance_s,
        n_shifts,
    )

    intervals = np.diff(truth_s)
    n = len(intervals)
    breaths = pd.DataFrame(
        {
            "t": truth_s[:-1],
            "freq_hz": 1.0 / intervals,
            "hit": [i in matched_truth for i in range(n)],
            "chance": chance[:n],
            "abs_err_s": [
                abs(truth_s[i] - pred_s[matched_truth[i]])
                if i in matched_truth
                else np.nan
                for i in range(n)
            ],
        }
    )

    # Each predicted onset inherits the frequency of the truth interval it
    # falls in; onsets before the first or after the last truth onset have no
    # local rate and are dropped rather than guessed.
    slot = np.searchsorted(truth_s, pred_s, side="right") - 1
    inside = (slot >= 0) & (slot < n)
    predictions = pd.DataFrame(
        {
            "t": pred_s[inside],
            "freq_hz": 1.0 / intervals[slot[inside]],
            "matched": [j in matched_pred for j in np.flatnonzero(inside)],
        }
    )
    return breaths, predictions


def window_table(
    clip: dict, window_s: float, hop_s: float, min_breaths: int
) -> pd.DataFrame:
    """Windowed Pearson ``r`` against the filtered truth, tagged with local rate.

    A window's frequency is the median instantaneous rate of the truth breaths
    starting inside it; windows holding fewer than *min_breaths* are dropped,
    since a rate read off one or two intervals is not a regime.
    """
    truth_on, _ = detect_inhalation_events(clip["truth_raw"], FS)
    onset_times = clip["times"][truth_on]
    freqs = 1.0 / np.diff(onset_times)
    onset_times = onset_times[:-1]

    width = round(window_s * FS)
    hop = max(1, round(hop_s * FS))

    rows = []
    for start in range(0, len(clip["times"]) - width + 1, hop):
        stop = start + width
        t0, t1 = clip["times"][start], clip["times"][stop - 1]
        in_window = (onset_times >= t0) & (onset_times < t1)
        if in_window.sum() < min_breaths:
            continue
        truth = clip["truth_filtered"][start:stop]
        pred = clip["prediction"][start:stop]
        if truth.std() == 0 or pred.std() == 0:
            continue
        rows.append(
            {
                "t": float(t0),
                "freq_hz": float(np.median(freqs[in_window])),
                "r": float(np.corrcoef(truth, pred)[0, 1]),
            }
        )
    return pd.DataFrame(rows, columns=["t", "freq_hz", "r"])


def wilson(
    hits: np.ndarray, total: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval -- honest near 0 and 1, where recall bins live."""
    total = np.maximum(total, 1)
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (z / (1 + z * z / total)) * np.sqrt(
        p * (1 - p) / total + z * z / (4 * total**2)
    )
    return centre - half, centre + half


def bin_index(freq: np.ndarray) -> np.ndarray:
    """Index into the ``FREQ_EDGES`` bins; -1 for anything outside them."""
    idx = np.digitize(freq, FREQ_EDGES) - 1
    return np.where((idx >= 0) & (idx < len(FREQ_EDGES) - 1), idx, -1)


def per_bin_events(breaths: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """TP / FN / FP counts per frequency bin, and the metrics built from them."""
    b_bin = bin_index(breaths["freq_hz"].to_numpy())
    p_bin = bin_index(predictions["freq_hz"].to_numpy())
    hit = breaths["hit"].to_numpy(dtype=bool)
    chance = breaths["chance"].to_numpy(dtype=float)
    err = breaths["abs_err_s"].to_numpy(dtype=float)
    spurious = ~predictions["matched"].to_numpy(dtype=bool)

    rows = []
    for b in range(len(BIN_CENTRES)):
        here = b_bin == b
        tp = int((here & hit).sum())
        fn = int((here & ~hit).sum())
        fp = int(((p_bin == b) & spurious).sum())
        matched_err = err[here & hit]
        recall = tp / (tp + fn) if tp + fn else np.nan
        chance_recall = float(np.mean(chance[here])) if here.any() else np.nan
        rows.append(
            {
                "bin": b,
                "freq_lo": float(FREQ_EDGES[b]),
                "freq_hi": float(FREQ_EDGES[b + 1]),
                "freq_centre": float(BIN_CENTRES[b]),
                "n_breaths": int(here.sum()),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": recall,
                "chance_recall": chance_recall,
                # Cohen's-kappa form: the share of the room above chance that
                # the model actually took.  1.0 is perfect, 0.0 is a
                # rate-matched but blind prediction.
                "skill": (
                    (recall - chance_recall) / (1.0 - chance_recall)
                    if np.isfinite(recall)
                    and np.isfinite(chance_recall)
                    and chance_recall < 1.0
                    else np.nan
                ),
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                "timing_mae_s": float(np.mean(matched_err))
                if len(matched_err)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def standardised_recall(
    per_session: dict[str, pd.DataFrame], pooled: pd.DataFrame
) -> dict:
    """Each session's recall reweighted onto the pooled frequency distribution.

    Restricted to bins where the session has ``MIN_BIN_BREATHS`` breaths:
    extrapolating it into a regime it never entered would invent the number the
    comparison exists to establish.  ``coverage`` reports what share of pooled
    breaths those bins hold, so a thin estimate looks thin.
    """
    weights = pooled["n_breaths"].to_numpy(dtype=float)
    out = {}
    for session, table in per_session.items():
        usable = (table["n_breaths"] >= MIN_BIN_BREATHS) & table["recall"].notna()
        w = np.where(usable, weights, 0.0)
        tp, fn = int(table["tp"].sum()), int(table["fn"].sum())

        def reweight(
            column: str, w: np.ndarray = w, table: pd.DataFrame = table
        ) -> float:
            values = table[column].to_numpy(dtype=float)
            keep = w * np.isfinite(values)
            return (
                float(np.sum(keep * np.nan_to_num(values)) / keep.sum())
                if keep.sum()
                else np.nan
            )

        out[session] = {
            "observed_recall": tp / (tp + fn) if tp + fn else np.nan,
            "standardised_recall": reweight("recall"),
            "standardised_skill": reweight("skill"),
            "standardised_chance": reweight("chance_recall"),
            "coverage": float(w.sum() / weights.sum()) if weights.sum() else np.nan,
            "mean_freq_hz": np.nan,  # filled by the caller, which holds the breaths
        }
    return out


def json_safe(value):
    """NaN -> null, numpy scalars -> python, so the sidecar is valid JSON.

    ``json.dumps`` would otherwise emit a bare ``NaN`` literal, which every
    strict reader rejects -- the same convention ``Score.to_dict`` uses.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _log_axis(ax) -> None:
    """Shared frequency axis -- fixed limits so panels can be read against
    each other rather than against their own autoscale."""
    ax.set_xscale("log")
    ax.set_xlim(FREQ_EDGES[0] * 0.95, FREQ_EDGES[-1] * 1.02)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.grid(alpha=0.25)


def _draw_rate_breakdown(
    windows: pd.DataFrame,
    per_session: dict[str, pd.DataFrame],
    per_clip: dict[tuple[str, str], pd.DataFrame],
    pooled: pd.DataFrame,
    standardised: dict,
    out: Path,
    title_suffix: str,
) -> None:
    sessions = list(per_session)
    colours = {
        s: SESSION_COLOURS[i % len(SESSION_COLOURS)] for i, s in enumerate(sessions)
    }

    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    fig.suptitle(
        "Held-out performance by ground-truth breathing frequency  "
        f"({title_suffix}).  Reserved sessions: never trained on, validated on or selected on.",
        fontsize=11,
    )

    def curve(ax, column: str, scale: float = 1.0) -> None:
        """Per-session curve over bins that clear MIN_BIN_BREATHS, plus pooled."""
        for session in sessions:
            table = per_session[session]
            keep = (
                (table["n_breaths"] >= MIN_BIN_BREATHS) & table[column].notna()
            ).to_numpy()
            ax.plot(
                BIN_CENTRES[keep],
                table[column].to_numpy()[keep] * scale,
                "o-",
                ms=4,
                lw=1.4,
                color=colours[session],
                label=str(session),
            )
        keep = (pooled[column].notna() & (pooled["n_breaths"] > 0)).to_numpy()
        ax.plot(
            BIN_CENTRES[keep],
            pooled[column].to_numpy()[keep] * scale,
            "s--",
            ms=4,
            lw=1.6,
            color=POOLED_COLOUR,
            alpha=0.75,
            label="pooled",
        )
        _log_axis(ax)

    # -- A: per-breath detection rate, the recall side of inhale_f1 ----------
    ax = axes[0, 0]
    curve(ax, "recall", scale=100.0)
    for session in sessions:
        table = per_session[session]
        keep = (
            (table["n_breaths"] >= MIN_BIN_BREATHS) & table["recall"].notna()
        ).to_numpy()
        lo, hi = wilson(
            table["tp"].to_numpy(dtype=float),
            (table["tp"] + table["fn"]).to_numpy(dtype=float),
        )
        ax.fill_between(
            BIN_CENTRES[keep],
            100 * lo[keep],
            100 * hi[keep],
            color=colours[session],
            alpha=0.13,
            lw=0,
        )
    # The null: a rate-matched but blind prediction. Above ~6 Hz the 50 ms
    # tolerance is most of a breathing cycle, so chance recall approaches 100%
    # and panel A's rise there is mostly free.
    keep = (pooled["chance_recall"].notna() & (pooled["n_breaths"] > 0)).to_numpy()
    ax.plot(
        BIN_CENTRES[keep],
        100 * pooled["chance_recall"].to_numpy()[keep],
        ":",
        lw=1.6,
        color="grey",
        label="chance (shifted prediction)",
    )
    ax.set_title("A  breath detected within 50 ms (recall)", fontsize=10)
    ax.set_ylabel("% of ground-truth breaths detected")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, loc="lower left")

    # -- B: the same recall with the free matches taken back out -------------
    ax = axes[0, 1]
    curve(ax, "skill")
    ax.axhline(0, color="grey", ls=":", lw=1.2)
    ax.set_title("B  chance-corrected detection (kappa)", fontsize=10)
    ax.set_ylabel("(recall - chance) / (1 - chance)")
    ax.set_ylim(-0.05, 1.02)

    # -- C: the scored metric itself, localised to a frequency band ----------
    ax = axes[0, 2]
    curve(ax, "f1")
    ax.set_title("C  local inhale F1 (TP / FP / FN binned by rate)", fontsize=10)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)

    # -- D: how tight the surviving matches are against the tolerance --------
    ax = axes[0, 3]
    curve(ax, "timing_mae_s", scale=1000.0)
    ax.axhline(1000 * EVENT_TOLERANCE_S, color="grey", ls=":", lw=1.2)
    ax.text(
        BIN_CENTRES[0],
        1000 * EVENT_TOLERANCE_S - 1.5,
        "scorer matching tolerance",
        fontsize=7,
        color="grey",
        va="top",
    )
    ax.set_title("D  onset timing error, matched breaths only", fontsize=10)
    ax.set_ylabel("mean |dt| (ms)")
    ax.set_ylim(0, 1000 * EVENT_TOLERANCE_S + 5)

    # -- E: waveform fidelity, independent of the event detector -------------
    ax = axes[1, 0]
    for session in sessions:
        w = windows[windows["recording"] == session]
        ax.plot(w["freq_hz"], w["r"], ".", ms=2.5, alpha=0.18, color=colours[session])
    for session in sessions:
        w = windows[windows["recording"] == session]
        idx = bin_index(w["freq_hz"].to_numpy())
        values = [w["r"].to_numpy()[idx == b] for b in range(len(BIN_CENTRES))]
        keep = [len(v) >= 5 for v in values]
        ax.plot(
            BIN_CENTRES[keep],
            [float(np.mean(v)) for v, k in zip(values, keep, strict=True) if k],
            "o-",
            ms=4,
            lw=1.4,
            color=colours[session],
            label=str(session),
        )
    ax.axhline(0, color="grey", lw=0.8)
    _log_axis(ax)
    ax.set_title("E  waveform correlation in 3 s windows", fontsize=10)
    ax.set_ylabel("Pearson r vs filtered truth")
    ax.set_xlabel("instantaneous breathing frequency (Hz)")
    ax.legend(fontsize=8, loc="lower left")

    # -- F: exposure -- the reason A-E cannot be compared across sessions ----
    ax = axes[1, 1]
    for session in sessions:
        table = per_session[session]
        share = table["n_breaths"].to_numpy(dtype=float)
        share = share / share.sum()
        ax.step(
            np.append(FREQ_EDGES[:-1], FREQ_EDGES[-1]),
            np.append(share, share[-1]),
            where="post",
            color=colours[session],
            lw=1.6,
            label=str(session),
        )
    _log_axis(ax)
    ax.set_title("F  where each session spends its breaths", fontsize=10)
    ax.set_ylabel("share of that session's breaths")
    ax.set_xlabel("instantaneous breathing frequency (Hz)")
    ax.legend(fontsize=8)

    # -- G: is the session hard, or just breathing fast? ---------------------
    ax = axes[1, 2]
    x = np.arange(len(sessions))
    observed = [100 * standardised[s]["observed_recall"] for s in sessions]
    adjusted = [100 * standardised[s]["standardised_recall"] for s in sessions]
    ax.bar(
        x - 0.2, observed, 0.4, color=[colours[s] for s in sessions], label="observed"
    )
    ax.bar(
        x + 0.2,
        adjusted,
        0.4,
        color=[colours[s] for s in sessions],
        alpha=0.45,
        hatch="//",
        edgecolor="white",
        label="rate-standardised",
    )
    for i, session in enumerate(sessions):
        ax.text(
            i,
            4,
            f"{standardised[session]['mean_freq_hz']:.1f} Hz\ncov {standardised[session]['coverage']:.0%}",
            ha="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sessions])
    ax.set_ylim(0, 118)
    ax.set_ylabel("% breaths detected")
    ax.set_xlabel("reserved session")
    ax.set_title("G  observed vs frequency-standardised recall", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")

    # -- H: does the session effect even hold across its two parts? ---------
    ax = axes[1, 3]
    styles = {1: "-", 2: "--"}
    for (session, part), table in per_clip.items():
        keep = (
            (table["n_breaths"] >= MIN_BIN_BREATHS) & table["recall"].notna()
        ).to_numpy()
        ax.plot(
            BIN_CENTRES[keep],
            100 * table["recall"].to_numpy()[keep],
            styles.get(part, ":"),
            marker="o",
            ms=3,
            lw=1.2,
            color=colours[session],
            label=f"{session} {part}",
        )
    _log_axis(ax)
    ax.set_ylim(0, 102)
    ax.set_title("H  per session-half, same axis as A", fontsize=10)
    ax.set_ylabel("% of ground-truth breaths detected")
    ax.set_xlabel("instantaneous breathing frequency (Hz)")
    ax.legend(fontsize=7, loc="lower left", ncol=1)

    for ax in axes[0]:
        ax.set_xlabel("instantaneous breathing frequency (Hz)")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def rate_breakdown(
    predictions: list[ClipPrediction],
    sessions: Sequence[str],
    out: Path,
    title_suffix: str,
    *,
    tolerance_s: float = EVENT_TOLERANCE_S,
    n_shifts: int = 5,
    corr_window_s: float = 3.0,
    corr_hop_s: float = 1.5,
    corr_min_breaths: int = 3,
) -> None:
    """Frequency-binned breakdown of held-out performance.

    Writes ``out`` (a PNG grid) and ``out.with_suffix(".json")`` (the numbers
    behind it).  *predictions* holds one :class:`ClipPrediction` per scored
    clip -- the caller has already run inference; nothing here does.
    """
    sessions = [short_label(s) for s in dict.fromkeys(sessions)]
    breath_rows, prediction_rows, window_rows = [], [], []
    for cp in predictions:
        clip = canonical_clip(cp.signal, cp.times, cp.truth)
        breaths, preds = breath_table(clip, tolerance_s, n_shifts)
        windows = window_table(clip, corr_window_s, corr_hop_s, corr_min_breaths)
        for frame in (breaths, preds, windows):
            frame["clip_id"] = cp.entry.clip_id
            frame["recording"] = short_label(cp.entry.recording)
            frame["clip"] = cp.entry.clip_id
        breath_rows.append(breaths)
        prediction_rows.append(preds)
        window_rows.append(windows)

    breaths = pd.concat(breath_rows, ignore_index=True)
    preds = pd.concat(prediction_rows, ignore_index=True)
    windows = pd.concat(window_rows, ignore_index=True)

    per_session = {
        s: per_bin_events(
            breaths[breaths["recording"] == s],
            preds[preds["recording"] == s],
        )
        for s in sessions
    }
    per_clip = {
        (session, part): per_bin_events(
            breaths[(breaths["recording"] == session) & (breaths["clip"] == part)],
            preds[(preds["recording"] == session) & (preds["clip"] == part)],
        )
        for session in sessions
        for part in sorted(breaths[breaths["recording"] == session]["clip"].unique())
    }
    pooled = per_bin_events(breaths, preds)
    standardised = standardised_recall(per_session, pooled)
    for s in sessions:
        standardised[s]["mean_freq_hz"] = float(
            breaths[breaths["recording"] == s]["freq_hz"].mean()
        )

    print("\n% of ground-truth breaths detected, by frequency bin (n breaths)")
    header = (
        f"{'bin (Hz)':<13s}"
        + "".join(f"{s!s:>16s}" for s in sessions)
        + f"{'pooled':>16s}"
    )
    print(header)
    print("-" * len(header))
    for b in range(len(BIN_CENTRES)):
        line = f"{FREQ_EDGES[b]:4.1f}-{FREQ_EDGES[b + 1]:<8.1f}"
        for s in sessions:
            row = per_session[s].iloc[b]
            cell = (
                "-"
                if row["n_breaths"] < MIN_BIN_BREATHS or not np.isfinite(row["recall"])
                else f"{100 * row['recall']:.1f} ({int(row['n_breaths'])})"
            )
            line += f"{cell:>16s}"
        row = pooled.iloc[b]
        cell = (
            "-"
            if not np.isfinite(row["recall"])
            else f"{100 * row['recall']:.1f} ({int(row['n_breaths'])})"
        )
        print(line + f"{cell:>16s}")

    print("\npooled, against the rate-preserving null")
    print(
        f"{'bin (Hz)':<13s}{'recall':>9s}{'chance':>9s}{'kappa':>9s}{'F1':>8s}{'n':>8s}"
    )
    for b in range(len(BIN_CENTRES)):
        row = pooled.iloc[b]
        if not np.isfinite(row["recall"]):
            continue
        print(
            f"{FREQ_EDGES[b]:4.1f}-{FREQ_EDGES[b + 1]:<8.1f}"
            f"{100 * row['recall']:8.1f}%{100 * row['chance_recall']:8.1f}%"
            f"{row['skill']:9.2f}{row['f1']:8.2f}{int(row['n_breaths']):8d}"
        )

    print("\nobserved vs frequency-standardised recall")
    for s in sessions:
        d = standardised[s]
        print(
            f"  {s}  mean rate {d['mean_freq_hz']:5.2f} Hz   observed "
            f"{100 * d['observed_recall']:5.1f}%   standardised "
            f"{100 * d['standardised_recall']:5.1f}%   standardised kappa "
            f"{d['standardised_skill']:.2f}   (covers {d['coverage']:.0%} of pooled breaths)"
        )

    _draw_rate_breakdown(
        windows, per_session, per_clip, pooled, standardised, out, title_suffix
    )

    out.with_suffix(".json").write_text(
        json.dumps(
            json_safe(
                {
                    "sessions": sessions,
                    "freq_edges_hz": FREQ_EDGES.tolist(),
                    "min_bin_breaths": MIN_BIN_BREATHS,
                    "tolerance_s": tolerance_s,
                    "corr_window_s": corr_window_s,
                    "null_shifts": n_shifts,
                    "per_session_bins": {
                        s: t.to_dict("records") for s, t in per_session.items()
                    },
                    "per_clip_bins": {
                        f"{session}|{part}": t.to_dict("records")
                        for (session, part), t in per_clip.items()
                    },
                    "pooled_bins": pooled.to_dict("records"),
                    "standardised": standardised,
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {out.with_suffix('.json')}")


# ---------------------------------------------------------------------------
# Reserved grid
# ---------------------------------------------------------------------------

WINDOW_LABELS = (
    (5.0, "worst  (5th pct)"),
    (50.0, "typical (median)"),
    (95.0, "best   (95th pct)"),
)
# Onset-head peaks: lower than a hard 0.5 cutoff, since the Gaussian bump
# target (sigma=20ms) tapers well below 0.5 a couple of frames off centre --
# a 0.5 height would silently miss onsets the head still localises correctly.
ONSET_HEAD_HEIGHT = 0.4
ONSET_HEAD_REFRACTORY_S = 0.050


def _clip_predictions(
    model: BreathingNet,
    entry: ClipEntry,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
) -> dict:
    """Whole-clip truth, prediction, onset-head probability and times, length-aligned."""
    signal, onset_prob = predict_clip(
        model,
        entry,
        mean,
        std,
        window=window,
        device=device,
        frame_chunk=frame_chunk,
        amp_dtype=amp_dtype,
    )
    truth = pd.read_parquet(entry.target)["signal"].to_numpy(np.float32)
    times = np.load(entry.times)
    n = min(len(truth), len(signal), len(times))
    return {
        "truth": truth[:n],
        "pred": signal[:n],
        "onset_prob": onset_prob[:n],
        "times": times[:n],
    }


def _window_correlations(clip: dict, window: int) -> list[dict]:
    """Whole-window Pearson ``r`` on every non-overlapping chunk of a clip."""
    n = len(clip["truth"])
    records = []
    for start in range(0, n - window + 1, window):
        t = clip["truth"][start : start + window]
        p = clip["pred"][start : start + window]
        r = float(np.corrcoef(t, p)[0, 1])
        records.append({"start": start, "r": r})
    return records


def _nearest_to_percentile(records: list[dict], pct: float) -> dict:
    target = np.percentile([rec["r"] for rec in records], pct)
    return min(records, key=lambda rec: abs(rec["r"] - target))


def _plot_panel(ax, clip: dict, start: int, window: int, label: str, r: float) -> None:
    sl = slice(start, start + window)
    t = clip["times"][sl]
    truth, pred, onset_prob = (
        clip["truth"][sl],
        clip["pred"][sl],
        clip["onset_prob"][sl],
    )

    onsets_truth, _ = detect_inhalation_events(truth, FS)
    onsets_submitted, _ = detect_inhalation_events(pred, FS)
    onset_head_distance = max(1, int(FS * ONSET_HEAD_REFRACTORY_S))
    onsets_head, _ = find_peaks(
        onset_prob, height=ONSET_HEAD_HEIGHT, distance=onset_head_distance
    )

    ax.plot(t, truth, color="black", lw=0.9, label="truth (thermistor)")
    ax.plot(t, pred, color="#1f6aa5", lw=0.9, label="prediction")
    # Onset-head probability drawn as a filled trace below the signal.
    ax.fill_between(
        t, onset_prob * 2 - 4, -4, color="#e8944a", alpha=0.6, label="onset-head prob."
    )

    top = max(truth.max(), pred.max()) + 0.6
    if len(onsets_truth):
        ax.plot(
            t[onsets_truth],
            np.full(len(onsets_truth), top),
            "v",
            ms=6,
            color="black",
            label="onsets: truth",
        )
    if len(onsets_submitted):
        ax.plot(
            t[onsets_submitted],
            np.full(len(onsets_submitted), top - 0.5),
            "o",
            ms=5,
            color="#1f6aa5",
            label="onsets: submitted",
        )
    if len(onsets_head):
        ax.plot(
            t[onsets_head],
            np.full(len(onsets_head), top - 1.0),
            "x",
            ms=6,
            color="#c0392b",
            label="onsets: head",
        )

    ax.set_title(f"{label}   r = {r:+.3f}", fontsize=10)
    ax.set_xlim(t[0], t[-1])


def reserved_grid(
    model: BreathingNet,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    entries: list[ClipEntry],
    sessions: Sequence[str],
    out: Path,
    *,
    window: int = 512,
    infer_window: int = 1024,
    frame_chunk: int = 256,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> None:
    """Worst / typical / best 3-column grid of held-out predictions, one row per session.

    For each session, every non-overlapping ``window``-frame chunk across both
    parts of it is scored with a whole-window Pearson ``r`` against ground
    truth, and the chunk nearest the session's own 5th / 50th / 95th
    percentile is plotted -- so each row spans that one session's own range of
    prediction quality rather than a fixed threshold that would not mean the
    same thing across sessions.

    Three event-onset series are shown per panel: the ground-truth onsets, the
    onsets that would be *submitted* (detected on the final z-scored
    prediction, exactly as the scorer would see it), and the onsets read
    directly off the model's onset-head probability curve.

    Writes ``out`` (a PNG grid) and ``out.with_suffix(".json")`` (which chunk
    was picked for each panel).
    """
    sessions = [short_label(s) for s in dict.fromkeys(sessions)]
    by_session: dict[str, list[ClipEntry]] = {s: [] for s in sessions}
    for entry in entries:
        label = short_label(entry.recording)
        if label in by_session and entry.has_target:
            by_session[label].append(entry)

    fig, axes = plt.subplots(len(sessions), 3, figsize=(15, 4.2 * len(sessions)))
    fig.suptitle(
        "Reserved sessions - never trained on, never validated on, never selected on.  "
        "Columns span each session's own range of prediction quality.",
        fontsize=11,
    )
    selections = []
    for row, session in enumerate(sessions):
        session_entries = sorted(by_session[session], key=lambda e: e.clip_id)
        clips = {
            e.clip_id: _clip_predictions(
                model,
                e,
                mean,
                std,
                device,
                window=infer_window,
                frame_chunk=frame_chunk,
                amp_dtype=amp_dtype,
            )
            for e in session_entries
        }
        records = []
        for part, clip in clips.items():
            for rec in _window_correlations(clip, window):
                records.append({**rec, "clip": part})

        for col, (pct, label) in enumerate(WINDOW_LABELS):
            rec = _nearest_to_percentile(records, pct)
            clip = clips[rec["clip"]]
            ax = axes[row, col] if len(sessions) > 1 else axes[col]
            _plot_panel(
                ax,
                clip,
                rec["start"],
                window,
                f"{session} {rec['clip']} {label}",
                rec["r"],
            )
            if row == len(sessions) - 1:
                ax.set_xlabel("time (s)")
            if col == 0:
                ax.set_ylabel("z-scored")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=1)
            selections.append(
                {
                    "session": session,
                    "clip": rec["clip"],
                    "window": label,
                    "t0": float(clip["times"][rec["start"]]),
                    "r": rec["r"],
                }
            )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    out.with_suffix(".json").write_text(json.dumps(selections, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")
