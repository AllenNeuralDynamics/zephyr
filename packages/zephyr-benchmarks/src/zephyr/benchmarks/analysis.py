"""What a benchmark reports beyond each run's ``evaluation.json``, clip by clip.

From one clip's prediction and truth: event matches, breathing rate in windows and
around task events (:func:`event_aligned`), onset timing, band-limited correlation and
the errors behind a bad clip; over clips: rate agreement, F1 by breathing rate, a
bootstrap over animals and how alike two methods' per-clip scores are. Events are
matched as the scorer matches them (:func:`zephyr.evaluation.match_events`); inhales
of a network with an onset head come from the head, the rest from the clip's own event
detector run on the trace, as the scorer does (:func:`score_prediction`). A prediction
is ``{"times", "signal", "onset", "head"?}``; a truth is :func:`truth`.
"""

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

from zephyr import events, features
from zephyr.evaluate import score_entry
from zephyr.evaluation import frame_tolerance_s, match_events
from zephyr.events import in_spans
from zephyr.signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

RATE_WINDOW_S = 10.0
"""Window over which a breathing rate is counted, for rate agreement."""
LAG_WINDOW_S = 0.1
"""Widest predicted-minus-true onset offset counted for the timing histogram, and the
distance within which a miss and a false event count as one near miss."""
BANDS_HZ = ((1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0))
"""Frequency bands the trace correlation is broken into."""
RATE_BINS_HZ = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
"""Bins of instantaneous breathing rate (1 / interval to the next inhale)."""
LOCAL_S = 0.25
"""Half-width of the window an error's motion and amplitude are measured over."""
TABLES = ("breaths", "false_positives", "lags", "windows", "bands")
"""The tables :func:`clip_tables` returns."""
HEAD_THRESHOLDS = tuple(float(x) for x in np.round(np.arange(0.20, 0.851, 0.05), 2))
"""Onset thresholds a checkpoint without one of its own is given one from, on its
validation clips (``evaluate.choose_head_threshold``)."""


def truth(entry: features.ClipEntry) -> dict:
    """A clip's target (on its output grid, z-scored), its inhale and exhale events,
    the spans excluded from scoring, the raw thermistor and the detector that found
    the events."""
    times = np.load(entry.times)
    target = pd.read_parquet(entry.target)["signal"].to_numpy(np.float64)[: len(times)]
    found = np.load(entry.events)
    raw = pd.read_parquet(entry.thermistor)
    excluded = found["excluded"] if "excluded" in found.files else np.empty((0, 2))
    return {
        "times": times[: len(target)],
        "signal": (target - target.mean()) / (target.std() or 1.0),
        "inhale": found["onset_times"],
        "exhale": found["offset_times"],
        "excluded": excluded,
        "raw_t": raw["Time"].to_numpy(),
        "raw": raw["Signal"].to_numpy(),
        "detector": getattr(entry, "detector", None),
    }


def _frame(p: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {TIME_COLUMN: p["times"], BREATHING_SIGNAL_COLUMN: p["signal"].astype(float)}
    )


def score_prediction(entry: features.ClipEntry, p: dict, head: bool) -> dict:
    """One clip's scores as ``evaluate`` gives them (:func:`zephyr.evaluate.score_entry`),
    from a prediction already made: correlation, DSP inhale and exhale F1 and KL-IBI,
    and with an onset head its inhale F1 (``head_inhale_f1``)."""
    predicted = _frame(p)
    row = score_entry(entry, predicted).to_dict()
    if head:
        row["head_inhale_f1"] = score_entry(
            entry, predicted, predicted_onset_times_s=p["head"]
        ).inhale_f1
    return row


def predicted_inhales(p: dict, head: bool, detector=None) -> np.ndarray:
    """Inhale times a method asserts: its head's events, or the inhales *detector*
    (the clip's :class:`zephyr.config.EventsSource`; the default detector without
    one) finds on its trace."""
    if head:
        return p["head"]
    method = detector.method if detector is not None else events.DEFAULT_METHOD
    params = detector.params if detector is not None else None
    return events.detect(method, _frame(p), params=params).inhale


def _scored(times: np.ndarray, t: dict, span: tuple[float, float]) -> np.ndarray:
    """*times* inside *span* and outside the clip's excluded spans."""
    keep = (times >= span[0]) & (times <= span[1])
    if len(t["excluded"]):
        keep &= ~in_spans(times, t["excluded"])
    return times[keep]


def clip_events(p: dict, t: dict, head: bool) -> dict:
    """True and predicted inhales of one clip in the span both cover, which of them
    matched (``hit``, ``used``), and each true breath's rate."""
    span = (max(p["times"][0], t["times"][0]), min(p["times"][-1], t["times"][-1]))
    true_on = _scored(t["inhale"], t, span)
    pred_on = _scored(predicted_inhales(p, head, t.get("detector")), t, span)
    pairs = match_events(true_on, pred_on, frame_tolerance_s(p["times"]))
    hit = np.zeros(len(true_on), bool)
    used = np.zeros(len(pred_on), bool)
    for i, j in pairs:
        hit[i], used[j] = True, True
    rate = np.full(len(true_on), np.nan)
    rate[:-1] = 1.0 / np.diff(true_on)
    return {
        "span": span,
        "true": true_on,
        "pred": pred_on,
        "hit": hit,
        "used": used,
        "rate": rate,
    }


def _local_rate(at: np.ndarray, true_on: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """The true breathing rate at times *at*: that of the breath they fall in."""
    if len(true_on) < 2:
        return np.full(len(at), np.nan)
    i = np.clip(np.searchsorted(true_on, at) - 1, 0, len(true_on) - 2)
    return rate[i]


def breathing_rate(inhales: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Instantaneous breathing rate (Hz) at times *at*: 1 / the interval of the breath
    each falls in; NaN before the first inhale and from the last."""
    out = np.full(len(at), np.nan)
    if len(inhales) < 2:
        return out
    i = np.searchsorted(inhales, at, side="right") - 1
    inside = (i >= 0) & (i < len(inhales) - 1)
    out[inside] = 1.0 / np.diff(inhales)[i[inside]]
    return out


def event_aligned(
    times: np.ndarray, values: np.ndarray, at: np.ndarray, lags: np.ndarray
) -> np.ndarray:
    """*values* (sampled at *times*) around each event in *at*, at *lags* (s) from it:
    one row per event, NaN where a lag falls outside *times*."""
    rows = [np.interp(t + lags, times, values, left=np.nan, right=np.nan) for t in at]
    return np.array(rows, float).reshape(len(at), len(lags))


def band_correlations(a: np.ndarray, b: np.ndarray, fs: float) -> dict[str, float]:
    """Correlation of *a* and *b* after both are band-passed to each of :data:`BANDS_HZ`."""
    out = {}
    for lo, hi in BANDS_HZ:
        sos = butter(2, (lo, min(hi, 0.45 * fs)), btype="bandpass", fs=fs, output="sos")
        r = np.corrcoef(sosfiltfilt(sos, a), sosfiltfilt(sos, b))[0, 1]
        out[f"{lo:g}-{hi:g} Hz"] = float(r)
    return out


def _rate_windows(ev: dict, t: dict) -> pd.DataFrame:
    rows = []
    edges = np.arange(ev["span"][0], ev["span"][1] - RATE_WINDOW_S, RATE_WINDOW_S)
    for w0 in edges:
        w1 = w0 + RATE_WINDOW_S
        ex = t["excluded"]
        if len(ex) and np.any((ex[:, 0] < w1) & (ex[:, 1] > w0)):
            continue
        n_true = np.sum((ev["true"] >= w0) & (ev["true"] < w1))
        n_pred = np.sum((ev["pred"] >= w0) & (ev["pred"] < w1))
        rows.append(
            {
                "start": w0,
                "true_hz": n_true / RATE_WINDOW_S,
                "pred_hz": n_pred / RATE_WINDOW_S,
            }
        )
    return pd.DataFrame(rows, columns=["start", "true_hz", "pred_hz"])


def clip_tables(p: dict, t: dict, head: bool) -> dict[str, pd.DataFrame]:
    """One clip's :data:`TABLES`: every true breath (rate, detected), every false event
    (true rate there), the lag of every true inhale with a prediction within
    :data:`LAG_WINDOW_S` (and that breath's rate), the counted rates per window, and the
    band correlations."""
    ev = clip_events(p, t, head)
    fp_t = ev["pred"][~ev["used"]]
    wide = match_events(ev["true"], ev["pred"], LAG_WINDOW_S)
    n = min(len(p["signal"]), len(t["signal"]))
    fs = 1.0 / float(np.median(np.diff(p["times"])))
    return {
        "breaths": pd.DataFrame(
            {"time": ev["true"], "rate_hz": ev["rate"], "detected": ev["hit"]}
        ),
        "false_positives": pd.DataFrame(
            {"time": fp_t, "rate_hz": _local_rate(fp_t, ev["true"], ev["rate"])}
        ),
        "lags": pd.DataFrame(
            {
                "lag_s": [ev["pred"][j] - ev["true"][i] for i, j in wide],
                "rate_hz": [ev["rate"][i] for i, _ in wide],
            },
            dtype=float,
        ),
        "windows": _rate_windows(ev, t),
        "bands": pd.DataFrame(
            [band_correlations(p["signal"][:n], t["signal"][:n], fs)]
        ),
    }


# --- summaries over clips --------------------------------------------------------------


def rate_agreement(
    windows: pd.DataFrame, by: Sequence[str] = ("method", "set")
) -> pd.DataFrame:
    """Per group of *by* (method and set by default): MAE, RMSE, MAPE (over windows
    with a true rate) and Pearson r of the windowed breathing rate, and the
    Bland-Altman bias and 95% limits (predicted minus true, in Hz)."""

    def one(g: pd.DataFrame) -> pd.Series:
        d = g["pred_hz"] - g["true_hz"]
        rated = g["true_hz"] > 0
        return pd.Series(
            {
                "MAE (Hz)": d.abs().mean(),
                "RMSE (Hz)": np.sqrt((d**2).mean()),
                "MAPE (%)": 100 * (d[rated].abs() / g.loc[rated, "true_hz"]).mean(),
                "r": np.corrcoef(g["pred_hz"], g["true_hz"])[0, 1],
                "bias (Hz)": d.mean(),
                "limits (Hz)": 1.96 * d.std(),
                "windows": len(g),
            }
        )

    return windows.groupby(list(by), sort=False).apply(one, include_groups=False)


def rate_bins(bins=RATE_BINS_HZ) -> list[str]:
    """Labels of the breathing-rate bins, ``"lo-hi"`` in Hz."""
    return [f"{lo:g}-{hi:g}" for lo, hi in itertools.pairwise(bins)]


def f1_by_rate(
    breaths: pd.DataFrame, false: pd.DataFrame, bins=RATE_BINS_HZ
) -> pd.DataFrame:
    """Inhale F1 per breathing-rate bin, per method and set (and fold, when the tables
    have a ``fold`` column): misses binned by the breath's own rate, false events by the
    true rate at their time; with the share of breaths in each bin."""
    cut = {"bins": list(bins), "labels": rate_bins(bins)}
    b = breaths.assign(bin=pd.cut(breaths["rate_hz"], **cut))
    f = false.assign(bin=pd.cut(false["rate_hz"], **cut))
    groups = ["method", *(["fold"] if "fold" in breaths else []), "set"]
    keys = [*groups, "bin"]
    tp = b[b["detected"]].groupby(keys, observed=False).size()
    fn = b[~b["detected"]].groupby(keys, observed=False).size()
    fp = f.groupby(keys, observed=False).size()
    out = pd.DataFrame({"tp": tp, "fn": fn, "fp": fp}).fillna(0)
    out["f1"] = (
        2 * out["tp"] / (2 * out["tp"] + out["fn"] + out["fp"]).replace(0, np.nan)
    )
    in_bin = out["tp"] + out["fn"]
    out["share"] = in_bin / in_bin.groupby(groups).transform("sum")
    return out.reset_index()


def rate_distribution(onsets: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """Share of breaths per rate bin in each named group of clips, from each clip's
    true inhale times."""
    rows = []
    for label, clips in onsets.items():
        rates = np.concatenate([1.0 / np.diff(on) for on in clips]) if clips else []
        share = pd.cut(
            pd.Series(rates, dtype=float), list(RATE_BINS_HZ), labels=rate_bins()
        ).value_counts(normalize=True, sort=False)
        rows += [{"clips": label, "bin": b, "share": s} for b, s in share.items()]
    return pd.DataFrame(rows)


def bootstrap(
    recordings: pd.DataFrame,
    metric: str,
    reference: str = "Zephyr",
    n: int = 2000,
    seed: int = 0,
    unit: str = "subject",
) -> pd.DataFrame:
    """Mean of *metric* over units with a 95% bootstrap interval, resampling units
    (each the mean of its rows: recordings, seeds), per ``set`` and ``method``; and the
    paired difference to *reference* on the same resamples. A unit is the value of the
    *unit* column: the animal by default, or e.g. an animal on one rig."""
    rng = np.random.default_rng(seed)
    per_animal = (
        recordings.groupby(["set", "method", unit])[metric].mean().unstack("method")
    )
    rows = []
    for name, table in per_animal.groupby(level="set"):
        table = table.droplevel("set")
        draws = rng.integers(0, len(table), size=(n, len(table)))
        values = table.to_numpy()
        means = np.stack([np.nanmean(values[d], axis=0) for d in draws])
        methods = list(table.columns)
        ref = methods.index(reference) if reference in methods else None
        for k, method in enumerate(methods):
            row = {
                "set": name,
                "method": method,
                "mean": np.nanmean(values[:, k]),
                "low": np.nanpercentile(means[:, k], 2.5),
                "high": np.nanpercentile(means[:, k], 97.5),
                "units": int(np.isfinite(values[:, k]).sum()),
            }
            if ref is not None and k != ref:
                diff = means[:, ref] - means[:, k]
                row |= {
                    f"{reference} minus": np.nanmean(values[:, ref] - values[:, k]),
                    "diff low": np.nanpercentile(diff, 2.5),
                    "diff high": np.nanpercentile(diff, 97.5),
                }
            rows.append(row)
    return pd.DataFrame(rows)


RIG_PAIR = "same animal, other rig"
ANIMAL_PAIR = "same rig, other animal"


def rig_animal_pairs(
    recordings: pd.DataFrame, metric: str, animal: str = "subject", rig: str = "rig"
) -> pd.DataFrame:
    """Every pair of units (an animal on one rig, *metric* averaged over its rows) that
    differ in exactly one of the two: the same animal on two rigs (``kind``
    :data:`RIG_PAIR`) or two animals on one rig (:data:`ANIMAL_PAIR`), with the
    absolute difference of their means (``difference``)."""
    units = recordings.groupby([animal, rig])[metric].mean().dropna().reset_index()
    a = units[animal].astype(str).to_numpy()
    r = units[rig].astype(str).to_numpy()
    v = units[metric].to_numpy()
    i, j = np.triu_indices(len(units), k=1)
    same_animal, same_rig = a[i] == a[j], r[i] == r[j]
    keep = same_animal != same_rig
    return pd.DataFrame(
        {
            "kind": np.where(same_animal[keep], RIG_PAIR, ANIMAL_PAIR),
            "animal_1": a[i][keep],
            "animal_2": a[j][keep],
            "rig_1": r[i][keep],
            "rig_2": r[j][keep],
            "difference": np.abs(v[i] - v[j])[keep],
        }
    )


def rig_animal_bootstrap(
    pairs: pd.DataFrame, n: int = 2000, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """Mean absolute difference of each kind of pair (:func:`rig_animal_pairs`) with an
    interval from resampling animals (a pair counts as often as the product of its two
    animals' draws), and the between-animal mean minus the between-rig mean on the
    same draws (``kind`` ``"difference"``)."""
    animals = np.unique(pairs[["animal_1", "animal_2"]].to_numpy().ravel())
    index = {name: k for k, name in enumerate(animals)}
    first = pairs["animal_1"].map(index).to_numpy()
    second = pairs["animal_2"].map(index).to_numpy()
    d = pairs["difference"].to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(animals), (n, len(animals)))
    counts = np.zeros((n, len(animals)))
    np.add.at(counts, (np.arange(n)[:, None], draws), 1)
    weight = counts[:, first] * counts[:, second]
    tail = 100 * (1 - level) / 2
    means, rows = {}, []
    for kind in (RIG_PAIR, ANIMAL_PAIR):
        mask = (pairs["kind"] == kind).to_numpy()
        w = weight[:, mask]
        with np.errstate(invalid="ignore"):
            means[kind] = (w * d[mask]).sum(axis=1) / w.sum(axis=1)
        rows.append(
            {
                "kind": kind,
                "mean": d[mask].mean(),
                "low": np.nanpercentile(means[kind], tail),
                "high": np.nanpercentile(means[kind], 100 - tail),
                "pairs": int(mask.sum()),
                "animals": len(
                    set(pairs.loc[mask, "animal_1"]) | set(pairs.loc[mask, "animal_2"])
                ),
            }
        )
    diff = means[ANIMAL_PAIR] - means[RIG_PAIR]
    rows.append(
        {
            "kind": "difference",
            "mean": rows[1]["mean"] - rows[0]["mean"],
            "low": np.nanpercentile(diff, tail),
            "high": np.nanpercentile(diff, 100 - tail),
            "pairs": len(pairs),
            "animals": len(animals),
        }
    )
    return pd.DataFrame(rows)


def familiarity(
    recordings: pd.DataFrame, trained: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Per row of *recordings* (a ``method``, ``subject`` and ``rig`` each), what that
    method trained on, from *trained* (method -> its training sessions, with ``subject``
    and ``rig``): ``animal`` (``"same rig"``, ``"other rig"`` or ``"new"``) and
    ``rig_trained`` (whether any training session came from the rig)."""
    animal, rig_trained = [], []
    for m, s, r in zip(
        recordings["method"], recordings["subject"], recordings["rig"], strict=True
    ):
        t = trained[m]
        mine = t[t["subject"].astype(str) == str(s)]
        animal.append(
            "new"
            if mine.empty
            else "same rig"
            if r in set(mine["rig"])
            else "other rig"
        )
        rig_trained.append(r in set(t["rig"]))
    return pd.DataFrame(
        {"animal": animal, "rig_trained": rig_trained}, index=recordings.index
    )


def across_folds(
    per_fold: pd.DataFrame,
    value: str,
    by: Sequence[str] = ("method",),
    fold: str = "fold",
    level: float = 0.95,
) -> pd.DataFrame:
    """Per group of *by*: the mean of *value* over folds (one row per fold), its
    *level* t-interval over folds and the number of folds; with one fold the interval is
    missing."""

    def one(g: pd.DataFrame) -> pd.Series:
        v = g[value].dropna().to_numpy(float)
        n = len(v)
        mean = v.mean() if n else np.nan
        half = (
            stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
            if n > 1
            else np.nan
        )
        return pd.Series(
            {"mean": mean, "low": mean - half, "high": mean + half, "folds": n}
        )

    return (
        per_fold.groupby(list(by), sort=False)[[value, fold]].apply(one).reset_index()
    )


def per_fold(
    recordings: pd.DataFrame,
    metric: str,
    unit: str = "unit",
    fold: str = "fold",
    by: Sequence[str] = ("set", "method"),
) -> pd.DataFrame:
    """Per group of *by* and fold: *metric* averaged within each unit, then over the
    units (column ``value``)."""
    keys = [*by, fold]
    return (
        recordings.groupby([*keys, unit], sort=False)[metric]
        .mean()
        .groupby(keys, sort=False)
        .mean()
        .rename("value")
        .reset_index()
    )


def fold_scores(
    recordings: pd.DataFrame,
    metric: str,
    unit: str = "unit",
    fold: str = "fold",
    by: Sequence[str] = ("set", "method"),
    reference: str | None = "Zephyr",
    level: float = 0.95,
) -> pd.DataFrame:
    """Per group of *by*: *metric* averaged within each unit, then over the units of
    each fold, then summarised over folds (:func:`across_folds`); with *reference*,
    also each method minus the reference, paired by fold."""
    folds = per_fold(recordings, metric, unit, fold, by)
    out = across_folds(folds, "value", by, fold, level)
    if reference is None or "method" not in by:
        return out
    others = [k for k in by if k != "method"]
    ref = folds[folds["method"] == reference].drop(columns="method")
    paired = folds.merge(ref, on=[*others, fold], suffixes=("", "_ref"))
    paired["value"] = paired["value"] - paired["value_ref"]
    diff = across_folds(paired, "value", by, fold, level).rename(
        columns={"mean": "minus ref", "low": "diff low", "high": "diff high"}
    )
    return out.merge(diff.drop(columns="folds"), on=list(by), how="left")


def clip_scores_wide(clips: pd.DataFrame, metric: str) -> pd.DataFrame:
    """*metric* of every clip (rows, indexed by ``set`` and ``clip_id``; seeds averaged)
    for every method (columns)."""
    return clips.groupby(["set", "clip_id", "method"])[metric].mean().unstack("method")


def score_agreement(
    clips: pd.DataFrame, metric: str, method: str = "spearman"
) -> pd.DataFrame:
    """How alike two methods' scores are over clips: the correlation (*method*, on the
    clips both scored) of their per-clip *metric*, for every pair of methods. High means
    they find the same clips hard, whatever their mean scores."""
    return clip_scores_wide(clips, metric).corr(method=method)


# --- frame embeddings ------------------------------------------------------------------


@torch.no_grad()
def frame_embeddings(
    model,
    mean,
    std,
    entry: features.ClipEntry,
    seconds: float,
    device,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """A per-frame encoder's (``model.encode``) embeddings over the first *seconds* of
    a clip, before they are resampled onto the output grid: ``(frame_times, (T, D))``."""
    times = np.load(entry.frame_times)
    n = int(np.searchsorted(times, times[0] + seconds))
    stack = features.reader(entry).read(slice(0, n)).astype(np.float32)
    frames = torch.from_numpy((stack - mean) / std).unsqueeze(0).to(device)
    return times[:n], model.encode(frames, chunk=chunk)[0].float().cpu().numpy()


def interpolation_r2(times: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per column of ``(T, D)`` *x*: R² of rebuilding each sample by linear
    interpolation between its two neighbours (at the real *times*); 1 is smooth,
    about -0.5 is white noise."""
    x = x.reshape(len(x), -1)
    weight = ((times[1:-1] - times[:-2]) / (times[2:] - times[:-2]))[:, None]
    rebuilt = x[:-2] + (x[2:] - x[:-2]) * weight
    middle = x[1:-1]
    return 1 - ((middle - rebuilt) ** 2).mean(0) / middle.var(0)


def mean_spectrum(times: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch density of each z-scored column of *x*, averaged and scaled to unit area."""
    x = x.reshape(len(x), -1)
    fs = 1.0 / float(np.median(np.diff(times)))
    z = (x - x.mean(0)) / x.std(0)
    freq, power = welch(z, fs=fs, nperseg=int(4 * fs), axis=0)
    power = power.mean(1)
    return freq, power / np.trapezoid(power, freq)


# --- failure modes ---------------------------------------------------------------------

MISSED, FALSE = "missed (FN)", "false (FP)"
CAUSES: dict[str, Callable[[pd.Series], bool]] = {
    "near miss (other error within 100 ms)": lambda e: e["near_miss"],
    "fast breathing (> 8 Hz)": lambda e: e["rate_hz"] > 8,
    "face motion (top 10% of the clip)": lambda e: e["motion"] > 0.9,
    "shallow breath (bottom 25%)": lambda e: e["amplitude"] < 0.25,
}
"""What each error is checked against, in order; an error can have several, or none."""
NO_CAUSE = "none of these"


def _around(
    times: np.ndarray, values: np.ndarray, at: np.ndarray, reduce
) -> np.ndarray:
    """``reduce`` of *values* within :data:`LOCAL_S` of each time in *at*."""
    out = np.full(len(at), np.nan)
    for k, a in enumerate(at):
        m = (times >= a - LOCAL_S) & (times <= a + LOCAL_S)
        if m.any():
            out[k] = reduce(values[m])
    return out


def errors(
    p: dict, t: dict, head: bool, frame_times: np.ndarray, frames: np.ndarray
) -> pd.DataFrame:
    """Every missed true inhale and unmatched predicted inhale of one clip, with the
    true rate there, the breath's size and the face motion around it (each a quantile
    of the clip's true breaths), and the :data:`CAUSES` it carries. *frames* are the
    gray crops ``(T, 1, H, W)`` at *frame_times*."""
    ev = clip_events(p, t, head)
    motion = np.abs(np.diff(frames[:, 0].astype(np.float32), axis=0)).mean(axis=(1, 2))
    motion_t = frame_times[1:]
    df = pd.concat(
        [
            pd.DataFrame({"time": ev["true"][~ev["hit"]], "kind": MISSED}),
            pd.DataFrame({"time": ev["pred"][~ev["used"]], "kind": FALSE}),
        ],
        ignore_index=True,
    ).sort_values("time", ignore_index=True)
    at = df["time"].to_numpy()
    df["rate_hz"] = _local_rate(at, ev["true"], ev["rate"])
    by_kind = {k: df.loc[df["kind"] == k, "time"].to_numpy() for k in (MISSED, FALSE)}
    other = [by_kind[FALSE] if k == MISSED else by_kind[MISSED] for k in df["kind"]]
    df["near_miss"] = np.array(
        [
            bool(len(o) and np.min(np.abs(o - t0)) <= LAG_WINDOW_S)
            for t0, o in zip(at, other, strict=True)
        ],
        dtype=bool,
    )  # bool even for a clip without errors, so clips concatenate as bool
    clip_motion = _around(motion_t, motion, ev["true"], np.mean)
    clip_size = _around(t["times"], t["signal"], ev["true"], np.ptp)
    df["motion"] = [
        np.mean(clip_motion <= v) for v in _around(motion_t, motion, at, np.mean)
    ]
    df["amplitude"] = [
        np.mean(clip_size <= v) for v in _around(t["times"], t["signal"], at, np.ptp)
    ]
    df["causes"] = [
        ", ".join(c for c, rule in CAUSES.items() if rule(row)) or NO_CAUSE
        for _, row in df.iterrows()
    ]
    return df


def cause_counts(err: pd.DataFrame) -> pd.DataFrame:
    """How many errors of each kind carry each cause (an error may carry several)."""
    rows = []
    for kind, g in err.groupby("kind"):
        for cause in [*CAUSES, NO_CAUSE]:
            n = int(g["causes"].str.contains(cause, regex=False).sum())
            rows.append({"kind": kind, "cause": cause, "errors": n, "of": len(g)})
    return pd.DataFrame(rows)


def timing_by_rate(
    lags: pd.DataFrame,
    breaths: pd.DataFrame,
    bins=RATE_BINS_HZ,
    tolerances_ms: Sequence[float] = (25.0, 50.0),
    cycle_share: float = 0.1,
) -> pd.DataFrame:
    """Per breathing-rate bin (and method, and fold when the tables have one): the share
    of true breaths found within each fixed tolerance and within *cycle_share* of the
    breath's own cycle, the shares found beyond the first tolerance but within
    :data:`LAG_WINDOW_S` (``near``) and not within it (``absent``), and the spread (IQR)
    of the lags in ms and as a percentage of the cycle (lag times rate)."""
    cut = {"bins": list(bins), "labels": rate_bins(bins)}
    keys = ["method", *(["fold"] if "fold" in breaths else [])]
    b = breaths.assign(bin=pd.cut(breaths["rate_hz"], **cut))
    lag = lags.assign(
        bin=pd.cut(lags["rate_hz"], **cut),
        ms=1000 * lags["lag_s"].abs(),
        signed_ms=1000 * lags["lag_s"],
        cycle=100 * lags["lag_s"] * lags["rate_hz"],
    )
    by = [*keys, "bin"]
    n = b.groupby(by, observed=False).size()
    g = lag.groupby(by, observed=False)
    out = pd.DataFrame({"breaths": n})
    for t in tolerances_ms:
        out[f"within {t:g} ms"] = g["ms"].apply(lambda s, t=t: (s <= t).sum()) / n
    out["within cycle"] = (
        g["cycle"].apply(lambda s: (s.abs() <= 100 * cycle_share).sum()) / n
    )
    first = tolerances_ms[0]
    out["near"] = g["ms"].apply(lambda s: (s > first).sum()) / n
    out["absent"] = 1 - g.size() / n
    out["iqr_ms"] = g["signed_ms"].quantile(0.75) - g["signed_ms"].quantile(0.25)
    out["iqr_cycle"] = g["cycle"].quantile(0.75) - g["cycle"].quantile(0.25)
    return out.reset_index()


def tolerance_curve(
    lags: pd.DataFrame,
    breaths: pd.DataFrame,
    false: pd.DataFrame,
    tolerances_ms: Sequence[float],
) -> pd.DataFrame:
    """Inhale F1 and the errors left (misses plus false inhales, as a share of true
    breaths; also apart, ``misses`` and ``false``) as the matching tolerance is relaxed,
    per method (and fold, when the tables have one): a true inhale counts as found when
    a prediction lies within the tolerance (pairs from the timing table, so up to
    :data:`LAG_WINDOW_S`)."""
    keys = ["method", *(["fold"] if "fold" in breaths else [])]
    rows = []
    n_false = false.groupby(keys).size()
    for key, b in breaths.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        m = np.ones(len(lags), bool)
        for k, v in zip(keys, key, strict=True):
            m &= lags[k].to_numpy() == v
        lag = np.abs(lags.loc[m, "lag_s"].to_numpy()) * 1000
        n_true = len(b)
        n_pred = int(b["detected"].sum()) + int(
            n_false.get(key if len(key) > 1 else key[0], 0)
        )
        for t in tolerances_ms:
            hits = int((lag <= t).sum())
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "tolerance_ms": float(t),
                    "f1": 2 * hits / (n_true + n_pred),
                    "errors": (n_true + n_pred - 2 * hits) / n_true,
                    "misses": (n_true - hits) / n_true,
                    "false": (n_pred - hits) / n_true,
                }
            )
    return pd.DataFrame(rows)


def error_profile(err: pd.DataFrame, fast_hz: float = 8.0) -> pd.DataFrame:
    """Per error kind: the share of errors that are near misses, then, among the rest
    (gross errors), the share in fast breathing (above *fast_hz*), in face motion (top
    10% of the clip) and on shallow breaths (bottom 25%), with ``n`` errors behind each
    share; per fold too, when *err* has a ``fold`` column."""
    keys = ["kind", *(["fold"] if "fold" in err else [])]
    rows = []
    for key, g in err.groupby(keys, sort=False):
        label = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        rows.append(
            {
                **label,
                "row": "near miss",
                "share": g["near_miss"].mean(),
                "n": len(g),
            }
        )
        gross = g[~g["near_miss"]]
        for row, flag in (
            (f"fast breathing (> {fast_hz:g} Hz)", gross["rate_hz"] > fast_hz),
            ("face motion (top 10%)", gross["motion"] > 0.9),
            ("shallow breath (bottom 25%)", gross["amplitude"] < 0.25),
        ):
            rows.append({**label, "row": row, "share": flag.mean(), "n": len(gross)})
    return pd.DataFrame(rows)


# --- quality control against the truth ---------------------------------------------------


def quality_agreement(
    table: pd.DataFrame, measures: dict[str, str | None], truth: str
) -> pd.DataFrame:
    """Per quality measure: Spearman correlation with the true score *truth* over the
    rows (videos or windows) both have, and the same oriented so that positive means
    the measure ranks as it should (*measures* maps each to ``"higher"``/``"lower"``
    being better, or ``None``: oriented as measured)."""
    rows = []
    for name, better in measures.items():
        if name not in table:
            continue
        pair = table[[name, truth]].dropna()
        rho = (
            float(pair[name].corr(pair[truth], method="spearman"))
            if len(pair) > 2
            else np.nan
        )
        rows.append(
            {
                "measure": name,
                "rho": rho,
                "oriented": -rho if better == "lower" else rho,
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows)


def retention(
    table: pd.DataFrame,
    measure: str,
    truth: str,
    better: str,
    fractions: tuple[float, ...] = tuple(i / 10 for i in range(1, 11)),
) -> pd.DataFrame:
    """Mean true score of the rows a measure keeps: the best *fraction* by the measure
    (``better`` is ``"higher"`` or ``"lower"``), for each fraction."""
    pair = table[list(dict.fromkeys([measure, truth]))].dropna()
    ranked = pair.sort_values(measure, ascending=better == "lower")[truth].to_numpy()
    out = []
    for fraction in fractions:
        n = max(1, round(fraction * len(ranked)))
        out.append({"kept": float(fraction), "mean": float(ranked[:n].mean())})
    return pd.DataFrame(out)
