"""Breathing rate of the recordings, and how well Zephyr does at each rate.

A breath's rate is ``1 / (time to the next inhale onset)`` (:mod:`zephyr.rates`).
Test breaths are the scorer's thermistor inhalations, matched to onset-head events
within one frame. Inference only; per-clip results are cached in ``notebooks/cache``.
"""

import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from zephyr import features
from zephyr.dataset import BALANCE_HOP_S
from zephyr.evaluation import EVENT_TOLERANCE_S, match_events
from zephyr.rates import (
    DEFAULT_RATE_BINS_HZ,
    balance_weights,
    breath_rates,
    rate_bin,
    rate_track,
    window_rates,
)
from zephyr.signal import (
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    filter_sniff_signal,
)

from . import results, style

PLOT_BINS_HZ: tuple[float, ...] = DEFAULT_RATE_BINS_HZ
"""Training's default bins, 2-15 Hz; breaths outside them are not plotted."""
WINDOW_S: float = 3.0
"""Length of the windows waveform correlation is measured in."""
CHANCE_SHIFTS_S: tuple[float, ...] = (7.3, 13.1, 21.7, 34.9, 55.3)
"""Circular shifts of the predicted events that give the chance level of recall."""
FS: float = CANONICAL_BREATHING_SAMPLING_RATE


def bin_centres(bins: tuple[float, ...] = PLOT_BINS_HZ) -> np.ndarray:
    """Geometric centre of each rate bin."""
    edges = np.asarray(bins)
    return np.sqrt(edges[:-1] * edges[1:])


def _with_bin(frame: pd.DataFrame) -> pd.DataFrame:
    rate = frame["rate"]
    frame = frame[(rate >= PLOT_BINS_HZ[0]) & (rate < PLOT_BINS_HZ[-1])]
    index = rate_bin(frame["rate"].to_numpy(), PLOT_BINS_HZ)
    return frame.assign(bin=index, centre=bin_centres()[index])


def share_by_bin(
    breaths: pd.DataFrame, by: list[str], within: str | None = None
) -> pd.DataFrame:
    """Share of each group's breaths in each rate bin; pooled rows have ``pooled``
    as every *by* value, pooled separately per *within* value when given."""
    if within is not None:
        return pd.concat(
            [
                share_by_bin(rows, by).assign(**{within: value})
                for value, rows in breaths.groupby(within)
            ],
            ignore_index=True,
        )
    binned = _with_bin(breaths)
    pooled = binned.assign(**{k: "pooled" for k in by})
    out = []
    for frame in (binned, pooled):
        counts = frame.groupby([*by, "bin", "centre"]).size().rename("n").reset_index()
        counts["share"] = counts["n"] / counts.groupby(by)["n"].transform("sum")
        out.append(counts)
    return pd.concat(out, ignore_index=True)


def time_share_by_bin(breaths: pd.DataFrame) -> pd.Series:
    """Share of breathing *time* in each rate bin, pooled: each breath weighted by
    its duration. This is the distribution random training windows see."""
    binned = _with_bin(breaths)
    seconds = (1.0 / binned["rate"]).groupby(binned["centre"]).sum()
    return seconds / seconds.sum()


def train_breaths() -> pd.DataFrame:
    """Every training breath: ``video``, ``recording``, ``rate`` (Hz)."""
    rows = []
    for entry in results.train_entries():
        onsets = np.load(entry.events)["onset_times"]
        rows.append(
            pd.DataFrame(
                {
                    "video": results.short_name(entry),
                    "recording": results.recording_label(entry),
                    "rate": breath_rates(onsets),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def drawn_share_by_bin(power: float | None = None) -> pd.Series:
    """Share of training windows drawn in each rate bin, by the network's own
    ``rate_balance`` and bins (``config.json`` of :data:`results.ZEPHYR_RUN`) unless
    *power* is given; 0 is the unbalanced draw.

    Candidate windows every :data:`BALANCE_HOP_S` of each training clip, each with
    its median rate, weighted as the training dataset weights them (one source).
    """
    params = json.loads((results.ZEPHYR_RUN / "config.json").read_text())["fold"][
        "train_params"
    ]
    power = params["rate_balance"] if power is None else power
    bins, window = params["rate_bins_hz"], params["window"]
    rates = []
    for entry in results.train_entries():
        times = np.load(entry.times)[: entry.n_output]
        track = rate_track(times, np.load(entry.events)["onset_times"])
        starts = np.arange(0, len(times) - window, round(BALANCE_HOP_S * FS))
        rates.append(window_rates(track, starts, window))
    rates = np.concatenate(rates)
    weights = balance_weights(rates, bins, power)
    mass = np.bincount(rate_bin(rates, bins), weights, minlength=len(bins) - 1)
    return pd.Series(mass, index=np.sqrt(np.array(bins[:-1]) * np.array(bins[1:])))


def train_spectra() -> pd.DataFrame:
    """Welch power spectrum of each training video's target, 1-15 Hz, unit area."""
    rows = []
    for entry in results.train_entries():
        signal = pd.read_parquet(entry.target)["signal"].to_numpy()
        freq, power = welch(signal, fs=FS, nperseg=int(8 * FS))
        keep = (freq >= PLOT_BINS_HZ[0]) & (freq <= PLOT_BINS_HZ[-1])
        power = power[keep] / np.trapezoid(power[keep], freq[keep])
        rows.append(
            pd.DataFrame(
                {
                    "video": results.short_name(entry),
                    "recording": results.recording_label(entry),
                    "freq": freq[keep],
                    "power": power,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


BANDS_HZ: tuple[float, ...] = tuple(
    float(e) for e in np.geomspace(PLOT_BINS_HZ[0], PLOT_BINS_HZ[-1], 11)
)
"""Edges of the 10 log-spaced bands correlation is broken down by, 2-15 Hz."""


def band_correlations(
    traces: pd.DataFrame, truth: pd.DataFrame, edges: tuple[float, ...] = BANDS_HZ
) -> pd.DataFrame:
    """Pearson correlation of every method with the thermistor within each band.

    Both are band-passed (zero-phase, so no lag is added) to a band of *edges* and
    correlated over the whole clip. The thermistor is filtered as the training
    target is and resampled onto the traces' grid. Rows are bands, indexed by their
    geometric centre (Hz); columns are methods.
    """
    times = traces[TIME_COLUMN].to_numpy()
    raw_t = truth[TIME_COLUMN].to_numpy()
    raw_fs = 1.0 / float(np.median(np.diff(raw_t)))
    filtered = filter_sniff_signal(truth["Signal"].to_numpy(dtype=float), raw_fs)
    reference = np.interp(times, raw_t, filtered)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = butter(4, (lo, hi), btype="bandpass", fs=FS, output="sos")
        band_reference = sosfiltfilt(sos, reference)
        rows.append(
            {
                method: float(
                    np.corrcoef(band_reference, sosfiltfilt(sos, traces[method]))[0, 1]
                )
                for method in style.METHODS
            }
        )
    return pd.DataFrame(rows, index=bin_centres(edges))


def band_correlation_summary(n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean band correlation of every method over all test clips, with a 95%
    bootstrap interval (clips resampled with replacement).

    Index is the band centre (Hz); columns are ``(method, stat)`` with stat one of
    ``mean``, ``lo``, ``hi``. Needs every test clip's traces (cached per clip).
    """
    per_clip = np.stack(
        [
            band_correlations(
                results.method_traces(c.entry), results.truth(c.entry)
            ).to_numpy()
            for c in results.clips().values()
        ]
    )  # clip, band, method
    rng = np.random.default_rng(seed)
    resampled = per_clip[rng.integers(0, len(per_clip), (n_boot, len(per_clip)))]
    boot = np.nanmean(resampled, axis=1)  # boot, band, method
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    mean = np.nanmean(per_clip, axis=0)
    columns = {
        (method, stat): values[:, i]
        for i, method in enumerate(style.METHODS)
        for stat, values in (("mean", mean), ("lo", lo), ("hi", hi))
    }
    return pd.DataFrame(columns, index=bin_centres(BANDS_HZ))


def _shifted(times: np.ndarray, shift: float, span: tuple[float, float]) -> np.ndarray:
    t0, t1 = span
    return np.sort((times - t0 + shift) % (t1 - t0) + t0)


TRAINING: str = "training"
"""Stratum label of the training clips, scored in-sample for comparison only."""


def _clip_tables(
    entry: features.ClipEntry, stratum: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-breath detection, unmatched head events, and 3 s windows of one clip."""
    truth = results.truth(entry)
    out = results.zephyr_outputs(entry)
    times = out[TIME_COLUMN].to_numpy()
    head = results.head_events(out)
    truth_on, _ = results.events(
        truth[TIME_COLUMN].to_numpy(), truth["Signal"].to_numpy()
    )
    labels = {
        "video": results.short_name(entry),
        "recording": results.recording_label(entry),
        "stratum": stratum,
    }

    rate = breath_rates(truth_on)  # the last breath has no rate and is left out
    # DSP on the trace is kept only as a comparison column for local F1.
    dsp = results.events(times, out["Zephyr"].to_numpy())[0]
    detected, false_rows = {}, []
    for source, predicted in (("head", head), ("DSP", dsp)):
        hit = np.zeros(len(truth_on), bool)
        matched = np.zeros(len(predicted), bool)
        for i, j in match_events(truth_on, predicted, EVENT_TOLERANCE_S):
            hit[i], matched[j] = True, True
        detected[source] = hit
        fp_rate = rate_track(predicted[~matched], truth_on)
        false_rows.append(
            pd.DataFrame({"rate": fp_rate[np.isfinite(fp_rate)], "source": source})
        )
    chance = np.zeros(len(truth_on))
    span = (float(times[0]), float(times[-1]))
    for shift in CHANCE_SHIFTS_S:
        for i, _ in match_events(truth_on, _shifted(head, shift, span)):
            chance[i] += 1 / len(CHANCE_SHIFTS_S)
    breaths = pd.DataFrame(
        {
            "rate": rate,
            "detected": detected["head"][:-1],
            "detected_dsp": detected["DSP"][:-1],
            "chance": chance[:-1],
        }
    ).assign(**labels)
    false_pos = pd.concat(false_rows, ignore_index=True).assign(**labels)

    raw_t = truth[TIME_COLUMN].to_numpy()
    raw_fs = 1.0 / float(np.median(np.diff(raw_t)))
    filtered = filter_sniff_signal(truth["Signal"].to_numpy(dtype=float), raw_fs)
    reference = np.interp(times, raw_t, filtered)
    predicted = out["Zephyr"].to_numpy()
    track = rate_track(times, truth_on)
    n = int(WINDOW_S * FS)
    rows = []
    for s in range(0, len(times) - n + 1, n):
        a, b, r = predicted[s : s + n], reference[s : s + n], track[s : s + n]
        if a.std() > 0 and b.std() > 0 and np.isfinite(r).any():
            rows.append(
                {"rate": float(np.nanmedian(r)), "r": float(np.corrcoef(a, b)[0, 1])}
            )
    windows = pd.DataFrame(rows).assign(**labels)
    return breaths, false_pos, windows


def _tables(
    prefix: str, clips: list[tuple[features.ClipEntry, str]]
) -> dict[str, pd.DataFrame]:
    names = ("breaths", "false_positives", "windows")
    paths = {
        k: results.CACHE / f"{prefix}-rates-{results.ZEPHYR_TAG}-{k}.parquet"
        for k in names
    }
    if all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    parts = [_clip_tables(entry, stratum) for entry, stratum in clips]
    tables = {k: pd.concat(t, ignore_index=True) for k, t in zip(names, zip(*parts))}
    results.CACHE.mkdir(exist_ok=True)
    for k, frame in tables.items():
        frame.to_parquet(paths[k])
    return tables


def test_tables() -> dict[str, pd.DataFrame]:
    """``breaths``, ``false_positives`` and ``windows`` over every test clip.

    Needs Zephyr's outputs on all 24 clips (about a minute each on CPU the first
    time); the tables themselves are cached.
    """
    return _tables("test", [(c.entry, c.stratum) for c in results.clips().values()])


def train_tables() -> dict[str, pd.DataFrame]:
    """The same over the 32 training clips, stratum ``training``.

    In-sample: the network was fitted to these clips, so this is a reference for
    the held-out rows, never a score. Inference as for :func:`test_tables`.
    """
    return _tables("train", [(e, TRAINING) for e in results.train_entries()])


def all_tables() -> dict[str, pd.DataFrame]:
    """Test and training tables stacked; ``stratum`` tells them apart."""
    test, train = test_tables(), train_tables()
    return {k: pd.concat([test[k], train[k]], ignore_index=True) for k in test}


def recall_by_bin(breaths: pd.DataFrame, column: str = "detected") -> pd.DataFrame:
    """Share of thermistor breaths detected per rate bin, per recording and pooled
    within each stratum (``recording == "pooled"``)."""
    binned = _with_bin(breaths)
    per = binned.groupby(["stratum", "recording", "bin", "centre"])[column].mean()
    pooled = binned.groupby(["stratum", "bin", "centre"])[column].mean()
    pooled = pooled.reset_index().assign(recording="pooled")
    return pd.concat([per.reset_index(), pooled], ignore_index=True).rename(
        columns={column: "recall"}
    )


def f1_by_bin(
    breaths: pd.DataFrame, false_pos: pd.DataFrame, source: str = "head"
) -> pd.DataFrame:
    """Local inhale F1 per rate bin: TP and FN binned by the breath's rate, FP by the
    thermistor's rate at the false event. Per recording and pooled. *source* is
    ``head`` or ``DSP`` (on the trace, for comparison only)."""
    column = "detected" if source == "head" else "detected_dsp"
    b = _with_bin(breaths)
    f = _with_bin(false_pos[false_pos["source"] == source])
    keys = ["stratum", "recording", "bin", "centre"]
    frames = []
    for pooled in (False, True):
        bb = b.assign(recording="pooled") if pooled else b
        ff = f.assign(recording="pooled") if pooled else f
        tp = bb.groupby(keys)[column].sum().rename("tp")
        fn = (~bb[column]).groupby([bb[k] for k in keys]).sum().rename("fn")
        fp = ff.groupby(keys).size().rename("fp")
        counts = pd.concat([tp, fn, fp], axis=1).fillna(0)
        counts["f1"] = 2 * counts.tp / (2 * counts.tp + counts.fp + counts.fn)
        frames.append(counts.reset_index())
    return pd.concat(frames, ignore_index=True)
