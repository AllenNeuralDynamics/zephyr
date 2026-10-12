"""Suggest which clips (sessions) of a clip list to use, and which stretches of them.

Clips are scored in rolling windows (SNR, amplitude, out-of-band breaths, alternation
breaks), with seconds around artifacts (large excursions) cut out; good runs at least
``--min-span-s`` long become ``included`` spans, bad stretches ``excluded`` spans.  A
clip is suggested ``exclude`` for a dead sensor (the list's low SNR mode), a small
amplitude, inconsistent events, or no accepted span.  Reviewed clips never change.

    zephyr clips select <list.toml> [--out <list.toml>] [--report <csv>] [--min-span-s 300]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .annotate_events import RATE_BAND_HZ, TuneState, parallel_map
from .rates import breath_rates


@dataclass(frozen=True)
class Settings:
    window_s: float = 30.0
    """Width of the rolling windows a clip is scored in."""
    min_span_s: float = 300.0
    """Shortest run of good windows written as an ``included`` span."""
    max_outside: float = 0.10
    """Most breaths a good window may have outside the rate band (fraction)."""
    max_breaks: float = 0.05
    """Most events of a good window that repeat the previous kind (fraction)."""
    outlier_z: float = 3.0
    """Robust z-score of event consistency above which a clip is suggested out."""
    min_snr_gap_db: float = 3.0
    """Smallest gap between the two SNR modes for the dead-sensor gate to apply."""
    snr_ceiling_db: float = 60.0
    """SNRs above this (a flat or quantised signal) count as this when the dead-sensor
    gate is fitted, so one such clip cannot move the gate."""
    min_amplitude: float = 20.0
    """Smallest raw amplitude (mean 5th-95th percentile range over 1 s blocks, in the
    thermistor's units) of a good window, and of a clip suggested include."""
    artifact_z: float = 8.0
    """|z| of the processed trace (as plotted: z-scored over the whole clip) above
    which a sample is an artifact."""
    artifact_pad_s: float = 5.0
    """Seconds excluded either side of an artifact."""


def _noise(x: np.ndarray) -> float:
    step = np.diff(x)
    return 1.4826 * float(np.median(np.abs(step - np.median(step)))) / np.sqrt(2)


def window_table(state: TuneState, i: int, settings: Settings) -> pd.DataFrame:
    """Per-window quality of clip *i*: ``start``, ``end``, ``snr_db``, ``amplitude``,
    ``breaths``, ``outside_frac`` and ``break_frac``."""
    trace = state.trace(i)
    raw = np.asarray(trace.x if trace.raw is None else trace.raw, float)
    t0, fs = float(trace.t[0]), trace.fs
    per = max(1, round(settings.window_s * fs))
    n = max(1, len(raw) // per)
    starts = t0 + np.arange(n) * settings.window_s

    def which(times: np.ndarray) -> np.ndarray:
        return np.clip(((times - t0) // settings.window_s).astype(int), 0, n - 1)

    rows = state.rows(i)
    points = rows[rows["kind"].isin(("inhale", "exhale"))].sort_values("time")
    kinds, times = points["kind"].to_numpy(), points["time"].to_numpy()
    events = np.bincount(which(times), minlength=n) if len(times) else np.zeros(n)
    same = kinds[1:] == kinds[:-1]
    breaks = np.bincount(which(times[:-1][same]), minlength=n) if same.any() else 0
    onsets = np.sort(times[kinds == "inhale"])
    rates = breath_rates(onsets) if len(onsets) > 1 else np.empty(0)
    slot = which(onsets[:-1]) if len(rates) else np.empty(0, int)
    out = (rates < RATE_BAND_HZ[0]) | (rates > RATE_BAND_HZ[1])
    breaths = np.bincount(slot, minlength=n)
    outside = np.bincount(slot[out], minlength=n) if out.any() else np.zeros(n)

    block = max(3, round(fs))
    snr = np.full(n, np.nan)
    amplitude = np.full(n, np.nan)
    for w in range(n):
        x = raw[w * per : (w + 1) * per]
        blocks = x[: len(x) // block * block].reshape(-1, block)
        noise = _noise(x)
        if len(blocks) and noise > 0:
            lo, hi = np.percentile(blocks, [5, 95], axis=1)
            amplitude[w] = float(np.mean(hi - lo))
            snr[w] = 20 * np.log10(amplitude[w] / noise)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(
            {
                "start": starts,
                "end": starts + settings.window_s,
                "snr_db": snr,
                "amplitude": amplitude,
                "breaths": breaths,
                "outside_frac": np.where(breaths > 0, outside / breaths, 1.0),
                "break_frac": np.where(events > 0, breaks / events, 1.0),
            }
        )


def artifacts(
    state: TuneState, i: int, settings: Settings
) -> list[tuple[float, float]]:
    """Merged ``(start, end)`` spans of ``artifact_pad_s`` either side of every
    stretch of clip *i* where the processed trace (the z-scored one the viewer plots)
    goes beyond ``artifact_z`` in either direction."""
    trace = state.trace(i)
    over = np.flatnonzero(np.abs(trace.x) > settings.artifact_z)
    if not len(over):
        return []
    times = trace.t[over]
    first = np.concatenate([[0], np.flatnonzero(np.diff(times) > 0.05) + 1])
    last = np.concatenate([first[1:] - 1, [len(times) - 1]])
    pad = settings.artifact_pad_s
    return merge([(times[a] - pad, times[b] + pad) for a, b in zip(first, last)])


def merge(spans) -> list[tuple[float, float]]:
    """Sorted spans with overlapping or touching ones joined."""
    out: list[list[float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([float(start), float(end)])
    return [(a, b) for a, b in out]


def subtract(spans, cuts) -> list[tuple[float, float]]:
    """*spans* with every part inside one of *cuts* removed."""
    out = list(spans)
    for c0, c1 in cuts:
        pieces = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                pieces.append((a, b))
                continue
            if a < c0:
                pieces.append((a, c0))
            if c1 < b:
                pieces.append((c1, b))
        out = pieces
    return out


def snr_threshold(snr_db: np.ndarray, min_gap_db: float) -> float | None:
    """The split between the two modes of the clips' SNR (Otsu), or None when the
    two groups are not at least *min_gap_db* apart (no dead-sensor mode)."""
    values = np.sort(np.asarray(snr_db, float)[np.isfinite(snr_db)])
    if len(values) < 4:
        return None
    best, cut = -1.0, None
    for k in range(1, len(values)):
        low, high = values[:k], values[k:]
        spread = len(low) * len(high) * (high.mean() - low.mean()) ** 2
        if spread > best:
            best, cut = spread, k
    if values[cut] - values[cut - 1] < min_gap_db:
        return None
    return float((values[cut] + values[cut - 1]) / 2)


def runs(good: np.ndarray, starts: np.ndarray, ends: np.ndarray, value: bool):
    """``(start, end)`` of each run of windows whose flag is *value*."""
    out, k = [], 0
    while k < len(good):
        if good[k] == value:
            j = k
            while j + 1 < len(good) and good[j + 1] == value:
                j += 1
            out.append((float(starts[k]), float(ends[j])))
            k = j + 1
        else:
            k += 1
    return out


def consistency(table: pd.DataFrame) -> pd.Series:
    """Robust z-score (median / MAD over the clips given) of how inconsistent each
    clip's events are: the mean of its positive z-scores of break fraction,
    out-of-band fraction, rate CV and amplitude CV."""
    columns = ["break_frac", "outside_frac", "rate_cv", "amp_cv"]
    z = []
    for column in columns:
        x = table[column].astype(float)
        mad = 1.4826 * float(np.median(np.abs(x - x.median()))) or 1e-9
        z.append(((x - x.median()) / mad).clip(lower=0))
    return pd.concat(z, axis=1).mean(axis=1)


def assess(
    state: TuneState, settings: Settings, progress=None, release: bool = False
) -> pd.DataFrame:
    """One row per clip: its features, spans and suggested selection.  With *release*
    each clip's trace and events are dropped from memory once scored (long lists)."""

    def score(i: int):
        windows = window_table(state, i, settings)
        amplitude, snr = state.signal_quality(i)
        stats = state.breath_stats(i)
        rates = stats["rates"]
        events = max(1, stats["inhales"] + stats["exhales"])
        trace = state.trace(i)
        raw = np.asarray(trace.x if trace.raw is None else trace.raw, float)
        block = max(3, round(trace.fs))
        blocks = raw[: len(raw) // block * block].reshape(-1, block)
        amp = np.subtract(*np.percentile(blocks, [95, 5], axis=1))
        record = {
            "clip": state.entries[i].clip.source.stem,
            "snr_db": snr,
            "amplitude": amplitude,
            "mean_rate": stats["mean"],
            "break_frac": (stats["ii_breaks"] + stats["ee_breaks"]) / events,
            "outside_frac": stats["outside"] / max(1, len(rates)),
            "rate_cv": float(np.std(rates) / np.mean(rates)) if len(rates) else 1.0,
            "amp_cv": float(np.std(amp) / np.mean(amp)) if len(amp) else 1.0,
        }
        result = record, (windows, artifacts(state, i, settings))
        if release:
            entry = state.entries[i]
            entry.trace = entry.rows = None
            entry.stats = {}
        return result

    scored = parallel_map(score, range(len(state.entries)), progress)
    records = [record for record, _ in scored]
    spans = [pair for _, pair in scored]
    table = pd.DataFrame(records)
    cut = snr_threshold(
        np.minimum(table["snr_db"].to_numpy(), settings.snr_ceiling_db),
        settings.min_snr_gap_db,
    )
    dead = table["snr_db"] < cut if cut is not None else pd.Series(False, table.index)
    table["dead_sensor"] = dead
    table["consistency_z"] = np.nan
    if (~dead).sum() >= 4:
        table.loc[~dead, "consistency_z"] = consistency(table[~dead])
    included, excluded, accepted_s = [], [], []
    for windows, cuts in spans:
        good = _good(windows, settings, cut)
        starts, ends = windows["start"].to_numpy(), windows["end"].to_numpy()
        keep = [
            s
            for s in subtract(runs(good, starts, ends, True), cuts)
            if s[1] - s[0] >= settings.min_span_s
        ]
        included.append(keep)
        excluded.append(merge(runs(good, starts, ends, False) + cuts))
        accepted_s.append(sum(e - s for s, e in keep))
    table["artifacts"] = [len(cuts) for _, cuts in spans]
    table["included"], table["excluded"] = included, excluded
    table["accepted_s"] = accepted_s
    table["good_frac"] = [
        float(np.mean(_good(w, settings, cut))) if len(w) else 0.0 for w, _ in spans
    ]
    reasons = []
    for _, row in table.iterrows():
        if row["dead_sensor"]:
            reasons.append(f"SNR {row['snr_db']:.1f} dB below {cut:.1f}")
        elif row["amplitude"] < settings.min_amplitude:
            reasons.append(
                f"amplitude {row['amplitude']:.1f} below {settings.min_amplitude:g}"
            )
        elif row["consistency_z"] > settings.outlier_z:
            reasons.append(f"inconsistent events (z {row['consistency_z']:.1f})")
        elif not row["included"]:
            reasons.append(f"no good run of {settings.min_span_s:g} s")
        else:
            reasons.append("")
    table["reason"] = reasons
    table["suggested"] = np.where(table["reason"] == "", "include", "exclude")
    table["reviewed"] = [e.reviewed for e in state.entries]
    table.attrs["snr_threshold"] = cut
    return table


def _good(windows: pd.DataFrame, settings: Settings, cut: float | None) -> np.ndarray:
    return (
        (windows["outside_frac"] <= settings.max_outside)
        & (windows["break_frac"] <= settings.max_breaks)
        & (windows["breaths"] > 0)
        & (windows["amplitude"] >= settings.min_amplitude)
        & ((windows["snr_db"] >= cut) if cut is not None else True)
    ).to_numpy()


def apply(state: TuneState, table: pd.DataFrame) -> int:
    """Write the suggestions into the unreviewed clips of *state* (selection and both
    span kinds replaced); returns how many clips changed."""
    changed = 0
    for i, row in table.iterrows():
        entry = state.entries[i]
        if entry.reviewed:
            continue
        before = (entry.selection, list(entry.excluded), list(entry.included))
        entry.selection = row["suggested"]
        entry.excluded = sorted(row["excluded"])
        entry.included = sorted(row["included"])
        if (entry.selection, entry.excluded, entry.included) != before:
            entry.dirty = True
            changed += 1
    if not state.accepted():
        raise ValueError("every clip would be excluded; nothing written")
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Write a copy of the list with the suggestions (the input if it is the "
        "same path); without it only the report is printed.",
    )
    parser.add_argument("--report", type=Path, help="Also save the table as CSV.")
    defaults = Settings()
    parser.add_argument("--min-span-s", type=float, default=defaults.min_span_s)
    parser.add_argument("--window-s", type=float, default=defaults.window_s)
    parser.add_argument("--min-amplitude", type=float, default=defaults.min_amplitude)
    parser.add_argument("--artifact-z", type=float, default=defaults.artifact_z)
    parser.add_argument("--artifact-pad-s", type=float, default=defaults.artifact_pad_s)
    args = parser.parse_args(argv)
    settings = Settings(
        window_s=args.window_s,
        min_span_s=args.min_span_s,
        min_amplitude=args.min_amplitude,
        artifact_z=args.artifact_z,
        artifact_pad_s=args.artifact_pad_s,
    )
    state = TuneState(args.clips)
    table = assess(state, settings)
    cut = table.attrs["snr_threshold"]
    print(
        f"SNR gate: {'none (no dead-sensor mode)' if cut is None else f'{cut:.1f} dB'}"
    )
    shown = table.assign(
        accepted_min=(table["accepted_s"] / 60).round(1),
        spans=table["included"].map(len),
        current=[e.selection or "-" for e in state.entries],
    )
    columns = ["clip", "current", "reviewed", "suggested", "snr_db", "consistency_z"]
    columns += ["good_frac", "artifacts", "accepted_min", "spans", "reason"]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(shown[columns].round(2).to_string(index=False))
    if args.report:
        table.drop(columns=["included", "excluded"]).to_csv(args.report, index=False)
    if args.out:
        if args.out.resolve().parent != args.clips.resolve().parent:
            raise SystemExit("--out must be in the same folder (paths are relative)")
        changed = apply(state, table)
        if args.out.resolve() != args.clips.resolve():
            args.out.write_text(args.clips.read_text())
            state.path = args.out
        state.save()
        print(f"{changed} unreviewed clip(s) updated in {args.out}")
