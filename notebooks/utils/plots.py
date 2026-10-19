"""Reusable figure components. Every colour and size comes from :mod:`.style`.

Times on a clip's traces are seconds from the clip's first output sample.
"""

import string
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from . import style

FIGURES: Path = Path(__file__).resolve().parents[1] / "figures"
CROP_EDGE: str = "#d55e00"
"""Outline of the crop the network sees, on the whole frame."""
METRIC_LABELS: dict[str, str] = {
    "correlation": "Correlation",
    "inhale": "Inhale F1",
    "exhale_f1": "Exhale F1 (DSP)",
    "kl_ibi": "KL-IBI (lower is better)",
}
TOLERANCE_MS = 25.0
"""The scorer's event tolerance at 60 Hz (1.5 frames), drawn on timing plots."""


def panel_letters(axes: Axes | Sequence[Axes] | NDArray) -> None:
    """Label panels a, b, c... in reading order."""
    for letter, ax in zip(string.ascii_lowercase, np.ravel(axes), strict=False):
        ax.text(-0.02, 1.05, letter, transform=ax.transAxes, weight="bold", size=9)


def save(fig: Figure, name: str) -> None:
    """Write *fig* as PNG and SVG into ``notebooks/figures``."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}")


def mark_events(
    ax: Axes, times: NDArray, y: float, color: str, marker: str = "v", size: float = 3
) -> None:
    """Mark event times along height *y* of an axes."""
    ax.plot(
        times,
        np.full(len(times), y),
        marker,
        color=color,
        markersize=size,
        clip_on=False,
        lw=0,
    )


def onset_head(ax: Axes, times: NDArray, probability: NDArray) -> None:
    """The onset head's probability, with its 0.5 threshold."""
    ax.plot(times, probability, color=style.ONSET)
    ax.axhline(0.5, color="0.6", linewidth=0.5, linestyle=":")
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Onset probability")


def _inside(x: NDArray, t0: float, t1: float) -> NDArray:
    return (x >= t0) & (x <= t1)


def _crop_image(ax: Axes, image: NDArray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=7, weight="normal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# --- one clip ---------------------------------------------------------------------------


def clip_figure(
    thumbnail: NDArray | None,
    box: tuple[int, int, int, int],
    crop: NDArray,
    now: float,
    truth: dict,
    prediction: dict,
    origin: float,
    t0: float,
    duration: float,
) -> Figure:
    """(a) The session's whole frame with the crop (orange), (b) the gray crop the
    networks see at time *now*, (c) the thermistor and Zephyr's trace over the
    excerpt, the cursor at *now*."""
    fig, ax = style.mosaic(
        [["frame", "crop", "trace"]], aspect=0.27, width_ratios=[1.35, 1, 3.4]
    )
    if thumbnail is not None:
        _crop_image(ax["frame"], thumbnail, "Session's first frame")
        ax["frame"].add_patch(
            Rectangle(
                box[:2], box[2], box[3], fill=False, edgecolor=CROP_EDGE, linewidth=1
            )
        )
    else:
        ax["frame"].set_axis_off()
    _crop_image(ax["crop"], crop, f"Crop at {now:.2f} s")
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    keep_t, keep_p = _inside(tt, t0, t0 + duration), _inside(tp, t0, t0 + duration)
    ax["trace"].plot(
        tt[keep_t], truth["signal"][keep_t], color=style.TRUTH, label="Thermistor"
    )
    ax["trace"].plot(
        tp[keep_p],
        prediction["signal"][keep_p],
        color=style.color("Zephyr"),
        label="Zephyr",
    )
    ax["trace"].axvline(now, color=CROP_EDGE, linewidth=0.8)
    ax["trace"].set_xlim(t0, t0 + duration)
    ax["trace"].set_xlabel("Time (s)")
    ax["trace"].set_ylabel("Breathing (z-score)")
    ax["trace"].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    panel_letters([ax["frame"], ax["crop"], ax["trace"]])
    return fig


def network_figure(
    truth: dict, prediction: dict, origin: float, t0: float, duration: float
) -> Figure:
    """Zephyr's trace over the thermistor (top) and its onset head (bottom), with the
    thermistor's inhales (black) and the head's (blue)."""
    fig, (top, bottom) = style.figure(
        "double", 0.36, nrows=2, sharex=True, height_ratios=[2, 1]
    )
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    t1 = t0 + duration
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    true_on = truth["inhale"] - origin
    head = prediction["head"] - origin
    true_on, head = true_on[_inside(true_on, t0, t1)], head[_inside(head, t0, t1)]
    top.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Thermistor")
    top.plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    mark_events(top, true_on, 3.3, style.TRUTH)
    mark_events(top, head, 2.8, style.color("Zephyr"), "^")
    top.set_ylabel("Breathing (z-score)")
    top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    onset_head(bottom, tp[kp], prediction["onset"][kp])
    mark_events(bottom, true_on, 1.1, style.TRUTH)
    mark_events(bottom, head, 1.02, style.color("Zephyr"), "^")
    bottom.set_xlim(t0, t1)
    bottom.set_xlabel("Time (s)")
    panel_letters([top, bottom])
    return fig


def trace_stack(
    predictions: dict[str, dict],
    truth: dict,
    scores: dict[str, float],
    origin: float,
    t0: float,
    duration: float,
) -> Figure:
    """Every method's trace on the same excerpt, one row each over the thermistor (grey),
    labelled with its correlation on the whole clip."""
    names = list(predictions)
    fig, ax = style.figure("double", 0.075 * len(names) + 0.08)
    t1 = t0 + duration
    tt = truth["times"] - origin
    kt = _inside(tt, t0, t1)
    for row, name in enumerate(names):
        y = -3.2 * row
        p = predictions[name]
        tp = p["times"] - origin
        kp = _inside(tp, t0, t1)
        ax.plot(tt[kt], truth["signal"][kt] + y, color="0.75", linewidth=0.7)
        ax.plot(tp[kp], p["signal"][kp] + y, color=style.color(name), linewidth=0.9)
        r = scores.get(name, np.nan)
        ax.text(
            t1 + 0.1,
            y,
            f"{name}\nr = {r:.2f}",
            va="center",
            fontsize=6.5,
            color=style.color(name),
        )
    ax.set_yticks([])
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.grid(False)
    return fig


# --- the benchmark --------------------------------------------------------------------


def benchmark_figure(
    ci: dict[str, pd.DataFrame], names: Sequence[str], sets: Sequence[str]
) -> Figure:
    """Each method's mean over animals with its 95% bootstrap interval (animals
    resampled), one panel per metric (rows) and test set (columns). *ci* maps a
    metric to :func:`.analysis.bootstrap`'s table."""
    metrics = list(ci)
    fig, axes = style.figure(
        "double",
        0.2 * len(metrics) + 0.12,
        nrows=len(metrics),
        ncols=len(sets),
        sharey="row",
        squeeze=False,
    )
    for r, metric in enumerate(metrics):
        table = ci[metric]
        for c, name in enumerate(sets):
            ax = axes[r, c]
            for x, method in enumerate(names):
                row = table[(table["set"] == name) & (table["method"] == method)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.errorbar(
                    x,
                    row["mean"],
                    yerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                    fmt="o",
                    ms=3.5,
                    color=style.color(method),
                    elinewidth=1,
                    capsize=0,
                )
            ax.set_xticks(
                range(len(names)),
                names if r == len(metrics) - 1 else [""] * len(names),
                rotation=40,
                ha="right",
            )
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
    return fig


def per_animal_figure(
    recordings: pd.DataFrame, names: Sequence[str], sets: Sequence[str]
) -> Figure:
    """Each animal's mean recording correlation (rows, worst first by Zephyr) for each
    method (columns), one panel per test set."""
    tables = []
    for name in sets:
        t = recordings[recordings["set"] == name].pivot_table(
            index="subject", columns="method", values="correlation", aggfunc="mean"
        )
        t = t[[n for n in names if n in t.columns]]
        tables.append(t.sort_values("Zephyr") if "Zephyr" in t else t)
    height = max(len(t) for t in tables)
    fig, axes = style.figure(
        "double", 0.02 * height + 0.25, ncols=len(tables), squeeze=False
    )
    image = None
    for ax, name, t in zip(axes[0], sets, tables, strict=True):
        image = ax.imshow(
            t.to_numpy(), aspect="auto", cmap=style.EMBEDDING_CMAP, vmin=-1, vmax=1
        )
        ax.set_yticks(range(len(t)), t.index, fontsize=5)
        ax.set_xticks(
            range(len(t.columns)), t.columns, rotation=50, ha="right", fontsize=6
        )
        ax.set_title(f"{name} ({len(t)} animals)")
        ax.grid(False)
    fig.colorbar(
        image, ax=axes[0].tolist(), label="Mean recording correlation", shrink=0.6
    )
    return fig


def agreement_figure(agreement: pd.DataFrame, label: str) -> Figure:
    """:func:`.analysis.score_agreement` as a matrix, each square annotated."""
    names = list(agreement.columns)
    fig, ax = style.figure("single", 0.85)
    image = ax.imshow(agreement.to_numpy(), cmap=style.EMBEDDING_CMAP, vmin=-1, vmax=1)
    for (i, j), r in np.ndenumerate(agreement.to_numpy()):
        if np.isfinite(r):
            ax.text(
                j,
                i,
                f"{r:.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if abs(r) > 0.6 else "black",
            )
    ax.set_xticks(range(len(names)), names, rotation=50, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(names)), names, fontsize=5.5)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=f"Spearman r of clip {label}", shrink=0.7)
    return fig


def pair_scatter_figure(wide: pd.DataFrame, a: str, b: str, label: str) -> Figure:
    """Every clip's score by method *a* (x) and *b* (y), coloured by test set, with the
    identity line; *wide* is :func:`.analysis.clip_scores_wide`."""
    both = wide[[a, b]].dropna()
    fig, ax = style.figure("single", 0.85)
    for name, part in both.groupby(level="set"):
        ax.scatter(
            part[a],
            part[b],
            s=5,
            alpha=0.6,
            lw=0,
            color=style.color(name),
            label=f"{name} ({len(part)})",
        )
    low = min(both[a].min(), both[b].min())
    high = max(both[a].max(), both[b].max())
    ax.plot([low, high], [low, high], color=style.POOLED, lw=0.6, ls="--")
    r = both[a].corr(both[b], method="spearman")
    ax.set_title(f"{a} vs {b}: Spearman r {r:.2f}, {len(both)} clips")
    ax.set_xlabel(f"{a}: clip {label}")
    ax.set_ylabel(f"{b}: clip {label}")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return fig


def bands_figure(
    bands: pd.DataFrame, names: Sequence[str], sets: Sequence[str]
) -> Figure:
    """Mean clip correlation of each method with the thermistor after both are
    band-passed to each frequency band, one panel per test set."""
    columns = [c for c in bands.columns if c.endswith("Hz")]
    fig, axes = style.figure("double", 0.3, ncols=len(sets), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], sets, strict=True):
        part = bands[bands["set"] == name]
        for method in names:
            values = part.loc[part["method"] == method, columns]
            if values.empty:
                continue
            ax.plot(
                range(len(columns)),
                values.mean(),
                "o-",
                ms=2.5,
                color=style.color(method),
                label=method,
            )
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_title(name)
    axes[0, 0].set_ylabel("Correlation in band")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=5.5)
    return fig


def timing_figure(timing: pd.DataFrame, accuracy: dict[str, pd.Series]) -> Figure:
    """(a) Inference seconds per minute of video, log scale, then each score of
    *accuracy* (label -> score per method) against it, one panel each: the cost and the
    benefit of each method."""
    t = timing.set_index("method")
    order = t["s per min of video"].sort_values().index
    fig, axes = style.figure(
        "double", 0.3, ncols=1 + len(accuracy), width_ratios=[1] + [1] * len(accuracy)
    )
    a = axes[0]
    a.barh(
        range(len(order)),
        t.loc[order, "s per min of video"],
        color=[style.color(n) for n in order],
    )
    a.set_yticks(
        range(len(order)),
        [f"{n} ({'CPU' if 'cpu' in t.loc[n, 'device'] else 'GPU'})" for n in order],
        fontsize=6,
    )
    a.set_xscale("log")
    a.set_xlabel("Inference s per min of video")
    a.axvline(60, color="0.5", linewidth=0.6, linestyle=":")
    a.text(60, len(order) - 0.4, " real time", fontsize=6, color="0.4")
    a.grid(axis="x")
    for ax, (label, score) in zip(axes[1:], accuracy.items(), strict=True):
        for name in order:
            if name not in score:
                continue
            x, y = t.loc[name, "s per min of video"], score[name]
            ax.scatter(x, y, s=18, color=style.color(name), zorder=3)
            ax.annotate(
                name, (x, y), fontsize=5.5, xytext=(3, 2), textcoords="offset points"
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Inference s per min of video")
        ax.set_ylabel(label)
    panel_letters(axes)
    return fig


def rate_agreement_figure(windows: pd.DataFrame, names: Sequence[str]) -> Figure:
    """Breathing rate counted in each 10 s window, predicted against true (top) and the
    Bland-Altman difference against the mean (bottom), test sets pooled."""
    fig, axes = style.figure("double", 0.42, nrows=2, ncols=len(names), squeeze=False)
    for c, name in enumerate(names):
        w = windows[windows["method"] == name]
        if w.empty:
            continue
        top, bottom = axes[0, c], axes[1, c]
        top.scatter(
            w["true_hz"],
            w["pred_hz"],
            s=1,
            alpha=0.15,
            color=style.color(name),
            rasterized=True,
        )
        hi = float(np.nanpercentile(w[["true_hz", "pred_hz"]].to_numpy(), 99.5)) + 0.5
        top.plot([0, hi], [0, hi], color="0.5", linewidth=0.6)
        top.set_xlim(0, hi)
        top.set_ylim(0, hi)
        top.set_title(name, fontsize=7)
        mean = (w["true_hz"] + w["pred_hz"]) / 2
        diff = w["pred_hz"] - w["true_hz"]
        bias, limit = diff.mean(), 1.96 * diff.std()
        bottom.scatter(
            mean, diff, s=1, alpha=0.15, color=style.color(name), rasterized=True
        )
        for y, ls in ((bias, "-"), (bias - limit, ":"), (bias + limit, ":")):
            bottom.axhline(y, color="0.3", linewidth=0.7, linestyle=ls)
        bottom.set_xlabel("Mean rate (Hz)")
        if c == 0:
            top.set_ylabel("Predicted rate (Hz)")
            bottom.set_ylabel("Predicted minus true (Hz)")
        top.set_xlabel("True rate (Hz)")
    return fig


def lag_figure(lags: pd.DataFrame, names: Sequence[str]) -> Figure:
    """Predicted minus true inhale time for every true inhale with a predicted one
    within 100 ms; the dotted lines are the scorer's tolerance (25 ms)."""
    fig, ax = style.figure("single", 0.75)
    bins = np.arange(-100, 101, 5)
    for name in names:
        values = lags.loc[lags["method"] == name, "lag_s"] * 1000
        if values.empty:
            continue
        ax.hist(
            values,
            bins,
            histtype="step",
            density=True,
            color=style.color(name),
            label=f"{name} (median {values.median():+.0f} ms)",
        )
    for x in (-TOLERANCE_MS, TOLERANCE_MS):
        ax.axvline(x, color="0.5", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Predicted minus true inhale (ms)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=5.5)
    return fig


def rate_f1_figure(
    f1: pd.DataFrame, shares: pd.DataFrame, names: Sequence[str], sets: Sequence[str]
) -> Figure:
    """Inhale F1 per breathing-rate bin (top, one panel per test set), and where the
    breaths of each split fall in rate (bottom)."""
    bins = list(dict.fromkeys(f1["bin"].astype(str)))
    fig, axes = style.figure(
        "double",
        0.48,
        nrows=2,
        ncols=len(sets),
        sharex=True,
        squeeze=False,
        height_ratios=[1.6, 1],
    )
    for c, name in enumerate(sets):
        ax = axes[0, c]
        part = f1[f1["set"] == name]
        for method in names:
            m = part[part["method"] == method]
            if m.empty:
                continue
            ax.plot(
                range(len(bins)),
                m["f1"],
                "o-",
                ms=2,
                color=style.color(method),
                label=method,
            )
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        if c == 0:
            ax.set_ylabel("Inhale F1 in bin")
        bottom = axes[1, c]
        for split in ("Train", name):
            s = shares[shares["clips"] == split]
            bottom.plot(
                range(len(bins)),
                s["share"],
                "o-" if split != "Train" else "--",
                ms=2,
                color=style.color(split),
                label=split,
            )
        bottom.set_xticks(range(len(bins)), bins, rotation=45, ha="right")
        bottom.set_xlabel("Breathing rate (Hz)")
        bottom.legend(fontsize=5.5)
        if c == 0:
            bottom.set_ylabel("Share of breaths")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=5.5)
    return fig


def splits_rate_figure(shares: pd.DataFrame) -> Figure:
    """Share of breaths per breathing-rate bin in each split."""
    fig, ax = style.figure("single", 0.7)
    for split, s in shares.groupby("clips", sort=False):
        ax.plot(
            range(len(s)),
            s["share"],
            "o-",
            ms=2.5,
            color=style.color(split),
            label=split,
        )
    bins = list(dict.fromkeys(shares["bin"].astype(str)))
    ax.set_xticks(range(len(bins)), bins, rotation=45, ha="right")
    ax.set_xlabel("Breathing rate (Hz)")
    ax.set_ylabel("Share of breaths")
    ax.legend(fontsize=6)
    return fig


def paired_figure(diff: pd.DataFrame, label: str, sets: Sequence[str]) -> Figure:
    """Per-recording difference in correlation and inhale F1 (one dot each) and the
    mean, per test set."""
    fig, axes = style.figure("double", 0.26, ncols=2, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes[0], ("correlation", "inhale"), strict=True):
        for x, name in enumerate(sets):
            v = diff.loc[diff["set"] == name, metric].dropna().to_numpy()
            ax.scatter(
                x + rng.uniform(-0.18, 0.18, len(v)),
                v,
                s=3,
                alpha=0.5,
                color=style.color(name),
            )
            ax.hlines(v.mean(), x - 0.25, x + 0.25, color="k", linewidth=1.4)
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_xticks(range(len(sets)), sets)
        ax.set_ylabel(f"{METRIC_LABELS[metric]}: {label}")
    panel_letters(axes[0])
    return fig


def examples_figure(
    rows: list[dict], origin_key: str = "origin", duration: float = 6.0
) -> Figure:
    """Short excerpts of chosen clips: thermistor and Zephyr, titled with what they are."""
    n = len(rows)
    cols = 3
    fig, axes = style.figure(
        "double",
        0.13 * int(np.ceil(n / cols)) + 0.05,
        nrows=int(np.ceil(n / cols)),
        ncols=cols,
        squeeze=False,
    )
    for ax, row in zip(axes.ravel(), rows, strict=False):
        t, p, t0 = row["truth"], row["prediction"], row["start"]
        tt, tp = t["times"] - row[origin_key], p["times"] - row[origin_key]
        kt, kp = _inside(tt, t0, t0 + duration), _inside(tp, t0, t0 + duration)
        ax.plot(tt[kt], t["signal"][kt], color=style.TRUTH, linewidth=0.7)
        ax.plot(tp[kp], p["signal"][kp], color=style.color("Zephyr"), linewidth=0.8)
        ax.set_title(row["title"], fontsize=6, weight="normal")
        ax.set_yticks([])
        ax.set_xlim(t0, t0 + duration)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    return fig


# --- failure modes --------------------------------------------------------------------


def failure_figure(
    truth: dict,
    prediction: dict,
    errors: pd.DataFrame,
    origin: float,
    at: float,
    frames: Sequence[tuple[float, NDArray]],
    half: float = 2.0,
) -> Figure:
    """A window around one error: the thermistor and Zephyr's trace with every true
    inhale (black), head event (blue), miss (FN) and false event (FP) marked, the onset
    head below, and the gray crops around the error's time."""
    n = len(frames)
    layout = [["trace"] * n, ["onset"] * n, [f"f{i}" for i in range(n)]]
    fig, ax = style.mosaic(layout, aspect=0.55, height_ratios=[1.6, 0.8, 1])
    t0, t1 = at - half, at + half
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    tr = truth["raw_t"] - origin
    kr = _inside(tr, t0, t1)
    if kr.any():
        raw = truth["raw"][kr]
        ax["trace"].plot(
            tr[kr],
            (raw - raw.mean()) / (raw.std() or 1.0),
            color="0.7",
            linewidth=0.5,
            label="Raw thermistor",
        )
    ax["trace"].plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Target")
    ax["trace"].plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    true_on = truth["inhale"] - origin
    head = prediction["head"] - origin
    mark_events(ax["trace"], true_on[_inside(true_on, t0, t1)], 3.4, style.TRUTH)
    mark_events(
        ax["trace"], head[_inside(head, t0, t1)], 2.9, style.color("Zephyr"), "^"
    )
    shown = errors[_inside(errors["time"].to_numpy() - origin, t0, t1)]
    for kind, g in shown.groupby("kind"):
        x = g["time"].to_numpy() - origin
        ax["trace"].plot(
            x,
            np.full(len(x), -3.4),
            "x" if "FP" in kind else "o",
            color=style.ERRORS[kind],
            ms=4,
            mfc="none",
            label=kind,
            clip_on=False,
        )
        for xi in x:
            ax["trace"].axvline(xi, color=style.ERRORS[kind], linewidth=0.5, alpha=0.6)
            ax["onset"].axvline(xi, color=style.ERRORS[kind], linewidth=0.5, alpha=0.6)
    ax["trace"].axvline(at, color="0.2", linewidth=0.8, linestyle="--")
    ax["trace"].set_xlim(t0, t1)
    ax["trace"].set_ylim(-3.8, 3.8)
    ax["trace"].set_ylabel("z-score")
    ax["trace"].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=4, fontsize=6)
    onset_head(ax["onset"], tp[kp], prediction["onset"][kp])
    ax["onset"].set_xlim(t0, t1)
    ax["onset"].set_xlabel("Time (s)")
    for i, (ft, image) in enumerate(frames):
        _crop_image(ax[f"f{i}"], image, f"{ft - at:+.2f} s")
    panel_letters([ax["trace"], ax["onset"], ax["f0"]])
    return fig


def causes_figure(counts: pd.DataFrame) -> Figure:
    """Share of misses and of false events that carry each cause."""
    fig, ax = style.figure("single", 0.6)
    causes = list(dict.fromkeys(counts["cause"]))
    kinds = list(dict.fromkeys(counts["kind"]))
    height = 0.8 / len(kinds)
    for k, kind in enumerate(kinds):
        c = counts[counts["kind"] == kind].set_index("cause").loc[causes]
        ax.barh(
            np.arange(len(causes)) + k * height,
            c["errors"] / c["of"].clip(lower=1),
            height,
            color=style.ERRORS[kind],
            label=f"{kind} (n = {int(c['of'].iloc[0])})",
        )
    ax.set_yticks(
        np.arange(len(causes)) + height * (len(kinds) - 1) / 2, causes, fontsize=6
    )
    ax.invert_yaxis()
    ax.set_xlabel("Share of errors")
    ax.grid(axis="x")
    ax.legend(fontsize=6, loc="lower right")
    return fig


# --- frame embeddings ------------------------------------------------------------------


def embedding_smoothness_figure(
    times: NDArray,
    embeddings: NDArray,
    reference: NDArray,
    spectra: dict[str, tuple[NDArray, NDArray]],
    r2: dict[str, NDArray],
    t0: float,
    duration: float,
) -> Figure:
    """Are the CNN's frame embeddings smooth enough to interpolate linearly?

    (a) The target at the frame times (*reference*) above (b) every embedding
    dimension, z-scored and ordered by its first-PC loading, from *t0* for *duration*
    s; the PCA is fitted on all of *embeddings*. (c) *spectra*: name -> (frequency,
    power density, unit area). (d) *r2*: name -> per-dimension R² of rebuilding each
    frame from its two neighbours.
    """
    fig, ax = style.mosaic(
        [["truth"] * 2, ["heat"] * 2, ["psd", "r2"]],
        aspect=0.7,
        height_ratios=[0.5, 1.6, 1.4],
    )
    shown = (times >= t0) & (times <= t0 + duration)
    z = (embeddings - embeddings.mean(0)) / embeddings.std(0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    order = np.argsort(vt[0])
    ax["truth"].plot(times[shown], reference[shown], color=style.TRUTH)
    ax["truth"].set_ylabel("Thermistor")
    ax["truth"].set_yticks([])
    ax["truth"].tick_params(labelbottom=False)
    ax["truth"].set_xlim(t0, t0 + duration)
    image = ax["heat"].imshow(
        z[shown][:, order].T,
        aspect="auto",
        cmap=style.EMBEDDING_CMAP,
        vmin=-3,
        vmax=3,
        interpolation="nearest",
        extent=(times[shown][0], times[shown][-1], len(order), 0),
    )
    ax["heat"].grid(False)
    ax["heat"].set_xlim(t0, t0 + duration)
    ax["heat"].set_xlabel("Time (s)")
    ax["heat"].set_ylabel("Embedding dim\n(by PC1 loading)")
    fig.colorbar(image, ax=ax["heat"], label="z-score", pad=0.01, aspect=15)
    colours = [style.color("Zephyr"), style.TRUTH, "0.6"]
    for (name, (freq, power)), colour in zip(spectra.items(), colours, strict=False):
        ax["psd"].semilogy(freq, power, color=colour, label=name)
    ax["psd"].set_xlabel("Frequency (Hz)")
    ax["psd"].set_ylabel("Power density")
    ax["psd"].legend(loc="lower left")
    bins = np.linspace(-1, 1, 41)
    for (name, values), colour in zip(r2.items(), colours, strict=False):
        if np.size(values) == 1:
            ax["r2"].axvline(float(np.ravel(values)[0]), color=colour, label=name)
        else:
            ax["r2"].hist(
                np.clip(values, -1, 1), bins, color=colour, alpha=0.6, label=name
            )
    ax["r2"].axvline(0, color="0.6", linewidth=0.5)
    ax["r2"].set_xlabel("R² of neighbour interpolation")
    ax["r2"].set_ylabel("Dimensions")
    ax["r2"].legend(loc="upper left")
    panel_letters([ax[k] for k in ("truth", "heat", "psd", "r2")])
    return fig


# --- quality control ----------------------------------------------------------------------


def _measure_label(name: str) -> str:
    from zephyr.quality_control.metrics import MEASURES

    return MEASURES[name][1]


def _family(name: str) -> str:
    from zephyr.quality_control.metrics import MEASURES

    return MEASURES[name][0]


def quality_scatter_figure(
    table: pd.DataFrame, measures: Sequence[str], truth: str, truth_label: str
) -> Figure:
    """Each measure (x) against the true score (y), one point per row, with its
    Spearman correlation."""
    measures = [m for m in measures if m in table and table[m].notna().any()]
    ncols = 5
    nrows = max(1, -(-len(measures) // ncols))
    fig, axes = style.figure("double", 0.22 * nrows, nrows=nrows, ncols=ncols)
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, measures, strict=False):
        pair = table[[name, truth]].dropna()
        rho = (
            pair[name].corr(pair[truth], method="spearman") if len(pair) > 2 else np.nan
        )
        ax.scatter(
            pair[name], pair[truth], s=4, color=style.QUALITY[_family(name)], lw=0
        )
        ax.set_title(f"{_measure_label(name)}\nρ = {rho:+.2f}", fontsize=6.5)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=5.5)
    for ax in axes[len(measures) :]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel(truth_label)
    return fig


def quality_agreement_figure(by_level: dict[str, pd.DataFrame]) -> Figure:
    """Oriented Spearman correlation of each judging measure with the truth, per level
    (videos, windows): above zero, the measure ranks predictions as it should."""
    first = next(iter(by_level.values()))
    names = list(first.sort_values("oriented")["measure"])
    fig, ax = style.figure("single", 1.1)
    height = 0.8 / len(by_level)
    for k, (level, df) in enumerate(by_level.items()):
        values = df.set_index("measure").reindex(names)["oriented"]
        y = np.arange(len(names)) + (k - (len(by_level) - 1) / 2) * height
        ax.barh(
            y,
            values,
            height=height,
            color=[style.QUALITY[_family(n)] for n in names],
            alpha=1.0 if k == 0 else 0.45,
            label=level,
        )
    ax.axvline(0, color="0.3", linewidth=0.6)
    ax.set_yticks(range(len(names)), [_measure_label(n) for n in names], fontsize=5.5)
    ax.set_xlabel("Spearman ρ with the truth, oriented (> 0: ranks correctly)")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    return fig


def retention_figure(
    curves: dict[str, pd.DataFrame], oracle: pd.DataFrame, truth_label: str
) -> Figure:
    """Mean true score of the videos a measure keeps, against the share kept; the
    oracle keeps by the truth itself, and keeping everything is the last point."""
    fig, ax = style.figure("single", 0.75)
    ax.plot(
        oracle["kept"],
        oracle["mean"],
        color=style.TRUTH,
        linestyle="--",
        label="Oracle",
    )
    for (name, df), colour in zip(
        curves.items(), style.series(len(curves)), strict=True
    ):
        ax.plot(
            df["kept"],
            df["mean"],
            marker="o",
            markersize=2,
            color=colour,
            label=_measure_label(name),
        )
    ax.axhline(oracle["mean"].iloc[-1], color="0.6", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Share of videos kept (best first)")
    ax.set_ylabel(f"Mean {truth_label}")
    ax.legend(fontsize=5.5)
    return fig


def quality_video_figure(
    traces: dict, windows: pd.DataFrame, measures: Sequence[str], t0: float, t1: float
) -> Figure:
    """One video: an excerpt of its trace with the draws' spread and its onset head,
    then the chosen measures and the true inhale F1 per window over the whole clip."""
    fig, axes = style.mosaic(
        [["trace"], ["onset"], ["windows"]],
        aspect=0.75,
        height_ratios=[1.2, 0.6, 1.4],
    )
    times = traces["times"] - traces["times"][0]
    inside = _inside(times, t0, t1)
    ax = axes["trace"]
    signal = traces["signal"]
    if len(traces["spread"]):
        ax.fill_between(
            times[inside],
            (signal - traces["spread"])[inside],
            (signal + traces["spread"])[inside],
            color=style.METHODS["Zephyr"],
            alpha=0.25,
            lw=0,
        )
    ax.plot(times[inside], signal[inside], color=style.METHODS["Zephyr"])
    inhale = traces["inhale"] - traces["times"][0]
    mark_events(ax, inhale[_inside(inhale, t0, t1)], ax.get_ylim()[1], style.ONSET)
    ax.set_ylabel("Trace (z)")
    onset_head(axes["onset"], times[inside], traces["onset"][inside])
    axes["onset"].set_xlabel("Time (s)")
    ax = axes["windows"]
    middle = (windows["start"] + windows["end"]) / 2 - traces["times"][0]
    for name in measures:
        ax.plot(
            middle,
            windows[name],
            marker="o",
            markersize=2,
            color=style.QUALITY[_family(name)],
            label=_measure_label(name),
        )
    if "truth_inhale_f1" in windows:
        ax.plot(
            middle,
            windows["truth_inhale_f1"],
            color=style.TRUTH,
            marker="s",
            markersize=2,
            label="True inhale F1",
        )
    ax.axvspan(t0, t1, color="0.9", zorder=0)
    ax.set_xlabel("Time in clip (s)")
    ax.legend(fontsize=5.5, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return fig


def calibration_figure(reference: dict) -> Figure:
    """Reliability of the calibrated onset head (on the calibration clips) and where
    an unseen animal's embedding distances fall."""
    fig, (left, right) = style.figure("double", 0.32, ncols=2)
    cal = reference.get("onset_calibration")
    if cal:
        table = pd.DataFrame(cal["reliability"])
        left.plot([0, 1], [0, 1], color="0.6", linestyle=":")
        left.scatter(
            table["predicted"],
            table["observed"],
            s=4 + 40 * table["n"] / table["n"].max(),
            color=style.ONSET,
        )
        left.set_title(
            f"Onset calibration (ECE {cal['ece_raw']:.3f} raw, {cal['ece']:.3f} calibrated)"
        )
    left.set_xlabel("Calibrated P(inhale is real)")
    left.set_ylabel("Share matched to a true inhale")
    q = reference["embedding"]["distance_quantiles"]
    right.plot(q, np.arange(len(q)), color=style.QUALITY["input"])
    right.set_xlabel("Embedding distance")
    right.set_ylabel("Percentile")
    right.set_title(
        f"Distances of unseen animals ({reference['embedding']['distances_from']})"
    )
    return fig
