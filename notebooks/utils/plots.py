"""Reusable figure components. Every colour and size comes from :mod:`.style`.

Times on a clip's traces are seconds from the clip's first output sample.
"""

import string
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
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


def save_panel(fig: Figure, path: Path) -> None:
    """Write a figure panel as SVG (text kept as text) at exactly the size it was made
    (:func:`utils.style.panel`): never cropped to fit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None would fall back to the style's "tight", which crops
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path.with_suffix(".svg"))


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


# --- manuscript panels ------------------------------------------------------------------
# Drawn at their printed size by utils.style.panel; panel letters are added where the
# figure is assembled, never here.


def breath_phases_panel(
    width_mm: float,
    height_mm: float,
    thumbnail: NDArray | None,
    box: tuple[float, float, float, float],
    frames: Sequence[NDArray],
    labels: Sequence[str],
    truth: dict,
    origin: float,
    t0: float,
    duration: float,
    marks: Sequence[float],
) -> Figure:
    """The whole face frame with the crop (left), the crop at a few moments of one
    breath (top), and the raw thermistor at its own rate (z-scored over the clip) with
    its inhales and exhales below them, the moments marked."""
    n = len(frames)
    fig, ax = style.panel(
        width_mm,
        height_mm,
        [["frame", *[f"f{i}" for i in range(n)]], ["frame", *["trace"] * n]],
        width_ratios=[1.5, *[1] * n],
        height_ratios=[1, 1.1],
    )
    if thumbnail is not None:
        _crop_image(ax["frame"], thumbnail, "")
        ax["frame"].add_patch(
            Rectangle(
                box[:2], box[2], box[3], fill=False, edgecolor=CROP_EDGE, linewidth=0.8
            )
        )
    else:
        ax["frame"].set_axis_off()
    for i, (image, label) in enumerate(zip(frames, labels, strict=True)):
        _crop_image(ax[f"f{i}"], image, label)
    tr = ax["trace"]
    tt = truth["raw_t"] - origin
    raw = (truth["raw"] - truth["raw"].mean()) / (truth["raw"].std() or 1.0)
    keep = _inside(tt, t0, t0 + duration)
    tr.plot(tt[keep], raw[keep], color=style.TRUTH)
    for kind, marker, y in (("inhale", "v", 1.08), ("exhale", "^", -0.08)):
        events = truth[kind] - origin
        events = events[_inside(events, t0, t0 + duration)]
        tr.plot(
            events,
            np.full(len(events), y),
            marker,
            color=style.TRUTH if kind == "inhale" else "0.55",
            markersize=3,
            lw=0,
            clip_on=False,
            transform=tr.get_xaxis_transform(),
            label=kind.capitalize(),
        )
    for t in marks:
        tr.axvline(t, color=CROP_EDGE, linewidth=0.6)
    tr.set_xlim(t0, t0 + duration)
    tr.set_ylabel("Thermistor (z)")
    tr.set_xlabel("Time (s)")
    tr.legend(loc="upper right", bbox_to_anchor=(1, -0.35), ncol=2)
    return fig


def prediction_panel(
    width_mm: float,
    height_mm: float,
    truth: dict,
    prediction: dict,
    origin: float,
    t0: float,
    duration: float,
) -> Figure:
    """Zephyr's breathing trace over the thermistor (top) and its onset probability
    (bottom), with the thermistor's inhales and the head's."""
    fig, ax = style.panel(
        width_mm, height_mm, [["trace"], ["onset"]], height_ratios=[2, 1], sharex=True
    )
    top, bottom = ax["trace"], ax["onset"]
    t1 = t0 + duration
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    true_on, head = truth["inhale"] - origin, prediction["head"] - origin
    true_on, head = true_on[_inside(true_on, t0, t1)], head[_inside(head, t0, t1)]
    top.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Thermistor")
    top.plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    for events, color, marker, y in (
        (true_on, style.TRUTH, "v", 1.12),
        (head, style.color("Zephyr"), "^", 1.02),
    ):
        top.plot(
            events,
            np.full(len(events), y),
            marker,
            color=color,
            markersize=2.5,
            lw=0,
            clip_on=False,
            transform=top.get_xaxis_transform(),
        )
    top.set_ylabel("Sniffing (z)")
    top.legend(loc="lower right", bbox_to_anchor=(1, 1.08), ncol=2)
    bottom.plot(tp[kp], prediction["onset"][kp], color=style.ONSET)
    bottom.set_ylim(0, 1.05)
    bottom.set_yticks([0, 1])
    bottom.set_ylabel("Onset (p)")
    bottom.set_xlim(t0, t1)
    bottom.set_xlabel("Time (s)")
    return fig


FOLD_SETS: dict[str, str] = {
    "train": "Train",
    "val": "Validation",
    "test_a": "A: new animals",
    "test_b": "B: new sessions",
    "test_c": "C: 359 Hz",
}
"""The split's set names (``folds.parquet``) and their labels in :data:`style.SETS`."""
RATE_BANDS: dict[str, str] = {"t_1-4": "1–4 Hz", "t_4-8": "4–8 Hz", "t_8-15": "8–15 Hz"}
"""The split's columns of time per breathing-rate band, and their labels."""


def sync_panel(
    width_mm: float, height_mm: float, offsets_s: NDArray, limit_s: float
) -> Figure:
    """Per session, the thermistor board's start minus the task controller's, on a
    log axis of its magnitude, with the exclusion limit."""
    fig, ax = style.panel(width_mm, height_mm, [["sync"]])
    a = ax["sync"]
    ms = np.abs(np.asarray(offsets_s, float)) * 1e3
    ms = ms[np.isfinite(ms) & (ms > 0)]
    bins = np.logspace(np.log10(max(ms.min(), 0.1)), np.log10(ms.max() * 1.2), 40)
    keep = ms <= limit_s * 1e3
    a.hist(ms[keep], bins, color=style.color("A: new animals"), label="kept")
    a.hist(ms[~keep], bins, color=CROP_EDGE, label=f"removed ({(~keep).sum()})")
    a.axvline(limit_s * 1e3, color="0.4", linewidth=0.6, linestyle="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("|Thermistor − task clock| (ms)")
    a.set_ylabel("Sessions")
    a.legend(loc="upper right")
    return fig


def rate_distribution_panel(
    width_mm: float,
    height_mm: float,
    rates: dict[str, NDArray],
    low: float = 2.0,
    high: float = 15.0,
    step: float = 0.25,
) -> Figure:
    """The distribution of breathing rate in each set (share of its breaths per Hz),
    overlaid, from *low* to *high* Hz; sets named as in :data:`style.SETS`. Breath
    intervals are whole samples of the thermistor, so rates take discrete values; bin
    edges sit halfway between them, about *step* Hz apart, so no bin is empty by
    construction."""
    fig, ax = style.panel(width_mm, height_mm, [["rates"]])
    a = ax["rates"]
    every = np.concatenate(list(rates.values()))
    sample = float(np.min(np.diff(np.unique(np.round(1.0 / every, 6)))))
    samples = np.round(1.0 / (np.arange(low, high + step / 2, step) * sample) - 0.5)
    bins = np.unique(1.0 / ((samples + 0.5) * sample))
    for name, values in rates.items():
        counts, _ = np.histogram(values, bins)
        density = 100 * counts / len(values) / np.diff(bins)
        a.stairs(density, bins, color=style.color(name), linewidth=0.9, label=name)
    a.set_xlim(low, high)
    a.set_ylim(bottom=0)
    a.set_xlabel("Breathing rate (Hz)")
    a.set_ylabel("Breaths (% per Hz)")
    a.legend(loc="upper right", fontsize=6, handlelength=1)
    return fig


def corpus_rate_panel(
    width_mm: float,
    height_mm: float,
    rates: dict[str, NDArray],
    low: float = 2.0,
    high: float = 15.0,
    step: float = 0.25,
) -> Figure:
    """The distribution of breathing rate over the whole corpus (black) and of each
    animal (grey), as a share of breaths per Hz, binned as in
    :func:`rate_distribution_panel`."""
    fig, ax = style.panel(width_mm, height_mm, [["rates"]])
    a = ax["rates"]
    every = np.concatenate(list(rates.values()))
    sample = float(np.min(np.diff(np.unique(np.round(1.0 / every, 6)))))
    samples = np.round(1.0 / (np.arange(low, high + step / 2, step) * sample) - 0.5)
    bins = np.unique(1.0 / ((samples + 0.5) * sample))

    def density(values: NDArray) -> NDArray:
        counts, _ = np.histogram(values, bins)
        return 100 * counts / len(values) / np.diff(bins)

    for values in rates.values():
        a.stairs(density(values), bins, color="0.8", linewidth=0.4)
    a.stairs(density(every), bins, color=style.TRUTH, linewidth=1.0)
    a.plot([], [], color="0.8", lw=0.6, label=f"Each animal ({len(rates)})")
    a.plot([], [], color=style.TRUTH, lw=1.0, label="All")
    a.set_xlim(low, high)
    a.set_ylim(bottom=0)
    a.set_xlabel("Breathing rate (Hz)")
    a.set_ylabel("Breaths (% per Hz)")
    a.legend(loc="upper right", fontsize=6, handlelength=1)
    return fig


def corpus_animals_panel(
    width_mm: float, height_mm: float, sessions: pd.DataFrame
) -> Figure:
    """Sessions per animal, stacked by how many rigs the animal was recorded on; the
    corpus' totals written in the corner. *sessions* has a row per session with
    ``subject_id``, ``rig`` and ``date``."""
    fig, ax = style.panel(width_mm, height_mm, [["animals"]])
    a = ax["animals"]
    per = sessions.groupby("subject_id").agg(
        n=("rig", "size"),
        rigs=("rig", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    groups = {
        "1 rig": per["rigs"] == 1,
        "2 rigs": per["rigs"] == 2,
        "3+ rigs": per["rigs"] >= 3,
    }
    bins = np.arange(0, per["n"].max() + 4, 3)
    shades = ["#c6dbef", "#6baed6", "#2171b5"]
    a.hist(
        [per.loc[k, "n"] for k in groups.values()],
        bins,
        stacked=True,
        color=shades,
        label=list(groups),
        edgecolor="white",
        linewidth=0.3,
    )
    span = (pd.to_datetime(per["last"]) - pd.to_datetime(per["first"])).dt.days
    a.text(
        0.97,
        0.6,
        f"{len(sessions)} sessions\n{len(per)} animals\n{sessions['rig'].nunique()} rigs\n"
        f"median {per['n'].median():g} sessions\nand {span.median():.0f} days per animal",
        transform=a.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
    )
    a.set_xlabel("Sessions per animal")
    a.set_ylabel("Animals")
    a.legend(loc="upper right", fontsize=5.5, handlelength=0.8)
    return fig


def split_panel(width_mm: float, height_mm: float, folds: pd.DataFrame) -> Figure:
    """One row per animal, one dot per assigned session at its date, coloured by its
    set; animals grouped by their role in the split."""
    fig, ax = style.panel(width_mm, height_mm, [["split"]])
    a = ax["split"]
    used = folds[folds["set"].notna()].copy()
    used["date"] = pd.to_datetime(used["date"])
    order = {"test_a": 0, "val": 1, "train": 2, "group_c": 3}
    first = used.groupby("subject_id")["date"].min()
    role = used.groupby("subject_id")["role"].first()
    animals = sorted(first.index, key=lambda s: (order.get(role[s], 9), first[s]))
    row = {s: i for i, s in enumerate(animals)}
    for key, label in FOLD_SETS.items():
        part = used[used["set"] == key]
        a.scatter(
            part["date"],
            part["subject_id"].map(row),
            s=2.5,
            color=style.color(label),
            label=label,
            linewidths=0,
        )
    a.set_ylim(len(animals) - 0.5, -0.5)
    a.set_yticks([])
    a.set_ylabel(f"Animals ({len(animals)})")
    a.grid(False)
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        markerscale=2.5,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    return fig


def gallery_panel(
    width_mm: float,
    height_mm: float,
    images: Sequence[NDArray],
    labels: Sequence[str] | None = None,
    rows: int = 2,
) -> Figure:
    """Whole face frames in a grid of `rows` rows, each labelled (e.g. by rig) if
    `labels` are given."""
    n = len(images)
    cols = int(np.ceil(n / rows))
    layout = [[f"g{r * cols + c}" for c in range(cols)] for r in range(rows)]
    fig, ax = style.panel(width_mm, height_mm, layout)
    for i in range(rows * cols):
        a = ax[f"g{i}"]
        if i < n:
            a.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            if labels is not None:
                a.set_title(labels[i], fontsize=6, weight="normal", pad=1.5)
        a.set_axis_off()
    return fig


DISPLAY: dict[str, str] = {
    "Pixel (motion)": "Motion energy",
    "Pixel (PCA)": "Pixel PCA",
    "Pixel (SNR)": "Pixel SNR",
}
"""Method names as printed in the manuscript, where they differ from the run's."""


def _shown(name: str) -> str:
    return DISPLAY.get(name, name)


def trace_stack_panel(
    width_mm: float,
    height_mm: float,
    predictions: dict[str, dict],
    truth: dict,
    scores: dict[str, float],
    origin: float,
    t0: float,
    duration: float,
) -> Figure:
    """Every method's trace (one row each, in the given order) over the thermistor,
    labelled with its correlation on the whole clip; the thermistor's inhales on top."""
    fig, ax = style.panel(width_mm, height_mm, [["stack"]])
    a = ax["stack"]
    t1 = t0 + duration
    tt = truth["times"] - origin
    kt = _inside(tt, t0, t1)
    step = 3.4
    a.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, linewidth=0.8)
    a.text(t0 - 0.02 * duration, 0, "Thermistor", ha="right", va="center")
    for row, name in enumerate(predictions, start=1):
        y = -step * row
        p = predictions[name]
        tp = p["times"] - origin
        kp = _inside(tp, t0, t1)
        a.plot(tt[kt], truth["signal"][kt] + y, color="0.85", linewidth=0.6)
        a.plot(tp[kp], p["signal"][kp] + y, color=style.color(name), linewidth=0.8)
        a.text(t0 - 0.02 * duration, y, _shown(name), ha="right", va="center")
        r = scores.get(name, np.nan)
        r = 0.0 if abs(r) < 0.005 else r  # no "-0.00"
        a.text(t1 + 0.01 * duration, y, f"r = {r:.2f}", ha="left", va="center")
    onsets = truth["inhale"] - origin
    onsets = onsets[_inside(onsets, t0, t1)]
    for x in onsets:
        a.axvline(x, color="0.9", linewidth=0.5, zorder=0)
    a.set_xlim(t0, t1)
    # Room for a z-scored trace's full swing above the first row and below the last.
    a.set_ylim(-step * len(predictions) - 3.5, 3.5)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xticks([t0, t0 + 1])
    a.set_xticklabels(["", "1 s"])
    a.tick_params(axis="x", length=0)
    a.spines["bottom"].set_bounds(t0, t0 + 1)
    a.grid(False)
    return fig


def benchmark_panel(
    width_mm: float,
    height_mm: float,
    recordings: pd.DataFrame,
    ci: dict[str, pd.DataFrame],
    names: Sequence[str],
    set_name: str,
    seconds: pd.Series | None = None,
    on_cpu: Sequence[str] = (),
) -> Figure:
    """One row per method (the last on top), one column per metric: each animal (dots,
    the mean of its recordings and seeds) and their mean with its 95% bootstrap
    interval, on one set. *ci* maps a metric to :func:`.analysis.bootstrap`'s table.
    With *seconds* (per method), a last column of inference time per clip, hatched for
    the *on_cpu* methods."""
    metrics = list(ci)
    layout = [metrics + (["time"] if seconds is not None else [])]
    fig, ax = style.panel(width_mm, height_mm, layout, sharey=True)
    rows = recordings[recordings["set"] == set_name]
    per_animal = rows.groupby(["method", "subject"])[metrics].mean()
    rng = np.random.default_rng(0)
    y_of = {m: i for i, m in enumerate(names)}
    for metric in metrics:
        a = ax[metric]
        table = ci[metric]
        for method, y in y_of.items():
            if method not in per_animal.index.get_level_values("method"):
                continue
            values = per_animal.loc[method, metric].to_numpy()
            jitter = rng.uniform(-0.2, 0.2, len(values))
            a.scatter(
                values, y + jitter, s=2, color=style.color(method), alpha=0.4, lw=0
            )
            row = table[(table["set"] == set_name) & (table["method"] == method)]
            if row.empty:
                continue
            row = row.iloc[0]
            a.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                fmt="o",
                ms=2.8,
                color=style.color(method),
                elinewidth=0.9,
                capsize=0,
            )
        a.set_xlim(-0.05, 1.0)
        a.set_xticks([0, 0.5, 1])
        a.set_xlabel(METRIC_LABELS[metric])
        a.grid(axis="y", visible=False)
    first = ax[metrics[0]]
    first.set_yticks(range(len(names)), [_shown(n) for n in names])
    first.set_ylim(-0.6, len(names) - 0.4)
    if seconds is not None:
        a = ax["time"]
        for method, y in y_of.items():
            if method not in seconds:
                continue
            cpu = method in on_cpu
            a.barh(
                y,
                seconds[method],
                height=0.65,
                color="white" if cpu else style.color(method),
                edgecolor=style.color(method),
                hatch="//////" if cpu else None,
                linewidth=0.6,
            )
        a.set_xscale("log")
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.set_xticks([0.5, 1, 2, 5])
        a.set_xlim(0.5, 8)
        a.set_xlabel("Inference (s per 5 min)")
        a.grid(axis="y", visible=False)
    return fig


def accuracy_time_panel(
    width_mm: float,
    height_mm: float,
    recordings: pd.DataFrame,
    ci: pd.DataFrame,
    seconds: pd.Series,
    names: Sequence[str],
    set_name: str,
    metric: str = "inhale",
    on_cpu: Sequence[str] = (),
    label_offsets: dict[str, tuple[float, float]] | None = None,
    unit: str = "subject",
) -> Figure:
    """*metric* against inference time per clip, per method: each unit (dots, the mean
    of its recordings; a unit is the value of the *unit* column) and the mean over
    units with its 95% bootstrap interval (*ci*, :func:`.analysis.bootstrap`'s table);
    open markers for the *on_cpu* methods."""
    fig, ax = style.panel(width_mm, height_mm, [["acc"]])
    a = ax["acc"]
    rows = recordings[recordings["set"] == set_name]
    per_animal = rows.groupby(["method", unit])[metric].mean()
    rng = np.random.default_rng(0)
    for method in names:
        if method not in seconds or method not in per_animal.index:
            continue
        x = float(seconds[method])
        values = per_animal.loc[method].to_numpy()
        colour = style.color(method)
        a.scatter(
            x * np.exp(rng.uniform(-0.06, 0.06, len(values))),
            values,
            s=2,
            color=colour,
            alpha=0.35,
            lw=0,
        )
        row = ci[(ci["set"] == set_name) & (ci["method"] == method)].iloc[0]
        a.errorbar(
            x,
            row["mean"],
            yerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
            fmt="o",
            ms=3.2,
            color=colour,
            mfc="white" if method in on_cpu else colour,
            mew=0.9,
            elinewidth=0.9,
            capsize=0,
            zorder=4,
        )
        a.annotate(
            _shown(method),
            (x, row["mean"]),
            xytext=(label_offsets or {}).get(method, (4, 0)),
            textcoords="offset points",
            va="center",
            fontsize=5.5,
            color=colour,
        )
    a.set_xscale("log")
    a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.set_xticks([0.5, 1, 2, 5, 10])
    a.set_xlim(0.6, 14)
    a.set_ylim(0, 1.02)
    a.set_xlabel("Inference (s per 5-min clip)")
    a.set_ylabel(f"{METRIC_LABELS[metric]} (test A)")
    return fig


def lag_rate_panel(
    width_mm: float,
    height_mm: float,
    lags: pd.DataFrame,
    low: float = 2.0,
    high: float = 12.0,
    step_hz: float = 1.0,
    widest_ms: float = 50.0,
) -> Figure:
    """Onset lag (predicted minus true inhale) against the breath's rate: the share of
    each rate bin's lags at each lag (each column sums to 1), with the median (line)
    and interquartile range (dashed) per bin; dotted, the scorer's tolerance."""
    fig, ax = style.panel(width_mm, height_mm, [["heat"]])
    a = ax["heat"]
    lag = lags["lag_s"].to_numpy() * 1000
    rate = lags["rate_hz"].to_numpy()
    keep = (
        np.isfinite(rate) & (rate >= low) & (rate < high) & (np.abs(lag) <= widest_ms)
    )
    lag, rate = lag[keep], rate[keep]
    x_bins = np.arange(low, high + step_hz / 2, step_hz)
    y_bins = np.arange(-widest_ms, widest_ms + 0.1, 2.5)
    counts, _, _ = np.histogram2d(rate, lag, [x_bins, y_bins])
    share = counts / counts.sum(axis=1, keepdims=True)
    image = a.pcolormesh(x_bins, y_bins, share.T, cmap="Blues", rasterized=True)
    centres = (x_bins[:-1] + x_bins[1:]) / 2
    which = np.digitize(rate, x_bins) - 1
    stats = np.array(
        [np.percentile(lag[which == i], [25, 50, 75]) for i in range(len(centres))]
    )
    a.plot(centres, stats[:, 1], color=style.TRUTH, lw=0.9)
    for k in (0, 2):
        a.plot(centres, stats[:, k], color=style.TRUTH, lw=0.6, ls="--")
    for y in (-TOLERANCE_MS, TOLERANCE_MS):
        a.axhline(y, color="0.4", linewidth=0.6, linestyle=":")
    a.set_xlabel("Breathing rate (Hz)")
    a.set_ylabel("Onset lag (ms)")
    a.grid(False)
    bar = fig.colorbar(image, ax=a, fraction=0.06, pad=0.03, aspect=25)
    bar.set_label("Share of the rate bin")
    bar.outline.set_visible(False)
    return fig


def lag_panel(
    width_mm: float, height_mm: float, lags: pd.DataFrame, names: Sequence[str]
) -> Figure:
    """Predicted minus true inhale time for matched inhales, one line per method; the
    dotted lines are the scorer's tolerance."""
    fig, ax = style.panel(width_mm, height_mm, [["lag"]])
    a = ax["lag"]
    bins = np.arange(-50, 51, 2.5)
    for name in names:
        values = lags.loc[lags["method"] == name, "lag_s"].to_numpy() * 1000
        if len(values) == 0:
            continue
        a.hist(
            values,
            bins,
            histtype="step",
            density=True,
            color=style.color(name),
            label=_shown(name),
            linewidth=0.8,
        )
    for x in (-TOLERANCE_MS, TOLERANCE_MS):
        a.axvline(x, color="0.5", linewidth=0.6, linestyle=":")
    for lo, hi in ((-50, -TOLERANCE_MS), (TOLERANCE_MS, 50)):
        a.axvspan(lo, hi, color="0.93", lw=0, zorder=0)
    a.set_xlim(-50, 50)
    a.set_xlabel("Onset lag (ms)")
    a.set_ylabel("Density")
    a.set_yticks([])
    a.legend(loc="upper left", fontsize=6, handlelength=1)
    return fig


def bland_altman_panel(
    width_mm: float,
    height_mm: float,
    windows: pd.DataFrame,
    name: str,
    min_hz: float = 2.0,
) -> Figure:
    """Breathing rate per window, predicted minus true against their mean, with the
    bias (solid) and 95% limits of agreement (dotted)."""
    fig, ax = style.panel(width_mm, height_mm, [["ba"]])
    a = ax["ba"]
    w = windows[(windows["method"] == name) & (windows["true_hz"] >= min_hz)]
    mean = (w["true_hz"] + w["pred_hz"]) / 2
    diff = w["pred_hz"] - w["true_hz"]
    bias, limit = float(diff.mean()), 1.96 * float(diff.std())
    lim = max(1.0, 1.5 * limit)
    # Counted rates come in steps of 1 / window, so bin the differences by that step.
    step = (
        float(np.min(np.diff(np.unique(np.round(diff, 6))))) if len(diff) > 1 else 0.1
    )
    y_bins = np.arange(-lim - step / 2, lim + step, step)
    x_bins = np.arange(np.floor(mean.min()), np.ceil(mean.max()) + 0.25, 0.25)
    *_, image = a.hist2d(
        mean,
        diff,
        [x_bins, y_bins],
        cmap="Blues",
        norm=mpl.colors.LogNorm(),
        rasterized=True,
    )
    for y, ls in ((bias, "-"), (bias - limit, ":"), (bias + limit, ":")):
        a.axhline(y, color="0.3", linewidth=0.6, linestyle=ls)
    a.set_ylim(-lim, lim)
    a.grid(False)
    bar = fig.colorbar(image, ax=a, fraction=0.08, pad=0.03, aspect=25)
    bar.set_label("Windows")
    bar.outline.set_visible(False)
    a.set_xlabel("Mean rate (Hz)")
    a.set_ylabel("Video − thermistor (Hz)")
    return fig


def sets_panel(
    width_mm: float,
    height_mm: float,
    recordings: pd.DataFrame,
    names: Sequence[str],
    metrics: Sequence[str],
    sets: Sequence[str],
    unit: str = "subject",
) -> Figure:
    """Per metric (side by side), each method's mean over units (values of the *unit*
    column) on each set, joined by a line, the method named at the last set."""
    fig, ax = style.panel(width_mm, height_mm, [list(metrics)], sharey=True)
    per_animal = recordings.groupby(["set", "method", unit])[list(metrics)].mean()
    means = per_animal.groupby(["set", "method"]).mean()
    for metric in metrics:
        a = ax[metric]
        ends = []
        for method in names:
            y = [means[metric].get((s, method), np.nan) for s in sets]
            a.plot(range(len(sets)), y, "-o", color=style.color(method), ms=2.5, lw=0.8)
            ends.append((y[-1], method))
        # Names at the last set, nudged apart so they do not overlap.
        ends.sort()
        placed: list[float] = []
        for y, method in ends:
            y_text = max(y, placed[-1] + 0.045) if placed else y
            placed.append(y_text)
            a.text(
                len(sets) - 1 + 0.12,
                y_text,
                _shown(method),
                va="center",
                fontsize=6,
                color=style.color(method),
            )
        a.set_xticks(range(len(sets)), [s.split(":")[0] for s in sets])
        a.set_xlim(-0.25, len(sets) - 1 + 0.9)
        a.set_xlabel("Test set")
        a.set_title(METRIC_LABELS[metric], fontsize=7)
    ax[metrics[0]].set_ylabel("Mean over units")
    return fig


def animals_panel(
    width_mm: float,
    height_mm: float,
    recordings: pd.DataFrame,
    names: Sequence[str],
    metric: str,
    set_name: str,
    unit: str = "subject",
    unit_label: str = "Animal",
) -> Figure:
    """Each unit (value of the *unit* column) of one set, ranked by the first method,
    one marker per method."""
    fig, ax = style.panel(width_mm, height_mm, [["animals"]])
    a = ax["animals"]
    rows = recordings[recordings["set"] == set_name]
    table = rows.groupby([unit, "method"])[metric].mean().unstack("method")
    table = table.sort_values(names[0])
    x = np.arange(len(table))
    for method in names:
        a.plot(
            x,
            table[method],
            "o",
            color=style.color(method),
            ms=3 if method == names[0] else 2.2,
            mfc="none" if method != names[0] else style.color(method),
            mew=0.7,
            label=_shown(method),
        )
    a.set_xticks(x, [])
    a.set_xlabel(f"{unit_label} (ranked)")
    a.set_ylabel(f"{METRIC_LABELS[metric]} (test A)")
    a.legend(loc="lower right", fontsize=6, handletextpad=0.2)
    return fig


def rate_band_panel(
    width_mm: float,
    height_mm: float,
    f1: pd.DataFrame,
    names: Sequence[str],
    legend: bool = True,
) -> Figure:
    """Inhale F1 per breathing-rate bin, one line per method, over the share of breaths
    in each bin (bars)."""
    fig, ax = style.panel(
        width_mm, height_mm, [["f1"], ["share"]], sharex=True, height_ratios=[3, 1]
    )
    bins = list(dict.fromkeys(f1["bin"].astype(str)))
    x = np.arange(len(bins))
    for method in names:
        g = f1[f1["method"] == method].set_index(
            f1["bin"].astype(str)[f1["method"] == method]
        )
        ax["f1"].plot(
            x,
            g["f1"].reindex(bins),
            "-o",
            ms=2,
            lw=0.8,
            color=style.color(method),
            label=_shown(method),
        )
    first = f1[f1["method"] == names[0]]
    share = first.set_index(first["bin"].astype(str))["share"].reindex(bins)
    ax["share"].bar(x, 100 * share, color="0.75", width=0.8)
    ax["share"].set_ylabel("Breaths (%)")
    ax["share"].set_xticks(x, [b.split("-")[0] for b in bins])
    ax["share"].set_xlabel("Breathing rate (Hz)")
    ax["f1"].set_ylabel("Inhale F1 (test A)")
    ax["f1"].set_ylim(0.4, 1.02)
    if legend:
        ax["f1"].legend(loc="lower center", fontsize=6, handlelength=1, ncol=3)
    return fig


def tolerance_panel(
    width_mm: float,
    height_mm: float,
    lags: pd.DataFrame,
    breaths: pd.DataFrame,
    false: pd.DataFrame,
    names: Sequence[str],
    widest_ms: float = 100.0,
) -> Figure:
    """Inhale F1 as the matching tolerance is relaxed, up to *widest_ms* (the timing
    table's window): a true inhale counts as found when a prediction lies within the
    tolerance (pairs from the timing table). The first method's F1 and errors left
    (misses plus false inhales) are written at the scorer's tolerance and the widest."""
    fig, ax = style.panel(width_mm, height_mm, [["tol"]])
    a = ax["tol"]
    tolerances = np.arange(0, widest_ms + 0.1, 2.5)
    for k, method in enumerate(names):
        lag = np.abs(lags.loc[lags["method"] == method, "lag_s"].to_numpy()) * 1000
        b = breaths[breaths["method"] == method]
        n_true = len(b)
        n_pred = int(b["detected"].sum()) + int((false["method"] == method).sum())
        hits = np.array([(lag <= t).sum() for t in tolerances])
        f1 = 2 * hits / (n_true + n_pred)
        a.plot(tolerances, f1, color=style.color(method), label=_shown(method), lw=0.9)
        if k == 0:
            # Marked on the curve; their values listed below the curves, clear of
            # every line.
            lines = []
            for t in (TOLERANCE_MS, 2 * TOLERANCE_MS, widest_ms):
                i = int(np.argmin(np.abs(tolerances - t)))
                errors = n_true + n_pred - 2 * hits[i]
                a.plot(t, f1[i], "o", ms=2.5, color=style.color(method), zorder=4)
                lines.append(f"{t:g} ms: F1 {f1[i]:.2f}, {errors / n_true:.1%} errors")
            a.text(
                0.2,
                0.04,
                "\n".join([_shown(method), *lines]),
                transform=a.transAxes,
                va="bottom",
                fontsize=5.5,
                color=style.color(method),
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
                zorder=5,
            )
    for t in (TOLERANCE_MS, 2 * TOLERANCE_MS):
        a.axvline(t, color="0.5", linewidth=0.6, linestyle=":")
    a.set_xlabel("Tolerance (ms)")
    a.set_ylabel("Inhale F1 (test A)")
    a.set_xlim(10, widest_ms)
    a.set_ylim(0.6, 1.02)
    a.legend(loc="center right", fontsize=6, handlelength=1)
    return fig


def failure_panel(
    width_mm: float,
    height_mm: float,
    truth: dict,
    prediction: dict,
    errors: pd.DataFrame,
    origin: float,
    at: float,
    frames: Sequence[tuple[float, NDArray]],
    half: float = 1.5,
) -> Figure:
    """A window around one error: the crops around it (top), the thermistor and
    Zephyr's trace with true inhales, head events and errors marked, and the onset
    probability."""
    n = len(frames)
    layout = [[f"f{i}" for i in range(n)], ["trace"] * n, ["onset"] * n]
    fig, ax = style.panel(width_mm, height_mm, layout, height_ratios=[1.1, 1.6, 0.8])
    t0, t1 = at - half, at + half
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    tr = ax["trace"]
    tr.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Thermistor")
    tr.plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    true_on = truth["inhale"] - origin
    head = prediction["head"] - origin
    mark_events(tr, true_on[_inside(true_on, t0, t1)], 3.6, style.TRUTH, size=2.5)
    mark_events(tr, head[_inside(head, t0, t1)], 3.0, style.color("Zephyr"), "^", 2.5)
    shown = errors[_inside(errors["time"].to_numpy() - origin, t0, t1)]
    for kind, g in shown.groupby("kind"):
        for xi in g["time"].to_numpy() - origin:
            for a in (tr, ax["onset"]):
                a.axvline(xi, color=style.ERRORS[kind], linewidth=0.6, alpha=0.8)
        tr.plot([], [], color=style.ERRORS[kind], lw=0.8, label=kind)
    tr.set_xlim(t0, t1)
    tr.set_ylim(-5.2, 3.9)
    tr.set_ylabel("Sniffing (z)")
    tr.set_xticklabels([])
    tr.legend(
        loc="lower left",
        ncol=4,
        fontsize=5.5,
        handlelength=1,
        bbox_to_anchor=(0, -0.08),
        frameon=False,
    )
    o = ax["onset"]
    o.plot(tp[kp], prediction["onset"][kp], color=style.ONSET)
    o.axhline(0.5, color="0.6", linewidth=0.5, linestyle=":")
    o.set_ylim(0, 1.05)
    o.set_yticks([0, 1])
    o.set_ylabel("Onset (p)")
    o.set_xlim(t0, t1)
    o.set_xlabel("Time (s)")
    for i, (ft, image) in enumerate(frames):
        f = ax[f"f{i}"]
        f.imshow(image, cmap="gray", vmin=0, vmax=255)
        f.set_title(f"{ft - at:+.2f} s", fontsize=6, weight="normal", pad=1.5)
        f.set_axis_off()
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
    traces: pd.DataFrame,
    windows: pd.DataFrame,
    measures: Sequence[str],
    t0: float,
    t1: float,
) -> Figure:
    """One video: an excerpt of its trace with its inhales and its onset head,
    then the chosen measures and the true inhale F1 per window over the whole clip."""
    fig, axes = style.mosaic(
        [["trace"], ["onset"], ["windows"]],
        aspect=0.75,
        height_ratios=[1.2, 0.6, 1.4],
    )
    origin = float(traces["time"].iloc[0])
    times = traces["time"].to_numpy() - origin
    inside = _inside(times, t0, t1)
    ax = axes["trace"]
    signal = traces["signal"].to_numpy()
    ax.plot(times[inside], signal[inside], color=style.METHODS["Zephyr"])
    inhale = times[traces["inhale"].to_numpy()]
    mark_events(ax, inhale[_inside(inhale, t0, t1)], ax.get_ylim()[1], style.ONSET)
    ax.set_ylabel("Trace (z)")
    onset_head(axes["onset"], times[inside], traces["onset"].to_numpy()[inside])
    axes["onset"].set_xlabel("Time (s)")
    ax = axes["windows"]
    middle = (windows["start"] + windows["end"]) / 2 - origin
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
    """Reliability of the calibrated onset head on the calibration clips."""
    fig, ax = style.figure("single", 0.8)
    cal = reference.get("onset_calibration")
    if cal:
        table = pd.DataFrame(cal["reliability"])
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
        ax.scatter(
            table["predicted"],
            table["observed"],
            s=4 + 40 * table["n"] / table["n"].max(),
            color=style.ONSET,
        )
        ax.set_title(f"ECE {cal['ece_raw']:.3f} raw, {cal['ece']:.3f} calibrated")
    ax.set_xlabel("Calibrated P(inhale is real)")
    ax.set_ylabel("Share matched to a true inhale")
    return fig


# --- manuscript panels: quality control -------------------------------------------------


def qc_video_panel(
    width_mm: float,
    height_mm: float,
    traces: pd.DataFrame,
    windows: pd.DataFrame,
    measures: Sequence[str],
    t0: float,
    t1: float,
) -> Figure:
    """One video: an excerpt of the trace and onset probability (left), and over the
    whole clip each chosen measure and the true inhale F1 per window, failing windows
    shaded (right)."""
    fig, ax = style.panel(
        width_mm,
        height_mm,
        [["trace", "windows"], ["onset", "windows"]],
        width_ratios=[1, 1.6],
        height_ratios=[1.6, 1],
    )
    origin = float(traces["time"].iloc[0])
    times = traces["time"].to_numpy() - origin
    inside = _inside(times, t0, t1)
    tr = ax["trace"]
    tr.plot(
        times[inside], traces["signal"].to_numpy()[inside], color=style.color("Zephyr")
    )
    inhale = times[traces["inhale"].to_numpy()]
    mark_events(
        tr, inhale[_inside(inhale, t0, t1)], 3.2, style.color("Zephyr"), "^", 2.5
    )
    tr.set_ylim(-3.5, 3.6)
    tr.set_xlim(t0, t1)
    tr.set_xticklabels([])
    tr.set_ylabel("Sniffing (z)")
    o = ax["onset"]
    o.plot(times[inside], traces["onset"].to_numpy()[inside], color=style.ONSET)
    o.set_ylim(0, 1.05)
    o.set_yticks([0, 1])
    o.set_xlim(t0, t1)
    o.set_ylabel("Onset (p)")
    o.set_xlabel("Time (s)")
    w = ax["windows"]
    start = windows["start"].to_numpy() - origin
    end = windows["end"].to_numpy() - origin
    middle = (start + end) / 2
    checks = [c for c in windows if c.startswith("ok_")]
    failed = ~windows[checks].fillna(True).astype(bool).all(axis=1).to_numpy()
    for a, b in zip(start[failed], end[failed], strict=True):
        w.axvspan(a, b, color=CROP_EDGE, alpha=0.15, lw=0)
    w.axvspan(t0, t1, color="0.9", zorder=0, lw=0)
    for name in measures:
        values = windows[name].to_numpy(float)
        lo, hi = np.nanmin(values), np.nanmax(values)
        w.plot(
            middle,
            (values - lo) / ((hi - lo) or 1.0),
            lw=0.8,
            color=style.QUALITY[_family(name)],
            label=_measure_label(name),
        )
    if "truth_inhale_f1" in windows:
        w.plot(
            middle,
            windows["truth_inhale_f1"],
            color=style.TRUTH,
            lw=1.0,
            label="True inhale F1",
        )
    w.set_ylim(-0.05, 1.05)
    w.set_xlim(start.min(), end.max())
    w.set_xlabel("Time in video (s)")
    w.set_ylabel("Per window (scaled)")
    w.legend(fontsize=5.5, ncols=2, loc="lower left", handlelength=1)
    return fig


def qc_agreement_panel(
    width_mm: float, height_mm: float, by_level: dict[str, pd.DataFrame]
) -> Figure:
    """Oriented Spearman correlation of each measure with the true inhale F1, per level
    (videos, windows): above zero, the measure ranks predictions as it should."""
    fig, ax = style.panel(width_mm, height_mm, [["rho"]])
    a = ax["rho"]
    first = next(iter(by_level.values()))
    names = list(first.sort_values("oriented")["measure"])
    height = 0.8 / len(by_level)
    for k, (level, df) in enumerate(by_level.items()):
        values = df.set_index("measure").reindex(names)["oriented"]
        y = np.arange(len(names)) + (k - (len(by_level) - 1) / 2) * height
        a.barh(
            y,
            values,
            height=height,
            color=[style.QUALITY[_family(n)] for n in names],
            alpha=1.0 if k == 0 else 0.45,
            label=level,
        )
    a.axvline(0, color="0.3", linewidth=0.6)
    a.set_yticks(range(len(names)), [_measure_label(n) for n in names], fontsize=6)
    a.set_xlabel("Spearman ρ with true F1")
    a.grid(axis="x")
    a.grid(axis="y", visible=False)
    a.legend(loc="lower right", fontsize=6)
    return fig


def qc_retention_panel(
    width_mm: float,
    height_mm: float,
    curves: dict[str, pd.DataFrame],
    oracle: pd.DataFrame,
    passing: tuple[float, float],
) -> Figure:
    """Mean true inhale F1 of the windows a measure keeps (best first) against the
    share kept, the oracle (keeping by the truth), and the windows that pass every
    check (star)."""
    fig, ax = style.panel(width_mm, height_mm, [["keep"]])
    a = ax["keep"]
    a.plot(
        oracle["kept"],
        oracle["mean"],
        color=style.TRUTH,
        ls="--",
        lw=0.8,
        label="Oracle",
    )
    for name, df in curves.items():
        a.plot(
            df["kept"],
            df["mean"],
            lw=0.9,
            color=style.QUALITY[_family(name)],
            label=_measure_label(name),
        )
    a.axhline(oracle["mean"].iloc[-1], color="0.6", linewidth=0.5, linestyle=":")
    a.plot(*passing, "*", color=CROP_EDGE, ms=6, label="All checks pass", zorder=5)
    a.set_xlim(0, 1.02)
    a.set_xlabel("Share of windows kept")
    a.set_ylabel("Mean true inhale F1")
    a.legend(fontsize=5.5, loc="lower left", handlelength=1)
    return fig


def calibration_panel(width_mm: float, height_mm: float, reference: dict) -> Figure:
    """Reliability of the calibrated onset output on the calibration clips: predicted
    probability that an inhale is real against the share matched to a true inhale."""
    fig, ax = style.panel(width_mm, height_mm, [["cal"]])
    a = ax["cal"]
    cal = reference["onset_calibration"]
    table = pd.DataFrame(cal["reliability"])
    a.plot([0, 1], [0, 1], color="0.6", linestyle=":", lw=0.8)
    a.plot(table["predicted"], table["observed"], "-", color=style.ONSET, lw=0.7)
    a.scatter(
        table["predicted"],
        table["observed"],
        s=3 + 25 * table["n"] / table["n"].max(),
        color=style.ONSET,
        zorder=3,
    )
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.02)
    a.set_xlabel("Predicted P(inhale is real)")
    a.set_ylabel("Observed precision")
    a.text(
        0.97,
        0.05,
        f"ECE {cal['ece']:.3f}",
        ha="right",
        transform=a.transAxes,
        fontsize=6,
    )
    return fig
