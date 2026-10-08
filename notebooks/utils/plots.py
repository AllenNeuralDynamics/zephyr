"""Reusable figure components. Every colour and size comes from :mod:`.style`."""

import string
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from . import style

FIGURES: Path = Path(__file__).resolve().parents[1] / "figures"
INPUT_LABELS: dict[str, str] = {
    "gray": "Gray",
    "gray-diff": "Gray + diff",
    "gray-flow": "Gray + flow",
    "gray-diff-flow": "Gray + diff + flow",
}
METRIC_LABELS: dict[str, str] = {
    "correlation": "Pearson correlation",
    "inhale_f1": "Inhalation F1",
    "exhale_f1": "Exhalation F1",
    "kl_ibi": "KL-IBI (lower is better)",
}
STRATUM_LABELS: dict[str, str] = {
    "new_animals": "Held-out animals",
    "known_animals_new_date": "Held-out sessions",
}
HEAD_COLUMNS: dict[str, str] = {
    "inhale_f1": "head_inhale_f1",
    "kl_ibi": "head_kl_ibi",
}
"""Metrics Zephyr can also be scored on with events from its onset head."""
RATE_ROWS: dict[str, str] = {**STRATUM_LABELS, "training": "Training (in-sample)"}
"""Rows of the by-rate figure: the test groups, then the clips the network fit."""
STRATUM_MARKERS: dict[str, str] = {"new_animals": "o", "known_animals_new_date": "s"}
CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "gray": (0, 255),
    "diff": (-40, 40),
    "flow_x": (-2, 2),
    "flow_y": (-2, 2),
}
CROP_EDGE: str = "#d55e00"
"""Outline of the crop the network sees, on the frame and around the crop itself."""
OBJECTIVE_MARKERS: dict[str, str] = dict(zip(style.OBJECTIVES, "oD^", strict=True))
SIGNAL, MULTI, HEAD = style.OBJECTIVES
"""Series of the ablation: ``(objective, column)`` of each is in :func:`ablation_figure`."""


def panel_letters(axes: Axes | Sequence[Axes] | NDArray) -> None:
    """Label panels a, b, c... in reading order."""
    for letter, ax in zip(string.ascii_lowercase, np.ravel(axes), strict=False):
        ax.text(-0.02, 1.05, letter, transform=ax.transAxes, weight="bold", size=9)


def save(fig: Figure, name: str) -> None:
    """Write *fig* as PNG and SVG into ``notebooks/figures``."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}")


def points(
    ax: Axes,
    x: float,
    values: NDArray,
    color: str,
    marker: str = "o",
    spread: float = 0.1,
) -> None:
    """One dot per run around *x*, and a bar at their mean."""
    jitter = np.random.default_rng(0).uniform(-spread, spread, len(values))
    ax.scatter(
        x + jitter,
        values,
        s=12,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.hlines(np.mean(values), x - 0.17, x + 0.17, color="k", linewidth=1.6, zorder=4)


def ablation_panel(
    ax: Axes, df: pd.DataFrame, series: list[tuple[str, str, str]]
) -> None:
    """Dots per input set for each ``(series, objective, column)``, side by side."""
    offsets = np.linspace(-0.15, 0.15, len(series)) if len(series) > 1 else [0.0]
    for (name, objective, column), offset in zip(series, offsets, strict=True):
        for i, inputs in enumerate(INPUT_LABELS):
            values = df[(df.inputs == inputs) & (df.objective == objective)][column]
            points(
                ax,
                i + offset,
                values.to_numpy(),
                style.OBJECTIVES[name],
                OBJECTIVE_MARKERS[name],
            )
    ax.set_xticks(
        range(len(INPUT_LABELS)), list(INPUT_LABELS.values()), rotation=20, ha="right"
    )


def ablation_figure(df: pd.DataFrame) -> Figure:
    """Input set x objective, one row per stratum: correlation, then inhalation F1
    from the trace (DSP event detection) and from the onset head, on a shared axis."""
    fig, axes = style.figure("double", 0.55, nrows=2, ncols=3, sharex=True)
    columns = {
        "correlation": [
            (SIGNAL, "signal", "correlation"),
            (MULTI, "multitask", "correlation"),
        ],
        "inhale_f1": [
            (SIGNAL, "signal", "inhale_f1"),
            (MULTI, "multitask", "inhale_f1"),
        ],
        "head_inhale_f1": [(HEAD, "multitask", "head_inhale_f1")],
    }
    ylabels = {
        "correlation": METRIC_LABELS["correlation"],
        "inhale_f1": "Inhalation F1 from trace\n(DSP, for comparison only)",
        "head_inhale_f1": "Inhalation F1\nfrom onset head",
    }
    for row, stratum in enumerate(STRATUM_LABELS):
        data = df[df.stratum == stratum]
        axes[row, 2].sharey(axes[row, 1])
        for col, (key, series) in enumerate(columns.items()):
            ablation_panel(axes[row, col], data, series)
            axes[row, col].set_ylabel(ylabels[key])
        axes[row, 0].annotate(
            STRATUM_LABELS[stratum], (-0.35, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", weight="bold",
        )  # fmt: skip
    handles = [
        Line2D(
            [],
            [],
            marker=OBJECTIVE_MARKERS[k],
            linestyle="",
            color=c,
            label=k,
            markersize=4,
        )
        for k, c in style.OBJECTIVES.items()
    ]
    handles.append(
        Line2D([], [], color="k", linewidth=1.6, label="Mean of five networks")
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    panel_letters(axes)
    return fig


def benchmark_means(df: pd.DataFrame) -> pd.DataFrame:
    """The means :func:`benchmark_figure` draws as bars: one row per method and
    test group, one column per metric, from the same columns (Zephyr's inhale F1
    and KL-IBI from its onset head), plus each method's inhale-event source."""
    rows = []
    for method in style.METHODS:
        for stratum, label in STRATUM_LABELS.items():
            runs = df[(df.method == method) & (df.stratum == stratum)]
            row: dict[str, object] = {"method": method, "test group": label}
            for metric, name in METRIC_LABELS.items():
                headed = method == "Zephyr" and metric in HEAD_COLUMNS
                row[name] = runs[HEAD_COLUMNS[metric] if headed else metric].mean()
            row["inhale events"] = "head" if method == "Zephyr" else "DSP"
            row["runs"] = len(runs)
            rows.append(row)
    return pd.DataFrame(rows)


def benchmark_figure(df: pd.DataFrame) -> Figure:
    """Every method on each metric; dots are runs, marker shape is the stratum.

    On the inhale and KL-IBI panels Zephyr is scored with its onset head's events
    and the headless methods with DSP on their trace; exhale F1 is DSP for all.
    Tick labels name the event source.
    """
    fig, axes = style.figure("double", 0.62, nrows=2, ncols=2)
    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items(), strict=True):
        ticks = []
        for i, method in enumerate(style.METHODS):
            headed = method == "Zephyr" and metric in HEAD_COLUMNS
            column = HEAD_COLUMNS[metric] if headed else metric
            for offset, stratum in zip((-0.2, 0.2), STRATUM_MARKERS, strict=True):
                values = df[(df.method == method) & (df.stratum == stratum)][column]
                points(
                    ax, i + offset, values.to_numpy(), style.METHODS[method],
                    STRATUM_MARKERS[stratum], spread=0.05,
                )  # fmt: skip
            source = "head" if headed else "DSP"
            ticks.append(method if metric == "correlation" else f"{method} ({source})")
        ax.set_xticks(range(len(ticks)), ticks, rotation=30, ha="right")
        ax.set_ylabel(label)
    handles = [
        Line2D(
            [], [], marker=m, linestyle="", color="0.4", label=STRATUM_LABELS[s],
            markersize=4,
        )
        for s, m in STRATUM_MARKERS.items()
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    panel_letters(axes)
    return fig


def frame_with_box(
    ax: Axes, frame: NDArray, box: tuple[int, int, int, int]
) -> AxesImage:
    """A video frame with the crop the network sees outlined."""
    image = ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=CROP_EDGE, linewidth=1))
    ax.axis("off")
    return image


class DomainPanel(NamedTuple):
    """One column of :func:`domain_figure`."""

    title: str
    frame: NDArray
    """The full video frame, at the working frame size the box is measured in."""
    box: tuple[int, int, int, int]
    thermistor: pd.DataFrame
    """``Time`` and ``Signal`` (filtered, ADC units)."""
    rate_hz: float
    """The clip's mean breathing rate."""


def domain_figure(panels: Sequence[DomainPanel], t0: float, duration: float) -> Figure:
    """What each recording looks like: one column per clip, rows are the full frame
    at *t0* with the crop outlined, the crop the network sees, and the thermistor
    from *t0* for *duration* seconds on a shared amplitude scale."""
    fig, axes = style.figure(
        "double",
        0.75,
        nrows=3,
        ncols=len(panels),
        height_ratios=[0.75, 1, 0.7],
        squeeze=False,
    )
    for col, panel in enumerate(panels):
        top, middle, bottom = axes[:, col]
        frame_with_box(top, panel.frame, panel.box)
        top.set_title(panel.title)
        x, y, w, h = panel.box
        middle.imshow(panel.frame[y : y + h, x : x + w], cmap="gray", vmin=0, vmax=255)
        middle.set_xticks([])
        middle.set_yticks([])
        middle.grid(False)
        for spine in middle.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(CROP_EDGE)
        shown = panel.thermistor[
            (panel.thermistor.Time >= t0) & (panel.thermistor.Time <= t0 + duration)
        ]
        bottom.plot(shown.Time, shown.Signal, color=style.TRUTH, linewidth=0.7)
        bottom.set_xlim(t0, t0 + duration)
        bottom.set_xlabel("Time (s)")
        bottom.set_title(f"{panel.rate_hz:.1f} breaths/s", fontweight="normal")
        if col:
            bottom.tick_params(labelleft=False)
    # One amplitude scale for every clip, from every excerpt's own range.
    low = min(ax.dataLim.y0 for ax in axes[2])
    high = max(ax.dataLim.y1 for ax in axes[2])
    pad = 0.05 * (high - low)
    for ax in axes[2]:
        ax.set_ylim(low - pad, high + pad)
    axes[2, 0].set_ylabel("Thermistor\n(ADC, filtered)")
    panel_letters(axes[:, 0])
    return fig


OOD_METRIC_LABELS: dict[str, str] = {
    "correlation": METRIC_LABELS["correlation"],
    "head_inhale_f1": "Inhalation F1 (head)",
}


def ood_figure(scores: pd.DataFrame) -> Figure:
    """Each out-of-distribution network on each kind of data it can be scored on:
    one dot per clip, bars are means; correlation, then head inhale F1."""
    networks = list(dict.fromkeys(scores["network"]))
    fig, axes = style.figure("double", 0.36, ncols=len(OOD_METRIC_LABELS))
    width = 0.8 / len(style.OOD_DATA)
    rng = np.random.default_rng(0)  # fixed jitter, so the figure never changes
    for ax, (metric, label) in zip(axes, OOD_METRIC_LABELS.items(), strict=True):
        for i, network in enumerate(networks):
            mine = scores[scores.network == network]
            present = [d for d in style.OOD_DATA if (mine.data == d).any()]
            for j, data in enumerate(present):
                x = i + (j - (len(present) - 1) / 2) * width
                values = mine.loc[mine.data == data, metric].to_numpy()
                colour = style.OOD_DATA[data]
                ax.bar(x, values.mean(), width * 0.9, color=colour, alpha=0.35)
                jitter = rng.uniform(-0.25, 0.25, len(values)) * width
                ax.plot(x + jitter, values, "o", color=colour, ms=2.5)
        ax.set_xticks(range(len(networks)), networks)
        ax.set_xlabel("Trained on")
        ax.set_ylabel(label)
        ax.set_ylim(min(0.0, scores[metric].min() - 0.05), 1)
        ax.axhline(0, color="0.6", linewidth=0.5)
    handles = [
        Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in style.OOD_DATA.values()
    ]
    fig.legend(handles, style.OOD_DATA, loc="outside upper center", ncol=3)
    panel_letters(axes)
    return fig


SHIFT_LIMIT: float = 6.0
"""Distances beyond this many robust z are drawn at the edge, with an arrow."""


def thermistor_figure(shifts: pd.DataFrame, labels: dict[str, str]) -> Figure:
    """Every clip's thermistor features as a distance from the training clips.

    One row per feature (*labels*, feature -> name), one lane per set of clips
    (``set`` column of *shifts*, from :func:`thermistor.shift_from_train`); a dot is
    a clip, a bar the set's median, the shaded band the training clips' usual range
    (2 robust z either side).
    """
    sets = list(style.CLIP_SETS)
    fig, ax = style.figure("double", 0.42)
    rng = np.random.default_rng(0)  # fixed jitter, so the figure never changes
    lane = 0.26
    ax.axvspan(-2, 2, color="0.92", linewidth=0)
    ax.axvline(0, color="0.6", linewidth=0.5)
    for row, feature in enumerate(labels):
        for j, name in enumerate(sets):
            y = row + (j - 1) * lane
            z = shifts.loc[shifts["set"] == name, feature].to_numpy()
            colour = style.CLIP_SETS[name]
            shown = np.clip(z, -SHIFT_LIMIT, SHIFT_LIMIT)
            jitter = rng.uniform(-0.06, 0.06, len(z))
            inside = np.abs(z) <= SHIFT_LIMIT
            ax.plot(
                shown[inside], y + jitter[inside], "o", color=colour, ms=2.4, alpha=0.8
            )
            if (~inside).any():  # off the scale: parked at the edge, pointing out
                ax.plot(
                    shown[~inside],
                    y + jitter[~inside],
                    ">",
                    color=colour,
                    ms=3,
                    alpha=0.8,
                )
            ax.plot(
                [np.median(z).clip(-SHIFT_LIMIT, SHIFT_LIMIT)] * 2,
                [y - 0.1, y + 0.1],
                color="black",
                linewidth=1.2,
            )
    ax.set_yticks(range(len(labels)), list(labels.values()))
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-SHIFT_LIMIT - 0.4, SHIFT_LIMIT + 0.4)
    ax.set_xlabel(
        "Distance from the training clips (robust z; log10 for amplitude and resolution)"
    )
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=3.5, label=n)
        for n, c in style.CLIP_SETS.items()
    ]
    handles.append(Line2D([], [], color="black", linewidth=1.2, label="median"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    return fig


def channel_image(ax: Axes, name: str, plane: NDArray) -> AxesImage:
    """One input channel on its own value range."""
    low, high = CHANNEL_RANGES[name]
    cmap = "gray" if name == "gray" else "RdBu_r"
    image = ax.imshow(plane, cmap=cmap, vmin=low, vmax=high)
    ax.axis("off")
    return image


def channel_strip(axes: Sequence[Axes], frames: dict[str, NDArray]) -> None:
    """The network's four inputs side by side, titled by channel."""
    for ax, (name, plane) in zip(axes, frames.items(), strict=True):
        channel_image(ax, name, plane)
        ax.set_title(name, color=style.CHANNELS[name])


def trace_panel(
    ax: Axes, traces: pd.DataFrame, truth: pd.DataFrame, t0: float, duration: float
) -> None:
    """The thermistor and Zephyr's trace over one excerpt."""
    shown = traces[(traces.Time >= t0) & (traces.Time <= t0 + duration)]
    shown_truth = truth[(truth.Time >= t0) & (truth.Time <= t0 + duration)]
    z = (shown_truth.Signal - shown_truth.Signal.mean()) / shown_truth.Signal.std()
    ax.plot(shown_truth.Time, z, color=style.TRUTH, label="Thermistor")
    ax.plot(shown.Time, shown.Zephyr, color=style.color("Zephyr"), label="Zephyr")
    ax.set_xlim(t0, t0 + duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Breathing (z-score)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)


def scrubber(
    frame: NDArray,
    channels: dict[str, NDArray],
    box: tuple[int, int, int, int],
    now: float,
    traces: pd.DataFrame,
    truth: pd.DataFrame,
    t0: float,
    duration: float,
) -> Figure:
    """The video and the four network inputs at time *now*, over the traces."""
    fig, axes = style.mosaic(
        [["video", *style.CHANNELS], ["trace"] * 5],
        "double",
        0.5,
        width_ratios=[1.33, 1, 1, 1, 1],
        height_ratios=[1, 0.9],
    )
    frame_with_box(axes["video"], frame, box)
    axes["video"].set_title(f"t = {now:.2f} s")
    for name in style.CHANNELS:
        channel_image(axes[name], name, channels[name])
        axes[name].set_title(name, color=style.CHANNELS[name])
    trace_panel(axes["trace"], traces, truth, t0, duration)
    axes["trace"].axvline(now, color="#d55e00", linewidth=1)
    return fig


def mark_events(
    ax: Axes, times: NDArray, y: float, color: str, marker: str = "v"
) -> None:
    """Mark event times along height *y* of an axes."""
    ax.plot(
        times, np.full(len(times), y), marker, color=color, markersize=3, clip_on=False
    )


def trace_stack(
    ax: Axes,
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    t0: float,
    duration: float,
) -> None:
    """The thermistor and every method's trace, stacked, over one excerpt."""
    names = ["Truth", *style.METHODS]
    for row, name in enumerate(reversed(names)):
        source = truth.rename(columns={"Signal": name}) if name == "Truth" else frame
        t, y = source["Time"].to_numpy(), source[name].to_numpy()
        keep = (t >= t0) & (t <= t0 + duration)
        y = y[keep] - y[keep].mean()
        ax.plot(t[keep], 4 * row + 1.6 * y / np.abs(y).max(), color=style.color(name))
        label = (
            name if name == "Truth" else f"{name}  r = {scores.correlation[name]:.2f}"
        )
        ax.text(
            t0 - 0.01 * duration,
            4 * row,
            label,
            ha="right",
            va="center",
            color=style.color(name),
        )
    ax.set_xlim(t0, t0 + duration)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    ax.grid(False)


def _band_lines(
    ax: Axes, summary: pd.DataFrame, series: Sequence[str], ylabel: str
) -> None:
    """One line per series over frequency bands, its 95% interval shaded. A series
    named ``<method> (DSP)`` is that method's colour, dashed."""
    for name in series:
        s = summary[name]
        method = name.split(" (")[0]
        colour = style.color(method)
        ax.fill_between(s.index, s["lo"], s["hi"], color=colour, alpha=0.2, linewidth=0)
        ax.plot(
            s.index,
            s["mean"],
            "o--" if name.endswith("(DSP)") else "o-",
            color=colour,
            markersize=2.5,
            linewidth=1.4 if method == "Zephyr" else 0.9,
            label=name,
        )
    ax.set_ylabel(ylabel)


def band_correlation_figure(summary: pd.DataFrame) -> Figure:
    """Each method's mean correlation with the thermistor by frequency band, over
    the test clips, with its 95% interval shaded (``summary`` from
    :func:`breathing.band_correlation_summary`)."""
    fig, ax = style.figure("double", 0.4)
    _band_lines(ax, summary, list(style.METHODS), "Pearson correlation")
    ax.axhline(0, color="0.6", linewidth=0.5)
    rate_axis(ax, (summary.index[0], summary.index[-1]))
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=len(style.METHODS))
    return fig


def band_comparison_figure(correlation: pd.DataFrame, f1: pd.DataFrame) -> Figure:
    """Waveform correlation and local inhale F1 by frequency band, side by side.

    *correlation* is :func:`breathing.band_correlation_summary`, *f1* is
    :func:`breathing.event_f1_band_summary`; both are means over the test clips with
    95% bootstrap intervals. Zephyr's F1 is shown from its onset head and, dashed,
    from DSP on its trace; the methods without a head are scored with DSP.
    """
    fig, axes = style.figure("double", 0.4, ncols=2)
    _band_lines(axes[0], correlation, list(style.METHODS), "Pearson correlation")
    axes[0].axhline(0, color="0.6", linewidth=0.5)
    axes[0].set_xlabel("Frequency (Hz)")
    series = [name for name in f1.columns.get_level_values(0).unique()]
    _band_lines(axes[1], f1, series, "Local inhale F1")
    axes[1].set_xlabel("Breathing rate (Hz)")
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        rate_axis(ax, (correlation.index[0], correlation.index[-1]))
    axes[0].set_xlabel("Frequency (Hz)")
    handles: dict[str, Line2D] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    fig.legend(
        list(handles.values()), list(handles), loc="outside upper center", ncol=4
    )
    panel_letters(axes)
    return fig


def onset_head(ax: Axes, times: pd.Series, probability: pd.Series) -> None:
    """The onset head's probability, with its 0.5 threshold."""
    ax.plot(times, probability, color=style.OBJECTIVES[HEAD])
    ax.axhline(0.5, color="0.6", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Onset probability")


def variant_traces_figure(
    traces: pd.DataFrame,
    variant: pd.DataFrame,
    truth: pd.DataFrame,
    events: dict[str, NDArray],
    t0: float,
    duration: float,
) -> Figure:
    """Zephyr and a training variant against the thermistor over one excerpt.

    *events* holds inhale times by name: ``"Truth"``, ``"Zephyr"`` and the variant's
    column name. Rows: Zephyr, the variant, then both onset heads overlaid.
    """
    name = next(c for c in variant.columns if c not in ("Time", "onset"))
    fig, axes = style.figure(
        "double", 0.5, nrows=3, sharex=True, height_ratios=[2, 2, 1.2]
    )
    shown_truth = truth[(truth.Time >= t0) & (truth.Time <= t0 + duration)]
    z = (shown_truth.Signal - shown_truth.Signal.mean()) / shown_truth.Signal.std()

    def inside(x: NDArray) -> NDArray:
        return x[(x >= t0) & (x <= t0 + duration)]

    for ax, label, frame, column in zip(
        axes[:2], ("Zephyr", name), (traces, variant), ("Zephyr", name), strict=True
    ):
        shown = frame[(frame.Time >= t0) & (frame.Time <= t0 + duration)]
        ax.plot(shown_truth.Time, z, color=style.TRUTH, label="Thermistor")
        ax.plot(shown.Time, shown[column], color=style.color(label), label=label)
        mark_events(ax, inside(events["Truth"]), 3.3, style.TRUTH)
        mark_events(ax, inside(events[label]), 2.8, style.color(label), "^")
        ax.set_ylabel("Breathing (z-score)")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    for label, frame in (("Zephyr", traces), (name, variant)):
        shown = frame[(frame.Time >= t0) & (frame.Time <= t0 + duration)]
        axes[2].plot(shown.Time, shown.onset, color=style.color(label), label=label)
    axes[2].axhline(0.5, color="0.6", linewidth=0.5, linestyle=":")
    axes[2].set_ylabel("Onset probability")
    axes[2].set_xlim(t0, t0 + duration)
    axes[2].set_xlabel("Time (s)")
    panel_letters(axes)
    return fig


RATE_TICKS: tuple[float, ...] = (2, 3, 4, 6, 8, 12)


def rate_axis(ax: Axes, bins: Sequence[float]) -> None:
    """Log breathing-rate x axis spanning *bins*."""
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(RATE_TICKS, [f"{t:g}" for t in RATE_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("Breathing rate (Hz)")


def _series_lines(
    ax: Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    colours: dict[str, str],
    by: str = "recording",
    **kwargs,
) -> None:
    """One line per *by* value, plus a heavier ``pooled`` line on top."""
    for name, rows in frame[frame[by] != "pooled"].groupby(by):
        rows = rows.sort_values(x)
        ax.plot(rows[x], rows[y], color=colours[name], linewidth=0.8, **kwargs)
    pooled = frame[frame[by] == "pooled"].sort_values(x)
    ax.plot(
        pooled[x], pooled[y], "s--", color=style.POOLED, markersize=2.5, linewidth=1.2
    )


def recording_colours(names: Sequence[str]) -> dict[str, str]:
    """A fixed colour per recording, in sorted order."""
    names = sorted(set(names) - {"pooled"}, key=lambda s: int(s.split()[-1]))
    return dict(zip(names, style.series(len(names)), strict=True))


def training_rates_figure(
    by_recording: pd.DataFrame,
    by_video: pd.DataFrame,
    spectra: pd.DataFrame,
    time_share: pd.Series,
    bins: Sequence[float],
) -> Figure:
    """Training breaths by rate: per recording (with the pooled share of time), per
    video, and power spectra."""
    fig, axes = style.figure("double", 0.34, ncols=3)
    colours = recording_colours(by_recording["recording"])
    _series_lines(axes[0], by_recording, "centre", "share", colours, marker="o", ms=2)
    axes[0].plot(
        time_share.index, time_share.values, ":", color=style.POOLED, linewidth=1.4
    )
    axes[0].legend(
        handles=[
            Line2D([], [], ls="--", marker="s", ms=2.5, color=style.POOLED),
            Line2D([], [], ls=":", color=style.POOLED, linewidth=1.4),
        ],
        labels=["pooled, share of breaths", "pooled, share of time"],
        loc="upper left",
    )
    axes[0].set_title("Per recording")
    recording_of = by_video.groupby("video")["recording"].first()
    for video, rows in by_video[by_video.video != "pooled"].groupby("video"):
        rows = rows.sort_values("centre")
        part_two = video.endswith("part_2")
        axes[1].plot(
            rows.centre,
            rows.share,
            "--" if part_two else "-",
            color=colours[recording_of[video]],
            linewidth=0.8,
        )
    axes[1].set_title("Per video (part 2 dashed)")
    for ax in axes[:2]:
        ax.set_ylabel("Share of breaths")
    mean = spectra.groupby("freq")["power"].mean()
    for _, rows in spectra.groupby("video"):
        rec = rows.recording.iloc[0]
        axes[2].plot(
            rows.freq, rows.power, color=colours[rec], linewidth=0.5, alpha=0.6
        )
    axes[2].plot(mean.index, mean.values, "--", color=style.POOLED, linewidth=1.2)
    axes[2].set_title("Power spectrum per video")
    axes[2].set_ylabel("Power (unit area)")
    for ax in axes:
        rate_axis(ax, bins)
    axes[2].set_xlabel("Frequency (Hz)")
    panel_letters(axes)
    return fig


def drawn_rates_figure(natural: pd.Series, balanced: pd.Series) -> Figure:
    """Share of training windows drawn per rate bin, without and with rate balancing."""
    fig, ax = style.figure("single", 0.8)
    ax.plot(natural.index, natural.values, "o--", ms=2.5, color=style.POOLED)
    ax.plot(balanced.index, balanced.values, "o-", ms=2.5, color=style.color("Zephyr"))
    ax.legend(["rate_balance 0", "as trained"], loc="upper right")
    ax.set_ylabel("Share of drawn windows")
    rate_axis(ax, [natural.index[0] * 0.8, natural.index[-1] * 1.25])
    return fig


def test_rates_figure(
    recall: pd.DataFrame,
    chance: pd.DataFrame,
    f1: pd.DataFrame,
    f1_dsp: pd.DataFrame,
    windows: pd.DataFrame,
    share: pd.DataFrame,
    centres: NDArray,
    bins: Sequence[float],
) -> Figure:
    """Performance by true breathing rate, one row per group in :data:`RATE_ROWS`
    present in *recall*: the two test groups, then the training clips (in-sample).

    Columns: recall within one frame (head), local inhale F1 from the head and,
    for comparison, from DSP on the trace, waveform correlation in 3 s windows,
    and where each recording's breaths fall.
    """
    groups = {k: v for k, v in RATE_ROWS.items() if k in set(recall.stratum)}
    fig, axes = style.figure(
        "double",
        0.26 * len(groups),
        nrows=len(groups),
        ncols=5,
        sharex=True,
        squeeze=False,
    )
    for row, stratum in enumerate(groups):
        ax_a, ax_c, ax_d, ax_e, ax_f = axes[row]
        r = recall[recall.stratum == stratum]
        colours = recording_colours(r["recording"])
        _series_lines(ax_a, r, "centre", "recall", colours, marker="o", ms=2)
        c = chance[(chance.stratum == stratum) & (chance.recording == "pooled")]
        c = c.sort_values("centre")
        ax_a.plot(c.centre, c.recall, ":", color="0.5", label="chance (shifted)")
        ax_a.set_ylabel(f"{groups[stratum]}\nRecall (1 frame)")
        for ax, table in ((ax_c, f1), (ax_d, f1_dsp)):
            part = table[table.stratum == stratum]
            _series_lines(ax, part, "centre", "f1", colours, marker="o", ms=2)
            ax.set_ylabel("Local inhale F1")
        for ax in (ax_a, ax_c, ax_d):
            ax.set_ylim(0, 1.02)
        w = windows[
            (windows.stratum == stratum)
            & (windows.rate >= bins[0])
            & (windows.rate < bins[-1])
        ].copy()
        for name, points_ in w.groupby("recording"):
            ax_e.plot(
                points_.rate, points_.r, ".", color=colours[name], ms=1.2, alpha=0.3
            )
        w["centre"] = centres[
            np.clip(np.searchsorted(bins, w.rate) - 1, 0, len(centres) - 1)
        ]
        per = w.groupby(["recording", "centre"])["r"].mean().reset_index()
        pooled = (
            w.groupby("centre")["r"].mean().reset_index().assign(recording="pooled")
        )
        _series_lines(ax_e, pd.concat([per, pooled]), "centre", "r", colours)
        ax_e.set_ylim(0, 1.02)
        ax_e.set_ylabel("Pearson r, 3 s windows")
        s = share[share.stratum == stratum]
        _series_lines(ax_f, s, "centre", "share", colours, drawstyle="steps-mid")
        ax_f.set_ylabel("Share of breaths")
        for ax in axes[row]:
            rate_axis(ax, bins)
            if row < len(groups) - 1:
                ax.set_xlabel("")
    axes[0, 0].legend(loc="lower right")
    for ax, title in zip(
        axes[0],
        (
            "Breaths detected (head)",
            "Local F1 (head)",
            "Local F1 (DSP)",
            "Waveform correlation",
            "Rate distribution",
        ),
        strict=True,
    ):
        ax.set_title(title, pad=12)
    panel_letters(axes)
    return fig


def _importance_row(
    axes: Sequence[Axes], maps: Sequence[NDArray], vmax: float
) -> AxesImage:
    for ax, m in zip(axes, maps, strict=True):
        image = ax.imshow(m, cmap=style.IMPORTANCE_CMAP, vmin=0, vmax=vmax)
        ax.axis("off")
    return image


def _row_label(ax: Axes, channel: str, size: float = 7) -> None:
    ax.text(
        -0.06,
        0.5,
        channel,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        color=style.CHANNELS.get(channel, style.TRUTH),
        weight="bold",
        size=size,
    )


def _clip_names(occlusion: dict[str, NDArray]) -> list[str]:
    """``test/video_face_6_part_1`` -> ``face_6_part_1`` for every clip."""
    return [str(c).removeprefix("test/video_") for c in occlusion["clips"]]


def channel_importance_figure(
    occlusion: dict[str, NDArray], frames: dict[str, NDArray]
) -> Figure:
    """Mean frame and mean importance map of each channel, per test stratum and overall.

    One row per group, one column per channel; the first column is the mean gray
    frame, for orientation.
    """
    rows = {**STRATUM_LABELS, "all": "All 24 clips"}
    # occlusion names groups after the clip-list stem (face_test_<stratum>)
    groups = np.array([str(g).removeprefix("face_test_") for g in occlusion["groups"]])
    mean_frames = np.stack([frames[name] for name in _clip_names(occlusion)])
    fig, axes = style.figure("double", 0.62, nrows=len(rows), ncols=5)
    vmax = max(occlusion[c].mean(0).max() for c in style.CHANNELS)
    image = None
    for row, (key, label) in zip(axes, rows.items(), strict=True):
        keep = slice(None) if key == "all" else groups == key
        row[0].imshow(mean_frames[keep].mean(0), cmap="gray")
        row[0].axis("off")
        maps = [occlusion[c][keep].mean(0) for c in style.CHANNELS]
        image = _importance_row(row[1:], maps, vmax)
        _row_label(row[0], label)
    axes[0, 0].set_title("mean frame")
    for ax, channel in zip(axes[0, 1:], style.CHANNELS, strict=True):
        ax.set_title(channel, color=style.CHANNELS[channel])
    fig.colorbar(
        image, ax=axes[:, 1:], shrink=0.6, label="Drop in event F1 when hidden"
    )
    return fig


def importance_by_clip_figure(
    occlusion: dict[str, NDArray], frames: dict[str, NDArray], per_block: int = 13
) -> Figure:
    """Per clip (columns), then the mean: its frame, then each channel's importance
    map. Each column is divided by its own largest value across all four channels, so
    channels compare within a clip and clips compare in where, not how much (no colour
    bar: every column runs 0 to 1). Clips wrap into blocks of *per_block* columns; the
    mean closes the last block."""
    names = _clip_names(occlusion)
    clip_frames = [frames[name] for name in names]
    maps = {c: [*occlusion[c], occlusion[c].mean(0)] for c in style.CHANNELS}
    peak = [max(maps[c][i].max() for c in maps) for i in range(len(names) + 1)]
    titles = [n.removeprefix("face_").replace("_part_", ".") for n in names] + ["mean"]
    columns = [*clip_frames, np.mean(clip_frames, axis=0)]
    blocks = [
        range(i, min(i + per_block, len(columns)))
        for i in range(0, len(columns), per_block)
    ]
    rows = 1 + len(maps)
    fig, axes = style.figure(
        "double",
        0.07 * rows * len(blocks) + 0.1,
        nrows=rows * len(blocks),
        ncols=per_block,
    )
    for ax in axes.flat:
        ax.axis("off")
    for k, block in enumerate(blocks):
        top = k * rows
        _row_label(axes[top, 0], "frame", size=5)
        for channel, row in zip(maps, axes[top + 1 : top + rows], strict=True):
            _row_label(row[0], channel, size=5)
        for j, i in enumerate(block):
            axes[top, j].imshow(columns[i], cmap="gray")
            axes[top, j].set_title(titles[i], size=5)
            for channel, row in zip(maps, axes[top + 1 : top + rows], strict=True):
                _importance_row([row[j]], [maps[channel][i] / peak[i]], 1.0)
    return fig
