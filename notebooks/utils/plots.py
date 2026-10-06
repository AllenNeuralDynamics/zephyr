"""Reusable figure components. Every colour and size comes from :mod:`.style`."""

import string
from collections.abc import Sequence
from pathlib import Path

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
STRATUM_MARKERS: dict[str, str] = {"new_animals": "o", "known_animals_new_date": "s"}
CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "gray": (0, 255),
    "diff": (-40, 40),
    "flow_x": (-2, 2),
    "flow_y": (-2, 2),
}
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
        "inhale_f1": "Inhalation F1\nfrom trace (DSP)",
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


def benchmark_figure(df: pd.DataFrame) -> Figure:
    """Every method on each metric; dots are runs, marker shape is the stratum.

    The inhale and KL-IBI panels add Zephyr scored with the onset head's events.
    """
    fig, axes = style.figure("double", 0.62, nrows=2, ncols=2)
    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items(), strict=True):
        columns = {method: (metric, style.METHODS[method]) for method in style.METHODS}
        if metric in HEAD_COLUMNS:
            columns["Zephyr (event head)"] = (
                HEAD_COLUMNS[metric],
                style.OBJECTIVES[HEAD],
            )
        for i, (name, (column, color)) in enumerate(columns.items()):
            method = name.removesuffix(" (event head)")
            for offset, stratum in zip((-0.2, 0.2), STRATUM_MARKERS, strict=True):
                values = df[(df.method == method) & (df.stratum == stratum)][column]
                points(
                    ax, i + offset, values.to_numpy(), color,
                    STRATUM_MARKERS[stratum], spread=0.05,
                )  # fmt: skip
        ax.set_xticks(range(len(columns)), list(columns), rotation=30, ha="right")
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
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#d55e00", linewidth=1))
    ax.axis("off")
    return image


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


def onset_head(ax: Axes, times: pd.Series, probability: pd.Series) -> None:
    """The onset head's probability, with its 0.5 threshold."""
    ax.plot(times, probability, color=style.OBJECTIVES[HEAD])
    ax.axhline(0.5, color="0.6", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Onset probability")


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
    groups = np.asarray(occlusion["groups"])
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
    occlusion: dict[str, NDArray], frames: dict[str, NDArray]
) -> Figure:
    """Per clip (columns), then the mean: its frame, then each channel's importance
    map (colour scale per channel)."""
    names = _clip_names(occlusion)
    fig, axes = style.figure("double", 0.3, nrows=5, ncols=len(names) + 1)
    clip_frames = [frames[name] for name in names]
    for ax, frame in zip(
        axes[0], [*clip_frames, np.mean(clip_frames, axis=0)], strict=True
    ):
        ax.imshow(frame, cmap="gray")
        ax.axis("off")
    _row_label(axes[0, 0], "frame", size=5)
    for row, channel in zip(axes[1:], style.CHANNELS, strict=True):
        maps = [*occlusion[channel], occlusion[channel].mean(0)]
        _importance_row(row, maps, max(m.max() for m in maps))
        _row_label(row[0], channel, size=5)
    titles = [name.removeprefix("face_").replace("_part_", ".") for name in names]
    for ax, title in zip(axes[0], [*titles, "mean"], strict=True):
        ax.set_title(title, size=5)
    return fig
