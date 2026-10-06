"""Figure style shared by every notebook: sizes, fonts, and one colour per entity."""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

COLUMN_IN: dict[str, float] = {"single": 3.5, "double": 7.2}
"""Figure widths in inches: one journal column, or the full page width."""

METHODS: dict[str, str] = {
    "Pixel": "#8c8c8c",
    "Facemap": "#cc79a7",
    "TS-CAN": "#e69f00",
    "PhysNet": "#56b4e9",
    "Zephyr": "#0072b2",
}
"""Compared methods. Zephyr is always the same blue."""

VARIANTS: dict[str, str] = {
    "Zephyr": METHODS["Zephyr"],
    "Zephyr, no stretch": "#7b3294",
}
"""Zephyr and its training variants, for ablations that compare networks directly."""

OBJECTIVES: dict[str, str] = {
    "Signal only: F1 from signal": "#0072b2",
    "Multitask: F1 from signal": "#d55e00",
    "Multitask: F1 from event head": "#009e73",
}
"""Ablation of the training objective, and where the F1 is read from."""

CHANNELS: dict[str, str] = {
    "gray": "#4d4d4d",
    "diff": "#117733",
    "flow_x": "#b8860b",
    "flow_y": "#aa4499",
}
"""The four input channels."""

TRUTH: str = "#000000"
"""The thermistor, wherever it is drawn."""

IMPORTANCE_CMAP: str = "inferno"
"""Colour map of every importance (drop) map."""


def use_style() -> None:
    """Apply the shared matplotlib settings; call once at the top of a notebook."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "figure.titlesize": 9,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 0.9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def figure(
    width: str = "double", aspect: float = 0.5, **kwargs: Any
) -> tuple[Figure, Any]:
    """A figure of a named width, as tall as *aspect* times its width.

    *kwargs* go to :func:`matplotlib.pyplot.subplots` (``nrows``, ``ncols``, ...);
    the layout is always constrained.  Returns ``(fig, axes)``.
    """
    w = COLUMN_IN[width]
    return plt.subplots(figsize=(w, w * aspect), layout="constrained", **kwargs)


def color(name: str) -> str:
    """The colour of a method, a variant, a channel, or ``"Truth"``."""
    return {**METHODS, **VARIANTS, **CHANNELS, "Truth": TRUTH}[name]


def mosaic(
    layout: list[list[str]], width: str = "double", aspect: float = 0.5, **kwargs: Any
) -> tuple[Figure, dict[str, Any]]:
    """Like :func:`figure`, for a named layout; returns the axes by name."""
    w = COLUMN_IN[width]
    fig = plt.figure(figsize=(w, w * aspect), layout="constrained")
    return fig, fig.subplot_mosaic(layout, **kwargs)
