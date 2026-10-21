"""Figure style shared by every notebook: sizes, fonts, and one colour per entity."""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

COLUMN_IN: dict[str, float] = {"single": 3.5, "double": 7.2}
"""Figure widths in inches: one journal column, or the full page width."""

METHODS: dict[str, str] = {
    "Zephyr": "#0072b2",
    "PhysNet": "#009e73",
    "TS-CAN": "#e69f00",
    "PhysFormer": "#d55e00",
    "RhythmFormer": "#882255",
    "Facemap": "#cc79a7",
    "Pixel (motion)": "#5f5f5f",
    "Pixel (PCA)": "#8c8c8c",
    "Pixel (SNR)": "#b3b3b3",
}
"""Compared methods (the keys of ``utils/runs.METHODS``). Zephyr is always the same blue."""

VARIANTS: dict[str, str] = {
    "Zephyr, more augmentation": "#7b3294",
}
"""Zephyr trained differently, compared with Zephyr directly."""

SETS: dict[str, str] = {
    "Train": "#7f7f7f",
    "Validation": "#bbbbbb",
    "A: new animals": "#0072b2",
    "B: new sessions": "#009e73",
    "C: 359 Hz": "#d55e00",
}
"""The training and validation clips and the three test sets."""

ERRORS: dict[str, str] = {
    "missed (FN)": "#d55e00",
    "false (FP)": "#cc79a7",
}
"""Event errors: a true inhale missed, a predicted inhale with no true one."""

ONSET: str = "#009e73"
"""The onset head's probability."""

TRUTH: str = "#000000"
"""The thermistor, wherever it is drawn."""

POOLED: str = "#333333"
"""A summary over recordings or videos, drawn over their individual lines."""

SERIES_CMAP: str = "tab20"
"""Colour map for many like series (recordings), which have no fixed colour."""


def series(n: int) -> list[str]:
    """*n* distinct colours for like series, in a fixed order."""
    cmap = mpl.colormaps[SERIES_CMAP]
    return [mpl.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]


EMBEDDING_CMAP: str = "RdBu_r"
"""Colour map of z-scored network features (diverging around 0)."""


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
    """The colour of a method, a variant, a set of clips, or ``"Truth"``."""
    return {**METHODS, **VARIANTS, **SETS, "Truth": TRUTH}[name]


def mosaic(
    layout: list[list[str]], width: str = "double", aspect: float = 0.5, **kwargs: Any
) -> tuple[Figure, dict[str, Any]]:
    """Like :func:`figure`, for a named layout; returns the axes by name."""
    w = COLUMN_IN[width]
    fig = plt.figure(figsize=(w, w * aspect), layout="constrained")
    return fig, fig.subplot_mosaic(layout, **kwargs)


QUALITY: dict[str, str] = {
    "rhythm": "#0072b2",
    "head": ONSET,
}
"""The families of quality-control measures (``zephyr.quality_control.metrics``)."""
