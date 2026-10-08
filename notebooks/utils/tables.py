"""Notebook tables that bold the best value of every column."""

from collections.abc import Sequence

import marimo as mo
import pandas as pd

BOLD: dict[str, str] = {"fontWeight": "bold"}


def best_cells(
    df: pd.DataFrame,
    lower: Sequence[str] = (),
    skip: Sequence[str] = (),
    group: str | None = None,
    across_columns: Sequence[str] | None = None,
) -> set[tuple[int, str]]:
    """``(row position, column)`` of the best value of each numeric column.

    Larger is better unless the column is in *lower*; *skip* columns are never
    marked. With *group*, the best is found within each value of that column; with
    *across_columns*, within each row among those columns instead (larger is
    better). Ties are all marked.
    """
    best: set[tuple[int, str]] = set()
    if across_columns is not None:
        for i, (_, row) in enumerate(df[list(across_columns)].iterrows()):
            best |= {(i, c) for c in across_columns if row[c] == row.max()}
        return best
    positions = pd.Series(range(len(df)), index=df.index)
    parts = [df] if group is None else [p for _, p in df.groupby(group, sort=False)]
    for column in df.select_dtypes("number").columns:
        if column in skip:
            continue
        for part in parts:
            x = part[column]
            target = x.min() if column in lower else x.max()
            best |= {(int(positions[i]), column) for i in x.index[x == target]}
    return best


def table(df: pd.DataFrame, **kwargs):
    """``mo.ui.table`` of *df* with the best cells bold; *kwargs* are those of
    :func:`best_cells`. The index is not shown, so reset it first if it matters."""
    df = df.reset_index(drop=True)
    best = best_cells(df, **kwargs)

    def style(row: str, column: str, _value) -> dict[str, str]:
        return BOLD if (int(row), column) in best else {}

    return mo.ui.table(df, selection=None, style_cell=style)
