import pandas as pd

from utils import tables


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": list("abcd"),
            "group": ["x", "x", "y", "y"],
            "r": [1, 3, 2, 2],
            "kl": [0.1, 0.2, 0.3, 0.1],
        }
    )


def test_best_is_largest_unless_listed_as_lower() -> None:
    best = tables.best_cells(_frame(), lower=["kl"])
    assert best == {(0, "kl"), (3, "kl"), (1, "r")}


def test_best_within_groups_marks_ties() -> None:
    best = tables.best_cells(_frame(), lower=["kl"], group="group", skip=["kl"])
    assert best == {(1, "r"), (2, "r"), (3, "r")}


def test_best_across_columns_of_each_row() -> None:
    best = tables.best_cells(_frame(), across_columns=["r", "kl"])
    assert best == {(i, "r") for i in range(4)}
