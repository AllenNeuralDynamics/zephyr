"""``cross_validation``: animal-grouped folds over the sessions ``folds`` assigned.

Animals outside group C are partitioned into ``n_folds`` groups (``tries`` seeded draws,
the one whose worst group is best balanced kept). Fold k tests group k whole (test A),
early-stops on ``val_animals`` drawn from the others, and tests one date block of every
other animal with ``test_b_min_sessions`` (test B): its sessions, by date, are cut into a
block for each fold it trains in. Group C sessions keep their ``folds.parquet`` set in
every fold. Writes ``sessions.parquet``: the assigned sessions' columns of
``folds.parquet`` (which it only reads) without its first draw, and the folds.
"""

import numpy as np
import pandas as pd

from . import storage
from .config import CrossValidation, DatasetSpec, Layout
from .folds import Draw, check, report

FIRST_DRAW = ["set", "role", "small"]
"""``folds.parquet``'s columns of its own draw, left out of ``sessions.parquet``."""

ALL = 10**6
"""A cap no animal reaches: :class:`folds.Draw` scores every session of an animal."""


def column(k: int) -> str:
    return f"fold_{k}"


def partition(
    draw: Draw, animals: list[str], s: CrossValidation, rng
) -> list[list[str]]:
    """*animals* in ``n_folds`` groups, the draw whose worst-balanced group scores best."""
    best, best_score = None, -np.inf
    for _ in range(s.tries):
        groups = [list(g) for g in np.array_split(rng.permutation(animals), s.n_folds)]
        score = min(
            draw.score(g, [a for a in animals if a not in g], ALL) for g in groups
        )
        if score > best_score:
            best, best_score = groups, score
    return [sorted(g) for g in best]


def assign(folds_df: pd.DataFrame, s: CrossValidation) -> pd.DataFrame:
    """The sessions *folds_df* assigns, with each animal's ``cv_group`` (-1: group C),
    each session's ``cv_b_fold`` (the fold it is test B in, -1: none) and its set in
    every fold (``fold_k``)."""
    used = folds_df[folds_df["set"].notna()].reset_index(drop=True)
    in_c = set(used.loc[used.role == "group_c", "subject_id"])
    animals = sorted(set(used.subject_id) - in_c)
    rng = np.random.default_rng(s.seed)
    draw = Draw(used, s.seed)
    groups = partition(draw, animals, s, rng)
    group = {a: k for k, g in enumerate(groups) for a in g}
    vals = [
        draw.best(
            [a for a in animals if a not in g],
            s.val_animals,
            animals,
            ALL,
            rng,
            s.tries,
        )
        for g in groups
    ]

    # test B: an animal's sessions by date, one block for each fold it trains in
    b_fold = pd.Series(-1, index=used.index)
    for a, g in used[~used.group_c].groupby("subject_id"):
        trains = [k for k in range(s.n_folds) if k != group.get(a) and a not in vals[k]]
        if len(g) < s.test_b_min_sessions or not trains:
            continue
        order = np.random.default_rng([s.seed, int(a)]).permutation(trains)
        dated = g.sort_values(["date", "session_id"]).index
        for k, block in zip(order, np.array_split(dated, min(len(trains), len(g)))):
            b_fold[block] = int(k)

    out = used.assign(cv_group=used.subject_id.map(group).fillna(-1).astype(int))
    out["cv_b_fold"] = b_fold
    fast = out.group_c.to_numpy(bool)
    for k, test in enumerate(groups):
        sets = np.where(out.cv_b_fold == k, "test_b", "train").astype(object)
        sets[out.subject_id.isin(vals[k]).to_numpy()] = "val"
        sets[out.subject_id.isin(test).to_numpy()] = "test_a"
        sets[fast] = out.loc[fast, "set"].to_numpy()
        out[column(k)] = sets
    return out


def fold_view(out: pd.DataFrame, k: int) -> pd.DataFrame:
    """Fold *k* in ``folds.parquet``'s shape (``set``, ``role``), for check and report."""
    sets = out[column(k)]
    role = pd.Series("train", index=out.index, dtype=object)
    for name in ("val", "test_a"):
        role[out.subject_id.isin(out.loc[sets == name, "subject_id"])] = name
    role[out.cv_group == -1] = "group_c"
    return out.assign(set=sets, role=role).drop(columns="small", errors="ignore")


def check_all(out: pd.DataFrame, n_folds: int) -> None:
    """Every fold passes :func:`folds.check`; each animal is test A in exactly one fold,
    and each session is test B in at most one, its ``cv_b_fold``."""
    for k in range(n_folds):
        check(fold_view(out, k))
    sets = out[[column(k) for k in range(n_folds)]].to_numpy()
    once = (sets[out.cv_group >= 0] == "test_a").sum(axis=1) == 1
    if not once.all():
        raise ValueError("an animal is not test A in exactly one fold")
    regular = ~out.group_c.to_numpy(bool)
    for i in np.flatnonzero(regular):
        where = list(np.flatnonzero(sets[i] == "test_b"))
        marked = int(out.cv_b_fold.iloc[i])
        if where != ([marked] if marked >= 0 else []):
            raise ValueError(f"{out.session_id.iloc[i]}: test B in folds {where}")


def run(spec: DatasetSpec) -> pd.DataFrame:
    s = spec.cross_validation
    if s is None:
        raise SystemExit("cross_validation: the spec has no [cross_validation]")
    layout = Layout(spec.root)
    if not layout.folds.exists():
        raise SystemExit("cross_validation: no folds.parquet (run folds first)")
    out = assign(storage.read_parquet(layout.folds), s)
    check_all(out, s.n_folds)
    storage.write_frame(
        out.drop(columns=FIRST_DRAW, errors="ignore"), layout.session_table
    )
    for k in range(s.n_folds):
        print(f"fold {k}:")
        report(fold_view(out, k))
    return out
