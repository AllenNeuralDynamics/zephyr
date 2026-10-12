"""``folds``: which selected sessions train, validate and test (``folds.parquet``).

Test A and validation hold out whole animals, drawn (``[folds] tries`` seeded draws, the
best kept) to cover rigs and months and match the corpus' breathing-rate bands; test B
holds out sessions of training animals per animal and rig; group C (a camera rate no
other animal has) is held out by session. Each set keeps at most a capped number of
sessions per animal and rig, spread over its dates. Refuses an assignment in which a
held-out animal trains. Why: the package README. Rate bands are cached in the file.
"""

import io
import itertools
import json
import zlib

import numpy as np
import pandas as pd

from zephyr import events, rates

from . import storage
from .config import DatasetSpec, Folds, Layout
from .remote import thread_map

BANDS = ["1-4", "4-8", "8-15"]
EDGES = [0, 1, 4, 8, 15, np.inf]
GRID_HZ = 60.0
SETS = ["train", "val", "test_a", "test_b", "test_c"]


def session_rates(spec: DatasetSpec, layout: Layout, s: dict) -> dict:
    """Seconds of a session's clips at each rate band, and its median rate."""
    frame = storage.read_parquet(layout.thermistor(s["session_id"]))
    found = events.detect(spec.events.method, frame, None, spec.events.params)
    onsets = np.asarray(found.inhale, float)
    tracks = []
    for span in s["clips"].values():
        track = rates.rate_track(
            np.arange(span["start"], span["end"], 1 / GRID_HZ), onsets
        )
        tracks.append(track[np.isfinite(track)])
    track = np.concatenate(tracks)
    counts = np.histogram(track, bins=EDGES)[0]
    return {
        "session_id": s["session_id"],
        "samples": len(track),
        **{f"t_{b}": int(c) for b, c in zip(BANDS, counts[1:4])},
        "q50": float(np.median(track)) if len(track) else np.nan,
    }


def table(spec: DatasetSpec, layout: Layout, workers: int) -> pd.DataFrame:
    """One row per selected session: subject, rig family, date, camera rate, bands."""
    s = spec.folds
    results = json.loads(layout.results.read_text(encoding="utf-8"))
    chosen = set(results["selected"])
    sessions = [x for x in results["sessions"] if x["session_id"] in chosen]
    frame = pd.DataFrame(
        {
            "session_id": [x["session_id"] for x in sessions],
            "subject_id": [str(x["subject_id"]) for x in sessions],
            "rig_name": [x["rig"] for x in sessions],
            "date": [x["date"] for x in sessions],
        }
    )
    frame["rig"] = frame["rig_name"].map(lambda r: s.rig_families.get(r, r))
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    cams = storage.read_parquet(layout.cameras)
    cams = cams[cams["camera"] == s.camera].set_index("session_id")
    frame["camera_hz"] = frame["session_id"].map(cams["after_rate_hz"])
    old = storage.read_parquet(layout.folds) if layout.folds.exists() else None
    cols = ["samples", *[f"t_{b}" for b in BANDS], "q50"]
    if old is not None and set(frame["session_id"]) <= set(old["session_id"]):
        band = old.set_index("session_id")[cols]  # cached: no thermistor read
    else:
        by_id = {x["session_id"]: x for x in sessions}
        rows = thread_map(
            lambda sid: session_rates(spec, layout, by_id[sid]),
            list(frame["session_id"]),
            workers,
        )
        band = pd.DataFrame(rows).set_index("session_id")[cols]
    return frame.join(band, on="session_id")


def spread(g: pd.DataFrame, k: int, rng) -> list:
    """*k* rows of *g* (sorted by date), one from each of *k* date segments."""
    if len(g) <= k:
        return list(g.index)
    edges = np.linspace(0, len(g), k + 1).astype(int)
    return [
        g.index[int(rng.integers(lo, max(lo + 1, hi)))]
        for lo, hi in itertools.pairwise(edges)
    ]


class Draw:
    """The balance score of a set of held-out animals (capped sessions per rig)."""

    def __init__(self, df: pd.DataFrame, seed: int) -> None:
        self.df, self.seed = df, seed
        self.rigs, self.months = sorted(df.rig.unique()), sorted(df.month.unique())
        self.target = self.stats(df)
        self.cache: dict = {}
        self.inc = {
            a: np.array([(g.rig == x).any() for x in self.rigs])
            for a, g in df.groupby("subject_id")
        }

    def stats(self, frame):
        t = frame[[f"t_{b}" for b in BANDS]].sum()
        return np.append(100 * t.values / frame["samples"].sum(), frame["q50"].mean())

    def capped(self, animal: str, cap: int) -> pd.DataFrame:
        out = []
        for rig, g in self.df[self.df.subject_id == animal].groupby("rig"):
            rng = np.random.default_rng(
                [self.seed, int(animal), zlib.crc32(rig.encode())]
            )
            out += spread(g.sort_values("date"), cap, rng)
        return self.df.loc[out]

    def vec(self, animal: str, cap: int):
        if (animal, cap) not in self.cache:
            c = self.capped(animal, cap)
            self.cache[(animal, cap)] = (
                np.array([(c.rig == x).sum() for x in self.rigs]),
                np.array([(c.month == x).sum() for x in self.months]),
                c[[f"t_{b}" for b in BANDS]].sum().values,
                c["samples"].sum(),
                c["q50"].sum(),
                len(c),
            )
        return self.cache[(animal, cap)]

    def score(self, chosen, rest, cap) -> float:
        """Rigs and months covered, rigs left with 2+ other animals, rate bands and
        median rate near the corpus'."""
        v = [self.vec(a, cap) for a in chosen]
        rigs_hit = (sum(x[0] for x in v) > 0).sum()
        months_hit = (sum(x[1] for x in v) > 0).sum()
        keeps = (sum(self.inc[a] for a in rest) >= 2).sum()
        t = sum(x[2] for x in v)
        shares = np.append(
            100 * t / sum(x[3] for x in v), sum(x[4] for x in v) / sum(x[5] for x in v)
        )
        miss = np.abs(shares[:3] - self.target[:3]).sum() + 3 * abs(
            shares[3] - self.target[3]
        )
        return rigs_hit * 3 + months_hit * 2 + keeps * 2 - 2 * miss

    def best(self, pool, n, rest_pool, cap, rng, tries) -> list[str]:
        if len(pool) < n:
            raise SystemExit(f"folds: {len(pool)} animals to draw {n} from")
        best, best_score = None, -np.inf
        for _ in range(tries):
            chosen = list(rng.choice(pool, n, replace=False))
            s = self.score(chosen, [a for a in rest_pool if a not in chosen], cap)
            if s > best_score:
                best, best_score = chosen, s
        return sorted(best)


def assign(df: pd.DataFrame, s: Folds) -> pd.DataFrame:
    """*df* with each animal's ``role`` and each session's ``set`` (None: unused)."""
    df = df.reset_index(drop=True)
    animals = sorted(df.subject_id.unique())
    raw = df.groupby("subject_id").size()
    fast = (
        df["camera_hz"] > s.group_c_min_hz
        if s.group_c_min_hz is not None
        else pd.Series(False, df.index)
    )
    in_c = sorted(df.loc[fast, "subject_id"].unique())
    if sorted(s.group_c) != in_c:
        raise SystemExit(
            f"folds: group_c names {sorted(s.group_c)}, the data has {in_c}"
        )
    draw = Draw(df, s.seed)
    rng = np.random.default_rng(s.seed)
    elig = [a for a in animals if raw[a] >= s.min_sessions and a not in s.group_c]
    test = draw.best(elig, s.test_animals, animals, s.cap_test, rng, s.tries)
    rest = [a for a in animals if a not in test]
    pool = [a for a in rest if raw[a] >= s.min_sessions and a not in s.group_c]
    val = draw.best(pool, s.val_animals, rest, s.cap_val, rng, s.tries)
    trainees = [a for a in rest if a not in val]

    label: dict[str, str] = {}
    for a in test:
        label.update(dict.fromkeys(draw.capped(a, s.cap_test).session_id, "test_a"))
    for a in val:
        label.update(dict.fromkeys(draw.capped(a, s.cap_val).session_id, "val"))
    for a, counts in s.group_c.items():
        g = df[(df.subject_id == a) & fast].sort_values("date")
        parts = [
            p for p, k in zip(["train", "val", "test_c"], counts) for _ in range(k)
        ]
        if len(parts) != len(g):
            raise SystemExit(
                f"folds: group_c {a} splits {len(parts)} of {len(g)} sessions"
            )
        parts = list(np.random.default_rng([s.seed, int(a), 359]).permutation(parts))
        label.update(dict(zip(g.session_id, parts)))
    for a in trainees:
        for rig, g in df[(df.subject_id == a) & ~fast].groupby("rig"):
            g = g.sort_values("date")
            rng2 = np.random.default_rng([s.seed, int(a), zlib.crc32(rig.encode()), 7])
            k = max(
                0, min(s.cap_test_b, len(g) // s.test_b_one_in, len(g) - s.train_keep)
            )
            b = spread(g, k, rng2) if k else []
            left = g.drop(index=b)
            t = spread(left, s.cap_train, rng2) if len(left) else []
            label.update(dict.fromkeys(g.loc[b].session_id, "test_b"))
            label.update(dict.fromkeys(g.loc[t].session_id, "train"))
    out = df.copy()
    out["set"] = out.session_id.map(label)
    role = {
        **dict.fromkeys(trainees, "train"),
        **dict.fromkeys(val, "val"),
        **dict.fromkeys(test, "test_a"),
        **dict.fromkeys(s.group_c, "group_c"),
    }
    out["role"] = out.subject_id.map(role)
    out["group_c"] = fast.values
    return out


def check(df: pd.DataFrame) -> None:
    """Refuse an assignment in which a held-out animal trains, or a set is unknown."""
    bad = sorted(set(df["set"].dropna()) - set(SETS))
    if bad:
        raise ValueError(f"unknown sets {bad}")
    held = set(df.loc[df["role"].isin(["val", "test_a"]), "subject_id"])
    used = df[df["set"].isin(["train", "test_b", "test_c"])]
    leak = sorted(held & set(used["subject_id"]))
    if leak:
        raise ValueError(f"held-out animals in training or test B/C: {leak}")
    for role, allowed in (("val", {"val"}), ("test_a", {"test_a"})):
        sets = set(df.loc[df["role"] == role, "set"].dropna())
        if not sets <= allowed:
            raise ValueError(
                f"{role} animals have sessions in {sorted(sets - allowed)}"
            )


def report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SETS:
        g = df[df["set"] == name]
        if g.empty:
            continue
        t = 100 * g[[f"t_{b}" for b in BANDS]].sum() / g["samples"].sum()
        rows.append(
            {
                "set": name,
                "animals": g.subject_id.nunique(),
                "sessions": len(g),
                "rigs": g.rig.nunique(),
                "months": g.month.nunique(),
                **{f"{b} Hz %": round(v, 1) for b, v in zip(BANDS, t)},
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print(f"folds: {int(df['set'].notna().sum())} of {len(df)} sessions assigned")
    return summary


def run(spec: DatasetSpec, workers: int = 6) -> pd.DataFrame:
    if spec.folds is None:
        raise SystemExit("folds: the spec has no [folds]")
    layout = Layout(spec.root)
    out = assign(table(spec, layout, workers), spec.folds)
    check(out)
    out["date"] = out["date"].astype(str)
    buffer = io.BytesIO()
    out.to_parquet(buffer, index=False)
    storage.write_bytes(layout.folds, buffer.getvalue())
    report(out)
    return out
