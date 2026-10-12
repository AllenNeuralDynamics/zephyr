"""``folds.assign``: whole animals held out, capped sets, test B and group C rules."""

import unittest

import numpy as np
import pandas as pd

from zephyr.corpus import folds
from zephyr.corpus.config import Folds

SETTINGS = Folds(
    seed=7,
    camera="FaceCamera",
    test_animals=3,
    val_animals=2,
    min_sessions=6,
    cap_train=4,
    cap_val=2,
    cap_test=3,
    cap_test_b=2,
    test_b_one_in=3,
    train_keep=2,
    tries=50,
    rig_families={"FRG.1-A": "1A"},
    group_c_min_hz=300.0,
    group_c={"199": (2, 1, 1)},
)


def sessions() -> pd.DataFrame:
    """20 animals on 1-2 of three rigs; animal 199 has 4 fast sessions besides."""
    rng = np.random.default_rng(0)
    rows = []
    for a in range(100, 120):
        rigs = ["1A", "2A"] if a % 3 else ["3A"]
        for rig in rigs:
            for k in range(int(rng.integers(3, 12))):
                rows.append((str(a), rig, 240.0, k))
    rows += [("199", "1A", 240.0, k) for k in range(5)]
    rows += [("199", "1A", 359.0, 20 + k) for k in range(4)]
    frame = pd.DataFrame(rows, columns=["subject_id", "rig", "camera_hz", "day"])
    frame["session_id"] = [
        f"{a}_{d}_{i}" for i, (a, d) in enumerate(zip(frame.subject_id, frame.day))
    ]
    frame["date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(frame["day"] * 9, "D")
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["samples"] = 1000
    shares = rng.dirichlet([5, 5, 1], len(frame)) * 1000
    for band, column in zip(folds.BANDS, shares.T):
        frame[f"t_{band}"] = column.astype(int)
    frame["q50"] = rng.uniform(3, 6, len(frame))
    return frame


class AssignTests(unittest.TestCase):
    def setUp(self):
        self.out = folds.assign(sessions(), SETTINGS)

    def test_held_out_animals_never_train(self):
        folds.check(self.out)  # raises on any leak
        roles = self.out.groupby("subject_id")["role"].first()
        self.assertEqual((roles == "test_a").sum(), 3)
        self.assertEqual((roles == "val").sum(), 2)
        held = set(roles[roles.isin(["test_a", "val"])].index)
        self.assertNotIn("199", held)  # group C animals are never held out whole
        trained = set(self.out.loc[self.out["set"] == "train", "subject_id"])
        self.assertFalse(held & trained)

    def test_sets_are_capped_per_animal_and_rig(self):
        counts = (
            self.out[~self.out.group_c].groupby(["set", "subject_id", "rig"]).size()
        )
        for name, cap in (("train", 4), ("val", 2), ("test_a", 3), ("test_b", 2)):
            if name in counts.index.get_level_values(0):
                self.assertLessEqual(counts.loc[name].max(), cap, name)

    def test_test_b_leaves_each_pair_sessions_to_train(self):
        regular = self.out[(self.out.role == "train") & ~self.out.group_c]
        for (_, _), g in regular.groupby(["subject_id", "rig"]):
            b = int((g["set"] == "test_b").sum())
            self.assertLessEqual(b, len(g) // 3)
            self.assertGreaterEqual(len(g) - b, 2)
            if b:
                self.assertGreater(int((g["set"] == "train").sum()), 0)

    def test_group_c_is_split_by_session(self):
        fast = self.out[self.out.group_c]
        self.assertEqual(sorted(fast["set"]), ["test_c", "train", "train", "val"])
        self.assertEqual(set(fast["role"]), {"group_c"})

    def test_the_draw_is_reproducible(self):
        again = folds.assign(sessions(), SETTINGS)
        pd.testing.assert_series_equal(again["set"], self.out["set"])

    def test_a_leak_is_refused(self):
        leaked = self.out.copy()
        animal = leaked.loc[leaked.role == "test_a", "subject_id"].iloc[0]
        leaked.loc[leaked.subject_id == animal, "set"] = "train"
        with self.assertRaises(ValueError):
            folds.check(leaked)

    def test_group_c_must_match_the_data(self):
        wrong = SETTINGS.model_copy(update={"group_c": {"199": (1, 1, 1)}})
        with self.assertRaises(SystemExit):
            folds.assign(sessions(), wrong)


if __name__ == "__main__":
    unittest.main()
