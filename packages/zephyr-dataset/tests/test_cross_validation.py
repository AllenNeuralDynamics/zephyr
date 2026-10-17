"""``cross_validation.assign``: animals tested once, date blocks for test B, group C kept."""

import unittest

import pandas as pd
from test_folds import SETTINGS, sessions

from zephyr.corpus import cross_validation as cv
from zephyr.corpus import folds
from zephyr.corpus.config import CrossValidation

CV = CrossValidation(seed=3, n_folds=4, val_animals=2, test_b_min_sessions=3, tries=20)


def published() -> pd.DataFrame:
    out = folds.assign(sessions(), SETTINGS)
    out["date"] = out["date"].astype(str)  # as folds.run writes it
    return out


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = published()
        self.out = cv.assign(self.base, CV)
        self.folds = [cv.column(k) for k in range(CV.n_folds)]

    def test_every_fold_passes_the_leakage_check(self):
        cv.check_all(self.out, CV.n_folds)  # raises on any leak

    def test_only_assigned_sessions_take_part(self):
        used = set(self.base.loc[self.base["set"].notna(), "session_id"])
        self.assertEqual(set(self.out.session_id), used)

    def test_each_animal_is_test_a_in_one_fold_whole(self):
        regular = self.out[self.out.cv_group >= 0]
        for _, g in regular.groupby("subject_id"):
            k = g.cv_group.iloc[0]
            self.assertTrue((g[cv.column(k)] == "test_a").all())
            others = g[[c for c in self.folds if c != cv.column(k)]]
            self.assertFalse((others == "test_a").any().any())

    def test_validation_animals_are_whole_and_counted(self):
        for c in self.folds:
            val = self.out[self.out[c] == "val"]
            animals = set(val.loc[val.cv_group >= 0, "subject_id"])
            self.assertEqual(len(animals), CV.val_animals)
            rest = self.out[self.out.subject_id.isin(animals)]
            self.assertTrue((rest.loc[~rest.group_c, c] == "val").all())

    def test_test_b_is_a_date_block_of_a_training_animal(self):
        for animal, g in self.out[~self.out.group_c].groupby("subject_id"):
            g = g.sort_values(["date", "session_id"])
            if len(g) < CV.test_b_min_sessions:
                self.assertTrue((g.cv_b_fold == -1).all(), animal)
                continue
            for k in set(g.cv_b_fold) - {-1}:
                block = (g.cv_b_fold == k).to_numpy().nonzero()[0]
                self.assertEqual(list(block), list(range(block[0], block[-1] + 1)))
                self.assertTrue(
                    (g.loc[g.cv_b_fold == k, cv.column(k)] == "test_b").all()
                )

    def test_group_c_keeps_its_published_sets(self):
        fast = self.out[self.out.group_c]
        for c in self.folds:
            self.assertTrue((fast[c] == fast["set"]).all())
        self.assertTrue(
            (self.out.loc[self.out.role == "group_c", "cv_group"] == -1).all()
        )

    def test_the_draw_is_reproducible(self):
        again = cv.assign(published(), CV)
        pd.testing.assert_frame_equal(again[self.folds], self.out[self.folds])

    def test_a_leak_is_refused(self):
        leaked = self.out.copy()
        animal = leaked.loc[leaked.fold_0 == "test_a", "subject_id"].iloc[0]
        other = leaked.subject_id == animal
        leaked.loc[other, "fold_1"] = "train"
        leaked.loc[other.idxmax(), "fold_0"] = "train"
        with self.assertRaises(ValueError):
            cv.check_all(leaked, CV.n_folds)


if __name__ == "__main__":
    unittest.main()
