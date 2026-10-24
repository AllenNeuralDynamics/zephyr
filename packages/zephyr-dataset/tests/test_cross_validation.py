"""``cross_validation``: animals tested once, date blocks for test B, group C kept;
``sessions.parquet`` and the READMEs it is published with."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from test_folds import SETTINGS, sessions

from zephyr.corpus import cross_validation as cv
from zephyr.corpus import folds, readme, storage
from zephyr.corpus.config import CrossValidation, DatasetSpec, Layout

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


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec = DatasetSpec.model_construct(root=self.tmp.name, cross_validation=CV)
        self.layout = Layout(self.tmp.name)
        self.layout.raw.mkdir()
        storage.write_frame(published().assign(small=False), self.layout.folds)

    def test_sessions_parquet_holds_the_folds_but_not_the_first_draw(self):
        cv.run(self.spec)
        table = pd.read_parquet(self.layout.session_table)
        base = published()
        self.assertEqual(len(table), base["set"].notna().sum())
        self.assertFalse({"set", "role", "small"} & set(table.columns))
        self.assertTrue({"subject_id", "camera_hz", "q50", "cv_group"} <= set(table))
        self.assertEqual(
            [c for c in table if c.startswith("fold_")],
            [cv.column(k) for k in range(CV.n_folds)],
        )
        self.assertFalse(Path(self.tmp.name, "cross_validation.parquet").exists())

    def test_readme_fills_every_count(self):
        cv.run(self.spec)
        table = pd.read_parquet(self.layout.session_table)
        counts = {
            "sessions": 40,
            "passed": 30,
            "clips_fit": 25,
            "dropped_for_cameras": 2,
            "selected": 20,
        }
        storage.write_text(self.layout.selection, json.dumps({"counts": counts}))
        storage.write_frame(
            pd.DataFrame({"session_id": [f"s{i}" for i in range(50)]}),
            self.layout.inventory,
        )
        clip = '[[clip]]\nvideo = "clips/{s}_span1_{c}.mp4"\ngroup = "{s}"\n'
        self.layout.clip_list.write_text(
            "".join(
                clip.format(s=s, c=c)
                for s in table.session_id
                for c in ("FaceCamera", "SideCamera")
            )
        )
        with self.assertRaises(SystemExit):  # the step that writes it comes first
            readme.run(self.spec.model_copy(update={"cross_validation": None}))
        readme.run(self.spec)
        top = self.layout.readme.read_text(encoding="utf-8")
        raw = self.layout.raw_readme.read_text(encoding="utf-8")
        for text in (top, raw):
            self.assertNotIn("{", text)
        self.assertIn(f"{len(table)} sessions", top)
        self.assertIn(f"| SideCamera | {len(table)} | {len(table)} |", top)
        self.assertIn(f"`fold_{CV.n_folds - 1}`", top)
        self.assertIn("3 sessions without a face or side video", raw)  # 25 - 20 - 2
        self.assertRegex(raw, r"50 +packaged sessions found")


if __name__ == "__main__":
    unittest.main()
