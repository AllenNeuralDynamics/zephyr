"""``folds.choose_small`` keeps the fold structure; ``write_small`` lists only its sessions."""

import unittest
import uuid

import pandas as pd
import tomlkit
from test_folds import SETTINGS, sessions

from zephyr.corpus import folds, lists, storage
from zephyr.corpus.config import Layout, Small

SMALL = Small(
    seed=3,
    tries=40,
    test_animals=2,
    val_animals=1,
    train_animals=4,
    cap_train=1,
    cap_val=1,
    cap_test=1,
    cap_test_b=1,
    group_c=(1, 1, 1),
)


class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.full = folds.assign(sessions(), SETTINGS)
        self.keep = folds.choose_small(self.full, SMALL)
        self.small = self.full[self.full.session_id.isin(self.keep)]

    def test_a_subset_of_the_assigned_sessions_with_the_structure_kept(self):
        self.assertTrue(self.small["set"].notna().all())
        self.assertLess(len(self.small), len(self.full[self.full["set"].notna()]))
        folds.check(self.small)  # held-out animals never train
        plain = self.small[~self.small.group_c]
        for name, n in (("test_a", 2), ("val", 1)):
            self.assertEqual(plain[plain["set"] == name].subject_id.nunique(), n)
        trained = set(plain.loc[plain["set"] == "train", "subject_id"])
        self.assertEqual(len(trained), 4)
        # test B only has sessions of the kept training animals
        self.assertLessEqual(
            set(plain.loc[plain["set"] == "test_b", "subject_id"]), trained
        )

    def test_the_caps_per_animal_and_rig_hold(self):
        cells = (
            self.small[~self.small.group_c].groupby(["subject_id", "rig", "set"]).size()
        )
        self.assertLessEqual(cells.max(), 1)

    def test_group_c_is_split_by_session_as_asked(self):
        c = self.small[self.small.group_c]
        self.assertEqual(
            c["set"].value_counts().to_dict(), {"train": 1, "val": 1, "test_c": 1}
        )

    def test_it_is_repeatable(self):
        self.assertEqual(self.keep, folds.choose_small(self.full, SMALL))


class ListTests(unittest.TestCase):
    def setUp(self):
        self.layout = Layout(f"memory://small-{uuid.uuid4().hex[:8]}")
        doc = tomlkit.document()
        doc.add("preprocess", {"target_size": [360, 270]})
        doc.add("events", {"method": "peak_detection"})
        entries = tomlkit.aot()
        for sid in ("a", "b"):
            for camera in ("FaceCamera", "SideCamera"):
                stem = f"clips/{sid}_span1_{camera}"
                entry = {
                    "video": f"{stem}.mp4",
                    "timestamps": f"{stem}.parquet",
                    "thermistor": f"clips/{sid}_span1_thermistor.parquet",
                    "group": sid,
                }
                if camera == "FaceCamera":
                    entry["box"] = [4, 8, 96, 96]
                else:
                    entry["selection"] = "exclude"
                entries.append(entry)
        doc.add("clip", entries)
        storage.write_text(self.layout.clip_list, tomlkit.dumps(doc))

    def folds(self, small):
        frame = pd.DataFrame({"session_id": ["a", "b"], "small": small})
        storage.write_frame(frame, self.layout.folds)

    def test_only_the_small_sessions_are_listed_with_their_boxes_and_marks(self):
        self.folds([True, False])
        self.assertEqual(lists.write_small(self.layout), 2)
        doc = tomlkit.parse(self.layout.clip_list_small.read_text(encoding="utf-8"))
        self.assertEqual({e["group"] for e in doc["clip"]}, {"a"})
        self.assertEqual(
            [("box" in e, e.get("selection")) for e in doc["clip"]],
            [(True, None), (False, "exclude")],
        )

    def test_no_small_sessions_removes_the_list(self):
        self.folds([True, False])
        lists.write_small(self.layout)
        self.folds([False, False])
        self.assertEqual(lists.write_small(self.layout), 0)
        self.assertFalse(self.layout.clip_list_small.exists())


if __name__ == "__main__":
    unittest.main()
