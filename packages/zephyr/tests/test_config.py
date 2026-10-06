import tempfile
import unittest
from pathlib import Path

from _clipfiles import touch_clip, write_fold, write_list

from zephyr.config import ClipList, Experiment, Fold, load


class _Dataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()


class ClipListTests(_Dataset):
    def test_paths_resolve_against_the_file_not_the_cwd(self):
        video = touch_clip(self.root / "data" / "a", 1, 1)
        path = write_list(self.root / "configs" / "x.toml", [video])
        clip = load(ClipList, path).resolve()[0]
        self.assertEqual(clip.video, video)
        self.assertEqual(clip.timestamps, video.with_suffix(".parquet"))
        self.assertEqual(clip.thermistor, video.parent / "thermistor_1_part_1.parquet")
        self.assertEqual(clip.group, "1")

    def test_side_right_underscore_is_part_of_the_camera_not_the_group(self):
        video = touch_clip(self.root / "ood", 13, 2, camera="side_right")
        clip = load(ClipList, write_list(self.root / "l.toml", [video])).resolve()[0]
        self.assertEqual(clip.group, "13")
        self.assertEqual(clip.thermistor.name, "thermistor_13_part_2.parquet")

    def test_thermistor_false_marks_a_clip_unlabelled(self):
        video = touch_clip(self.root / "t", 1, 1, thermistor=False)
        path = write_list(self.root / "l.toml", [video])
        path.write_text(
            path.read_text().replace("[[clip]]", "[[clip]]\nthermistor = false")
        )
        clip = load(ClipList, path).resolve()[0]
        self.assertIsNone(clip.thermistor)
        self.assertFalse(clip.labelled)

    def test_derived_but_missing_thermistor_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1, thermistor=False)
        with self.assertRaisesRegex(ValueError, "do not exist"):
            load(ClipList, write_list(self.root / "l.toml", [video]))

    def test_missing_video_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1)
        path = write_list(self.root / "l.toml", [video])
        video.unlink()
        with self.assertRaisesRegex(ValueError, "do not exist"):
            load(ClipList, path)

    def test_box_outside_the_target_frame_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1)
        path = write_list(self.root / "l.toml", [video], box=(80, 10, 32, 32))
        with self.assertRaisesRegex(ValueError, "outside"):
            load(ClipList, path)

    def test_nonpositive_box_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1)
        path = write_list(self.root / "l.toml", [video], box=(0, 0, 0, 8))
        with self.assertRaisesRegex(ValueError, "w, h > 0"):
            load(ClipList, path)

    def test_unknown_key_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1)
        path = write_list(self.root / "l.toml", [video], extra="selct_fs = 60")
        with self.assertRaisesRegex(ValueError, "selct_fs"):
            load(ClipList, path)

    def test_underivable_clip_needs_explicit_values(self):
        folder = self.root / "t"
        folder.mkdir()
        for name in ("odd.mp4", "odd_ts.parquet", "odd_th.parquet"):
            (folder / name).touch()
        path = write_list(self.root / "l.toml", [folder / "odd.mp4"])
        with self.assertRaisesRegex(ValueError, "cannot derive"):
            load(ClipList, path)
        path.write_text(
            path.read_text().replace(
                "[[clip]]",
                '[[clip]]\ngroup = "g"\ntimestamps = "t/odd_ts.parquet"\n'
                'thermistor = "t/odd_th.parquet"',
            )
        )
        clip = load(ClipList, path).resolve()[0]
        self.assertEqual((clip.group, clip.timestamps.name), ("g", "odd_ts.parquet"))

    def test_duplicate_video_is_an_error(self):
        video = touch_clip(self.root / "t", 1, 1)
        with self.assertRaisesRegex(ValueError, "more than once"):
            load(ClipList, write_list(self.root / "l.toml", [video, video]))

    def test_mixed_box_sizes_are_an_error(self):
        videos = [touch_clip(self.root / "t", 1, p) for p in (1, 2)]
        path = write_list(self.root / "l.toml", videos)
        path.write_text(path.read_text().replace("32, 32", "16, 16", 1))
        with self.assertRaisesRegex(ValueError, "mixed sizes"):
            load(ClipList, path)

    def test_derive_pattern_must_capture_group(self):
        video = touch_clip(self.root / "t", 1, 1)
        path = write_list(self.root / "l.toml", [video])
        path.write_text(path.read_text() + '\n[derive]\npattern = "(?P<x>[0-9]+)"\n')
        with self.assertRaisesRegex(ValueError, "group"):
            load(ClipList, path)


class FoldTests(_Dataset):
    def _fold(self, train_videos, test_videos):
        train = write_list(self.root / "c" / "train.toml", train_videos)
        test = write_list(self.root / "c" / "test.toml", test_videos)
        return write_fold(self.root / "f.toml", [(train, 1.0)], {"held": test})

    def test_valid_fold_loads(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        fold = load(Fold, self._fold([a], [b]))
        self.assertEqual(fold.train_params.epochs, 200)
        self.assertEqual(list(fold.test), ["held"])

    def test_a_fold_cannot_choose_another_network(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        path = self._fold([a], [b])
        path.write_text(path.read_text() + '\n[train_params]\narch = "tscan"\n')
        with self.assertRaisesRegex(ValueError, "arch"):
            load(Fold, path)

    def test_same_video_in_train_and_test_is_refused(self):
        a = touch_clip(self.root / "d", 1, 1)
        with self.assertRaisesRegex(ValueError, "also a training clip"):
            load(Fold, self._fold([a], [a]))

    def test_other_part_of_a_training_recording_is_refused(self):
        a = touch_clip(self.root / "d", 1, 1)
        a2 = touch_clip(self.root / "d", 1, 2)
        with self.assertRaisesRegex(ValueError, "shares recording"):
            load(Fold, self._fold([a], [a2]))

    def test_same_group_label_in_two_folders_is_two_recordings(self):
        a = touch_clip(self.root / "train", 1, 1)
        b = touch_clip(self.root / "test", 1, 1)
        load(Fold, self._fold([a], [b]))

    def test_unlabelled_training_clip_is_refused(self):
        a = touch_clip(self.root / "d", 1, 1, thermistor=False)
        b = touch_clip(self.root / "d", 2, 1)
        train = write_list(self.root / "c" / "train.toml", [a])
        train.write_text(
            train.read_text().replace("[[clip]]", "[[clip]]\nthermistor = false")
        )
        test = write_list(self.root / "c" / "test.toml", [b])
        path = write_fold(self.root / "f.toml", [(train, 1.0)], {"held": test})
        with self.assertRaisesRegex(ValueError, "need a thermistor"):
            load(Fold, path)

    def test_lists_must_share_a_recipe(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        train = write_list(self.root / "c" / "train.toml", [a])
        test = write_list(self.root / "c" / "test.toml", [b], target_size=(100, 90))
        path = write_fold(self.root / "f.toml", [(train, 1.0)], {"held": test})
        with self.assertRaisesRegex(ValueError, "disagree on"):
            load(Fold, path)

    def test_unboxed_clip_is_refused(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        train = write_list(self.root / "c" / "train.toml", [a])
        train.write_text(train.read_text().replace("box = [10, 10, 32, 32]", ""))
        test = write_list(self.root / "c" / "test.toml", [b])
        path = write_fold(self.root / "f.toml", [(train, 1.0)], {"held": test})
        with self.assertRaisesRegex(ValueError, "no box"):
            load(Fold, path)

    def test_all_is_a_reserved_group_name(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        train = write_list(self.root / "c" / "train.toml", [a])
        test = write_list(self.root / "c" / "test.toml", [b])
        path = write_fold(self.root / "f.toml", [(train, 1.0)], {"all": test})
        with self.assertRaisesRegex(ValueError, "reserved"):
            load(Fold, path)

    def _one_list_fold(self, train_groups, test_groups) -> Path:
        """A fold that trains on and scores groups of one shared clip list."""
        videos = [touch_clip(self.root / "d", g, p) for g in (1, 2, 3) for p in (1, 2)]
        clips = write_list(self.root / "c" / "all.toml", videos)
        path = self.root / "f.toml"
        path.write_text(
            '[[train]]\nclips = "c/all.toml"\n'
            f"groups = {[str(g) for g in train_groups]}\n\n"
            f'[test]\nheld = {{ clips = "c/all.toml", groups = '
            f"{[str(g) for g in test_groups]} }}\n"
        )
        self.assertTrue(clips.exists())
        return path

    def test_groups_select_sessions_from_one_list(self):
        fold = load(Fold, self._one_list_fold([1, 2], [3]))
        (train,) = fold.train_clips()
        self.assertEqual(sorted({c.group for c in train}), ["1", "2"])
        self.assertEqual(len(train), 4)
        self.assertEqual([c.group for c in fold.test_clips()["held"]], ["3", "3"])

    def test_overlapping_group_selections_are_refused(self):
        with self.assertRaisesRegex(ValueError, "also a training clip"):
            load(Fold, self._one_list_fold([1, 2], [2]))

    def test_unknown_group_is_an_error(self):
        with self.assertRaisesRegex(ValueError, r"groups \['9'\] are not in the list"):
            load(Fold, self._one_list_fold([1, 2], [9]))

    def test_experiment_loads_its_folds_and_rejects_duplicate_seeds(self):
        a = touch_clip(self.root / "d", 1, 1)
        b = touch_clip(self.root / "d", 2, 1)
        self._fold([a], [b])
        exp = self.root / "e.toml"
        exp.write_text(
            'folds = ["f.toml"]\nseeds = [1, 2]\n'
            'output_dir = "out"\nfeatures_dir = "feat"\n'
        )
        self.assertEqual(load(Experiment, exp).output_dir, self.root / "out")
        exp.write_text(exp.read_text().replace("[1, 2]", "[1, 1]"))
        with self.assertRaisesRegex(ValueError, "distinct"):
            load(Experiment, exp)


if __name__ == "__main__":
    unittest.main()
