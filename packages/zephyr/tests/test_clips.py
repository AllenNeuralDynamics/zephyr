import tempfile
import unittest
from pathlib import Path

import numpy as np
from _clipfiles import touch_clip, write_list

from zephyr import features
from zephyr.annotate import BoxState, initial_state, recordings_of
from zephyr.clips import scan, write_boxes
from zephyr.config import OUTPUT_FS, ClipList, dump, load
from zephyr.preprocess import output_times


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()


class ScanTests(_Tmp):
    def test_scan_lists_videos_in_natural_order_with_a_loadable_result(self):
        for part in (10, 2, 1):
            touch_clip(self.root / "d", 1, part)
        out = self.root / "cfg" / "list.toml"
        scan(self.root / "d", ["video_face_*.mp4"], out, target_size=(100, 80))
        names = [c.video.name for c in load(ClipList, out).resolve()]
        self.assertEqual(
            names,
            [f"video_face_1_part_{p}.mp4" for p in (1, 2, 10)],
        )
        self.assertIn("../d/video_face_1_part_1.mp4", out.read_text())

    def test_scan_refuses_to_write_an_unloadable_list(self):
        touch_clip(self.root / "d", 1, 1, thermistor=False)
        out = self.root / "list.toml"
        with self.assertRaisesRegex(SystemExit, "--unlabelled"):
            scan(self.root / "d", ["*.mp4"], out, target_size=(100, 80))
        self.assertFalse(out.exists())
        scan(self.root / "d", ["*.mp4"], out, target_size=(100, 80), unlabelled=True)
        self.assertFalse(load(ClipList, out).resolve()[0].labelled)

    def test_write_boxes_keeps_comments_and_other_keys(self):
        video = touch_clip(self.root / "d", 1, 1)
        path = write_list(self.root / "l.toml", [video])
        path.write_text("# my note\n" + path.read_text().replace("box", "# kept\nbox"))
        write_boxes(path, {video: (1, 2, 32, 32)}, target_size=(100, 80))
        text = path.read_text()
        self.assertIn("# my note", text)
        self.assertIn("# kept", text)
        self.assertEqual(load(ClipList, path).resolve()[0].box, (1, 2, 32, 32))


class AnnotateStateTests(_Tmp):
    def _state(self) -> BoxState:
        videos = [touch_clip(self.root / "d", g, p) for g in (1, 2) for p in (1, 2)]
        path = write_list(self.root / "l.toml", videos)
        text = path.read_text().replace("box = [10, 10, 32, 32]\n", "")
        path.write_text(text)
        clips = load(ClipList, path).resolve()
        return initial_state(clips, (200, 160), (100, 80), 32)

    def test_one_box_per_recording_and_only_placed_ones_are_written(self):
        state = self._state()
        self.assertEqual(len(state.recordings), 2)
        self.assertEqual(state.clip_boxes(), {})
        state.place(50, 40)
        boxes = state.clip_boxes()
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(set(boxes.values())), 1)

    def test_override_gives_one_clip_its_own_box(self):
        state = self._state()
        state.place(50, 40)
        state.toggle_override()
        state.place(30, 30)
        boxes = state.clip_boxes()
        first, second = state.recording.clips
        self.assertNotEqual(boxes[first.video], boxes[second.video])
        state.toggle_override()
        self.assertEqual(state.clip_boxes()[first.video], boxes[second.video])

    def test_recordings_group_by_folder_and_group(self):
        a = touch_clip(self.root / "x", 1, 1)
        b = touch_clip(self.root / "y", 1, 1)
        clips = load(ClipList, write_list(self.root / "l.toml", [a, b])).resolve()
        self.assertEqual(len(recordings_of(clips)), 2)


class FeatureKeyTests(_Tmp):
    def test_key_changes_with_box_and_recipe_only(self):
        video = touch_clip(self.root / "d", 1, 1)
        cl = load(ClipList, write_list(self.root / "l.toml", [video]))
        clip = cl.resolve()[0]
        base = features.prefix(clip, cl.preprocess)
        self.assertEqual(base, features.prefix(clip, cl.preprocess))
        moved = clip.model_copy(update={"box": (11, 10, 32, 32)})
        self.assertNotEqual(base, features.prefix(moved, cl.preprocess))
        other = cl.preprocess.model_copy(update={"motion_tau_s": 0.02})
        self.assertNotEqual(base, features.prefix(clip, other))
        self.assertTrue(base.startswith(video.stem + "-"))

    def test_output_fs_keys_only_off_its_default(self):
        # Caches made before output_fs existed must keep matching at 60 Hz.
        video = touch_clip(self.root / "d", 1, 1)
        cl = load(ClipList, write_list(self.root / "l.toml", [video]))
        clip = cl.resolve()[0]
        self.assertEqual(cl.preprocess.output_fs, OUTPUT_FS)
        key = features.cache_key(clip, cl.preprocess)["preprocess"]
        self.assertNotIn("output_fs", key)
        recipe = dump(cl.preprocess)
        del recipe["output_fs"]
        self.assertEqual(key, recipe)
        faster = cl.preprocess.model_copy(update={"output_fs": 120.0})
        self.assertEqual(
            features.cache_key(clip, faster)["preprocess"]["output_fs"], 120.0
        )
        self.assertNotEqual(
            features.prefix(clip, cl.preprocess), features.prefix(clip, faster)
        )

    def test_output_grid_follows_output_fs(self):
        anchors = np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(np.diff(output_times(anchors)), 1 / OUTPUT_FS)
        grid = output_times(anchors, 120.0)
        self.assertEqual(len(grid), 121)
        np.testing.assert_allclose(np.diff(grid), 1 / 120.0)
        self.assertEqual(grid[0], 0.5)

    def test_require_names_what_is_missing(self):
        video = touch_clip(self.root / "d", 1, 1)
        cl = load(ClipList, write_list(self.root / "l.toml", [video]))
        with self.assertRaisesRegex(SystemExit, "zephyr preprocess"):
            features.require(cl.resolve(), cl.preprocess, self.root / "cache")
