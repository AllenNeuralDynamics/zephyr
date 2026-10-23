import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _clipfiles import touch_clip, write_list

from zephyr import features
from zephyr.annotate import (
    BoxState,
    frame_path,
    frame_size,
    initial_state,
    recordings_of,
)
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

    def test_a_list_of_clips_kept_elsewhere_loads_without_its_files(self):
        out = self.root / "list.toml"
        out.write_text(
            "[preprocess]\ntarget_size = [100, 80]\n\n[[clip]]\n"
            'video = "clips/a_span1_FaceCamera.mp4"\n'
            'timestamps = "clips/a_span1_FaceCamera.parquet"\n'
            'thermistor = "clips/a_span1_thermistor.parquet"\ngroup = "a"\n'
        )
        with self.assertRaises(ValueError):
            load(ClipList, out)
        clips = load(ClipList, out, check_files=False).resolve()
        self.assertEqual([c.group for c in clips], ["a"])
        frames = self.root / "frames"
        frames.mkdir()
        np.save(frame_path(frames, clips[0].video), np.zeros((54, 72), np.uint8))
        self.assertEqual(frame_size(clips, frames), (72, 54))
        with self.assertRaises(SystemExit):
            frame_size(clips, self.root / "none")

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

    def test_write_boxes_rereads_a_list_changed_since_it_wrote(self):
        video = touch_clip(self.root / "d", 1, 1)
        path = write_list(self.root / "l.toml", [video])
        write_boxes(path, {video: (1, 2, 32, 32)})
        path.write_text("# edited elsewhere\n" + path.read_text())
        write_boxes(path, {video: (3, 4, 32, 32)})
        self.assertIn("# edited elsewhere", path.read_text())
        self.assertEqual(load(ClipList, path).resolve()[0].box, (3, 4, 32, 32))


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

    def _scene(self, centre, rng) -> np.ndarray:
        """A native frame of noise with a bright blob (a snout) at target *centre*."""
        y, x = np.mgrid[0:160, 0:200]
        cx, cy = 2 * centre[0], 2 * centre[1]
        blob = 160 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 12.0**2))
        return np.clip(40 + blob + rng.normal(0, 4, x.shape), 0, 255).astype(np.uint8)

    def test_auto_matches_the_hand_placed_box_and_leaves_poor_matches(self):
        videos = [touch_clip(self.root / "d", g, 1) for g in (1, 2, 3)]
        path = write_list(self.root / "l.toml", videos)
        path.write_text(path.read_text().replace("box = [10, 10, 32, 32]\n", ""))
        state = initial_state(load(ClipList, path).resolve(), (200, 160), (100, 80), 32)
        rng = np.random.default_rng(0)
        frames = {
            state.recordings[0].key: [self._scene((60, 50), rng)],
            state.recordings[1].key: [self._scene((30, 25), rng)],
            state.recordings[2].key: [rng.integers(0, 255, (160, 200), np.uint8)],
        }
        with self.assertRaises(ValueError):  # nothing to look for yet
            state.auto(lambda r: frames[r.key])
        state.place(60, 50)
        scores = state.auto(lambda r: frames[r.key])
        first, second, noise = (r.key for r in state.recordings)
        self.assertEqual(set(scores), {second, noise})
        self.assertGreater(scores[second], 0.9)
        self.assertEqual(state.boxes[second], (14, 9, 32, 32))  # centred on (30, 25)
        self.assertEqual(set(state.matched), {second})
        self.assertEqual(state.placed, {first, second})
        state.select(second)
        state.nudge(1, 0)  # moved by hand: no longer a match
        self.assertEqual(state.matched, {})

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
        other = cl.preprocess.model_copy(update={"select_fs": 30.0})
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
        self.assertEqual(key, cl.preprocess.KEY_RETIRED | recipe)
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


class FeatureCacheTests(_Tmp):
    T, H, W = 50, 6, 8

    def _cached(self, *, frames: bool = True):
        video = touch_clip(self.root / "d", 1, 1)
        cl = load(ClipList, write_list(self.root / "l.toml", [video]))
        (clip,) = cl.resolve()
        cache = self.root / "cache"
        cache.mkdir(exist_ok=True)
        files = features.paths(cache, features.prefix(clip, cl.preprocess))
        files["sidecar"].write_text(
            json.dumps(
                {
                    "key": features.cache_key(clip, cl.preprocess),
                    "thermistor": clip.thermistor.as_posix(),
                    "events": features.events_identity(clip),
                    "n_frames": self.T,
                    "n_output": self.T,
                    "stats": {},
                }
            )
        )
        if frames:
            rows = np.arange(self.T, dtype=np.uint8)[:, None, None, None]
            np.save(files["frames"], np.broadcast_to(rows, (self.T, 1, self.H, self.W)))
        return clip, cl.preprocess, cache

    def test_an_entry_reads_its_gray_frames_by_row(self):
        clip, params, cache = self._cached()
        (entry,) = features.require([clip], params, cache)
        self.assertEqual(entry.frames, features.paths(cache, entry.stem.name)["frames"])
        reader = features.reader(entry)
        self.assertEqual(len(reader), self.T)
        for rows in (slice(2, 5), np.array([1, 4, 9])):
            block = reader.read(rows)
            self.assertEqual(block.dtype, np.uint8)
            self.assertEqual(block.shape[1:], (1, self.H, self.W))
            np.testing.assert_array_equal(block[:, 0, 0, 0], np.arange(self.T)[rows])

    def test_an_entry_without_its_frames_is_not_cached(self):
        clip, params, cache = self._cached(frames=False)
        self.assertIsNone(features.lookup(clip, params, cache))
        with self.assertRaisesRegex(SystemExit, "zephyr preprocess"):
            features.require([clip], params, cache)

    def test_the_key_keeps_retired_fields_at_their_old_values(self):
        clip, params, _ = self._cached()
        key = features.cache_key(clip, params)["preprocess"]
        for name, value in params.KEY_RETIRED.items():
            self.assertEqual(key[name], value)


if __name__ == "__main__":
    unittest.main()
