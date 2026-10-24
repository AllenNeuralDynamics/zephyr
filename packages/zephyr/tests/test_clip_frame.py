"""A clip's own ``frame``: validation, cache key, preprocessing, ``clips frame``, Boxes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tomlkit
from _clipfiles import touch_clip, write_list
from _synthetic import HAS_FFMPEG, make_clip

from zephyr import features
from zephyr.annotate import initial_state, save_boxes
from zephyr.clips import main as clips_main
from zephyr.clips import scale_box, set_frames
from zephyr.config import ClipList, load
from zephyr.preprocess import preprocess_clips


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def _list(self, n: int = 2) -> tuple[Path, list[Path]]:
        videos = [touch_clip(self.root / "d", g, 1) for g in range(1, n + 1)]
        return write_list(self.root / "l.toml", videos, box=(10, 10, 16, 16)), videos

    def _frame(self, path: Path, index: int, frame, box) -> None:
        doc = tomlkit.parse(path.read_text())
        doc["clip"][index]["frame"] = list(frame)
        doc["clip"][index]["box"] = list(box)
        path.write_text(tomlkit.dumps(doc))


class FrameConfigTests(_Tmp):
    def test_a_box_is_checked_against_its_own_frame(self):
        path, _ = self._list()
        self._frame(path, 1, (200, 160), (150, 120, 16, 16))  # outside 100x80
        clips = load(ClipList, path).resolve()
        self.assertEqual([c.frame for c in clips], [None, (200, 160)])
        self._frame(path, 1, (100, 80), (150, 120, 16, 16))
        with self.assertRaisesRegex(ValueError, "outside the 100x80 frame"):
            load(ClipList, path)

    def test_boxes_must_still_be_one_size_across_frames(self):
        path, _ = self._list()
        self._frame(path, 1, (200, 160), (20, 20, 32, 32))
        with self.assertRaisesRegex(ValueError, "mixed sizes"):
            load(ClipList, path)

    def test_a_thermistor_only_clip_has_no_frame(self):
        path = self.root / "t.toml"
        (self.root / "t.parquet").touch()
        path.write_text(
            '[preprocess]\ntarget_size = [100, 80]\n\n[[clip]]\nthermistor = "t.parquet"'
            "\nframe = [200, 160]\n"
        )
        with self.assertRaisesRegex(ValueError, "no timestamps, box or frame"):
            load(ClipList, path)


class FrameKeyTests(_Tmp):
    def test_the_key_is_a_lists_at_that_frame_and_unchanged_without_one(self):
        path, _ = self._list(1)
        plain = load(ClipList, path)
        clip = plain.resolve()[0]
        key = features.prefix(clip, plain.preprocess)
        same = clip.model_copy(update={"frame": (100, 80)})  # the list's own size
        self.assertEqual(features.prefix(same, plain.preprocess), key)
        own = clip.model_copy(update={"frame": (200, 160)})
        wide = plain.preprocess.model_copy(update={"target_size": (200, 160)})
        self.assertNotEqual(features.prefix(own, plain.preprocess), key)
        self.assertEqual(
            features.prefix(own, plain.preprocess), features.prefix(clip, wide)
        )


class ClipsFrameTests(_Tmp):
    def test_input_scale_carries_the_box_onto_the_frame_and_one_undoes_it(self):
        path, videos = self._list()
        before = path.read_text()
        with self.assertRaisesRegex(ValueError, "mixed sizes"):  # 16 and 32 in use
            set_frames(path, 2, match=f"{videos[1].stem}")
        self.assertEqual(path.read_text(), before)  # refused: nothing written
        doc = tomlkit.parse("# kept\n" + before)
        doc["clip"][1]["selection"] = "exclude"  # as the dataset lists other cameras
        path.write_text(tomlkit.dumps(doc))
        self.assertEqual(set_frames(path, 2, match=f"{videos[1].stem}"), 1)
        doc = tomlkit.parse(path.read_text())
        self.assertNotIn("frame", doc["clip"][0])
        self.assertEqual(list(doc["clip"][1]["frame"]), [200, 160])
        self.assertEqual(list(doc["clip"][1]["box"]), [20, 20, 32, 32])
        self.assertIn("# kept", path.read_text())
        self.assertEqual(set_frames(path, 1), 1)  # every clip: back to the list's
        doc = tomlkit.parse(path.read_text())
        self.assertNotIn("frame", doc["clip"][1])
        self.assertEqual(list(doc["clip"][1]["box"]), [10, 10, 16, 16])

    def test_the_command_scales_every_matching_clip(self):
        path, _ = self._list()
        clips_main(["frame", str(path), "--input-scale", "2"])
        clips = load(ClipList, path).resolve()
        self.assertEqual({c.frame for c in clips}, {(200, 160)})
        self.assertEqual({c.box for c in clips}, {(20, 20, 32, 32)})
        with self.assertRaises(SystemExit):
            clips_main(["frame", str(path), "--input-scale", "0"])

    def test_scale_box_keeps_the_same_part_of_the_image(self):
        self.assertEqual(
            scale_box((156, 70, 48, 48), (360, 270), (720, 540)), (312, 140, 96, 96)
        )


class FrameBoxStateTests(_Tmp):
    def _state(self):
        path, videos = self._list()
        self._frame(path, 1, (200, 160), (20, 20, 16, 16))
        clips = load(ClipList, path).resolve()
        return path, videos, initial_state(clips, (200, 160), (100, 80), 16)

    def test_a_clip_on_its_own_frame_keeps_its_box_and_frame(self):
        path, videos, state = self._state()
        self.assertEqual(state.frames, {videos[1]: (200, 160)})
        self.assertEqual(state.overrides[videos[1]], (20, 20, 16, 16))
        state.set_target(50, 40)  # the list's frame: the other clip's box moves
        self.assertEqual(state.frames[videos[1]], (200, 160))
        self.assertEqual(state.overrides[videos[1]], (20, 20, 16, 16))
        state.select(state.recordings[1].key)
        state.toggle_override()  # cannot drop the box of a clip on its own frame
        self.assertIn(videos[1], state.overrides)
        save_boxes(path, state)
        clips = load(ClipList, path).resolve()
        self.assertEqual(clips[1].frame, (200, 160))
        self.assertEqual(clips[1].box, (20, 20, 16, 16))
        self.assertEqual(clips[0].frame, None)

    def test_input_scale_moves_the_displayed_clip_and_back(self):
        path, _, state = self._state()
        state.select(state.recordings[0].key)
        state.set_input_scale(2)
        self.assertEqual(state.frame, (200, 160))
        self.assertEqual(state.input_scale, 2)
        self.assertEqual(state.box, (28, 28, 16, 16))  # centre (18, 18) -> (36, 36)
        state.select(state.recordings[1].key)
        state.set_input_scale(1)
        self.assertEqual(state.frame, (100, 80))
        save_boxes(path, state)
        clips = load(ClipList, path).resolve()
        self.assertEqual([c.frame for c in clips], [(200, 160), None])
        self.assertEqual(clips[1].box, (6, 6, 16, 16))  # (28, 28) -> (14, 14) centre


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class FramePreprocessTests(_Tmp):
    def test_a_clip_on_its_own_frame_is_cropped_from_that_frame(self):
        video = make_clip(self.root / "data", 1, 1)  # 64x48
        half = write_list(
            self.root / "half.toml", [video], box=(12, 8, 16, 16), target_size=(32, 24)
        )
        full = write_list(
            self.root / "full.toml", [video], box=(24, 16, 16, 16), target_size=(64, 48)
        )
        self._frame(half, 0, (64, 48), (24, 16, 16, 16))
        framed, native = load(ClipList, half), load(ClipList, full)
        (a,) = preprocess_clips(framed.resolve(), framed.preprocess, self.root / "a")
        (b,) = preprocess_clips(native.resolve(), native.preprocess, self.root / "b")
        self.assertEqual(a.stem.name, b.stem.name)  # one entry: the same key
        np.testing.assert_array_equal(np.load(a.frames), np.load(b.frames))
        self.assertEqual(np.load(a.frames).shape[1:], (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
