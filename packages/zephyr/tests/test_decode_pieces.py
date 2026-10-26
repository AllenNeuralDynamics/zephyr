"""Decoding a video in pieces from its keyframes gives its frames as one pass does,
and the progress line that counts them."""

import io
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _clipfiles import write_list
from _synthetic import HAS_FFMPEG, write_video

from zephyr.config import ClipList, load
from zephyr.preprocess import pieces_of, preprocess_clip
from zephyr.progress import Progress
from zephyr.video import keyframes

GOP = 15


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class PiecesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (301, 48, 64), dtype=np.uint8)  # every one new
        self.video = self.root / "data" / "session.mp4"
        self.video.parent.mkdir()
        # Lossy, B-frames and a keyframe every GOP frames: decoding from one only
        # gives the same frames if every reference it needs follows it
        codec = ("-crf", "30", "-bf", "3", "-g", str(GOP), "-keyint_min", str(GOP))
        write_video(self.video, frames, codec=codec)
        path = write_list(
            self.root / "l.toml", [self.video], box=(8, 4, 32, 32), target_size=(64, 48)
        )
        path.write_text(
            path.read_text().replace(
                "box =", 'frame_rate = 120.0\nthermistor = false\ngroup = "s"\nbox ='
            )
        )
        self.list = load(ClipList, path)

    def test_the_keyframes_are_found_in_presentation_order(self):
        n, index, seek = keyframes(self.video)
        self.assertEqual(n, 301)
        self.assertEqual(index[0], 0)
        self.assertGreater(len(index), 301 // GOP)  # x264 may add more
        self.assertTrue(np.all(np.diff(index) <= GOP))
        np.testing.assert_allclose(seek, (index - 0.5) / 60, atol=1e-5)  # ms times

    def test_pieces_start_at_keyframes_and_cover_the_video_once(self):
        cut = keyframes(self.video)
        pieces = pieces_of(301, cut, 4)
        self.assertEqual(len(pieces), 4)
        self.assertEqual(pieces[0][:2], (0, pieces[1][0]))
        self.assertEqual(pieces[-1][1], 301)
        for (_, stop, _), (first, _, seek) in itertools.pairwise(pieces):
            self.assertEqual(stop, first)
            self.assertIn(first, cut[1])
            self.assertIsNotNone(seek)
        self.assertEqual(pieces_of(301, None, 4), [(0, 301, None)])
        self.assertEqual(pieces_of(300, cut, 4), [(0, 300, None)])  # count disagrees

    def test_pieces_give_the_frames_of_one_pass(self):
        (clip,) = self.list.resolve()
        params = self.list.preprocess
        one = preprocess_clip(clip, params, self.root / "one")
        shown = io.StringIO()
        progress = Progress("decoding", 1, "frames", stream=shown)
        many = preprocess_clip(
            clip, params, self.root / "many", pieces=5, progress=progress
        )
        np.testing.assert_array_equal(np.load(many.frames), np.load(one.frames))
        np.testing.assert_array_equal(
            np.load(many.frame_times), np.load(one.frame_times)
        )
        self.assertEqual(progress.done, 301)
        lines = shown.getvalue().splitlines()  # not a terminal: at most one a tenth
        self.assertLessEqual(len(lines), 10)
        self.assertIn("decoding: 301/301 frames 100%", lines[-1])


if __name__ == "__main__":
    unittest.main()
