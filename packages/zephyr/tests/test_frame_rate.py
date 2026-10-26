"""A clip's ``frame_rate`` in place of timestamps: the list, cache key, scan and check."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tomlkit
from _clipfiles import write_list
from _synthetic import HAS_FFMPEG, make_clip

from zephyr import features
from zephyr.clips import check
from zephyr.clips import main as clips_main
from zephyr.config import ClipList, load
from zephyr.preprocess import frame_times, preprocess_clips


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def _rated(self, path: Path, rate: float = 60.0, **extra) -> Path:
        doc = tomlkit.parse(path.read_text())
        for entry in doc["clip"]:
            entry["frame_rate"] = rate
            entry["thermistor"] = False
            entry.update(extra)
        path.write_text(tomlkit.dumps(doc))
        return path


class FrameRateConfigTests(_Tmp):
    def _video(self) -> Path:
        video = self.root / "d" / "session.mp4"
        video.parent.mkdir()
        video.touch()  # no timestamps beside it
        return video

    def test_a_clip_on_its_frame_rate_needs_no_timestamps(self):
        path = self._rated(write_list(self.root / "l.toml", [self._video()]))
        with self.assertRaises(ValueError):  # the default pattern names no group
            load(ClipList, path)
        doc = tomlkit.parse(path.read_text())
        doc["clip"][0]["group"] = "s"
        path.write_text(tomlkit.dumps(doc))
        (clip,) = load(ClipList, path).resolve()
        self.assertEqual((clip.timestamps, clip.frame_rate), (None, 60.0))

    def test_timestamps_and_a_frame_rate_are_refused_together(self):
        path = write_list(self.root / "l.toml", [self._video()])
        self._rated(path, timestamps="times.parquet", group="s")
        with self.assertRaisesRegex(ValueError, "timestamps or frame_rate, not both"):
            load(ClipList, path, check_files=False)

    def test_the_cache_key_names_the_rate_instead_of_a_file(self):
        path = self._rated(write_list(self.root / "l.toml", [self._video()]), group="s")
        clip_list = load(ClipList, path)
        (clip,) = clip_list.resolve()
        key = features.cache_key(clip, clip_list.preprocess)
        self.assertEqual(key["frame_rate"], 60.0)
        self.assertNotIn("timestamps", key)
        other = clip.model_copy(update={"frame_rate": 120.0})
        self.assertNotEqual(
            features.prefix(other, clip_list.preprocess),
            features.prefix(clip, clip_list.preprocess),
        )


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class FrameRateVideoTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.video = make_clip(self.root / "data", 1, 1, seconds=2.0, thermistor=False)
        self.video.with_suffix(".parquet").unlink()  # the video alone

    def test_scan_writes_the_rate_and_check_counts_the_frames(self):
        path = self.root / "videos.toml"
        clips_main(
            ["scan", str(self.root / "data"), "--glob", "*.mp4", "-o", str(path)]
            + ["--pattern", "(?P<group>.+)", "--thermistor", "{stem}.t.parquet"]
            + ["--unlabelled", "--frame-rate", "60"]
        )
        (clip,) = load(ClipList, path).resolve()
        self.assertEqual((clip.frame_rate, clip.timestamps), (60.0, None))
        (found,) = check(path)
        self.assertEqual(found.facts["frames"], 120)
        self.assertEqual(found.facts["camera_hz"], 60.0)
        self.assertEqual(
            found.problems, ["no crop box: place one in `zephyr gui` (Boxes)"]
        )
        with self.assertRaises(SystemExit):
            clips_main(
                ["scan", str(self.root / "data"), "--glob", "*.mp4", "-o", str(path)]
                + ["--frame-rate", "60", "--timestamps", "{stem}.times.parquet"]
            )

    def test_preprocessing_picks_frames_on_the_rate(self):
        path = write_list(
            self.root / "l.toml",
            [self.video],
            box=(16, 8, 32, 32),
            target_size=(64, 48),
        )
        self._rated(path, rate=120.0, group="1")  # as if it ran at 120 Hz
        clip_list = load(ClipList, path)
        (clip,) = clip_list.resolve()
        np.testing.assert_allclose(frame_times(clip), np.arange(120) / 120.0)
        (entry,) = preprocess_clips([clip], clip_list.preprocess, self.root / "cache")
        self.assertEqual(entry.n_frames, 60)  # every other frame, at 60 Hz
        np.testing.assert_allclose(np.load(entry.frame_times), np.arange(60) / 60.0)


if __name__ == "__main__":
    unittest.main()
