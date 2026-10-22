"""``zephyr clips check`` reads every file of a list and names, per clip, what
preprocessing or training would refuse; ``scan`` writes a ``[derive]`` for other names
and the adaptive detector as ``[events]``."""

import tempfile
import tomllib
import unittest
from pathlib import Path

import pandas as pd
from _synthetic import HAS_FFMPEG, make_clip, make_dataset

from zephyr.clips import SCAN_EVENTS, check, main, scan
from zephyr.config import ClipList, Derive, load


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class CheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.list = make_dataset(self.root, groups=(1, 2))
        self.data = self.root / "data"

    def tearDown(self):
        self._tmp.cleanup()

    def _problems(self) -> dict[str, list[str]]:
        return {c.group: c.problems for c in check(self.list)}

    def test_a_good_list_passes_with_what_was_measured(self):
        checks = check(self.list)
        self.assertTrue(all(c.ok for c in checks), [c.problems for c in checks])
        facts = checks[0].facts
        self.assertEqual((facts["frames"], facts["timestamps"]), (1200, 1200))
        self.assertAlmostEqual(facts["camera_hz"], 60.0, places=1)
        self.assertAlmostEqual(facts["sensor_hz"], 500.0, places=0)
        self.assertGreater(facts["overlap_s"], 19)
        main(["check", str(self.list)])  # returns: every clip passed

    def test_a_timestamp_row_short_of_the_frames(self):
        stamps = self.data / "video_face_1_part_1.parquet"
        pd.read_parquet(stamps).iloc[:-1].to_parquet(stamps)
        self.assertRegex(" ".join(self._problems()["1"]), "1200 frames .* 1199 rows")
        self.assertEqual(self._problems()["2"], [])
        with self.assertRaises(SystemExit) as stopped:
            main(["check", str(self.list)])
        self.assertEqual(stopped.exception.code, 1)

    def test_a_slow_sensor_and_one_on_another_clock(self):
        make_clip(self.data, 1, 1, sensor_hz=60.0)
        thermistor = self.data / "thermistor_2_part_1.parquet"
        table = pd.read_parquet(thermistor)
        table["Time"] += 10_000.0
        table.to_parquet(thermistor)
        problems = self._problems()
        self.assertRegex(" ".join(problems["1"]), "sampled at 60.0 Hz")
        self.assertRegex(" ".join(problems["2"]), "share no time")

    def test_a_camera_slower_than_select_fs(self):
        self.list.write_text(
            self.list.read_text().replace(
                "[preprocess]", "[preprocess]\nselect_fs = 90"
            )
        )
        self.assertRegex(" ".join(self._problems()["1"]), "below select_fs 90")

    def test_missing_files_columns_and_boxes(self):
        (self.data / "thermistor_1_part_1.parquet").unlink()
        stamps = self.data / "video_face_2_part_1.parquet"
        pd.DataFrame({"t": [0.0, 1.0]}).to_parquet(stamps)
        self.list.write_text(
            self.list.read_text().replace("box = [16, 8, 32, 32]\n", "", 1)
        )
        problems = self._problems()
        self.assertRegex(" ".join(problems["1"]), "thermistor .* does not exist")
        self.assertRegex(" ".join(problems["1"]), "no crop box")
        self.assertRegex(" ".join(problems["2"]), r"no column \['Time'\]")


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class ScanDeriveTests(unittest.TestCase):
    def test_other_names_are_written_as_derive_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_clip(root, 3, 1)
            video.rename(root / "mouse3.mp4")
            (root / "video_face_3_part_1.parquet").rename(root / "mouse3_times.parquet")
            (root / "thermistor_3_part_1.parquet").rename(
                root / "mouse3_breath.parquet"
            )
            rule = Derive(
                pattern="(?P<group>.+)",
                timestamps="{stem}_times.parquet",
                thermistor="{stem}_breath.parquet",
            )
            out = root / "list.toml"
            scan(root, ["*.mp4"], out, derive=rule)
            written = tomllib.loads(out.read_text())
            self.assertEqual(written["derive"]["timestamps"], "{stem}_times.parquet")
            clip = load(ClipList, out).resolve()[0]
            self.assertEqual(
                (clip.group, clip.thermistor.name), ("mouse3", "mouse3_breath.parquet")
            )

    def test_the_default_naming_writes_no_derive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_clip(root, 1, 1)
            out = root / "list.toml"
            main(["scan", str(root), "--glob", "*.mp4", "-o", str(out)])
            self.assertNotIn("derive", tomllib.loads(out.read_text()))

    def test_a_scanned_list_uses_the_adaptive_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_clip(root, 1, 1)
            out = root / "list.toml"
            main(["scan", str(root), "--glob", "*.mp4", "-o", str(out)])
            events = load(ClipList, out).resolve()[0].events
            self.assertEqual(events, SCAN_EVENTS)
            self.assertEqual(events.method, "adaptive_prominence")


if __name__ == "__main__":
    unittest.main()
