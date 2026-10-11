import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from _clipfiles import write_list

from zephyr.annotate_events import TuneState
from zephyr.config import ClipList, load
from zephyr.session_selection import Settings, apply, assess, runs, snr_threshold

FS = 250.0
SETTINGS = Settings(window_s=2.0, min_span_s=6.0, min_amplitude=0.5)


def make_clip(root: Path, group: int, signal) -> Path:
    folder = root / "clips"
    folder.mkdir(exist_ok=True)
    t = np.arange(0, 40.0, 1 / FS)
    pd.DataFrame({"Time": t, "Signal": signal(t)}).to_parquet(
        folder / f"thermistor_{group}_part_1.parquet"
    )
    video = folder / f"video_face_{group}_part_1.mp4"
    video.touch()
    (folder / f"video_face_{group}_part_1.parquet").touch()
    return video


def breathing(t, seed):
    rng = np.random.default_rng(seed)
    depth = 1 + 0.15 * np.sin(2 * np.pi * 0.13 * t + seed)  # breaths vary in depth
    return depth * np.sin(2 * np.pi * 5.0 * t) + 0.01 * rng.standard_normal(len(t))


def noise(t, seed):
    return np.random.default_rng(seed).standard_normal(len(t))


def broken(t, seed):
    """Breathing for the first 20 s, sensor noise after."""
    return np.where(t < 20, breathing(t, seed), noise(t, seed))


def spiky(t, seed):
    """Breathing with one large artifact at 25 s."""
    return breathing(t, seed) - 8 * np.exp(-(((t - 25.0) / 0.05) ** 2))


class SessionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        signals = [breathing] * 4 + [noise] * 3 + [broken, spiky]
        videos = [
            make_clip(root, g, lambda t, f=f, g=g: f(t, g))
            for g, f in enumerate(signals, start=1)
        ]
        self.path = write_list(root / "list.toml", videos)
        self.state = TuneState(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dead_sensors_are_excluded_and_bad_stretches_become_spans(self):
        table = assess(self.state, SETTINGS)
        self.assertIsNotNone(table.attrs["snr_threshold"])
        self.assertEqual(list(table["suggested"][:4]), ["include"] * 4)
        self.assertEqual(list(table["suggested"][4:7]), ["exclude"] * 3)
        self.assertTrue(table["dead_sensor"][4:7].all())
        last = table.iloc[7]
        self.assertEqual(len(last["included"]), 1)
        start, end = last["included"][0]
        self.assertLess(start, 1.0)
        self.assertAlmostEqual(end, 20.0, delta=2.0)
        self.assertTrue(any(s >= 18.0 for s, _ in last["excluded"]))

    def test_a_good_run_shorter_than_the_minimum_is_not_included(self):
        table = assess(
            self.state, Settings(window_s=2.0, min_span_s=25.0, min_amplitude=0.5)
        )
        self.assertEqual(table.iloc[7]["included"], [])
        self.assertEqual(table.iloc[7]["suggested"], "exclude")
        self.assertEqual(table.iloc[0]["suggested"], "include")  # 40 s of good data

    def test_a_small_amplitude_fails_windows_and_the_clip(self):
        table = assess(
            self.state, Settings(window_s=2.0, min_span_s=6.0, min_amplitude=5)
        )
        self.assertTrue((table["suggested"][:4] == "exclude").all())
        self.assertIn("amplitude", table.iloc[0]["reason"])
        self.assertEqual(table.iloc[0]["included"], [])

    def test_apply_skips_reviewed_clips_and_saves(self):
        self.state.set_selection([5], "include")
        self.state.set_reviewed([5])
        table = assess(self.state, SETTINGS)
        changed = apply(self.state, table)
        self.assertEqual(self.state.entries[5].selection, "include")  # untouched
        self.assertEqual(self.state.entries[4].selection, "exclude")
        self.assertGreater(changed, 0)
        self.state.save()
        text = self.path.read_text()
        self.assertEqual(text.count("reviewed = true"), 1)
        self.assertIn("included = ", text)
        clip_list = load(ClipList, self.path)
        self.assertTrue(clip_list.clip[5].reviewed)
        again = TuneState(self.path)
        self.assertTrue(again.entries[5].reviewed)
        self.assertEqual(len(again.spans(7, kind="include")), 1)

    def test_an_artifact_event_cuts_its_surroundings_out_of_the_include_spans(self):
        table = assess(self.state, SETTINGS)
        spiky_row = table.iloc[8]
        self.assertGreaterEqual(spiky_row["artifacts"], 1)
        self.assertTrue(any(s <= 25 <= e for s, e in spiky_row["excluded"]))
        self.assertFalse(any(s < 25 < e for s, e in spiky_row["included"]))
        self.assertTrue(all(e <= 20.1 or s >= 29.9 for s, e in spiky_row["included"]))
        self.assertEqual(table.iloc[0]["artifacts"], 0)

    def test_snr_threshold_needs_two_separated_modes(self):
        self.assertIsNone(snr_threshold(np.array([30.0, 31, 32, 33, 34]), 3.0))
        cut = snr_threshold(np.array([9.0, 10, 11, 30, 31, 33]), 3.0)
        self.assertTrue(11 < cut < 30)

    def test_one_absurd_snr_does_not_move_the_gate(self):
        snr = np.array([9.0, 10, 11, 12, 30, 31, 33, 34, 238])
        capped = np.minimum(snr, Settings().snr_ceiling_db)
        cut = snr_threshold(capped, 3.0)
        self.assertTrue(12 < cut < 30)

    def test_runs(self):
        good = np.array([True, True, False, True])
        starts = np.arange(4.0)
        self.assertEqual(runs(good, starts, starts + 1, True), [(0.0, 2.0), (3.0, 4.0)])
        self.assertEqual(runs(good, starts, starts + 1, False), [(2.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
