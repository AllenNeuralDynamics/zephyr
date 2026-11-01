"""The Predict page's quality control, headless: a predictions folder's reports set
beside the checkpoint's reference, split at a true F1 the user sets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zephyr.config import QualityParams
from zephyr.gui.quality import GOOD_F1, QualityState
from zephyr.predict import SETTINGS
from zephyr.quality_control.metrics import Video, find_events
from zephyr.quality_control.report import REFERENCE, quality_report, write_report

FS = 60.0


def _video(seconds: float, rate: float = 4.0) -> Video:
    times = np.arange(0, seconds, 1 / FS)
    signal = np.sin(2 * np.pi * rate * times)
    onset = np.exp(-((((rate * times) % 1.0 - 0.25) / 0.08) ** 2))
    return Video(times, signal, onset, find_events(times, signal, onset))


class QualityStateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folder = self.root / "predictions"
        self.folder.mkdir()
        checkpoint = self.root / "run" / "best.pt"
        checkpoint.parent.mkdir()
        validation = [
            {"peak_prob": 0.99, "head_trace_f1": 0.99, "truth_inhale_f1": f1}
            for f1 in (0.95, 0.95, 0.85, 0.5)
        ]
        (checkpoint.parent / REFERENCE).write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "a" * 64,
                    "onset_calibration": None,
                    "validation": {"window_s": 30.0, "windows": validation},
                }
            )
        )
        self.settings = {"checkpoint": str(checkpoint), "checkpoint_sha256": "a" * 64}
        (self.folder / SETTINGS).write_text(json.dumps(self.settings))
        for stem, seconds in (("one", 90), ("two", 60)):
            report = quality_report(_video(seconds), QualityParams(), None, True, {})
            write_report(report, self.folder / f"{stem}.qc.json")

    def test_the_reference_is_split_at_the_chosen_f1_then_each_video(self):
        state = QualityState(self.folder)
        self.assertEqual(state.good_f1, GOOD_F1)
        rows = state.rows("peak_prob")
        self.assertEqual([r.kind for r in rows], ["good", "poor", "video", "video"])
        self.assertEqual((len(rows[0].values), len(rows[1].values)), (2, 2))
        state.good_f1 = 0.8
        rows = state.rows("peak_prob")
        self.assertEqual((len(rows[0].values), len(rows[1].values)), (3, 1))
        self.assertEqual([r.stem for r in rows[2:]], ["one", "two"])
        self.assertEqual(state.good_range("peak_prob"), (0.99, 0.99))

    def test_a_reference_of_another_checkpoint_is_not_used(self):
        settings = self.settings | {"checkpoint_sha256": "b" * 64}
        (self.folder / SETTINGS).write_text(json.dumps(settings))
        state = QualityState(self.folder)
        self.assertIsNone(state.reference)
        self.assertEqual([r.kind for r in state.rows("peak_prob")], ["video", "video"])

    def test_each_video_is_summarised_and_found_by_time(self):
        state = QualityState(self.folder)
        self.assertEqual(
            state.measures()[:2], ["head_trace_f1", "alternation_break_frac"]
        )
        table = state.table().set_index("stem")
        self.assertAlmostEqual(table.loc["one", "minutes"], 1.5, places=2)
        windows = state.over_time("one", "peak_prob")
        self.assertEqual(len(windows), 3)
        self.assertEqual(
            state.window_at("one", 45.0), tuple(windows.iloc[1][["start", "end"]])
        )
        self.assertIsNone(state.window_at("one", 500.0))

    def test_a_folder_without_reports_is_empty(self):
        self.assertTrue(QualityState(self.root).empty)


if __name__ == "__main__":
    unittest.main()
