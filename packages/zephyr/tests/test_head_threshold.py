"""The onset threshold: chosen on validation clips, read from the checkpoint, overridden
by a flag, and scored as ``zephyr evaluate`` scores the head."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pydantic import ValidationError

from zephyr.config import TrainParams
from zephyr.evaluate import (
    HEAD_THRESHOLD,
    checkpoint_head_threshold,
    choose_head_threshold,
    head_event_indices,
    head_inhale_counts,
    head_threshold,
    score_entry,
)
from zephyr.signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

FS = 60.0


def _bumps(times, at, heights, sigma=0.02):
    onset = np.zeros(len(times))
    for t, h in zip(at, heights, strict=True):
        onset = np.maximum(onset, h * np.exp(-0.5 * ((times - t) / sigma) ** 2))
    return onset


class ClipTests(unittest.TestCase):
    """400 inhales at 2 Hz: 390 the head is sure of, 2 it half sees and 8 it misses,
    and 5 spurious peaks it half sees between them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.times = np.arange(0, 201, 1 / FS)
        inhale = 0.5 + 0.5 * np.arange(400)
        heights = np.r_[np.full(390, 0.9), [0.4, 0.4], np.full(8, 0.2)]
        spurious = inhale[:5] + 0.25
        self.onset = _bumps(
            self.times, np.r_[inhale, spurious], np.r_[heights, np.full(5, 0.4)]
        )
        events = root / "events.npz"
        np.savez(events, onset_times=inhale, offset_times=inhale + 0.25)
        thermistor = root / "thermistor.parquet"
        pd.DataFrame(
            {
                TIME_COLUMN: self.times,
                BREATHING_SIGNAL_COLUMN: np.sin(2 * np.pi * 2 * self.times),
            }
        ).to_parquet(thermistor)
        self.entry = SimpleNamespace(events=events, thermistor=thermistor)
        self.clips = [(self.entry, self.times, self.onset)]

    def test_counts_score_the_head_as_evaluate_does(self):
        for threshold in (0.3, 0.5):
            counts = head_inhale_counts(self.entry, self.times, self.onset, threshold)
            at = self.times[self.onset > 0]  # any predicted signal will do
            predicted = pd.DataFrame(
                {TIME_COLUMN: at, BREATHING_SIGNAL_COLUMN: np.zeros(len(at))}
            )
            head = self.times[head_event_indices(self.onset, self.times, threshold)]
            score = score_entry(self.entry, predicted, predicted_onset_times_s=head)
            self.assertAlmostEqual(counts["f1"], score.inhale_f1)
        self.assertEqual(
            head_inhale_counts(self.entry, self.times, self.onset, 0.3)["n_head"], 397
        )

    def test_the_best_f1_wins_when_nothing_ties(self):
        chosen = choose_head_threshold(self.clips, (0.3, 0.5), f1_tie=0.002)
        self.assertEqual(chosen["value"], 0.5)  # 10 misses beat 8 misses + 5 false
        self.assertEqual(chosen["chosen_on"], "validation")
        self.assertAlmostEqual(chosen["val_overcount_per_min"], -10 / (201 / 60), 1)
        self.assertEqual([r["threshold"] for r in chosen["sweep"]], [0.3, 0.5])

    def test_a_tie_goes_to_the_smallest_overcount(self):
        chosen = choose_head_threshold(self.clips, (0.3, 0.5), f1_tie=0.01)
        self.assertEqual(chosen["value"], 0.3)  # -3 per clip, not -10


class PrecedenceTests(unittest.TestCase):
    def test_the_flag_then_the_checkpoint_then_the_default(self):
        own = {"head_threshold": {"value": 0.7, "chosen_on": "validation"}}
        self.assertEqual(head_threshold(own, 0.6), {"value": 0.6, "source": "flag"})
        self.assertEqual(head_threshold(own), {"value": 0.7, "source": "checkpoint"})
        self.assertEqual(
            head_threshold({}), {"value": HEAD_THRESHOLD, "source": "default"}
        )
        self.assertEqual(checkpoint_head_threshold({}), HEAD_THRESHOLD)  # old ones

    def test_a_flag_outside_zero_and_one_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "not in"):
            head_threshold({}, 1.5)


class ConfigTests(unittest.TestCase):
    def test_no_sweep_by_default_and_a_grid_is_sorted_inside_zero_and_one(self):
        self.assertEqual(TrainParams().head_thresholds, ())
        params = TrainParams(head_thresholds=(0.7, 0.2, 0.7))
        self.assertEqual(params.head_thresholds, (0.2, 0.7))
        with self.assertRaises(ValidationError):
            TrainParams(head_thresholds=(0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
