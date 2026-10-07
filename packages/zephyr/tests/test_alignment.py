"""Video and thermistor need not start together: scoring and targets go by time."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from zephyr.evaluation import score_clip
from zephyr.signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from zephyr.targets import load_target

THERMISTOR_FS = 250.0
OUTPUT_FS = 60.0
LEAD_S = 0.9
"""How long the video runs before the thermistor starts, as on the side rig."""


def breathing(t: np.ndarray) -> np.ndarray:
    """A sharp-peaked, slightly irregular 3 Hz trace: easy to misalign visibly."""
    phase = 2 * np.pi * (3.0 * t + 0.1 * np.sin(2 * np.pi * 0.2 * t))
    return np.sin(phase) + 0.5 * np.sin(2 * phase + 0.3)


def thermistor(duration_s: float = 60.0) -> pd.DataFrame:
    t = np.arange(0.0, duration_s, 1 / THERMISTOR_FS)
    return pd.DataFrame({TIME_COLUMN: t, BREATHING_SIGNAL_COLUMN: 100 * breathing(t)})


def prediction(start_s: float, duration_s: float = 60.0) -> pd.DataFrame:
    t = np.arange(start_s, duration_s, 1 / OUTPUT_FS)
    return pd.DataFrame({TIME_COLUMN: t, BREATHING_SIGNAL_COLUMN: breathing(t)})


class ScoreAlignmentTests(unittest.TestCase):
    def test_a_perfect_prediction_scores_perfectly_whenever_the_video_starts(self):
        truth = thermistor()
        aligned = score_clip(truth, prediction(0.0))
        early = score_clip(truth, prediction(-LEAD_S))
        self.assertGreater(aligned.correlation, 0.999)
        self.assertAlmostEqual(early.correlation, aligned.correlation, places=3)
        self.assertAlmostEqual(early.inhale_f1, aligned.inhale_f1, places=2)

    def test_a_shifted_prediction_is_not_rescued(self):
        shifted = prediction(-LEAD_S)
        shifted[TIME_COLUMN] += 0.1  # the trace now lags the truth by 100 ms
        self.assertLess(score_clip(thermistor(), shifted).correlation, 0.9)


class TargetAlignmentTests(unittest.TestCase):
    def test_no_extrapolation_before_the_thermistor_starts(self):
        times = np.arange(-LEAD_S, 30.0, 1 / OUTPUT_FS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermistor.parquet"
            thermistor(30.0).to_parquet(path)
            target = load_target(path, times)
        before = times < 0
        covered = target.signal[~before]
        # The stretch without a thermistor holds one value inside the covered range.
        self.assertEqual(len(np.unique(target.signal[before])), 1)
        self.assertLessEqual(np.abs(target.signal[before]).max(), np.abs(covered).max())
        # z-scored over the covered span alone.
        self.assertAlmostEqual(covered.mean(), 0.0, places=6)
        self.assertAlmostEqual(covered.std(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
