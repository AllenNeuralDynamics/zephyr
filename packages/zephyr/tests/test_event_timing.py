"""Onset-head targets centred on event times; event matching counted in frames."""

import unittest

import numpy as np
import pandas as pd

from zephyr.evaluation import (
    EVENT_TOLERANCE_FRAMES,
    event_f1,
    frame_tolerance_s,
    score_clip,
)
from zephyr.signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from zephyr.targets import onset_heatmap


class OnsetHeatmapTests(unittest.TestCase):
    def test_an_event_between_frames_peaks_symmetrically_about_it(self):
        times = np.arange(120) / 60.0
        event = (40 + 0.5) / 60.0  # halfway between frames 40 and 41
        heatmap = onset_heatmap(times, np.array([event]), sigma_s=0.02)
        self.assertAlmostEqual(heatmap[40], heatmap[41], places=6)
        self.assertLess(heatmap[40], 1.0)
        self.assertGreater(heatmap[40], heatmap[39])
        self.assertAlmostEqual(heatmap[39], heatmap[42], places=6)

    def test_an_event_on_a_frame_peaks_there_at_one(self):
        times = np.arange(120) / 60.0
        heatmap = onset_heatmap(times, np.array([times[50]]), sigma_s=0.02)
        self.assertEqual(int(np.argmax(heatmap)), 50)
        self.assertAlmostEqual(float(heatmap[50]), 1.0, places=6)

    def test_events_outside_the_window_make_no_peak_at_its_edges(self):
        times = 10.0 + np.arange(120) / 60.0
        far = np.array([0.0, 5.0, 30.0])
        self.assertFalse(onset_heatmap(times, far, sigma_s=0.02).any())
        # One just outside the start leaves only its tail, below its peak.
        near = np.array([times[0] - 0.02])
        edge = onset_heatmap(times, near, sigma_s=0.02)
        self.assertAlmostEqual(float(edge[0]), np.exp(-0.5), places=5)


class FrameToleranceTests(unittest.TestCase):
    def _accepted(self, fs: float, offset_frames: float) -> list[int]:
        """Frames whose prediction matches a true event *offset_frames* past frame 10."""
        frames = np.arange(30) / fs
        truth = np.array([(10 + offset_frames) / fs])
        tolerance = frame_tolerance_s(frames)
        return [
            k for k in range(30) if event_f1(truth, frames[k : k + 1], tolerance) == 1.0
        ]

    def test_the_nearest_frame_and_its_neighbours_match_at_any_rate(self):
        for fs in (30.0, 60.0, 90.0):
            for offset, nearest in ((0.0, 10), (0.2, 10), (0.4, 10), (0.7, 11)):
                with self.subTest(fs=fs, offset=offset):
                    self.assertEqual(
                        self._accepted(fs, offset), [nearest - 1, nearest, nearest + 1]
                    )

    def test_score_clip_counts_frames_of_the_prediction(self):
        fs, n = 30.0, 900
        time = np.arange(n) / fs
        wave = np.sin(2 * np.pi * 2.0 * time)
        truth = pd.DataFrame({TIME_COLUMN: time, BREATHING_SIGNAL_COLUMN: wave})
        truth_on = np.arange(0.125, time[-1], 0.5)
        # 1.4 frames late at 30 Hz is 47 ms: inside the rule, outside 25 ms.
        late = truth_on + 1.4 / fs
        score = score_clip(
            truth,
            truth,
            truth_onset_times_s=truth_on,
            predicted_onset_times_s=late,
        )
        self.assertEqual(score.inhale_f1, 1.0)
        self.assertEqual(EVENT_TOLERANCE_FRAMES, 1.5)


if __name__ == "__main__":
    unittest.main()
