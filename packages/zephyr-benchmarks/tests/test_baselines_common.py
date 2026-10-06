import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zephyr.benchmarks import common

FS = 60.0


def breathing(n=1800, rate=3.0, fs=FS):
    """Asymmetric breath: slow rise (exhale), fast fall (inhale)."""
    phase = (np.arange(n) / fs * rate) % 1.0
    return np.where(phase < 0.65, phase / 0.65, (1.0 - phase) / 0.35)


class CommonTests(unittest.TestCase):
    def test_bandpass_keeps_breathing_removes_drift(self):
        t = np.arange(3600) / FS
        x = np.sin(2 * np.pi * 3 * t) + 5 * t / t[-1]
        y = common.bandpass(x)
        self.assertGreater(np.corrcoef(y, np.sin(2 * np.pi * 3 * t))[0, 1], 0.99)

    def test_shift_delays_positive_lag(self):
        x = np.arange(10.0)
        np.testing.assert_array_equal(common.shift(x, 2)[2:], x[:-2])
        np.testing.assert_array_equal(common.shift(x, -2)[:-2], x[2:])
        self.assertEqual(len(common.shift(x, 3)), 10)

    def test_blind_polarity_restores_fast_fall(self):
        x = breathing()
        np.testing.assert_allclose(common.blind_polarity(-x), x)
        np.testing.assert_allclose(common.blind_polarity(x), x)

    def test_fit_sign_lag_recovers_flip_and_delay(self):
        truth = common.bandpass(breathing())
        pred = -common.shift(truth, 5)
        sign, lag, r = common.fit_sign_lag([(pred, truth)], max_lag=10)
        self.assertEqual(sign, -1)
        self.assertEqual(lag, -5)
        self.assertGreater(r, 0.95)

    def test_to_output_grid_interpolates_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = SimpleNamespace(
                frame_times=Path(tmp) / "f.npy", times=Path(tmp) / "t.npy"
            )
            np.save(entry.frame_times, np.array([0.0, 1.0, 2.0]))
            np.save(entry.times, np.array([0.5, 1.5]))
            out = common.to_output_grid(
                entry, np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
            )
            np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 6.0]])
            np.testing.assert_allclose(
                common.to_output_grid(entry, np.array([0.0, 2.0, 4.0])), [1.0, 3.0]
            )
