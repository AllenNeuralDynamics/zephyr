import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from zephyr import rates
from zephyr.config import TrainParams
from zephyr.dataset import WindowDataset
from zephyr.features import ClipEntry

FS = 60.0


class RateTests(unittest.TestCase):
    def test_breath_rate_is_the_inverse_interval_to_the_next_onset(self):
        np.testing.assert_allclose(rates.breath_rates([0.0, 0.5, 0.75]), [2.0, 4.0])

    def test_rate_track_holds_each_breath_and_is_nan_outside(self):
        track = rates.rate_track(np.array([-1, 0.1, 0.6, 2.0]), [0.0, 0.5, 0.75])
        self.assertTrue(np.isnan(track[[0, 3]]).all())
        np.testing.assert_allclose(track[[1, 2]], [2.0, 4.0])

    def test_window_without_a_complete_breath_has_rate_zero(self):
        track = np.array([np.nan] * 5 + [3.0] * 5)
        np.testing.assert_allclose(rates.window_rates(track, [0, 5], 5), [0.0, 3.0])

    def test_out_of_range_rates_join_the_end_bins(self):
        index = rates.rate_bin(np.array([0.5, 2.0, 14.9, 40.0]), (2, 4, 15))
        self.assertEqual(index.tolist(), [0, 0, 1, 1])

    def test_full_balance_gives_every_occupied_bin_equal_mass(self):
        r = np.array([3.0] * 9 + [9.0])
        p = rates.balance_weights(r, (2, 6, 15), 1.0)
        self.assertAlmostEqual(p[r == 3.0].sum(), 0.5)
        np.testing.assert_allclose(rates.balance_weights(r, (2, 6, 15), 0.0), 0.1)

    def test_a_nearly_empty_bin_is_not_boosted_past_the_floor(self):
        r = np.array([3.0] * 999 + [9.0])  # 0.1% of windows in the fast bin
        p = rates.balance_weights(r, (2, 6, 15), 1.0)
        boost = p[-1] / p[0]
        self.assertAlmostEqual(boost, (0.999 / rates.MIN_BIN_SHARE), places=6)


def _clip(root: Path, fast_s: float = 540.0, slow_s: float = 60.0) -> ClipEntry:
    """A clip breathing at 9 Hz, then at 2.5 Hz for its last *slow_s* seconds."""
    n = int((fast_s + slow_s) * FS)
    times = np.arange(n) / FS
    onsets = np.concatenate(
        [np.arange(0, fast_s, 1 / 9), np.arange(fast_s, fast_s + slow_s, 1 / 2.5)]
    )
    paths = {k: root / f"{k}" for k in ("features", "frame_times", "times", "base")}
    np.save(paths["features"].with_suffix(".npy"), np.zeros((n, 4, 4, 4), np.uint8))
    np.save(paths["frame_times"].with_suffix(".npy"), times)
    np.save(paths["times"].with_suffix(".npy"), times)
    pd.DataFrame({"time": times, "signal": np.zeros(n)}).to_parquet(root / "t.parquet")
    np.savez(root / "events.npz", onset_times=onsets)
    return ClipEntry(
        clip_id="fake/clip",
        recording="fake#1",
        n_frames=n,
        n_output=n,
        features=paths["features"].with_suffix(".npy"),
        frame_times=paths["frame_times"].with_suffix(".npy"),
        baselines=paths["base"],
        times=paths["times"].with_suffix(".npy"),
        target=root / "t.parquet",
        events=root / "events.npz",
    )


class BalancedDrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.entry = _clip(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _slow_share(self, rate_balance: float) -> float:
        dataset = WindowDataset(
            [[self.entry]],
            window=60,
            mean=np.zeros(4),
            std=np.ones(4),
            select_fs=FS,
            output_fs=FS,
            motion_tau_s=1 / FS,
            length=4000,
            rate_balance=rate_balance,
        )
        starts = np.array([dataset.draw(i)[1] for i in range(4000)])
        return float(np.mean(starts / FS > 541.0))

    def test_unbalanced_draws_follow_the_data(self):
        self.assertLess(self._slow_share(0.0), 0.15)

    def test_full_balance_draws_slow_windows_about_half_the_time(self):
        self.assertAlmostEqual(self._slow_share(1.0), 0.5, delta=0.06)

    def test_gridded_windows_refuse_balancing(self):
        with self.assertRaises(ValueError):
            WindowDataset(
                [[self.entry]],
                window=60,
                mean=np.zeros(4),
                std=np.ones(4),
                select_fs=FS,
                output_fs=FS,
                motion_tau_s=1 / FS,
                stride=60,
                rate_balance=1.0,
            )


class ConfigTests(unittest.TestCase):
    def test_defaults_keep_todays_draw(self):
        params = TrainParams()
        self.assertEqual(params.rate_balance, 0.0)
        self.assertEqual(params.rate_bins_hz[0], 2)
        self.assertEqual(params.rate_bins_hz[-1], 15)

    def test_bins_must_increase(self):
        for bins in ([2.0], [3.0, 2.0], [0.0, 2.0]):
            with self.subTest(bins=bins), self.assertRaises(ValidationError):
                TrainParams(rate_bins_hz=bins)


if __name__ == "__main__":
    unittest.main()
