import numpy as np

from utils import test_sets


def test_best_lag_finds_a_delayed_copy() -> None:
    fs = 60.0
    t = np.arange(0, 60, 1 / fs)
    truth = np.sin(2 * np.pi * 4 * t + 0.3 * np.sin(2 * np.pi * 0.1 * t))
    later = np.roll(truth, 7)  # the signal 7 samples (117 ms) after the truth
    lag, r, r0 = test_sets.best_lag(later, truth, fs)
    assert abs(lag - 7 / fs) < 1e-9
    assert r > 0.99
    assert r0 < r


def test_best_lag_of_an_aligned_signal_is_zero() -> None:
    x = np.random.default_rng(0).normal(size=3000)
    lag, r, r0 = test_sets.best_lag(x, x, 60.0)
    assert lag == 0.0
    assert r == r0
