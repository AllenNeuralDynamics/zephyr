import numpy as np
import pandas as pd
import pytest

from utils import thermistor


def recording(path, rate_hz: float, fs: float = 250.0, seconds: float = 60.0):
    t = np.arange(0.0, seconds, 1 / fs)
    signal = 1000 + 300 * np.sin(2 * np.pi * rate_hz * t)
    pd.DataFrame({"Time": t, "Signal": np.round(signal)}).to_parquet(path)
    return path


def test_a_clean_oscillation_has_its_own_rate_and_regular_breaths(tmp_path) -> None:
    f = thermistor.clip_features(recording(tmp_path / "a.parquet", 3.0))
    assert f["rate_hz"] == pytest.approx(3.0, rel=0.02)
    assert f["ibi_cv"] < 0.05
    assert f["breath_cv"] < 0.05
    assert f["snr_db"] > 20
    assert f["amp"] == pytest.approx(
        2 * 300 * np.cos(np.pi * 0.05), rel=0.01
    )  # p95 - p5
    assert f["raw_levels"] > 100


def test_the_training_clips_are_the_reference_of_the_shift() -> None:
    rows = [
        {"set": "Train", **{k: v for k, v in zip(thermistor.FEATURES, [x] * 6)}}
        for x in (1, 2, 3, 4, 5)
    ]
    rows += [{"set": "OOD (side camera)", **{k: 3.0 for k in thermistor.FEATURES}}]
    shifts = thermistor.shift_from_train(pd.DataFrame(rows))
    train = shifts[shifts["set"] == "Train"]
    assert train["rate_hz"].median() == pytest.approx(0.0)
    assert shifts[shifts["set"] == "OOD (side camera)"]["rate_hz"].iloc[
        0
    ] == pytest.approx(0.0)


def test_a_slower_set_sits_below_the_training_clips() -> None:
    train = [
        {"set": "Train", **{k: x for k in thermistor.FEATURES}}
        for x in (4.0, 4.5, 5.0, 5.5, 6.0)
    ]
    slow = [{"set": "OOD (side camera)", **{k: 2.0 for k in thermistor.FEATURES}}]
    shifts = thermistor.shift_from_train(pd.DataFrame(train + slow))
    assert shifts[shifts["set"] == "OOD (side camera)"]["rate_hz"].iloc[0] < -2
