import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from zephyr.benchmarks import baselines


class ScoreTracesTests(unittest.TestCase):
    def test_perfect_trace_scores_correlation_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            times = np.arange(1800) / 60.0
            np.save(Path(tmp) / "t.npy", times)
            signal = np.sin(2 * np.pi * 3 * times)
            pd.DataFrame({"Time": times, "Signal": signal}).to_parquet(
                Path(tmp) / "thermistor.parquet"
            )
            np.savez(
                Path(tmp) / "events.npz",
                onset_times=times[15::20],
                offset_times=times[5::20],
            )
            entry = SimpleNamespace(
                clip_id="c1",
                recording="r1",
                times=Path(tmp) / "t.npy",
                features=Path(tmp) / "feat-c1.npy",
                thermistor=Path(tmp) / "thermistor.parquet",
                events=Path(tmp) / "events.npz",
            )
            result = baselines.score_traces({"feat-c1": signal}, {"g": [entry]})
        self.assertAlmostEqual(result["summary"]["correlation"], 1.0, places=3)
        self.assertEqual(result["groups"]["g"]["recordings"], ["r1"])
