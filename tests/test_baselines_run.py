import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from zephyr.baselines import run


class ScoreTracesTests(unittest.TestCase):
    def test_perfect_trace_scores_correlation_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            times = np.arange(1800) / 60.0
            np.save(Path(tmp) / "t.npy", times)
            entry = SimpleNamespace(
                clip_id="c1",
                recording="r1",
                times=Path(tmp) / "t.npy",
                features=Path(tmp) / "feat-c1.npy",
            )
            signal = np.sin(2 * np.pi * 3 * times)
            truth = pd.DataFrame({"Time": times, "Signal": signal})
            with patch.object(run, "truth_frame", return_value=truth):
                result = run.score_traces({"feat-c1": signal}, {"g": [entry]})
        self.assertAlmostEqual(result["summary"]["correlation"], 1.0, places=3)
        self.assertEqual(result["groups"]["g"]["recordings"], ["r1"])
