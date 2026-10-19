import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from zephyr.benchmarks import analysis
from zephyr.benchmarks.config import ReportedRun, ResultsManifest
from zephyr.config import load

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FS = 60.0


def _clip(true_on, pred_on, seconds=60.0):
    """A clip whose target and prediction are a 3 Hz sine, with the given inhales."""
    times = np.arange(0, seconds, 1 / FS)
    signal = np.sin(2 * np.pi * 3 * times)
    t = {
        "times": times,
        "signal": signal,
        "inhale": np.asarray(true_on, float),
        "exhale": np.empty(0),
        "excluded": np.empty((0, 2)),
    }
    p = {"times": times, "signal": signal, "onset": np.zeros_like(times)}
    p["head"] = np.asarray(pred_on, float)
    return p, t


class EventTests(unittest.TestCase):
    def test_matches_within_the_scorers_tolerance(self):
        true_on = np.arange(1, 50, 1 / 3)
        pred = true_on + 0.01  # 10 ms late: within 1.5 frames (25 ms)
        p, t = _clip(true_on, pred)
        ev = analysis.clip_events(p, t, head=True)
        self.assertTrue(ev["hit"].all())
        self.assertTrue(ev["used"].all())
        np.testing.assert_allclose(ev["rate"][:-1], 3.0)

    def test_a_breath_placed_40_ms_off_is_one_miss_and_one_false_event_a_near_miss(
        self,
    ):
        true_on = np.arange(1, 50, 1 / 3)
        pred = true_on.copy()
        pred[10] += 0.04
        p, t = _clip(true_on, pred)
        tables = analysis.clip_tables(p, t, head=True)
        self.assertEqual(int((~tables["breaths"]["detected"]).sum()), 1)
        self.assertEqual(len(tables["false_positives"]), 1)
        self.assertEqual(set(tables), set(analysis.TABLES))
        frames = np.zeros((len(p["times"]), 1, 4, 4), np.uint8)
        err = analysis.errors(p, t, True, p["times"], frames)
        self.assertEqual(sorted(err["kind"]), sorted([analysis.MISSED, analysis.FALSE]))
        self.assertTrue(err["near_miss"].all())
        self.assertTrue(err["causes"].str.contains("near miss").all())

    def test_excluded_spans_are_not_scored(self):
        true_on = np.arange(1, 50, 1 / 3)
        p, t = _clip(true_on, [])
        t["excluded"] = np.array([[0.0, 30.0]])
        ev = analysis.clip_events(p, t, head=True)
        self.assertTrue(np.all(ev["true"] > 30.0))

    def test_rate_windows_count_breaths(self):
        true_on = np.arange(0.1, 60, 1 / 3)  # the whole clip, so every window is full
        p, t = _clip(true_on, true_on)
        w = analysis.clip_tables(p, t, head=True)["windows"]
        np.testing.assert_allclose(w["true_hz"], w["pred_hz"])
        self.assertTrue(np.all(np.abs(w["true_hz"] - 3.0) <= 0.1))


class SummaryTests(unittest.TestCase):
    def test_rate_agreement_of_a_perfect_count_is_exact(self):
        w = pd.DataFrame(
            {
                "true_hz": [2.0, 3.0, 5.0],
                "pred_hz": [2.0, 3.0, 5.0],
                "method": "m",
                "set": "A",
            }
        )
        row = analysis.rate_agreement(w).iloc[0]
        self.assertEqual(row["MAE (Hz)"], 0)
        self.assertEqual(row["bias (Hz)"], 0)
        self.assertAlmostEqual(row["r"], 1.0)

    def test_f1_by_rate_bins_misses_and_false_events_by_rate(self):
        breaths = pd.DataFrame(
            {
                "rate_hz": [2.5, 2.5, 9.0, 9.0],
                "detected": [True, True, True, False],
                "method": "m",
                "set": "A",
            }
        )
        false = pd.DataFrame({"rate_hz": [9.0], "method": "m", "set": "A"})
        f1 = analysis.f1_by_rate(breaths, false).set_index("bin")
        self.assertEqual(f1.loc["2-3", "f1"], 1.0)
        self.assertAlmostEqual(f1.loc["8-10", "f1"], 2 / 4)
        self.assertAlmostEqual(f1["share"].sum(), 1.0)

    def test_bootstrap_resamples_animals_and_pairs_the_difference(self):
        rows = [
            {"set": "A", "subject": s, "method": m, "correlation": base + offset}
            for s, base in zip("abcd", (0.5, 0.6, 0.7, 0.8), strict=True)
            for m, offset in (("Zephyr", 0.1), ("Other", 0.0))
        ]
        ci = analysis.bootstrap(pd.DataFrame(rows), "correlation", n=200).set_index(
            "method"
        )
        self.assertAlmostEqual(ci.loc["Zephyr", "mean"], 0.75)
        self.assertLessEqual(ci.loc["Zephyr", "low"], 0.75)
        self.assertGreaterEqual(ci.loc["Zephyr", "high"], 0.75)
        # every animal is 0.1 better, so every resample's difference is exactly 0.1
        self.assertAlmostEqual(ci.loc["Other", "diff low"], 0.1)
        self.assertAlmostEqual(ci.loc["Other", "diff high"], 0.1)

    def test_score_agreement_ranks_clips_across_methods_and_averages_seeds(self):
        scores = [0.2, 0.5, 0.9, 0.4]
        rows = [
            {"set": "A", "clip_id": f"c{i}", "method": m, "seed": seed, "r": value}
            for i, s in enumerate(scores)
            for m, value in (("Zephyr", s), ("Same order", s / 2), ("Reversed", -s))
            for seed in ("17", "42")
        ]
        rows.append({"set": "A", "clip_id": "c0", "method": "Zephyr", "seed": "42"})
        rows[-1]["r"] = 0.2  # a second seed equal to the first leaves the mean alone
        wide = analysis.clip_scores_wide(pd.DataFrame(rows), "r")
        self.assertEqual(wide.shape, (4, 3))
        agree = analysis.score_agreement(pd.DataFrame(rows), "r")
        self.assertAlmostEqual(agree.loc["Zephyr", "Same order"], 1.0)
        self.assertAlmostEqual(agree.loc["Zephyr", "Reversed"], -1.0)

    def test_rate_distribution_shares_sum_to_one_per_group(self):
        shares = analysis.rate_distribution(
            {
                "a": [np.arange(0, 10, 0.22)],
                "b": [np.arange(0, 10, 0.1)],
            }  # 4.5 and 10 Hz
        )
        np.testing.assert_allclose(shares.groupby("clips")["share"].sum(), 1.0)
        self.assertEqual(
            shares.set_index(["clips", "bin"]).loc[("a", "4-5"), "share"], 1.0
        )

    def test_interpolation_r2_is_one_for_a_line_and_low_for_noise(self):
        t = np.arange(200) / FS
        np.testing.assert_allclose(analysis.interpolation_r2(t, 3 * t), 1.0)
        noise = np.random.default_rng(0).normal(size=(200, 4))
        self.assertTrue(np.all(analysis.interpolation_r2(t, noise) < 0))

    def test_band_correlation_of_a_signal_with_itself_is_one(self):
        x = np.random.default_rng(1).normal(size=3600)
        for r in analysis.band_correlations(x, x, FS).values():
            self.assertAlmostEqual(r, 1.0)


class TruthTests(unittest.TestCase):
    def test_truth_reads_the_cache_entry(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            times = np.arange(600) / FS
            np.save(d / "time.npy", times)
            pd.DataFrame({"signal": np.sin(times) * 5 + 2}).to_parquet(
                d / "target.parquet"
            )
            np.savez(d / "events.npz", onset_times=[1.0, 2.0], offset_times=[1.5])
            pd.DataFrame({"Time": times, "Signal": times}).to_parquet(
                d / "therm.parquet"
            )

            class Entry:
                times = d / "time.npy"
                target = d / "target.parquet"
                events = d / "events.npz"
                thermistor = d / "therm.parquet"

            t = analysis.truth(Entry())
        self.assertAlmostEqual(float(t["signal"].mean()), 0.0)
        self.assertAlmostEqual(float(t["signal"].std()), 1.0)
        self.assertEqual(t["excluded"].shape, (0, 2))
        np.testing.assert_array_equal(t["inhale"], [1.0, 2.0])


class ManifestTests(unittest.TestCase):
    def test_every_shipped_manifest_loads(self):
        paths = sorted((CONFIGS / "results").glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                m = load(ResultsManifest, path)
                self.assertTrue(m.method)

    def test_a_variant_is_only_for_a_baseline_and_a_head_only_for_a_network(self):
        with self.assertRaises(ValueError):
            ReportedRun(label="x", run="r", variant="both")
        with self.assertRaises(ValueError):
            ReportedRun(label="x", run="r", kind="facemap")
        with self.assertRaises(ValueError):
            ReportedRun(
                label="x", run="r", kind="pixel", variant="pca/blind", head=True
            )

    def test_labels_are_distinct(self):
        run = {"label": "Zephyr", "run": "r"}
        with self.assertRaises(ValueError):
            ResultsManifest(root="s3://b", method=[run], variant=[run])
