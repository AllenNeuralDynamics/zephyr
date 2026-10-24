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


class TaskEventTests(unittest.TestCase):
    def test_breathing_rate_is_that_of_the_breath_each_time_falls_in(self):
        inhales = np.array([0.0, 0.5, 1.0, 1.25])
        rate = analysis.breathing_rate(inhales, np.array([-0.1, 0.2, 0.7, 1.1, 1.3]))
        np.testing.assert_allclose(rate, [np.nan, 2.0, 2.0, 4.0, np.nan])

    def test_event_aligned_rows_are_nan_outside_the_samples(self):
        times = np.arange(0, 10, 0.5)
        rows = analysis.event_aligned(
            times, times * 2, np.array([1.0, 9.0]), np.array([-1.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(rows, [[0.0, 2.0, 4.0], [16.0, 18.0, np.nan]])
        empty = analysis.event_aligned(times, times, np.empty(0), np.zeros(3))
        self.assertEqual(empty.shape, (0, 3))


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

    def test_a_clip_without_errors_keeps_near_miss_bool(self):
        true_on = np.arange(1, 50, 1 / 3)
        p, t = _clip(true_on, true_on)
        frames = np.zeros((len(p["times"]), 1, 4, 4), np.uint8)
        err = analysis.errors(p, t, True, p["times"], frames)
        self.assertTrue(err.empty)
        self.assertEqual(err["near_miss"].dtype, bool)

    def test_each_lag_carries_its_breaths_rate(self):
        true_on = np.arange(1, 50, 1 / 3)
        pred = true_on + 0.01
        p, t = _clip(true_on, pred)
        lags = analysis.clip_tables(p, t, head=True)["lags"]
        np.testing.assert_allclose(lags["lag_s"], 0.01, atol=1e-9)
        np.testing.assert_allclose(lags["rate_hz"].dropna(), 3.0)

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

    def test_rate_agreement_gives_percentage_error_per_group(self):
        w = pd.DataFrame(
            {
                "true_hz": [2.0, 4.0, 5.0, 5.0],
                "pred_hz": [2.2, 4.4, 5.0, 5.0],
                "method": "m",
                "subject": ["a", "a", "b", "b"],
            }
        )
        rows = analysis.rate_agreement(w, by=["method", "subject"])
        self.assertAlmostEqual(rows.loc[("m", "a"), "MAPE (%)"], 10.0)
        self.assertEqual(rows.loc[("m", "b"), "MAPE (%)"], 0.0)

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

    def test_bootstrap_takes_any_column_as_the_unit(self):
        rows = [
            {"set": "A", "unit": u, "method": "Zephyr", "correlation": v}
            for u, v in (("a@1", 0.2), ("a@2", 0.4), ("b@1", 0.9))
        ]
        ci = analysis.bootstrap(pd.DataFrame(rows), "correlation", n=50, unit="unit")
        self.assertEqual(int(ci["units"].iloc[0]), 3)
        self.assertAlmostEqual(ci["mean"].iloc[0], 0.5)

    def test_rig_and_animal_pairs_differ_in_one_of_the_two(self):
        # a on rigs 1 and 2 (0.2 apart), b on rig 1 (0.5 from a@1): the a@2-b@1 pair
        # differs in both, so it is neither kind.
        rows = pd.DataFrame(
            {
                "subject": ["a", "a", "a", "b"],
                "rig": ["1", "1", "2", "1"],
                "inhale": [0.4, 0.6, 0.7, 1.0],
            }
        )
        pairs = analysis.rig_animal_pairs(rows, "inhale").set_index("kind")
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(pairs.loc[analysis.RIG_PAIR, "difference"], 0.2)
        self.assertAlmostEqual(pairs.loc[analysis.ANIMAL_PAIR, "difference"], 0.5)

    def test_rig_animal_bootstrap_resamples_animals(self):
        # Every same-rig pair of animals is 0.3 apart and every animal's two rigs 0.1,
        # so every resample's means, and their difference, are exact.
        rows = pd.DataFrame(
            [
                {"subject": s, "rig": r, "inhale": base + (0.1 if r == "2" else 0.0)}
                for s, base in zip("abc", (0.0, 0.3, 0.6), strict=True)
                for r in ("1", "2")
            ]
        )
        pairs = analysis.rig_animal_pairs(rows, "inhale")
        ci = analysis.rig_animal_bootstrap(pairs, n=200).set_index("kind")
        self.assertAlmostEqual(ci.loc[analysis.RIG_PAIR, "low"], 0.1)
        self.assertAlmostEqual(ci.loc[analysis.RIG_PAIR, "high"], 0.1)
        self.assertEqual(ci.loc[analysis.RIG_PAIR, "pairs"], 3)
        self.assertGreater(ci.loc[analysis.ANIMAL_PAIR, "mean"], 0.3)
        self.assertGreaterEqual(ci.loc["difference", "low"], 0.2 - 1e-9)

    def test_familiarity_tells_animal_and_rig_apart_per_method(self):
        rows = pd.DataFrame(
            {
                "method": ["small", "small", "small", "full"],
                "subject": [1, 1, 2, 2],
                "rig": ["4A", "5B", "5B", "5B"],
            }
        )
        trained = {
            "small": pd.DataFrame({"subject": ["1"], "rig": ["4A"]}),
            "full": pd.DataFrame({"subject": ["1", "2"], "rig": ["4A", "5B"]}),
        }
        seen = analysis.familiarity(rows, trained)
        self.assertEqual(
            seen["animal"].tolist(), ["same rig", "other rig", "new", "same rig"]
        )
        self.assertEqual(seen["rig_trained"].tolist(), [True, False, False, True])

    def test_cross_validation_runs_take_their_fold_from_folds(self):
        from zephyr.benchmarks.config import CrossValidationRuns

        run = ReportedRun(label="Zephyr", run="cv", head=True)
        cv = CrossValidationRuns(folds=["cv_0", "cv_1"], method=[run])
        self.assertEqual(cv.folds, ["cv_0", "cv_1"])
        with self.assertRaises(ValueError):
            CrossValidationRuns(
                folds=["cv_0"], method=[run.model_copy(update={"fold": "x"})]
            )
        with self.assertRaises(ValueError):
            CrossValidationRuns(folds=["cv_0"], method=[run, run])
        other = run.model_copy(update={"label": "Zephyr, 30 Hz", "run": "cv-30"})
        cv = CrossValidationRuns(folds=["cv_0"], method=[run], variant=[other])
        self.assertEqual(cv.variant, [other])
        with self.assertRaises(ValueError):
            CrossValidationRuns(folds=["cv_0"], method=[run], variant=[run])
        with self.assertRaises(ValueError):
            CrossValidationRuns(
                folds=["cv_0"],
                method=[run],
                variant=[other.model_copy(update={"fold": "x"})],
            )

    def test_across_folds_gives_a_t_interval_and_none_for_one_fold(self):
        rows = pd.DataFrame(
            {
                "method": ["a"] * 3 + ["b"],
                "fold": [0, 1, 2, 0],
                "v": [1.0, 2.0, 3.0, 5.0],
            }
        )
        out = analysis.across_folds(rows, "v").set_index("method")
        self.assertAlmostEqual(out.loc["a", "mean"], 2.0)
        self.assertAlmostEqual(out.loc["a", "high"] - 2.0, 4.302653 / np.sqrt(3), 5)
        self.assertEqual(int(out.loc["b", "folds"]), 1)
        self.assertTrue(np.isnan(out.loc["b", "low"]))

    def test_fold_scores_average_units_then_folds_and_pair_by_fold(self):
        rows = pd.DataFrame(
            {
                "set": "A",
                "method": ["Zephyr"] * 4 + ["Other"] * 4,
                "fold": [0, 0, 1, 1] * 2,
                "unit": ["u", "v", "w", "x"] * 2,
                "f1": [0.9, 0.7, 1.0, 0.8, 0.8, 0.6, 0.8, 0.6],
            }
        )
        out = analysis.fold_scores(rows, "f1").set_index("method")
        self.assertAlmostEqual(out.loc["Zephyr", "mean"], 0.85)
        self.assertAlmostEqual(out.loc["Other", "minus ref"], -0.15)
        self.assertEqual(int(out.loc["Other", "folds"]), 2)

    def test_tolerance_curve_counts_hits_within_each_tolerance(self):
        breaths = pd.DataFrame({"method": "m", "detected": [True, True, False]})
        lags = pd.DataFrame({"method": "m", "lag_s": [0.01, -0.04]})
        false = pd.DataFrame({"method": ["m"]})
        out = analysis.tolerance_curve(lags, breaths, false, [25, 50]).set_index(
            "tolerance_ms"
        )
        # 3 true, 3 predicted (2 detected + 1 false): 1 hit at 25 ms, 2 at 50 ms
        self.assertAlmostEqual(out.loc[25.0, "f1"], 2 / 6)
        self.assertAlmostEqual(out.loc[50.0, "f1"], 4 / 6)
        self.assertAlmostEqual(out.loc[50.0, "errors"], 2 / 3)
        # at 25 ms: 2 of the 3 true breaths missed, 2 of the 3 predictions false
        self.assertAlmostEqual(out.loc[25.0, "misses"], 2 / 3)
        self.assertAlmostEqual(out.loc[25.0, "false"], 2 / 3)
        self.assertAlmostEqual(out.loc[50.0, "misses"] + out.loc[50.0, "false"], 2 / 3)

    def test_error_profile_splits_near_misses_from_gross_errors(self):
        err = pd.DataFrame(
            {
                "kind": ["missed (FN)"] * 4,
                "near_miss": [True, True, False, False],
                "rate_hz": [10.0, 3.0, 9.0, 3.0],
                "motion": [0.5, 0.5, 0.95, 0.2],
                "amplitude": [0.5, 0.5, 0.1, 0.9],
            }
        )
        prof = analysis.error_profile(err).set_index("row")
        self.assertAlmostEqual(prof.loc["near miss", "share"], 0.5)
        self.assertEqual(int(prof.loc["near miss", "n"]), 4)
        self.assertAlmostEqual(prof.loc["fast breathing (> 8 Hz)", "share"], 0.5)
        self.assertEqual(int(prof.loc["face motion (top 10%)", "n"]), 2)

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
                self.assertTrue(load(ResultsManifest, path).method)

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


class QualityTests(unittest.TestCase):
    def test_a_measure_that_tracks_the_truth_is_oriented_positive(self):
        from zephyr.benchmarks.analysis import quality_agreement, retention

        truth = np.linspace(0, 1, 20)
        table = pd.DataFrame({"good": truth, "bad": -truth, "f1": truth})
        out = quality_agreement(table, {"good": "higher", "bad": "lower"}, "f1")
        self.assertEqual(list(out["oriented"]), [1.0, 1.0])
        curve = retention(table, "good", "f1", "higher")
        oracle = retention(table, "f1", "f1", "higher")
        self.assertAlmostEqual(oracle["mean"].iloc[0], truth[-2:].mean())  # top 2 of 20
        self.assertGreater(curve["mean"].iloc[0], curve["mean"].iloc[-1])
