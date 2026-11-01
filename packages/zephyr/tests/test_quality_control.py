import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from _clipfiles import write_list

from zephyr import features
from zephyr.config import (
    ClipList,
    QualityParams,
    TrainParams,
    load,
)
from zephyr.evaluate import load_checkpoint
from zephyr.model import BreathingNet
from zephyr.quality_control import configure, load_reports, quality_report
from zephyr.quality_control.metrics import (
    MEASURES,
    Video,
    find_events,
    measure,
    spans,
)
from zephyr.quality_control.reference import (
    Reference,
    fit_calibration,
    percentiles,
)
from zephyr.quality_control.report import (
    _spread_out,
    load_reference,
    measure_reference,
    write_report,
)

FS = 60.0
CEILING = 15.0


def _breathing(seconds=40.0, rate=4.0, noise=0.0, seed=0):
    """A clean (or noisy) trace at *rate*, and an onset head peaking on its inhales."""
    times = np.arange(0, seconds, 1 / FS)
    rng = np.random.default_rng(seed)
    signal = np.sin(2 * np.pi * rate * times) + noise * rng.standard_normal(len(times))
    phase = (rate * times) % 1.0
    onset = np.exp(-(((phase - 0.25) / 0.08) ** 2))  # sin peaks at a quarter cycle
    return times, signal, onset


def _video(times, signal, onset):
    return Video(
        times=times,
        signal=signal,
        onset=onset,
        events=find_events(times, signal, onset),
    )


def _measure(video, span=None, reference=None):
    span = span or (video.times[0], np.nextafter(video.times[-1], np.inf))
    return measure(
        video, span, ceiling_hz=CEILING, ambiguous_floor=0.2, reference=reference
    )


def _reference(**overrides) -> Reference:
    fields = {"checkpoint_sha256": "x", "onset_calibration": None, "record": {}}
    return Reference(**(fields | overrides))


class MeasureTests(unittest.TestCase):
    def test_a_clean_rhythm_is_plausible(self):
        m = _measure(_video(*_breathing()))
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)
        self.assertEqual(m["too_fast_frac"], 0.0)
        self.assertEqual(m["alternation_break_frac"], 0.0)
        self.assertGreater(m["head_trace_f1"], 0.95)
        self.assertGreater(m["peak_prob"], 0.9)
        self.assertLess(m["onset_background"], 0.05)

    def test_noise_is_less_plausible_than_breathing(self):
        clean = _measure(_video(*_breathing()))
        times, _, onset = _breathing()
        noise = np.random.default_rng(1).standard_normal(len(times))
        noisy = _measure(_video(times, noise, onset))
        self.assertLess(noisy["head_trace_f1"], clean["head_trace_f1"])

    def test_without_a_head_its_measures_are_nan(self):
        times, signal, _ = _breathing()
        m = _measure(_video(times, signal, None))
        for name in ("head_trace_f1", "peak_prob"):
            self.assertTrue(np.isnan(m[name]), name)
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)  # DSP inhales instead

    def test_peaks_are_ambiguous_below_the_checkpoints_threshold(self):
        times, signal, onset = _breathing(rate=2.0)
        half = 0.6 * np.exp(-((((2.0 * times) % 1.0 - 0.75) / 0.04) ** 2))
        onset = np.maximum(onset, half)
        for threshold, inhales, ambiguous in ((0.5, 160, 0.0), (0.8, 80, 0.5)):
            video = Video(
                times=times,
                signal=signal,
                onset=onset,
                events=find_events(times, signal, onset, threshold),
            )
            m = _measure(video)
            self.assertEqual(m["n_inhales"], inhales)
            self.assertAlmostEqual(m["ambiguous_peak_frac"], ambiguous, places=2)

    def test_expected_precision_needs_a_calibration(self):
        times, signal, onset = _breathing()
        self.assertTrue(
            np.isnan(_measure(_video(times, signal, onset))["expected_precision"])
        )
        fit = {"slope": 1.0, "intercept": 0.0}
        m = _measure(
            _video(times, signal, onset), reference=_reference(onset_calibration=fit)
        )
        self.assertAlmostEqual(m["expected_precision"], m["peak_prob"], delta=0.05)

    def test_only_rates_above_the_ceiling_are_judged(self):
        for rate in (2.0, 4.0, 12.0):  # slow, usual and sniffing are all fine
            times, signal, _ = _breathing(rate=rate, seconds=20)
            m = _measure(_video(times, signal, None))
            self.assertEqual(m["too_fast_frac"], 0.0, rate)
        times, signal, _ = _breathing(rate=20.0, seconds=20)
        m = _measure(_video(times, signal, None))
        self.assertGreater(m["too_fast_frac"], 0.9)

    def test_a_stretch_the_network_flattens_counts_as_flat_breaths(self):
        times, signal, onset = _breathing(seconds=40)
        flat = signal.copy()
        flat[times >= 30] *= 0.05
        m = _measure(_video(times, flat, onset), span=(30.0, 40.0))
        self.assertGreater(m["flat_breath_frac"], 0.9)
        self.assertEqual(_measure(_video(times, signal, onset))["flat_breath_frac"], 0)

    def test_windows_cover_the_clip_once(self):
        times = np.arange(0, 95, 1 / FS)
        windows = spans(times, 30.0)
        self.assertEqual(len(windows), 3)
        inside = sum(((times >= a) & (times < b)).sum() for a, b in windows)
        self.assertEqual(inside, len(times))

    def test_every_judging_measure_says_which_way_is_better(self):
        times, signal, onset = _breathing()
        fit = {"slope": 1.0, "intercept": 0.0}
        video = _video(times, signal, onset)
        produced = set(_measure(video, reference=_reference(onset_calibration=fit)))
        self.assertEqual(produced, set(MEASURES))
        for family, _, better in MEASURES.values():
            self.assertIn(family, ("rhythm", "head"))
            self.assertIn(better, ("higher", "lower", None))


class PercentileTests(unittest.TestCase):
    def test_a_value_is_placed_among_the_validation_windows(self):
        windows = [{"peak_prob": v} for v in (0.1, 0.2, 0.3, 0.4)]
        self.assertEqual(
            percentiles({"peak_prob": 0.25}, windows, ["peak_prob"]),
            {"peak_prob": 50.0},
        )
        self.assertEqual(
            percentiles({"peak_prob": 0.2}, windows, ["peak_prob"]), {"peak_prob": 37.5}
        )
        self.assertEqual(
            percentiles({"peak_prob": 9.0}, windows, ["peak_prob"]),
            {"peak_prob": 100.0},
        )

    def test_a_missing_value_or_distribution_has_no_percentile(self):
        windows = [{"peak_prob": 0.5, "too_fast_frac": None}]
        out = percentiles(
            {"peak_prob": None, "too_fast_frac": 0.0},
            windows,
            ["peak_prob", "too_fast_frac"],
        )
        self.assertEqual(out, {"peak_prob": None, "too_fast_frac": None})


class ReferenceTests(unittest.TestCase):
    def test_calibration_recovers_how_often_a_peak_is_real(self):
        rng = np.random.default_rng(0)
        peaks = rng.uniform(0.5, 1.0, 4000)
        real = (rng.uniform(size=4000) < peaks**4).astype(float)  # over-confident head
        fit = fit_calibration(peaks, real)
        ref = _reference(onset_calibration=fit)
        np.testing.assert_allclose(ref.calibrate(np.array([0.9])), 0.9**4, atol=0.05)
        self.assertLess(fit["ece"], fit["ece_raw"])

    def test_too_few_events_give_no_calibration(self):
        self.assertIsNone(fit_calibration(np.full(5, 0.9), np.ones(5)))

    def test_clips_are_spread_over_the_list(self):
        self.assertEqual(_spread_out(list(range(10)), 3), [0, 4, 9])
        self.assertEqual(_spread_out(list(range(2)), 3), [0, 1])
        self.assertEqual(_spread_out(list(range(4)), None), [0, 1, 2, 3])


T_FRAMES, SIZE = 600, 16
NET = {"widths": (4, 4, 4, 4), "embed": 8, "tcn_channels": 8, "dilations": (1, 2)}


class ReportTests(unittest.TestCase):
    """A report from a prediction alone."""

    def _report(self, reference=None):
        video = _video(*_breathing(seconds=90))
        return quality_report(
            video, QualityParams(window_s=30), reference, True, {"checkpoint": "c"}
        )

    def test_a_report_holds_scores_and_no_verdict(self):
        data = self._report()
        json.dumps(data, allow_nan=False)  # NaN would not parse in strict readers
        self.assertEqual(data["inhale_events"], "onset head")
        self.assertEqual(set(data["summary"]), set(MEASURES))
        self.assertEqual(len(data["windows"]), 3)
        self.assertEqual(data["params"]["window_s"], 30.0)

    def test_a_reference_places_each_measure_among_its_validation_windows(self):
        windows = [{"peak_prob": p, "head_trace_f1": 0.95} for p in (0.0, 1.0)]
        reference = _reference(validation=windows)
        data = self._report(reference)
        for span in (data["summary"], *data["windows"]):
            self.assertEqual(set(span["percentiles"]), set(MEASURES))
            self.assertEqual(span["percentiles"]["peak_prob"], 50.0)

    def test_reports_read_back_as_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = _reference(validation=[{"peak_prob": 0.5}])
            write_report(self._report(reference), Path(tmp) / "clip.qc.json")
            videos, windows = load_reports(tmp)
        self.assertEqual(list(videos["stem"]), ["clip"])
        self.assertEqual(len(windows), 3)
        self.assertIn("pct_peak_prob", windows.columns)


class ReferenceLoadingTests(unittest.TestCase):
    def _write(self, tmp, **fields) -> Path:
        path = Path(tmp) / "qc-reference.json"
        path.write_text(json.dumps({"checkpoint_sha256": "a" * 64} | fields))
        return path

    def test_a_reference_of_another_checkpoint_or_threshold_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, onset_calibration=None, head_threshold=0.7)
            self.assertEqual(
                load_reference(path, "a" * 64, 0.7).checkpoint_sha256, "a" * 64
            )
            with self.assertRaisesRegex(SystemExit, "another checkpoint"):
                load_reference(path, "b" * 64, 0.7)
            with self.assertRaisesRegex(SystemExit, "onset threshold 0.7"):
                load_reference(path, "a" * 64, 0.5)


class CalibrationTests(unittest.TestCase):
    """A tiny checkpoint and one unlabelled clip in a real cache entry."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        data = self.root / "data"
        data.mkdir()
        video = data / "video_face_1_part_1.mp4"
        clips = write_list(
            self.root / "clips.toml",
            [video],
            box=(0, 0, SIZE, SIZE),
            target_size=(SIZE, SIZE),
        )
        clips.write_text(
            clips.read_text().replace("box =", "thermistor = false\nbox =")
        )
        self.clip_list = load(ClipList, clips, check_files=False)
        (clip,) = self.clip_list.resolve()
        cache = self.root / "cache"
        cache.mkdir()
        name = features.prefix(clip, self.clip_list.preprocess)
        files = features.paths(cache, name)
        rng = np.random.default_rng(0)
        np.save(
            files["frames"],
            rng.integers(0, 255, (T_FRAMES, 1, SIZE, SIZE), dtype=np.uint8),
        )
        times = 1000 + np.arange(T_FRAMES) / FS
        np.save(files["frame_times"], times)
        np.save(files["times"], times)
        files["sidecar"].write_text(
            json.dumps(
                {
                    "key": features.cache_key(clip, self.clip_list.preprocess),
                    "thermistor": None,
                    "n_frames": T_FRAMES,
                    "n_output": T_FRAMES,
                }
            )
        )
        self.checkpoint = self._save("best.pt", w_onset=0.5)

    def _save(self, name: str, w_onset: float, **extra) -> Path:
        torch.manual_seed(0)
        model = BreathingNet(**NET)
        path = self.root / name
        torch.save(
            extra
            | {
                "model": model.state_dict(),
                "arch_kwargs": NET,
                "mean": 128.0,
                "std": 64.0,
                "params": {"w_onset": w_onset},
                "train_videos": [],
                "epoch": 1,
            },
            path,
        )
        load_checkpoint(path, torch.device("cpu"))  # the file loads
        return path

    def _calibrate(self, checkpoint: Path):
        return measure_reference(
            checkpoint,
            self.clip_list.resolve(),
            self.clip_list.preprocess,
            self.root / "cache",
            self.root / "ref.json",
            None,
            TrainParams(infer_window=256),
            "cpu",
            "off",
        )

    def test_calibration_clips_need_a_thermistor(self):
        with self.assertRaisesRegex(SystemExit, "need a thermistor"):
            self._calibrate(self.checkpoint)

    def test_a_network_without_a_head_has_nothing_to_calibrate(self):
        with self.assertRaisesRegex(SystemExit, "no onset head"):
            self._calibrate(self._save("headless.pt", w_onset=0.0))


class CommandLineTests(unittest.TestCase):
    def test_reports_come_from_predict_not_a_command_of_their_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = write_list(
                Path(tmp) / "clips.toml",
                [Path(tmp) / "video_face_1_part_1.mp4"],
                box=(0, 0, SIZE, SIZE),
                target_size=(SIZE, SIZE),
            )
            argv = ["run", "--checkpoint=x.pt", f"--clips={clips}", "--cache=c"]
            with self.assertRaises(SystemExit):
                configure([*argv, "--out=o"])

    def test_groups_are_kept_per_list_and_unknown_ones_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            box = {"box": (0, 0, SIZE, SIZE), "target_size": (SIZE, SIZE)}
            one = write_list(
                root / "one.toml", [root / "video_face_1_part_1.mp4"], **box
            )
            two = write_list(
                root / "two.toml", [root / "video_face_2_part_1.mp4"], **box
            )
            argv = ["reference", "--checkpoint=x.pt", "--clips", str(one), str(two)]
            config = configure([*argv, "--cache=c", "--out=r.json", "--groups", "2"])
            self.assertEqual([s.groups for s in config.calibration], [["2"]])
            with self.assertRaisesRegex(SystemExit, "in none of the clip lists"):
                configure([*argv, "--cache=c", "--out=r.json", "--groups", "9"])


if __name__ == "__main__":
    unittest.main()
