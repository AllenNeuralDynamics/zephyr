import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from _clipfiles import write_list

from zephyr import features
from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.config import ClipList, QualityControl, QualityReference, load
from zephyr.evaluate import load_checkpoint
from zephyr.infer import predict_clip, predict_samples
from zephyr.model import BreathingNet
from zephyr.quality_control import load_reports, run
from zephyr.quality_control.metrics import MEASURES, Video, find_events, measure, spans
from zephyr.quality_control.reference import (
    INTERVAL_EDGES,
    Reference,
    embedding_statistics,
    fit_calibration,
    interval_density,
)

FS = 60.0
BAND = (2.0, 15.0)


def _breathing(seconds=40.0, rate=4.0, noise=0.0, seed=0):
    """A clean (or noisy) trace at *rate*, and an onset head peaking on its inhales."""
    times = np.arange(0, seconds, 1 / FS)
    rng = np.random.default_rng(seed)
    signal = np.sin(2 * np.pi * rate * times) + noise * rng.standard_normal(len(times))
    phase = (rate * times) % 1.0
    onset = np.exp(-(((phase - 0.25) / 0.08) ** 2))  # sin peaks at a quarter cycle
    return times, signal, onset


def _video(times, signal, onset, draws=(), reference_distances=None):
    draws = np.asarray(draws, float).reshape(-1, len(times))
    return Video(
        times=times,
        signal=signal,
        onset=onset,
        events=find_events(times, signal, onset),
        draws=draws,
        draw_events=[find_events(times, d, onset) for d in draws],
        frame_times=times,
        distances=reference_distances,
    )


def _measure(video, span=None, reference=None):
    span = span or (video.times[0], np.nextafter(video.times[-1], np.inf))
    return measure(
        video, span, band=BAND, jump=2.0, ambiguous_floor=0.2, reference=reference
    )


def _reference(**overrides) -> Reference:
    fields = {
        "checkpoint_sha256": "x",
        "embedding_mean": np.zeros(2),
        "embedding_precision": np.eye(2),
        "distance_quantiles": np.linspace(0, 3, 101),
        "image_quantiles": None,
        "interval_edges": INTERVAL_EDGES,
        "interval_density": interval_density(np.full(500, 0.25)),
        "onset_calibration": None,
        "record": {},
    }
    return Reference(**(fields | overrides))


class MeasureTests(unittest.TestCase):
    def test_a_clean_rhythm_is_plausible(self):
        m = _measure(_video(*_breathing()))
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)
        self.assertEqual(m["out_of_band_frac"], 0.0)
        self.assertEqual(m["rate_jump_frac"], 0.0)
        self.assertEqual(m["alternation_break_frac"], 0.0)
        self.assertGreater(m["band_power_frac"], 0.9)
        self.assertGreater(m["cycle_similarity"], 0.99)
        self.assertGreater(m["head_trace_f1"], 0.95)
        self.assertGreater(m["peak_prob"], 0.9)
        self.assertLess(m["onset_background"], 0.05)

    def test_noise_is_less_plausible_than_breathing(self):
        clean = _measure(_video(*_breathing()))
        times, _, onset = _breathing()
        noise = np.random.default_rng(1).standard_normal(len(times))
        noisy = _measure(_video(times, noise, onset))
        self.assertLess(noisy["cycle_similarity"], clean["cycle_similarity"])
        self.assertLess(noisy["band_power_frac"], clean["band_power_frac"])
        self.assertLess(noisy["head_trace_f1"], clean["head_trace_f1"])

    def test_draws_equal_to_the_prediction_agree_fully(self):
        times, signal, onset = _breathing()
        m = _measure(_video(times, signal, onset, draws=[signal, signal]))
        self.assertEqual(m["agreement_f1"], 1.0)
        self.assertEqual(m["event_support"], 1.0)
        self.assertAlmostEqual(m["agreement_corr"], 1.0)
        self.assertEqual(m["trace_spread"], 0.0)

    def test_draws_unlike_the_prediction_agree_less(self):
        times, signal, onset = _breathing()
        _, other, _ = _breathing(noise=2.0, seed=3)
        m = _measure(_video(times, signal, onset, draws=[other]))
        self.assertLess(m["agreement_corr"], 0.7)
        self.assertGreater(m["trace_spread"], 0.0)

    def test_without_draws_or_head_those_measures_are_nan(self):
        times, signal, _ = _breathing()
        m = _measure(_video(times, signal, None))
        for name in ("agreement_f1", "head_trace_f1", "peak_prob"):
            self.assertTrue(np.isnan(m[name]), name)
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)  # DSP inhales instead

    def test_measures_needing_a_reference_appear_only_with_one(self):
        times, signal, onset = _breathing()
        self.assertNotIn("interval_kl", _measure(_video(times, signal, onset)))
        video = _video(times, signal, onset, reference_distances=np.ones(len(times)))
        m = _measure(video, reference=_reference())
        self.assertLess(m["interval_kl"], 0.5)  # 4 Hz: the population's 0.25 s
        self.assertEqual(m["embedding_distance"], 1.0)
        self.assertAlmostEqual(m["embedding_percentile"], 100 / 3, places=5)
        self.assertEqual(m["unfamiliar_frac"], 0.0)

    def test_a_rate_unlike_the_population_diverges(self):
        times, signal, onset = _breathing(rate=2.5)
        m = _measure(_video(times, signal, onset), reference=_reference())
        times, signal, onset = _breathing(rate=4.0)
        usual = _measure(_video(times, signal, onset), reference=_reference())
        self.assertGreater(m["interval_kl"], usual["interval_kl"] + 1)

    def test_windows_cover_the_clip_once(self):
        times = np.arange(0, 95, 1 / FS)
        windows = spans(times, 30.0)
        self.assertEqual(len(windows), 3)
        inside = sum(((times >= a) & (times < b)).sum() for a, b in windows)
        self.assertEqual(inside, len(times))


class ConfigTests(unittest.TestCase):
    def test_the_examples_load(self):
        examples = Path(__file__).resolve().parents[3] / "examples"
        config = load(QualityControl, examples / "quality-control.toml")
        self.assertEqual(config.params.agreement, "dropout")
        reference = load(QualityReference, examples / "quality-reference.toml")
        self.assertEqual(reference.calibration, [])

    def test_every_judging_measure_says_which_way_is_better(self):
        times, signal, onset = _breathing()
        video = _video(times, signal, onset, draws=[signal])
        produced = set(_measure(video, reference=_reference()))
        self.assertLessEqual(produced, set(MEASURES))
        for family, _, better in MEASURES.values():
            self.assertIn(family, ("rhythm", "head", "agreement", "input"))
            self.assertIn(better, ("higher", "lower", None))


class ReferenceTests(unittest.TestCase):
    def test_distance_is_mahalanobis(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5000, 3)) * [1.0, 2.0, 4.0]
        mean, precision = embedding_statistics(x, shrinkage=0.0)
        ref = _reference(embedding_mean=mean, embedding_precision=precision)
        d = ref.distance(np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(d, [1.0, 1.0], atol=0.05)

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


T_FRAMES, SIZE = 600, 16


def _tiny_net() -> BreathingNet:
    torch.manual_seed(0)
    return BreathingNet(
        channels=ChannelSet.parse("gray"),
        widths=(4, 4, 4, 4),
        embed=8,
        tcn_channels=8,
        dilations=(1, 2),
        dropout=0.3,
    )


class PredictSamplesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        rng = np.random.default_rng(0)
        name = "clip"
        stack = rng.integers(0, 255, (T_FRAMES, 1, SIZE, SIZE), dtype=np.uint8)
        np.save(self.root / f"gray-{name}.npy", stack)
        times = np.arange(T_FRAMES) / FS
        np.save(self.root / f"ftime-{name}.npy", times)
        np.save(self.root / f"time-{name}.npy", times)
        self.entry = features.ClipEntry(
            clip_id=name,
            recording=name,
            n_frames=T_FRAMES,
            n_output=T_FRAMES,
            stem=self.root / name,
            frame_times=self.root / f"ftime-{name}.npy",
            baselines=self.root / f"dt-{name}.npy",
            times=self.root / f"time-{name}.npy",
            target=None,
            events=None,
            groups=("gray",),
        )
        self.model = _tiny_net().eval()
        self.mean = np.full(len(CHANNEL_NAMES), 128.0)
        self.std = np.full(len(CHANNEL_NAMES), 64.0)

    def _samples(self, passes, seed=0):
        return predict_samples(
            self.model,
            self.entry,
            self.mean,
            self.std,
            passes=passes,
            seed=seed,
            window=256,
            device="cpu",
            amp_dtype=None,
        )

    def test_the_prediction_is_predict_clips(self):
        signal, onset = predict_clip(
            self.model,
            self.entry,
            self.mean,
            self.std,
            window=256,
            device="cpu",
            amp_dtype=None,
        )
        samples = self._samples(4)
        np.testing.assert_allclose(samples.signal, signal, atol=1e-4)
        np.testing.assert_allclose(samples.onset, onset, atol=1e-5)

    def test_draws_differ_reproducibly_and_dropout_is_left_off(self):
        a, b = self._samples(4, seed=1), self._samples(4, seed=1)
        np.testing.assert_array_equal(a.signal_draws, b.signal_draws)
        self.assertEqual(a.signal_draws.shape, (4, T_FRAMES))
        self.assertGreater(np.abs(a.signal_draws[0] - a.signal_draws[1]).max(), 1e-3)
        self.assertFalse(any(m.training for m in self.model.modules()))

    def test_every_frame_has_one_embedding(self):
        samples = self._samples(0)
        self.assertEqual(samples.embeddings.shape, (T_FRAMES, 8))
        self.assertTrue(np.isfinite(samples.embeddings).all())


class RunTests(unittest.TestCase):
    """End to end: a tiny checkpoint, one unlabelled clip in a real cache entry."""

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
        clip_list = load(ClipList, clips, check_files=False)
        (clip,) = clip_list.resolve()
        cache = self.root / "cache"
        cache.mkdir()
        name = features.prefix(clip, clip_list.preprocess)
        files = features.paths(cache, name)
        rng = np.random.default_rng(0)
        np.save(
            files["gray"],
            rng.integers(0, 255, (T_FRAMES, 1, SIZE, SIZE), dtype=np.uint8),
        )
        times = 1000 + np.arange(T_FRAMES) / FS
        np.save(files["frame_times"], times)
        np.save(files["times"], times)
        files["sidecar"].write_text(
            json.dumps(
                {
                    "key": features.cache_key(clip, clip_list.preprocess),
                    "thermistor": None,
                    "n_frames": T_FRAMES,
                    "n_output": T_FRAMES,
                    "groups": ["gray"],
                }
            )
        )
        model = _tiny_net()
        self.checkpoint = self.root / "best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "channels": ["gray"],
                "arch_kwargs": {
                    "widths": (4, 4, 4, 4),
                    "embed": 8,
                    "tcn_channels": 8,
                    "dilations": (1, 2),
                    "dropout": 0.3,
                },
                "mean": np.full(len(CHANNEL_NAMES), 128.0),
                "std": np.full(len(CHANNEL_NAMES), 64.0),
                "params": {"w_onset": 0.5},
                "train_videos": [],
                "epoch": 1,
            },
            self.checkpoint,
        )
        load_checkpoint(self.checkpoint, torch.device("cpu"))  # the file loads

    def _config(self, **params) -> QualityControl:
        lines = [
            'checkpoints = ["best.pt"]',
            'features_dir = "cache"',
            'output_dir = "qc"',
            'device = "cpu"',
            'amp = "off"',
            "infer_window = 256",
            '[[clips]]\nclips = "clips.toml"',
            "[params]\nwindow_s = 3.0\ndropout_passes = 4",
            *(f"{k} = {json.dumps(v)}" for k, v in params.items()),
        ]
        path = self.root / "qc.toml"
        path.write_text("\n".join(lines) + "\n")
        return load(QualityControl, path)

    def test_writes_one_report_per_video_and_reads_back(self):
        (report,) = run(self._config(), "cpu", "off")
        self.assertEqual(report.name, "video_face_1_part_1.qc.json")
        data = json.loads(report.read_text())  # NaN would not parse in strict readers
        self.assertEqual(data["draws"], 4)
        self.assertEqual(data["inhale_events"], "onset head")
        self.assertFalse(data["labelled"])
        self.assertNotIn("truth", data["summary"])
        self.assertEqual(len(data["windows"]), 3)
        self.assertIn("agreement_f1", data["summary"])
        self.assertIn("sharpness", data["windows"][0])
        self.assertTrue(report.with_name("video_face_1_part_1.qc.npz").is_file())
        videos, windows = load_reports(report.parent)
        self.assertEqual(len(videos), 1)
        self.assertEqual(len(windows), 3)

    def test_a_finished_report_is_kept_and_other_settings_are_refused(self):
        (report,) = run(self._config(), "cpu", "off")
        before = report.stat().st_mtime_ns
        run(self._config(), "cpu", "off")
        self.assertEqual(report.stat().st_mtime_ns, before)
        with self.assertRaisesRegex(SystemExit, "other settings"):
            run(self._config(seed=7), "cpu", "off")

    def test_agreement_with_checkpoints_needs_several(self):
        with self.assertRaisesRegex(ValueError, "at least two checkpoints"):
            self._config(agreement="checkpoints")


if __name__ == "__main__":
    unittest.main()
