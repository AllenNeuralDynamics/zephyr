import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from _clipfiles import touch_clip, write_list

from zephyr import features
from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.config import (
    ClipList,
    Fold,
    FoldReference,
    QualityControl,
    QualityReference,
    ReferenceSampling,
    TrainParams,
    canonical,
    load,
)
from zephyr.evaluate import load_checkpoint
from zephyr.model import BreathingNet
from zephyr.quality_control import embeddings, load_reports, run
from zephyr.quality_control.metrics import MEASURES, Video, find_events, measure, spans
from zephyr.quality_control.reference import (
    Reference,
    checkpoint_hash,
    embedding_statistics,
    fit_calibration,
)
from zephyr.quality_control.report import _spread_out, fold_reference, measure_reference
from zephyr.run import run_fold
from zephyr.train import Machine

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


def _video(times, signal, onset, distances=None):
    return Video(
        times=times,
        signal=signal,
        onset=onset,
        events=find_events(times, signal, onset),
        distances=distances,
    )


def _measure(video, span=None, reference=None):
    whole = span is None
    span = span or (video.times[0], np.nextafter(video.times[-1], np.inf))
    return measure(
        video,
        span,
        ceiling_hz=CEILING,
        ambiguous_floor=0.2,
        reference=reference,
        whole=whole,
    )


def _reference(**overrides) -> Reference:
    fields = {
        "checkpoint_sha256": "x",
        "embedding_mean": np.zeros(2),
        "embedding_precision": np.eye(2),
        "distance_quantiles": np.linspace(0, 3, 101),
        "onset_calibration": None,
        "record": {},
    }
    return Reference(**(fields | overrides))


class MeasureTests(unittest.TestCase):
    def test_a_clean_rhythm_is_plausible(self):
        m = _measure(_video(*_breathing()))
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)
        self.assertEqual(m["too_fast_frac"], 0.0)
        self.assertEqual(m["alternation_break_frac"], 0.0)
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
        self.assertLess(noisy["head_trace_f1"], clean["head_trace_f1"])

    def test_without_a_head_its_measures_are_nan(self):
        times, signal, _ = _breathing()
        m = _measure(_video(times, signal, None))
        for name in ("head_trace_f1", "peak_prob", "inhale_phase_spread"):
            self.assertTrue(np.isnan(m[name]), name)
        self.assertAlmostEqual(m["rate_hz"], 4.0, places=1)  # DSP inhales instead

    def test_embedding_measures_need_a_reference_and_the_whole_clip(self):
        times, signal, onset = _breathing()
        self.assertNotIn("embedding_distance", _measure(_video(times, signal, onset)))
        video = _video(times, signal, onset, distances=np.ones(5))
        m = _measure(video, reference=_reference())
        self.assertEqual(m["embedding_distance"], 1.0)
        self.assertAlmostEqual(m["embedding_percentile"], 100 / 3, places=5)
        self.assertEqual(m["unfamiliar_frac"], 0.0)
        window = _measure(video, span=(0.0, 10.0), reference=_reference())
        self.assertNotIn("embedding_distance", window)  # five frames: no window

    def test_only_rates_above_the_ceiling_are_judged(self):
        for rate in (2.0, 4.0, 12.0):  # slow, usual and sniffing are all fine
            times, signal, _ = _breathing(rate=rate, seconds=20)
            m = _measure(_video(times, signal, None))
            self.assertEqual(m["too_fast_frac"], 0.0, rate)
        times, signal, _ = _breathing(rate=20.0, seconds=20)
        m = _measure(_video(times, signal, None))
        self.assertGreater(m["too_fast_frac"], 0.9)

    def test_inhales_at_one_point_of_the_cycle_have_no_phase_spread(self):
        times, signal, onset = _breathing()
        steady = _measure(_video(times, signal, onset))["inhale_phase_spread"]
        rng = np.random.default_rng(0)
        anywhere = np.zeros(len(times))
        anywhere[rng.choice(len(times), 150, replace=False)] = 1.0
        wandering = _measure(_video(times, signal, anywhere))["inhale_phase_spread"]
        self.assertLess(steady, 0.05)
        self.assertGreater(wandering, 0.5)

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


class ConfigTests(unittest.TestCase):
    def test_the_examples_load(self):
        examples = Path(__file__).resolve().parents[3] / "examples"
        config = load(QualityControl, examples / "quality-control.toml")
        self.assertEqual(config.params.embedding_frames, 5)
        reference = load(QualityReference, examples / "quality-reference.toml")
        self.assertEqual(reference.calibration, [])

    def test_every_judging_measure_says_which_way_is_better(self):
        times, signal, onset = _breathing()
        video = _video(times, signal, onset, distances=np.ones(5))
        produced = set(_measure(video, reference=_reference()))
        self.assertLessEqual(produced, set(MEASURES))
        for family, _, better in MEASURES.values():
            self.assertIn(family, ("rhythm", "head", "agreement", "input"))
            self.assertIn(better, ("higher", "lower", None))


class FoldReferenceTests(unittest.TestCase):
    def test_training_writes_a_reference_by_default(self):
        self.assertTrue(FoldReference().enabled)
        self.assertEqual(FoldReference().max_clips, 30)

    def test_a_fold_can_turn_it_off(self):
        fold = mock.Mock(quality_reference=FoldReference(enabled=False))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                fold_reference(fold, Path(tmp), Path(tmp), "cpu", "off", smoke=False)
            )
        fold.train_clips.assert_not_called()

    def test_clips_are_spread_over_the_list(self):
        self.assertEqual(_spread_out(list(range(10)), 3), [0, 4, 9])
        self.assertEqual(_spread_out(list(range(2)), 3), [0, 1])
        self.assertEqual(_spread_out(list(range(4)), None), [0, 1, 2, 3])


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


class EmbeddingSampleTests(unittest.TestCase):
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

    def test_sampled_frames_embed_as_in_a_whole_clip_pass(self):
        for kernel in (1, 3):
            torch.manual_seed(0)
            model = BreathingNet(
                channels=ChannelSet.parse("gray"),
                widths=(4, 4, 4, 4),
                embed=8,
                tcn_channels=8,
                dilations=(1, 2),
                temporal_kernel=kernel,
            ).eval()
            vectors, times = embeddings.sample(
                model, self.mean, self.std, self.entry, 5, "cpu", None
            )
            self.assertEqual(vectors.shape, (5, 8))
            rows = embeddings.spread_rows(T_FRAMES, 5)
            np.testing.assert_array_equal(times, rows / FS)
            stack = features.reader(self.entry).read(slice(None), model.channels)
            block = (torch.from_numpy(stack.astype(np.float32)) - 128) / 64
            with torch.no_grad():
                whole = model.encode(block.unsqueeze(0))[0].numpy()
            np.testing.assert_allclose(vectors, whole[rows], atol=1e-5)

    def test_frames_are_spread_inside_the_clip(self):
        self.assertEqual(
            list(embeddings.spread_rows(600, 5)), [100, 200, 300, 400, 500]
        )
        self.assertEqual(len(embeddings.spread_rows(3, 5)), 3)


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
            'checkpoint = "best.pt"',
            'features_dir = "cache"',
            'output_dir = "qc"',
            'device = "cpu"',
            'amp = "off"',
            "infer_window = 256",
            '[[clips]]\nclips = "clips.toml"',
            "[params]\nwindow_s = 3.0",
            *(f"{k} = {json.dumps(v)}" for k, v in params.items()),
        ]
        path = self.root / "qc.toml"
        path.write_text("\n".join(lines) + "\n")
        return load(QualityControl, path)

    def test_writes_one_report_per_video_and_reads_back(self):
        (report,) = run(self._config(), "cpu", "off")
        self.assertEqual(report.name, "video_face_1_part_1.qc.json")
        data = json.loads(report.read_text())  # NaN would not parse in strict readers
        self.assertEqual(data["inhale_events"], "onset head")
        self.assertFalse(data["labelled"])
        self.assertNotIn("truth", data["summary"])
        self.assertEqual(len(data["windows"]), 3)
        self.assertIn("inhale_phase_spread", data["summary"])
        self.assertNotIn("embedding_distance", data["summary"])  # no reference
        self.assertTrue(report.with_name("video_face_1_part_1.qc.parquet").is_file())
        videos, windows = load_reports(report.parent)
        self.assertEqual(len(videos), 1)
        self.assertEqual(len(windows), 3)

    def test_a_finished_report_is_kept_and_other_settings_are_refused(self):
        (report,) = run(self._config(), "cpu", "off")
        before = report.stat().st_mtime_ns
        run(self._config(), "cpu", "off")
        self.assertEqual(report.stat().st_mtime_ns, before)
        with self.assertRaisesRegex(SystemExit, "other settings"):
            run(self._config(flat_ratio=0.3), "cpu", "off")

    def test_the_traces_are_one_parquet_row_per_sample(self):
        (report,) = run(self._config(), "cpu", "off")
        traces = pd.read_parquet(report.with_name("video_face_1_part_1.qc.parquet"))
        self.assertEqual(len(traces), T_FRAMES)
        self.assertEqual(list(traces.columns), ["time", "signal", "onset", "inhale"])
        self.assertEqual(traces["inhale"].dtype, bool)

    def test_a_reference_is_built_then_used(self):
        reference = self.root / "qc-reference.json"
        clip_list = load(ClipList, self.root / "clips.toml", check_files=False)
        measure_reference(
            self.checkpoint,
            clip_list.resolve(),
            [],
            clip_list.preprocess,
            self.root / "cache",
            reference,
            ReferenceSampling(frames_per_clip=100),
            TrainParams(infer_window=256),  # what training passes
            "cpu",
            "off",
        )
        record = json.loads(reference.read_text())
        self.assertEqual(record["embedding"]["distances_from"], "clips")
        self.assertIsNone(record["onset_calibration"])  # no calibration clips
        config = self._config().model_copy(update={"reference": reference})
        (report,) = run(config, "cpu", "off")
        summary = json.loads(report.read_text())["summary"]
        self.assertIsNotNone(summary["embedding_percentile"])
        self.assertEqual(record["embedding"]["frames"], 100)  # sampled, not the clip
        self.assertIsNone(summary["expected_precision"])

    def test_a_reference_of_another_checkpoint_is_refused(self):
        other = self.root / "other.json"
        other.write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
        with mock.patch.object(Reference, "load", return_value=_reference()):
            config = self._config().model_copy(update={"reference": other})
            with self.assertRaisesRegex(SystemExit, "another checkpoint"):
                run(config, "cpu", "off")

    def _trained_fold(self, extra: str = "") -> tuple:
        """A fold training on one labelled clip with a cache entry, and a run folder
        whose ``best.pt`` is already there (so ``run_fold`` skips training)."""
        data = self.root / "labelled"
        videos = [touch_clip(data, g, 1) for g in (1, 2)]
        clips = write_list(
            self.root / "labelled.toml",
            videos,
            box=(0, 0, SIZE, SIZE),
            target_size=(SIZE, SIZE),
        )
        clip_list = load(ClipList, clips)
        clip = clip_list.resolve()[0]
        cache = self.root / "cache"
        files = features.paths(cache, features.prefix(clip, clip_list.preprocess))
        rng = np.random.default_rng(1)
        np.save(
            files["gray"],
            rng.integers(0, 255, (T_FRAMES, 1, SIZE, SIZE), dtype=np.uint8),
        )
        times = np.arange(T_FRAMES) / FS
        np.save(files["frame_times"], times)
        np.save(files["times"], times)
        files["sidecar"].write_text(
            json.dumps(
                {
                    "key": features.cache_key(clip, clip_list.preprocess),
                    "thermistor": canonical(clip.thermistor),
                    "events": features.events_identity(clip),
                    "n_frames": T_FRAMES,
                    "n_output": T_FRAMES,
                    "groups": ["gray"],
                }
            )
        )
        fold_file = self.root / "fold.toml"
        fold_file.write_text(
            'name = "fold"\n[[train]]\nclips = "labelled.toml"\ngroups = ["1"]\n'
            '[test]\nheld = { clips = "labelled.toml", groups = ["2"] }\n'
            '[train_params]\nchannels = "gray"\ninfer_window = 256\n' + extra
        )
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "best.pt").write_bytes(self.checkpoint.read_bytes())
        return load(Fold, fold_file), run_dir, cache

    def test_training_ends_with_the_reference(self):
        fold, run_dir, cache = self._trained_fold()
        with mock.patch("zephyr.run.evaluate_groups") as evaluate:
            run_fold(fold, 1, run_dir, cache, Machine(device="cpu", amp="off"))
        evaluate.assert_called_once()
        record = json.loads((run_dir / "qc-reference.json").read_text())
        self.assertEqual(record["clips"], [fold.train_clips()[0][0].name])
        self.assertEqual(record["checkpoint_sha256"], checkpoint_hash(self.checkpoint))

    def test_a_fold_without_the_reference_writes_none(self):
        extra = "[quality_reference]\nenabled = false\n"
        fold, run_dir, cache = self._trained_fold(extra)
        with mock.patch("zephyr.run.evaluate_groups"):
            run_fold(fold, 1, run_dir, cache, Machine(device="cpu", amp="off"))
        self.assertFalse((run_dir / "qc-reference.json").exists())


if __name__ == "__main__":
    unittest.main()
