"""The shipped benchmark config reproduces the reference runs' inputs.

These read local data (``data/``, ``benchmarks/`` at the repository root), which
is not in the repository, so each test skips when its inputs are absent.
"""

import ast
import json
import re
import unittest
from pathlib import Path

import numpy as np

from zephyr import features
from zephyr.benchmarks.config import BenchmarkExperiment
from zephyr.channels import ChannelSet
from zephyr.config import load
from zephyr.dataset import WindowDataset
from zephyr.train import augment_config

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiments/benchmark-gray-diff-flow-multitask.toml"
RUNS = ROOT / "benchmarks/input-objective-v1/runs"
FIXTURE = Path(__file__).parent / "fixtures/window_draws_seed42.json"
CACHE = ROOT / "data/features-v2"

needs_data = unittest.skipUnless((ROOT / "data/train").is_dir(), "no local data")


def _arg(args: dict, key: str):
    return ast.literal_eval(args[key])


def _assert_params_match_runs(case, fold, runs, seeds) -> None:
    """Every training-relevant ``args.json`` key of each run equals the fold's."""
    checked = 0
    for run, seed in zip(runs, seeds):
        path = run / "args.json"
        if not path.exists():
            continue
        args = json.loads(path.read_text())
        case.assertEqual(_arg(args, "seed"), seed)
        p = fold.train_params
        expected = {
            "arch": args.get("arch", "zephyr"),
            "channels": args["channels"],
            "window": _arg(args, "window"),
            "batch_size": _arg(args, "batch_size"),
            "epochs": _arg(args, "epochs"),
            "steps_per_epoch": _arg(args, "steps_per_epoch"),
            "lr": _arg(args, "lr"),
            "weight_decay": _arg(args, "weight_decay"),
            "warmup_steps": _arg(args, "warmup_steps"),
            "grad_clip": _arg(args, "grad_clip"),
            "dropout": _arg(args, "dropout"),
            "ema_decay": _arg(args, "ema_decay"),
            "w_corr": _arg(args, "w_corr"),
            "w_onset": _arg(args, "w_onset"),
            "scales": tuple(_arg(args, "scales")),
            "val_fraction": _arg(args, "val_fraction"),
            "score_every": _arg(args, "score_every"),
            "patience": _arg(args, "patience"),
            "min_epochs": _arg(args, "min_epochs"),
            "val_windows": _arg(args, "val_windows"),
            "infer_window": _arg(args, "infer_window"),
            "frame_chunk": _arg(args, "frame_chunk"),
        }
        if expected["arch"] == "tscan":
            expected["tscan_img_size"] = _arg(args, "tscan_img_size")
        for key, value in expected.items():
            case.assertEqual(getattr(p, key), value, f"{run.name}: {key}")
        aug = {
            "time_stretch": _arg(args, "time_stretch"),
            "rate_range": tuple(_arg(args, "rate_range")),
            "scale_motion": not _arg(args, "no_scale_motion"),
            "select_jitter": _arg(args, "select_jitter"),
            "motion_noise": _arg(args, "motion_noise"),
            "shift_px": _arg(args, "shift_px"),
            "brightness": _arg(args, "brightness"),
            "contrast": _arg(args, "contrast"),
            "noise": _arg(args, "noise"),
            "flip": _arg(args, "flip"),
        }
        for key, value in aug.items():
            case.assertEqual(getattr(p.augmentation, key), value, f"{run.name}: {key}")
        checked += 1
    if not checked:
        case.skipTest("no reference runs on disk")


@needs_data
class BaselineFoldTests(unittest.TestCase):
    """The network baseline folds reproduce benchmarks/baselines-v1's runs."""

    def test_train_params_equal_the_reference_runs(self):
        experiment = load(
            BenchmarkExperiment, CONFIGS / "experiments/baselines-nets.toml"
        )
        self.assertEqual(experiment.seeds, [17, 42])
        for arch in ("tscan", "physnet"):
            with self.subTest(arch=arch):
                (fold,) = experiment.folds([f"baseline-{arch}"])
                root = ROOT / f"benchmarks/baselines-v1/{arch}/runs"
                runs = [root / f"gray__signal__seed-{s}" for s in experiment.seeds]
                _assert_params_match_runs(self, fold, runs, experiment.seeds)


@needs_data
class BenchmarkFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.fold,) = load(BenchmarkExperiment, EXPERIMENT).fold

    def test_train_params_equal_every_reference_run(self):
        seeds = load(BenchmarkExperiment, EXPERIMENT).seeds
        self.assertEqual(seeds, [17, 42, 101, 202, 314])
        runs = [RUNS / f"gray-diff-flow__multitask__seed-{seed}" for seed in seeds]
        _assert_params_match_runs(self, self.fold, runs, seeds)

    def test_training_clips_are_the_reference_manifest_in_order(self):
        order = json.loads(FIXTURE.read_text())["train_clip_order"]
        train = (ROOT / "data/train").resolve()  # loaded paths are resolved too
        wanted = [
            train / ("video_face_" + m.group(1) + ".mp4")
            for m in (re.fullmatch(r"train_(\d+_part_\d+)", c) for c in order)
        ]
        (clips,) = self.fold.train_clips()
        self.assertEqual([c.video for c in clips], wanted)

    @unittest.skipUnless(CACHE.is_dir(), "features-v2 not preprocessed")
    def test_window_draws_match_the_pre_refactor_dataset(self):
        fixture = json.loads(FIXTURE.read_text())
        params = self.fold.train_params
        recipe = self.fold.preprocess
        (clips,) = self.fold.train_clips()
        entries = features.require(clips, recipe, CACHE)
        dataset = WindowDataset(
            [entries],
            window=params.window,
            mean=np.zeros(4),
            std=np.ones(4),
            length=params.epochs * params.steps_per_epoch * params.batch_size,
            seed=fixture["seed"],
            augment=augment_config(params),
            channels=ChannelSet.parse(params.channels),
            select_fs=recipe.select_fs,
            output_fs=60.0,
            motion_tau_s=recipe.motion_tau_s,
            onset_sigma_s=recipe.onset_sigma_s,
        )
        for draw in fixture["draws"]:
            clip_i, start, stretch = dataset.draw(draw["i"])
            session_part = draw["clip"].removeprefix("train_")
            self.assertEqual(
                dataset.usable[clip_i].clip_id, f"train/video_face_{session_part}"
            )
            self.assertEqual(start, draw["start"])
            self.assertEqual(stretch, draw["stretch"])


if __name__ == "__main__":
    unittest.main()
