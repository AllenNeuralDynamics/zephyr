"""The shipped baseline config matches the reference runs' inputs.

Every setting the config still has; the runs were trained on the gray channel.

These read local data (``data/``, ``benchmarks/`` at the repository root), which
is not in the repository, so each test skips when its inputs are absent.
"""

import ast
import json
import re
import unittest
from pathlib import Path

from zephyr.benchmarks.config import BenchmarkExperiment
from zephyr.config import load

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiments/baselines-nets.toml"
FIXTURE = Path(__file__).parent / "fixtures/window_draws_seed42.json"

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
        case.assertEqual(args["channels"], "gray", run.name)  # the only input now
        p = fold.train_params
        expected = {
            "arch": args.get("arch", "zephyr"),
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
            "select_jitter": _arg(args, "select_jitter"),
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
        experiment = load(BenchmarkExperiment, EXPERIMENT)
        self.assertEqual(experiment.seeds, [17, 42])
        for arch in ("tscan", "physnet"):
            with self.subTest(arch=arch):
                (fold,) = experiment.folds([f"baseline-{arch}"])
                root = ROOT / f"benchmarks/baselines-v1/{arch}/runs"
                runs = [root / f"gray__signal__seed-{s}" for s in experiment.seeds]
                _assert_params_match_runs(self, fold, runs, experiment.seeds)


@needs_data
class TrainingClipTests(unittest.TestCase):
    def test_training_clips_are_the_reference_manifest_in_order(self):
        experiment = load(BenchmarkExperiment, EXPERIMENT)
        order = json.loads(FIXTURE.read_text())["train_clip_order"]
        train = (ROOT / "data/train").resolve()  # loaded paths are resolved too
        wanted = [
            train / ("video_face_" + m.group(1) + ".mp4")
            for m in (re.fullmatch(r"train_(\d+_part_\d+)", c) for c in order)
        ]
        for fold in experiment.fold:
            with self.subTest(fold=fold.name):
                (clips,) = fold.train_clips()
                self.assertEqual([c.video for c in clips], wanted)


if __name__ == "__main__":
    unittest.main()
