import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import torch

from zephyr.benchmarks import nets, transformer_losses
from zephyr.benchmarks.config import (
    BenchmarkExperiment,
    BenchmarkFold,
    BenchmarkTrainParams,
)
from zephyr.benchmarks.runner import RUNNER, architecture, load_checkpoint
from zephyr.config import load
from zephyr.model import BreathingLoss, BreathingNet
from zephyr.train import ZEPHYR

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATA = Path(__file__).resolve().parents[3] / "data"


def _fold(**params) -> BenchmarkFold:
    """A fold's train_params without loading clip lists: only the params matter."""
    return BenchmarkFold.model_construct(train_params=BenchmarkTrainParams(**params))


class ConfigTests(unittest.TestCase):
    def test_arch_names_match_the_networks(self):
        arch = BenchmarkTrainParams.model_fields["arch"].annotation
        self.assertEqual(typing.get_args(arch), nets.ARCHS)

    def test_arch_defaults_to_zephyr(self):
        self.assertEqual(BenchmarkTrainParams().arch, "zephyr")

    def test_only_zephyr_has_an_onset_head(self):
        for arch in nets.ARCHS[1:]:
            with self.assertRaisesRegex(ValueError, "only zephyr has an onset head"):
                BenchmarkTrainParams(arch=arch, w_onset=0.5)
            BenchmarkTrainParams(arch=arch, w_onset=0.0)
        BenchmarkTrainParams(arch="zephyr", w_onset=0.5)

    def test_signal_pool_is_zephyr_only(self):
        with self.assertRaisesRegex(ValueError, "are zephyr options"):
            BenchmarkTrainParams(arch="physnet", w_onset=0.0, signal_pool=2)
        BenchmarkTrainParams(signal_pool=2, onset_input="both")

    def test_experiment_loads_its_folds_as_benchmark_folds(self):
        fold_type = BenchmarkExperiment.model_fields["fold"].annotation
        self.assertEqual(fold_type, list[BenchmarkFold])
        self.assertIs(RUNNER.experiment, BenchmarkExperiment)

    @unittest.skipUnless((DATA / "train").is_dir(), "no local data")
    def test_every_shipped_config_loads(self):
        for path in sorted((CONFIGS / "experiments").glob("*.toml")):
            with self.subTest(path=path.name):
                load(BenchmarkExperiment, path)


class ArchitectureTests(unittest.TestCase):
    def test_zephyr_folds_train_zephyr_itself(self):
        self.assertIs(architecture(_fold()), ZEPHYR)

    def test_tscan_trains_on_the_derivative_at_its_input_size(self):
        arch = architecture(_fold(arch="tscan", w_onset=0.0, tscan_img_size=48))
        params = BenchmarkTrainParams(arch="tscan", w_onset=0.0, tscan_img_size=48)
        self.assertEqual(arch.name, "tscan")
        self.assertEqual(arch.kwargs(params), {"img_size": 48})
        self.assertIsInstance(arch.criterion(params, 60.0), nets.DerivativeMSELoss)
        model = arch.build(params, 120.0, 1.0, **arch.kwargs(params))
        self.assertIsInstance(model, nets.TSCANBreathing)
        self.assertEqual(model.img_size, 48)
        self.assertAlmostEqual(float(model.gray_mean), 120.0)

    def test_physnet_trains_on_zephyrs_loss(self):
        params = BenchmarkTrainParams(arch="physnet", w_onset=0.0)
        arch = architecture(_fold(arch="physnet", w_onset=0.0))
        self.assertEqual(arch.kwargs(params), {})
        self.assertIsInstance(arch.criterion(params, 60.0), BreathingLoss)

    def test_transformers_train_on_their_papers_losses(self):
        for name, loss in (
            ("physformer", transformer_losses.PhysFormerLoss),
            ("rhythmformer", transformer_losses.RhythmFormerLoss),
        ):
            params = BenchmarkTrainParams(arch=name, w_onset=0.0)
            arch = architecture(_fold(arch=name, w_onset=0.0))
            self.assertIsInstance(arch.criterion(params, 60.0), loss, name)


class LoadCheckpointTests(unittest.TestCase):
    def _roundtrip(self, arch: str, **kwargs):
        model = nets.build_model(arch, **kwargs)
        state = {
            "model": model.state_dict(),
            "feature_config": {},
            "mean": 0.0,
            "std": 1.0,
            "arch": arch,
            "arch_kwargs": kwargs,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(state, path)
            loaded, *_ = load_checkpoint(path, torch.device("cpu"))
        return loaded

    def test_rebuilds_every_network(self):
        self.assertIsInstance(self._roundtrip("zephyr"), BreathingNet)
        pooled = self._roundtrip("zephyr", signal_pool=2, onset_input="both")
        self.assertEqual(pooled.temporal.onset_input, "both")
        tscan = self._roundtrip("tscan", img_size=24)
        self.assertEqual(tscan.img_size, 24)
        self.assertIsInstance(self._roundtrip("physnet"), nets.PhysNetBreathing)


class EntrypointTests(unittest.TestCase):
    def test_cli_runs_as_a_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "zephyr.benchmarks.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("zephyr-benchmarks", result.stdout)


if __name__ == "__main__":
    unittest.main()
