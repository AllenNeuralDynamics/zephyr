import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import numpy as np
import torch

from zephyr.benchmarks import nets
from zephyr.benchmarks.config import (
    BenchmarkExperiment,
    BenchmarkFold,
    BenchmarkTrainParams,
)
from zephyr.benchmarks.runner import RUNNER, architecture, load_checkpoint
from zephyr.channels import CHANNEL_NAMES, ChannelSet
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

    def test_baseline_archs_read_gray(self):
        with self.assertRaisesRegex(ValueError, "needs channels = 'gray'"):
            BenchmarkTrainParams(arch="tscan", channels="gray+diff")
        BenchmarkTrainParams(arch="physnet", channels="gray")

    def test_experiment_loads_its_folds_as_benchmark_folds(self):
        self.assertIs(BenchmarkExperiment.fold_model, BenchmarkFold)
        self.assertIs(RUNNER.experiment, BenchmarkExperiment)
        self.assertIs(RUNNER.fold, BenchmarkFold)

    @unittest.skipUnless((DATA / "train").is_dir(), "no local data")
    def test_every_shipped_config_loads(self):
        for path in sorted((CONFIGS / "experiments").glob("*.toml")):
            with self.subTest(path=path.name):
                load(BenchmarkExperiment, path)


class ArchitectureTests(unittest.TestCase):
    def test_zephyr_folds_train_zephyr_itself(self):
        self.assertIs(architecture(_fold()), ZEPHYR)

    def test_tscan_trains_on_the_derivative_at_its_input_size(self):
        arch = architecture(_fold(arch="tscan", channels="gray", tscan_img_size=48))
        params = BenchmarkTrainParams(arch="tscan", channels="gray", tscan_img_size=48)
        self.assertEqual(arch.name, "tscan")
        self.assertEqual(arch.kwargs(params), {"img_size": 48})
        self.assertIsInstance(arch.criterion(params), nets.DerivativeMSELoss)
        model = arch.build(
            ChannelSet.parse("gray"),
            params,
            np.full(len(CHANNEL_NAMES), 120.0),
            np.ones(len(CHANNEL_NAMES)),
            **arch.kwargs(params),
        )
        self.assertIsInstance(model, nets.TSCANBreathing)
        self.assertEqual(model.img_size, 48)
        self.assertAlmostEqual(float(model.gray_mean), 120.0)

    def test_physnet_trains_on_zephyrs_loss(self):
        params = BenchmarkTrainParams(arch="physnet", channels="gray")
        arch = architecture(_fold(arch="physnet", channels="gray"))
        self.assertEqual(arch.kwargs(params), {})
        self.assertIsInstance(arch.criterion(params), BreathingLoss)


class LoadCheckpointTests(unittest.TestCase):
    def _roundtrip(self, arch: str, channels: str, **kwargs):
        model = nets.build_model(arch, ChannelSet.parse(channels), **kwargs)
        state = {
            "model": model.state_dict(),
            "channels": [channels],
            "feature_config": {"channel_names": list(CHANNEL_NAMES)},
            "mean": np.zeros(len(CHANNEL_NAMES)),
            "std": np.ones(len(CHANNEL_NAMES)),
            "arch": arch,
            "arch_kwargs": kwargs,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(state, path)
            loaded, *_ = load_checkpoint(path, torch.device("cpu"))
        return loaded

    def test_rebuilds_every_network(self):
        self.assertIsInstance(self._roundtrip("zephyr", "gray"), BreathingNet)
        tscan = self._roundtrip("tscan", "gray", img_size=24)
        self.assertEqual(tscan.img_size, 24)
        self.assertIsInstance(self._roundtrip("physnet", "gray"), nets.PhysNetBreathing)


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
