"""The pooled branch: shapes, alignment, which head reads it, default unchanged."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.config import TrainParams
from zephyr.evaluate import load_checkpoint
from zephyr.model import BreathingNet, PooledBranch, TemporalNet
from zephyr.train import ZEPHYR

GRAY = ChannelSet.parse("gray")


class PooledBranchTests(unittest.TestCase):
    def test_keeps_the_sequence_length_for_any_factor(self):
        for factor in (2, 3, 4, 5):
            branch = PooledBranch(8, factor).eval()
            for t in (37, 40, 64):
                self.assertEqual(branch(torch.randn(2, 8, t)).shape, (2, 8, t))

    def test_adds_no_lag(self):
        # Identity body, flat upsampling kernel, identity norms: an impulse comes
        # back centred on the frames its pooled sample averaged.
        for factor in (2, 3, 4):
            branch = PooledBranch(1, factor).eval()
            with torch.no_grad():
                branch.body[0].weight.zero_()
                branch.body[0].weight[0, 0, 1] = 1.0
                branch.up[0].weight.fill_(1.0)
            for module in (branch.body, branch.up):
                module[1] = nn.Identity()
                module[2] = nn.Identity()
            x = torch.zeros(1, 1, 60)
            cell = 5 * factor
            x[0, 0, cell : cell + factor] = 1.0
            y = branch(x)[0, 0].detach().numpy()
            centre = (np.arange(60) * y).sum() / y.sum()
            self.assertAlmostEqual(centre, cell + (factor - 1) / 2, places=4)

    def test_refuses_a_factor_below_two(self):
        with self.assertRaises(ValueError):
            PooledBranch(8, 1)


class TemporalNetTests(unittest.TestCase):
    def test_default_network_has_no_branch(self):
        keys = TemporalNet(16).state_dict().keys()
        self.assertFalse(any(k.startswith("signal_branch") for k in keys))

    def test_onset_head_does_not_read_the_branch(self):
        net = TemporalNet(16, channels=8, signal_pool=2).eval()
        x = torch.randn(1, 50, 16)
        _, onset = net(x)
        with torch.no_grad():
            for p in net.signal_branch.parameters():
                p.add_(1.0)
        signal_after, onset_after = net(x)
        torch.testing.assert_close(onset, onset_after)
        self.assertEqual(signal_after.shape, (1, 50))

    def test_pooled_and_both_onset_heads_read_the_branch(self):
        x = torch.randn(1, 50, 16)
        for onset_input, width in (("pooled", 8), ("both", 16)):
            net = TemporalNet(
                16, channels=8, signal_pool=2, onset_input=onset_input
            ).eval()
            self.assertEqual(net.onset_head.in_channels, width)
            _, onset = net(x)
            with torch.no_grad():
                for p in net.signal_branch.parameters():
                    p.add_(1.0)
            _, onset_after = net(x)
            self.assertFalse(torch.allclose(onset, onset_after))
            self.assertEqual(onset_after.shape, (1, 50))

    def test_onset_input_needs_the_branch(self):
        with self.assertRaises(ValueError):
            TemporalNet(16, onset_input="both")

    def test_branch_widens_the_receptive_field(self):
        self.assertGreater(
            TemporalNet(16, signal_pool=4).receptive_field,
            TemporalNet(16).receptive_field,
        )


class SignalPoolPlumbingTests(unittest.TestCase):
    def test_default_is_off_and_not_recorded(self):
        params = TrainParams()
        self.assertEqual(params.signal_pool, 1)
        self.assertEqual(ZEPHYR.kwargs(params), {})

    def test_a_non_positive_pool_is_an_error(self):
        with self.assertRaises(ValueError):
            TrainParams(signal_pool=0)

    def test_onset_input_without_the_branch_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "needs signal_pool > 1"):
            TrainParams(onset_input="pooled")
        with self.assertRaises(ValueError):
            TrainParams(signal_pool=2, onset_input="smoothed")

    def test_checkpoint_rebuilds_the_branch(self):
        params = TrainParams(channels="gray", signal_pool=2, onset_input="both")
        kwargs = ZEPHYR.kwargs(params)
        self.assertEqual(kwargs, {"signal_pool": 2, "onset_input": "both"})
        model = ZEPHYR.build(GRAY, params, None, None, **kwargs)
        state = {
            "model": model.state_dict(),
            "channels": ["gray"],
            "feature_config": {"channel_names": list(CHANNEL_NAMES)},
            "mean": np.zeros(len(CHANNEL_NAMES)),
            "std": np.ones(len(CHANNEL_NAMES)),
            "arch": "zephyr",
            "arch_kwargs": kwargs,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(state, path)
            loaded, *_ = load_checkpoint(path, torch.device("cpu"))
        self.assertIsInstance(loaded, BreathingNet)
        self.assertIsNotNone(loaded.temporal.signal_branch)
        self.assertEqual(loaded.temporal.onset_input, "both")


if __name__ == "__main__":
    unittest.main()
