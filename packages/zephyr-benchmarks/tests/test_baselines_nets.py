import unittest

import numpy as np
import torch

from zephyr.benchmarks import nets
from zephyr.channels import ChannelSet


class TsmTests(unittest.TestCase):
    def test_temporal_shift_directions(self):
        x = torch.arange(4.0).view(4, 1, 1, 1).repeat(1, 3, 1, 1)  # 4 frames, 3 ch
        out = nets.temporal_shift(x, n_segment=4)
        self.assertEqual(out[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0, 0.0])  # from t+1
        self.assertEqual(out[:, 1, 0, 0].tolist(), [0.0, 0.0, 1.0, 2.0])  # from t-1
        self.assertEqual(out[:, 2, 0, 0].tolist(), [0.0, 1.0, 2.0, 3.0])  # unchanged

    def test_attention_mask_mean_is_half(self):
        g = torch.rand(2, 1, 5, 5)
        self.assertTrue(
            torch.allclose(
                nets.attention_mask(g).mean(dim=(2, 3)), torch.full((2, 1), 0.5)
            )
        )


class TscanCoreTests(unittest.TestCase):
    def test_one_output_per_frame(self):
        net = nets.TSCAN(frame_depth=10, img_size=36)
        out = net(torch.randn(20, 1, 36, 36), torch.randn(20, 1, 36, 36))
        self.assertEqual(out.shape, (20,))


class TscanWrapperTests(unittest.TestCase):
    def test_forward_shapes_and_zero_onset(self):
        model = nets.TSCANBreathing(
            ChannelSet.parse("gray"), gray_mean=100.0, gray_std=20.0
        )
        features = torch.randn(2, 33, 1, 96, 96)
        t_in = torch.arange(33.0).repeat(2, 1) / 60
        t_out = torch.arange(30.0).repeat(2, 1) / 60
        signal, onset = model(features, t_in, t_out)
        self.assertEqual(signal.shape, (2, 30))
        self.assertTrue(torch.equal(onset, torch.zeros_like(signal)))

    def test_rejects_non_gray_channels(self):
        with self.assertRaises(ValueError):
            nets.TSCANBreathing(ChannelSet.parse("gray+diff"))


class ReconstructionTests(unittest.TestCase):
    def test_sparse_detrend_matches_dense_formula(self):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(size=120))
        n, lam = len(x), 10.0
        d = np.zeros((n - 2, n))
        for i in range(n - 2):
            d[i, i : i + 3] = [1, -2, 1]
        dense = (np.eye(n) - np.linalg.inv(np.eye(n) + lam**2 * d.T @ d)) @ x
        np.testing.assert_allclose(nets.tarvainen_detrend(x, lam), dense, atol=1e-8)

    def test_reconstruct_from_derivative_recovers_signal_without_a_lag(self):
        t = np.arange(3600) / 60.0
        signal = np.sin(2 * np.pi * 3 * t)
        forward = np.diff(signal, append=signal[-1])  # y[k+1] - y[k]
        rebuilt = nets.reconstruct_from_derivative(forward, 60.0)
        inner = slice(600, -600)
        self.assertGreater(np.corrcoef(rebuilt[inner], signal[inner])[0, 1], 0.99)
        # exact alignment: correlation peaks at zero lag, not one sample either side
        corr = {
            lag: np.corrcoef(np.roll(rebuilt, lag)[inner], signal[inner])[0, 1]
            for lag in (-1, 0, 1)
        }
        self.assertEqual(max(corr, key=corr.get), 0)


class DerivativeLossTests(unittest.TestCase):
    def test_target_is_the_forward_difference(self):
        y = torch.sin(torch.linspace(0, 20, 512)).repeat(3, 1)
        d = y[:, 1:] - y[:, :-1]
        d = d / d.std(dim=-1, keepdim=True)
        pred = torch.cat([d, torch.full((3, 1), 99.0)], dim=-1)  # last sample unscored
        loss, stats = nets.DerivativeMSELoss()(
            pred, torch.zeros_like(pred), y, torch.zeros_like(y)
        )
        self.assertLess(float(loss), 1e-6)
        self.assertGreater(stats["corr"], 0.999)

    def test_a_central_difference_prediction_is_not_perfect(self):
        y = torch.sin(torch.linspace(0, 20, 512)).repeat(3, 1)
        d = torch.gradient(y, dim=-1)[0]
        d = d / d.std(dim=-1, keepdim=True)
        loss, _ = nets.DerivativeMSELoss()(
            d, torch.zeros_like(d), y, torch.zeros_like(y)
        )
        self.assertGreater(float(loss), 1e-5)


class TscanAlignmentTests(unittest.TestCase):
    def test_each_output_is_placed_at_the_earlier_frame_of_its_pair(self):
        model = nets.TSCANBreathing(ChannelSet.parse("gray"))

        class Ramp(torch.nn.Module):
            def forward(self, motion, appearance):
                return torch.arange(motion.shape[0], dtype=torch.float32)

        model.net = Ramp()
        t = torch.arange(21.0).unsqueeze(0) / 60
        signal, _ = model(torch.randn(1, 21, 1, 96, 96), t, t)
        # output j describes frames (j, j+1), so it sits at frame j's time
        self.assertEqual(signal[0, :20].tolist(), list(range(20)))


class PhysNetTests(unittest.TestCase):
    def test_core_preserves_length(self):
        net = nets.PhysNet()
        self.assertEqual(net(torch.randn(1, 1, 16, 32, 32)).shape, (1, 16))

    def test_wrapper_pads_odd_lengths(self):
        model = nets.PhysNetBreathing(ChannelSet.parse("gray"))
        features = torch.randn(1, 18, 1, 32, 32)
        t_in = torch.arange(18.0).unsqueeze(0) / 60
        signal, _ = model(features, t_in, t_in[:, 1:-1])
        self.assertEqual(signal.shape, (1, 16))


class PhysNetEventTests(unittest.TestCase):
    def test_dual_head_returns_two_outputs_of_same_shape(self):
        net = nets.PhysNetDualHead()
        signal, onset = net(torch.randn(1, 1, 16, 32, 32))
        self.assertEqual(signal.shape, (1, 16))
        self.assertEqual(onset.shape, (1, 16))

    def test_wrapper_returns_non_zero_onset_logits(self):
        model = nets.PhysNetBreathingEvent(ChannelSet.parse("gray"))
        features = torch.randn(1, 18, 1, 32, 32)
        t_in = torch.arange(18.0).unsqueeze(0) / 60
        signal, onset = model(features, t_in, t_in[:, 1:-1])
        self.assertEqual(signal.shape, (1, 16))
        self.assertEqual(onset.shape, (1, 16))
        self.assertFalse(torch.all(onset == 0))

    def test_rejects_non_gray_channels(self):
        with self.assertRaises(ValueError):
            nets.PhysNetBreathingEvent(ChannelSet.parse("gray+diff"))


class BuildModelTests(unittest.TestCase):
    def test_build_each_arch(self):
        mean, std = np.full(4, 120.0), np.full(4, 30.0)
        self.assertIsInstance(
            nets.build_model("zephyr", ChannelSet.parse("gray+diff")),
            nets.BreathingNet,
        )
        tscan = nets.build_model("tscan", ChannelSet.parse("gray"), mean=mean, std=std)
        self.assertAlmostEqual(float(tscan.gray_mean), 120.0)
        self.assertIsInstance(
            nets.build_model("physnet", ChannelSet.parse("gray")),
            nets.PhysNetBreathing,
        )
        self.assertIsInstance(
            nets.build_model("physnet_event", ChannelSet.parse("gray")),
            nets.PhysNetBreathingEvent,
        )
        with self.assertRaises(ValueError):
            nets.build_model("nope", ChannelSet.parse("gray"))


class TscanOptionsTests(unittest.TestCase):
    def test_build_model_forwards_the_input_size(self):
        model = nets.build_model("tscan", ChannelSet.parse("gray"), img_size=96)
        self.assertEqual(model.img_size, 96)
        out = model(
            torch.randn(1, 21, 1, 96, 96),
            torch.arange(21.0).unsqueeze(0) / 60,
            torch.arange(20.0).unsqueeze(0) / 60,
        )[0]
        self.assertEqual(out.shape, (1, 20))

    def test_unsupported_options_are_rejected(self):
        gray = ChannelSet.parse("gray")
        with self.assertRaises(ValueError):
            nets.build_model("physnet", gray, img_size=96)
        with self.assertRaises(ValueError):
            nets.build_model("zephyr", gray, img_size=96)

    def test_there_is_no_reconstruction_delay_option(self):
        gray = ChannelSet.parse("gray")
        for arch in ("tscan", "physnet", "physnet_event", "zephyr"):
            with self.assertRaises(ValueError):
                nets.build_model(arch, gray, delay=1.0)
