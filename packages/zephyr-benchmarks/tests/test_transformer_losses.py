import math
import unittest

import numpy as np
import torch

from zephyr.benchmarks import transformer_losses as tl

FS = 60.0


def _sine(hz: float, n: int = 320, batch: int = 2) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / FS
    return torch.sin(2 * math.pi * hz * t).repeat(batch, 1)


class RateTests(unittest.TestCase):
    def test_bins_are_whole_rates_per_minute_over_the_band(self):
        bins = tl.rate_bins((1.0, 15.0))
        self.assertEqual((bins[0], bins[-1], len(bins)), (60.0, 899.0, 840))

    def test_window_rate_finds_a_sine(self):
        for hz in (2.0, 4.5, 10.0):
            rate = tl.window_rate(_sine(hz)[0].numpy(), FS)
            self.assertLess(abs(rate - hz * 60), 3.0, hz)


class SpectrumTests(unittest.TestCase):
    def test_matches_the_authors_loop(self):
        x = torch.randn(1, 160)
        bins = np.arange(40, 180, dtype=np.float64)
        fs = 30.0
        # TorchLossComputer.compute_complex_absolute_given_k, on the CPU
        n = x.shape[1]
        k = torch.tensor(bins / 60.0 / (fs / n), dtype=torch.float32).view(1, -1, 1)
        two_pi_n_over_n = (2 * math.pi * torch.arange(0, n).float() / n).view(1, 1, -1)
        hanning = torch.from_numpy(np.hanning(n)).float().view(1, -1)
        out = (x * hanning).view(1, 1, -1)
        power = (out * torch.sin(k * two_pi_n_over_n)).sum(-1) ** 2 + (
            out * torch.cos(k * two_pi_n_over_n)
        ).sum(-1) ** 2
        reference = power / power.sum()
        np.testing.assert_allclose(
            tl.spectrum(x, fs, bins).numpy(), reference.numpy(), rtol=1e-4, atol=1e-7
        )

    def test_peaks_at_the_signal_rate_and_sums_to_one(self):
        bins = tl.rate_bins()
        power = tl.spectrum(_sine(3.0), FS, bins)
        self.assertTrue(torch.allclose(power.sum(-1), torch.ones(2)))
        self.assertLess(abs(bins[int(power[0].argmax())] - 180.0), 3.0)


class LossTests(unittest.TestCase):
    def test_gaussian_label_peaks_at_the_target(self):
        label = tl.gaussian_label(torch.tensor([5, 30]), 40, std=1.0)
        self.assertEqual(label.argmax(-1).tolist(), [5, 30])
        self.assertAlmostEqual(float(label[0, 5]), 1 / math.sqrt(2 * math.pi), 6)
        self.assertGreaterEqual(float(label.min()), 1e-15)

    def test_physformer_frequency_weight_rises_from_one_to_five(self):
        loss = tl.PhysFormerLoss(FS, epochs=200)
        self.assertEqual((loss.a, loss.b), (0.1, 1.0))
        loss.set_epoch(100)
        self.assertAlmostEqual(loss.b, math.sqrt(5.0))
        loss.set_epoch(199)
        self.assertAlmostEqual(loss.b, 5.0 ** (199 / 200))

    def test_the_right_rate_scores_lower_and_gradients_flow(self):
        truth = _sine(3.0)
        onset_true = torch.zeros_like(truth)
        for loss in (
            tl.PhysFormerLoss(FS, epochs=10),
            tl.RhythmFormerLoss(FS),
        ):
            right = (truth + 0.1 * torch.randn_like(truth)).requires_grad_()
            wrong = _sine(8.0)
            onset = torch.zeros_like(truth)
            low, stats = loss(right, onset, truth, onset_true)
            high, _ = loss(wrong, onset, truth, onset_true)
            self.assertLess(float(low), float(high), type(loss).__name__)
            self.assertGreater(stats["corr"], 0.9)
            self.assertEqual(
                set(stats), {"loss", "corr", "onset", "pearson", "frequency"}
            )
            low.backward()
            self.assertTrue(torch.isfinite(right.grad).all())
            self.assertGreater(float(right.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
