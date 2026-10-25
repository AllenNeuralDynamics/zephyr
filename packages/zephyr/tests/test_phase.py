"""The breath-phase target, head and loss: values, wrap, mask, off by default."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from _synthetic import HAS_FFMPEG, make_dataset, write_experiment

from zephyr import run
from zephyr.config import TrainParams
from zephyr.evaluate import load_checkpoint
from zephyr.model import BreathingLoss, BreathingNet, phase_loss
from zephyr.preprocess import main as preprocess
from zephyr.targets import breath_phase
from zephyr.train import TARGETS, ZEPHYR, _loss


class BreathPhaseTests(unittest.TestCase):
    times = np.arange(0.0, 3.0, 0.25)
    onsets = np.array([0.5, 1.0, 2.0])

    def test_runs_linearly_from_zero_to_two_pi_in_each_cycle(self):
        phase, mask = breath_phase(self.times, self.onsets)
        by_time = dict(zip(self.times[mask], phase[mask], strict=True))
        self.assertEqual(by_time[0.5], 0.0)
        self.assertAlmostEqual(by_time[0.75], np.pi)
        self.assertAlmostEqual(by_time[1.25], np.pi / 2)
        self.assertAlmostEqual(by_time[1.75], 3 * np.pi / 2)

    def test_wraps_to_zero_at_each_inhalation(self):
        phase, _ = breath_phase(self.times, self.onsets)
        self.assertEqual(phase[self.times == 1.0][0], 0.0)
        self.assertTrue(((phase >= 0) & (phase < 2 * np.pi)).all())

    def test_masks_before_the_first_and_from_the_last_inhalation(self):
        _, mask = breath_phase(self.times, self.onsets)
        np.testing.assert_array_equal(mask, (self.times >= 0.5) & (self.times < 2.0))

    def test_masks_every_cycle_an_excluded_span_touches(self):
        _, mask = breath_phase(self.times, self.onsets, np.array([[1.4, 1.6]]))
        np.testing.assert_array_equal(mask, (self.times >= 0.5) & (self.times < 1.0))

    def test_fewer_than_two_inhalations_give_no_phase(self):
        _, mask = breath_phase(self.times, np.array([1.0]))
        self.assertFalse(mask.any())


class PhaseLossTests(unittest.TestCase):
    def test_zero_when_the_angle_matches_whatever_the_length(self):
        truth = torch.tensor([[0.3, 2.0, 5.0]])
        pred = 3.0 * torch.stack([torch.sin(truth), torch.cos(truth)], dim=-1)
        mask = torch.ones_like(truth, dtype=torch.bool)
        self.assertAlmostEqual(float(phase_loss(pred, truth, mask)), 0.0, places=6)

    def test_reads_only_masked_samples(self):
        truth = torch.zeros(1, 2)
        pred = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])  # right, then opposite
        mask = torch.tensor([[True, False]])
        self.assertEqual(float(phase_loss(pred, truth, mask)), 0.0)
        self.assertEqual(float(phase_loss(pred, truth, ~mask)), 2.0)
        self.assertEqual(float(phase_loss(pred, truth, mask & False)), 0.0)


def _batch(t=16):
    g = torch.Generator().manual_seed(0)
    return {
        "features": torch.randn(2, t + 4, 1, 32, 32, generator=g),
        "t_in": torch.arange(t + 4, dtype=torch.float32).expand(2, -1) / 60 - 2 / 60,
        "t_out": torch.arange(t, dtype=torch.float32).expand(2, -1) / 60,
        "signal": torch.randn(2, t, generator=g),
        "onset": torch.rand(2, t, generator=g),
        "phase": 2 * np.pi * torch.rand(2, t, generator=g),
        "phase_mask": torch.rand(2, t, generator=g) > 0.3,
    }


class PhaseHeadTests(unittest.TestCase):
    def test_default_has_no_phase_head_and_records_nothing(self):
        params = TrainParams()
        self.assertEqual(params.w_phase, 0.0)
        self.assertEqual(ZEPHYR.kwargs(params), {})
        model = ZEPHYR.build(params, 0.0, 1.0)
        self.assertIsNone(model.temporal.phase_head)
        self.assertFalse(any("phase" in k for k in model.state_dict()))

    def test_no_phase_term_when_w_phase_is_zero(self):
        batch = _batch()
        model = BreathingNet().eval()
        with torch.no_grad():
            outputs = model(batch["features"], batch["t_in"], batch["t_out"])
        self.assertEqual(len(outputs), 2)
        loss, stats = _loss(ZEPHYR.criterion(TrainParams(), 60.0), outputs, batch)
        reference, _ = BreathingLoss()(*outputs, batch["signal"], batch["onset"])
        self.assertEqual(float(loss), float(reference))
        self.assertNotIn("phase", stats)

    def test_a_weight_of_zero_adds_nothing_even_with_a_head(self):
        batch = _batch()
        model = BreathingNet(phase_head=True).eval()
        with torch.no_grad():
            outputs = model(batch["features"], batch["t_in"], batch["t_out"])
        loss, stats = _loss(BreathingLoss(w_phase=0.0), outputs, batch)
        reference, _ = BreathingLoss()(*outputs[:2], batch["signal"], batch["onset"])
        self.assertEqual(float(loss), float(reference))
        self.assertGreater(stats["phase"], 0.0)

    def test_one_training_step_moves_the_phase_head(self):
        params = TrainParams(w_phase=0.5)
        kwargs = ZEPHYR.kwargs(params)
        self.assertEqual(kwargs, {"phase_head": True})
        model = ZEPHYR.build(params, 0.0, 1.0, **kwargs)
        criterion = ZEPHYR.criterion(params, 60.0)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
        batch = _batch()
        before = model.temporal.phase_head.weight.detach().clone()
        outputs = model(batch["features"], batch["t_in"], batch["t_out"])
        self.assertEqual(outputs[2].shape, (2, 16, 2))
        loss, stats = _loss(criterion, outputs, {k: batch[k] for k in TARGETS})
        loss.backward()
        optimiser.step()
        self.assertTrue(np.isfinite(stats["phase"]))
        self.assertFalse(torch.equal(before, model.temporal.phase_head.weight))

    def test_checkpoint_rebuilds_the_head(self):
        params = TrainParams(w_phase=0.25)
        kwargs = ZEPHYR.kwargs(params)
        model = ZEPHYR.build(params, 0.0, 1.0, **kwargs)
        state = {"model": model.state_dict(), "mean": 0.0, "std": 1.0}
        state |= {"arch": "zephyr", "arch_kwargs": kwargs}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(state, path)
            loaded, *_ = load_checkpoint(path, torch.device("cpu"))
        self.assertIsNotNone(loaded.temporal.phase_head)


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class PhaseRunTests(unittest.TestCase):
    def test_a_run_with_a_phase_loss_trains_reports_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = make_dataset(root)
            path = write_experiment(root, clips)
            text = path.read_text().replace(
                "[fold.train_params]\n", "[fold.train_params]\nw_phase = 0.25\n"
            )
            path.write_text(text)
            preprocess([str(clips), "--cache", str(root / "cache"), "--workers", "1"])
            run.main([str(path)])
            seed = root / "runs" / "experiment" / "fold" / "seed-1"
            history = json.loads((seed / "history.json").read_text())
            self.assertTrue(all(np.isfinite(r["train_phase"]) for r in history))
            self.assertIn("val_phase", history[0])
            state = torch.load(seed / "best.pt", weights_only=False)
            self.assertEqual(state["arch_kwargs"], {"phase_head": True})
            self.assertTrue((seed / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
