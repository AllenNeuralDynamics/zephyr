"""Noisy frames: the noise wrapper, and a probe run on a real (synthetic) video."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# The tiny real dataset zephyr's own tests decode, train and predict on.
sys.path.insert(0, str(Path(__file__).parents[2] / "zephyr" / "tests"))
from _synthetic import HAS_FFMPEG, make_dataset

from zephyr.benchmarks.probe import Noisy, load_probes, main, probe_traces
from zephyr.config import ClipList, load
from zephyr.model import BreathingNet
from zephyr.preprocess import preprocess_clips

NET = {"widths": (4, 4, 4, 4), "embed": 8, "tcn_channels": 8, "dilations": (1, 2)}


class _Echo(torch.nn.Module):
    receptive_field = 1

    def forward(self, frames, *args, **kwargs):
        return frames


class NoisyTests(unittest.TestCase):
    def test_noise_is_in_gray_levels_and_repeatable(self):
        noisy = Noisy(_Echo(), std=20.0)
        frames = torch.zeros(1, 2000, 1, 4, 4)
        noisy.reseed(0.0, 1)
        self.assertTrue(torch.equal(noisy(frames), frames))
        noisy.reseed(10.0, 1)
        first = noisy(frames)
        self.assertAlmostEqual(float(first.std()), 10.0 / 20.0, delta=0.01)
        noisy.reseed(10.0, 1)
        self.assertTrue(torch.equal(noisy(frames), first))


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class ProbeTests(unittest.TestCase):
    def test_each_noise_level_is_measured_beside_its_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            clips = make_dataset(root, groups=(1,))
            clip_list = load(ClipList, clips)
            preprocess_clips(
                clip_list.resolve(), clip_list.preprocess, root / "cache", workers=1
            )
            torch.manual_seed(0)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model": BreathingNet(**NET).state_dict(),
                    "arch_kwargs": NET,
                    "mean": 80.0,
                    "std": 30.0,
                    "params": {"w_onset": 0.5},
                    "train_videos": [],
                    "epoch": 1,
                },
                checkpoint,
            )
            main(
                [
                    f"--checkpoint={checkpoint}",
                    f"--clips={clips}",
                    f"--cache={root / 'cache'}",
                    f"--out={root / 'probes'}",
                    "--sigmas",
                    "8",
                    "64",
                    "--window-s=4",
                    "--device=cpu",
                    "--amp=off",
                    "--infer-window=256",
                    "--save-traces",
                ]
            )
            (report,) = (root / "probes").glob("*.probe.json")
            data = json.loads(report.read_text())
            self.assertEqual([c["sigma"] for c in data["conditions"]], [0.0, 8.0, 64.0])
            ordinary = data["conditions"][0]["summary"]
            self.assertAlmostEqual(ordinary["agreement"]["trace_r"], 1.0, places=6)
            self.assertIn("inhale_f1", ordinary["truth"])
            noisy = data["conditions"][2]["summary"]["agreement"]["trace_r"]
            self.assertLess(noisy, 1.0)

            table = load_probes(root / "probes")
            self.assertEqual(sorted(table["sigma"]), [0.0, 8.0, 64.0])
            self.assertTrue(np.isfinite(table["truth_inhale_f1"]).all())
            self.assertEqual(len(load_probes(root / "probes", windows=True)), 15)
            traces = probe_traces(
                root / "probes", report.name.removesuffix(".probe.json")
            )
            self.assertEqual(sorted(traces), [0.0, 8.0, 64.0])
            self.assertEqual(
                list(traces[8.0].columns), ["time", "signal", "onset", "inhale"]
            )


if __name__ == "__main__":
    unittest.main()
