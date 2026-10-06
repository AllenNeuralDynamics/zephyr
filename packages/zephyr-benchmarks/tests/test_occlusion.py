import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zephyr.benchmarks.occlusion import (
    Occluded,
    head_entry,
    occlusion_maps,
    patch_origins,
)
from zephyr.features import ClipEntry


class PatchGridTests(unittest.TestCase):
    def test_default_grid_is_eleven_by_eleven(self):
        self.assertEqual(patch_origins(96, 16, 8), list(range(0, 81, 8)))

    def test_last_patch_is_flush_with_the_edge(self):
        self.assertEqual(patch_origins(10, 4, 4), [0, 4, 6])


class Echo(nn.Module):
    channels = ("a", "b", "c")
    receptive_field = 1

    def forward(self, features, *args, **kwargs):
        return features, features


class OccludedTests(unittest.TestCase):
    def test_hides_one_patch_of_one_channel_only(self):
        model = Occluded(Echo(), patch=3)
        features = torch.ones(1, 2, 3, 8, 8)
        self.assertTrue((model(features)[0] == 1).all())
        model.hidden = (1, 2, 4)
        out = model(features)[0]
        changed = out != 1
        self.assertTrue((out[:, :, 1, 2:5, 4:7] == 0).all())
        self.assertEqual(changed.sum().item(), 2 * 3 * 3)
        self.assertFalse(changed[:, :, [0, 2]].any())
        self.assertTrue((features == 1).all())  # the input is not modified


class MapTests(unittest.TestCase):
    def test_overlapping_patches_are_averaged(self):
        # Hiding a patch costs 1 only when it contains pixel (0, 0) of channel 0.
        def score(hidden):
            if hidden is None:
                return 1.0
            channel, y, x = hidden
            return 1.0 - float(channel == 0 and y == 0 and x == 0)

        base, maps = occlusion_maps(score, ["a", "b"], patch=4, stride=2, size=8)
        self.assertEqual(base, 1.0)
        self.assertEqual(maps["b"].sum(), 0)
        # Only the patch at (0, 0) covers pixel (0, 0); it is 1 of the 4 patches
        # (origins 0, 2, 4 per axis -> 9 in all) covering pixel (3, 3).
        self.assertEqual(maps["a"][0, 0], 1.0)
        self.assertAlmostEqual(maps["a"][3, 3], 1 / 4)
        self.assertEqual(maps["a"][7, 7], 0.0)
        self.assertEqual(maps["a"].shape, (8, 8))


class HeadEntryTests(unittest.TestCase):
    def test_keeps_the_first_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "f.npy", np.zeros((600, 4, 6, 6), np.uint8))
            np.save(tmp / "ft.npy", np.arange(600) / 60.0)
            np.save(tmp / "t.npy", np.arange(600) / 60.0)
            entry = ClipEntry(
                "c", "r", 600, 600, tmp / "f.npy", tmp / "ft.npy", tmp / "dt.npy",
                tmp / "t.npy", None, None,
            )  # fmt: skip
            out = tmp / "out"
            out.mkdir()
            head = head_entry(entry, 2.0, out)
            self.assertEqual(len(np.load(head.times)), 120)
            self.assertEqual(head.n_output, 120)
            self.assertGreaterEqual(len(np.load(head.frame_times)), 120)
            self.assertEqual(np.load(head.features).shape[1:], (4, 6, 6))


if __name__ == "__main__":
    unittest.main()
