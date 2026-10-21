import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zephyr.dataset import frame_stats, mixture_weights
from zephyr.features import ClipEntry


def _entry(folder: Path, name: str, value: int, t: int = 50) -> ClipEntry:
    """An entry whose gray frames are all *value*."""
    frames = folder / f"gray-{name}.npy"
    np.save(frames, np.full((t, 1, 6, 8), value, np.uint8))
    return ClipEntry(
        clip_id=name,
        recording=name,
        n_frames=t,
        n_output=t,
        stem=folder / name,
        frames=frames,
        frame_times=folder / f"ftime-{name}.npy",
        times=folder / f"time-{name}.npy",
        target=None,
        events=None,
    )


class FrameStatsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_stats_are_scalars_over_every_clip(self):
        entries = [_entry(self.root, "a", 10), _entry(self.root, "b", 30)]
        mean, std = frame_stats(entries, self.root / "s.json", frames_per_clip=10)
        self.assertIsInstance(mean, float)
        self.assertAlmostEqual(mean, 20.0)
        self.assertAlmostEqual(std, 10.0)

    def test_the_cache_is_reused_only_for_the_same_clips(self):
        cache = self.root / "s.json"
        a = _entry(self.root, "a", 10)
        self.assertEqual(frame_stats([a], cache, frames_per_clip=10)[0], 10.0)
        self.assertEqual(json.loads(cache.read_text())["clips"], ["a"])
        _entry(self.root, "a", 90)  # the cached value wins for the same clips
        self.assertEqual(frame_stats([a], cache, frames_per_clip=10)[0], 10.0)
        b = _entry(self.root, "b", 50)
        mean, _ = frame_stats([a, b], cache, frames_per_clip=10)
        self.assertAlmostEqual(mean, 70.0)


class MixtureWeightsTests(unittest.TestCase):
    def test_one_list_is_proportional_to_offsets(self):
        offsets = np.array([300.0, 100.0, 50.0])
        np.testing.assert_array_equal(
            mixture_weights(offsets, [0, 0, 0], [1.0]), offsets / offsets.sum()
        )

    def test_lists_get_their_share_and_clips_stay_proportional_within(self):
        offsets = np.array([300.0, 100.0, 50.0, 50.0])
        weights = mixture_weights(offsets, [0, 0, 1, 2], [0.5, 0.25, 0.25])
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[:2].sum(), 0.5)
        self.assertAlmostEqual(weights[0] / weights[1], 3.0)
        self.assertAlmostEqual(weights[2], 0.25)
        self.assertAlmostEqual(weights[3], 0.25)

    def test_weights_are_relative(self):
        offsets = np.array([1.0, 1.0, 1.0])
        a = mixture_weights(offsets, [0, 1, 2], [1, 1, 2])
        b = mixture_weights(offsets, [0, 1, 2], [0.25, 0.25, 0.5])
        np.testing.assert_allclose(a, b)

    def test_a_list_with_no_usable_clip_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no clip long enough"):
            mixture_weights(np.array([1.0, 1.0]), [0, 0], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
