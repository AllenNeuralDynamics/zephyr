"""``split_cache``: a combined feature cache rewritten one file per channel group."""

import json
import tempfile
import unittest
import uuid
from pathlib import Path

import fsspec
import numpy as np

from zephyr import features
from zephyr.corpus import split_cache

T, H, W = 20, 4, 5


def _combined_entry(fs, folder: str, name: str) -> np.ndarray:
    """One entry in the layout made before the split: all four channels in ``feat-``."""
    stack = np.random.default_rng(len(name)).integers(
        0, 255, (T, 4, H, W), dtype=np.uint8
    )
    with tempfile.TemporaryDirectory() as tmp:
        for key, array in {
            "features": stack,
            "frame_times": np.arange(T) / 60.0,
        }.items():
            path = Path(tmp) / features.paths(Path(), name)[key].name
            np.save(path, array)
            fs.put_file(str(path), f"{folder}/{path.name}")
    sidecar = {"key": {"video": name}, "n_frames": T, "n_output": T, "stats": {}}
    fs.pipe_file(f"{folder}/{name}.json", json.dumps(sidecar).encode())
    return stack


def _load(fs, path: str) -> np.ndarray:
    with fs.open(path, "rb") as handle:
        return np.load(handle)


class SplitCacheTests(unittest.TestCase):
    def setUp(self):
        self.fs = fsspec.filesystem("memory")
        self.folder = f"memory://cache-{uuid.uuid4().hex}"
        self.stacks = {
            n: _combined_entry(self.fs, self.folder, n) for n in ("a-1", "b-2")
        }
        self.fs.pipe_file(f"{self.folder}/channel_stats-0.json", b"{}")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_in_place_on_a_remote_cache_then_a_rerun_does_nothing(self):
        split_cache.main(
            [self.folder, "--scratch", str(self.scratch), "--workers", "2"]
        )
        names = {Path(p).name for p in self.fs.ls(self.folder, detail=False)}
        for name, stack in self.stacks.items():
            self.assertNotIn(f"feat-{name}.npy", names)
            sidecar = json.loads(self.fs.cat_file(f"{self.folder}/{name}.json"))
            self.assertEqual(sidecar["groups"], ["gray", "diff", "flow"])
            np.testing.assert_array_equal(
                _load(self.fs, f"{self.folder}/gray-{name}.npy"), stack[:, :1]
            )
            np.testing.assert_array_equal(
                _load(self.fs, f"{self.folder}/diff-{name}.npy"), stack[:, 1:2]
            )
            np.testing.assert_array_equal(
                _load(self.fs, f"{self.folder}/flow-{name}.npy"), stack[:, 2:]
            )
        self.assertEqual(split_cache.entries(self.fs, self.folder), ["a-1", "b-2"])
        self.assertEqual(
            split_cache.split_entry(self.fs, self.folder, "a-1", self.scratch, None),
            "already",
        )

    def test_into_out_keeps_the_source_and_copies_the_rest(self):
        out = Path(self._tmp.name) / "out"
        split_cache.main(
            [self.folder, "--out", str(out), "--scratch", str(self.scratch)]
        )
        self.assertTrue(self.fs.exists(f"{self.folder}/feat-a-1.npy"))
        self.assertTrue((out / "channel_stats-0.json").is_file())
        self.assertTrue((out / "ftime-a-1.npy").is_file())
        recombined = np.concatenate(
            [np.load(out / f"{g}-b-2.npy") for g in features.GROUPS], axis=1
        )
        np.testing.assert_array_equal(recombined, self.stacks["b-2"])
        self.assertIn("groups", json.loads((out / "b-2.json").read_text()))


if __name__ == "__main__":
    unittest.main()
