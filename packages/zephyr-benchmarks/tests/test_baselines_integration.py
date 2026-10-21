import tempfile
import unittest
from pathlib import Path

import numpy as np

from zephyr.benchmarks.nets import build_model
from zephyr.dataset import ClipEntry
from zephyr.infer import predict_clip


def fake_entry(tmp: Path, n: int = 150) -> ClipEntry:
    rng = np.random.default_rng(0)
    np.save(tmp / "gray-c.npy", rng.integers(0, 255, (n, 1, 32, 32), dtype=np.uint8))
    times = np.arange(n) / 60.0
    np.save(tmp / "ft.npy", times)
    np.save(tmp / "t.npy", times)
    return ClipEntry(
        clip_id="fake",
        recording="rec",
        n_frames=n,
        n_output=n,
        stem=tmp / "c",
        frames=tmp / "gray-c.npy",
        frame_times=tmp / "ft.npy",
        times=tmp / "t.npy",
        target=None,
        events=None,
    )


class PredictClipTests(unittest.TestCase):
    def test_new_archs_stitch_whole_clips(self):
        mean, std = 128.0, 50.0
        with tempfile.TemporaryDirectory() as tmp:
            entry = fake_entry(Path(tmp))
            for arch in ("tscan", "physnet"):
                model = build_model(arch, mean=mean, std=std)
                signal, _ = predict_clip(
                    model, entry, mean, std, window=64, device="cpu", amp_dtype=None
                )
                self.assertEqual(signal.shape, (150,), arch)
                self.assertTrue(np.isfinite(signal).all(), arch)
