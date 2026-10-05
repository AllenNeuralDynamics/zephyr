import tempfile
import unittest
from pathlib import Path

import numpy as np

from zephyr.baselines.nets import build_model
from zephyr.channels import ChannelSet
from zephyr.dataset import ClipEntry
from zephyr.infer import predict_clip


def fake_entry(tmp: Path, n: int = 150) -> ClipEntry:
    rng = np.random.default_rng(0)
    np.save(tmp / "feat.npy", rng.integers(0, 255, (n, 4, 32, 32), dtype=np.uint8))
    times = np.arange(n) / 60.0
    np.save(tmp / "ft.npy", times)
    np.save(tmp / "t.npy", times)
    np.save(tmp / "dt.npy", np.full(n, 1 / 60, np.float32))
    return ClipEntry(
        clip_id="fake",
        recording="rec",
        n_frames=n,
        n_output=n,
        features=tmp / "feat.npy",
        frame_times=tmp / "ft.npy",
        baselines=tmp / "dt.npy",
        times=tmp / "t.npy",
        target=None,
        events=None,
    )


class PredictClipTests(unittest.TestCase):
    def test_new_archs_stitch_whole_clips(self):
        mean, std = np.full(4, 128.0), np.full(4, 50.0)
        with tempfile.TemporaryDirectory() as tmp:
            entry = fake_entry(Path(tmp))
            for arch in ("tscan", "physnet"):
                model = build_model(arch, ChannelSet.parse("gray"), mean=mean, std=std)
                signal, _ = predict_clip(
                    model, entry, mean, std, window=64, device="cpu", amp_dtype=None
                )
                self.assertEqual(signal.shape, (150,), arch)
                self.assertTrue(np.isfinite(signal).all(), arch)
