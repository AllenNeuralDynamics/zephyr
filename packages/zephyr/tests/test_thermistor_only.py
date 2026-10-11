"""Thermistor-only clip-list entries (whole recordings without a video)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit
from _clipfiles import touch_clip, write_fold, write_list

from zephyr.annotate_events import TuneState
from zephyr.config import ClipList, Fold, load, require_videos
from zephyr.session_selection import Settings, apply, assess

FS = 250.0


def breathing(t, seed):
    rng = np.random.default_rng(seed)
    depth = 1 + 0.15 * np.sin(2 * np.pi * 0.13 * t + seed)
    return depth * np.sin(2 * np.pi * 5.0 * t) + 0.01 * rng.standard_normal(len(t))


def write_sessions(root: Path, n: int = 4) -> Path:
    """A list of *n* whole-session thermistors, no videos."""
    folder = root / "thermistor"
    folder.mkdir(parents=True)
    t = np.arange(0, 40.0, 1 / FS)
    lines = ["[preprocess]\ntarget_size = [100, 80]\n"]
    for k in range(n):
        name = f"session_{k}.parquet"
        pd.DataFrame({"Time": t, "Signal": breathing(t, k)}).to_parquet(folder / name)
        lines.append(
            f'[[clip]]\nthermistor = "thermistor/{name}"\ngroup = "animal_{k}"\n'
        )
    path = root / "sessions.toml"
    path.write_text("\n".join(lines))
    return path


class ThermistorOnlyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def test_an_entry_without_a_video_resolves_to_its_thermistor(self):
        path = write_sessions(self.root, n=2)
        clips = load(ClipList, path).resolve()
        self.assertIsNone(clips[0].video)
        self.assertIsNone(clips[0].timestamps)
        self.assertEqual(
            clips[0].thermistor, self.root / "thermistor/session_0.parquet"
        )
        self.assertEqual(clips[0].group, "animal_0")
        self.assertEqual(clips[0].name, "thermistor/session_0")

    def test_without_a_video_it_needs_a_thermistor_and_no_box(self):
        bad = [
            'group = "a"',
            'thermistor = "thermistor/session_0.parquet"\nbox = [1, 1, 8, 8]',
            "thermistor = false",
        ]
        write_sessions(self.root, n=1)
        for body in bad:
            path = self.root / "bad.toml"
            path.write_text(
                f"[preprocess]\ntarget_size = [100, 80]\n[[clip]]\n{body}\n"
            )
            with self.assertRaises(ValueError, msg=body):
                load(ClipList, path)

    def test_the_same_thermistor_twice_is_refused(self):
        path = write_sessions(self.root, n=1)
        text = path.read_text()
        path.write_text(text + text[text.index("[[clip]]") :])
        with self.assertRaisesRegex(ValueError, "more than once"):
            load(ClipList, path)

    def test_everything_needing_frames_refuses_them(self):
        clips = load(ClipList, write_sessions(self.root, n=1)).resolve()
        with self.assertRaisesRegex(ValueError, "thermistor only"):
            require_videos(clips, "preprocessing")
        video = touch_clip(self.root / "v", 1, 1)
        trained = write_list(self.root / "train.toml", [video])
        fold = write_fold(
            self.root / "fold.toml",
            [(trained, 1.0)],
            {"s": self.root / "sessions.toml"},
        )
        with self.assertRaisesRegex(ValueError, "thermistor only"):
            load(Fold, fold)

    def test_the_tuner_and_selection_read_and_save_them(self):
        path = write_sessions(self.root)
        state = TuneState(path)
        self.assertEqual(len(state.entries), 4)
        table = assess(state, Settings(window_s=2.0, min_span_s=6.0, min_amplitude=0.5))
        self.assertEqual(list(table["suggested"]), ["include"] * 4)
        state.entries[1].reviewed = True
        state.entries[1].selection = "exclude"
        apply(state, table)
        state.save()
        doc = tomlkit.parse(path.read_text())
        self.assertNotIn("video", doc["clip"][0])
        self.assertEqual(str(doc["clip"][0]["selection"]), "include")
        self.assertTrue(doc["clip"][0]["included"])
        self.assertEqual(str(doc["clip"][1]["selection"]), "exclude")
        self.assertTrue(doc["clip"][1]["reviewed"])
        self.assertEqual(TuneState(path).entries[0].included, state.entries[0].included)


if __name__ == "__main__":
    unittest.main()
