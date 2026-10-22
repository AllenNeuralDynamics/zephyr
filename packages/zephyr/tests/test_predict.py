"""Training's ``on_scored`` hook, ``zephyr predict``, and the validation snippets the
app cuts from the hook, on a tiny real run (ffmpeg, CPU, a few optimiser steps)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from _synthetic import HAS_FFMPEG, make_dataset, write_experiment

from zephyr import run
from zephyr.config import ClipList, load
from zephyr.gui.worker import snippets
from zephyr.predict import SETTINGS, SUFFIX, main, select_clips
from zephyr.preprocess import main as preprocess

RUN = Path("runs/experiment/fold/seed-1")


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class TrainedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.list = make_dataset(cls.root)
        cls.experiment = write_experiment(cls.root, cls.list)
        preprocess(
            [str(cls.list), "--cache", str(cls.root / "cache"), "--workers", "1"]
        )
        cls.scored: list = []
        run.main(
            [str(cls.experiment)],
            on_scored=lambda epoch, clips: cls.scored.append((epoch, clips)),
        )
        cls.run_dir = cls.root / RUN
        cls.checkpoint = cls.run_dir / "best.pt"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_hook_sees_every_scoring_of_the_validation_clips(self):
        self.assertEqual([epoch for epoch, _ in self.scored], [0, 1])
        for _, clips in self.scored:
            self.assertEqual(
                [c.entry.clip_id for c in clips], ["data/video_face_2_part_1"]
            )
            clip = clips[0]
            self.assertEqual(len(clip.times), len(clip.signal))
            self.assertEqual(len(clip.truth), len(clip.onset))
            self.assertEqual(clip.span, (0.0, 1.0))

    def test_the_hook_holds_the_predictions_best_pt_was_chosen_on(self):
        history = json.loads((self.run_dir / "history.json").read_text())
        best = [r["epoch"] for r in history if r.get("since_best") == 0][-1]
        chosen = dict(self.scored)[best][0].scores
        saved = json.loads((self.run_dir / "best_per_clip.json").read_text())[0]
        self.assertAlmostEqual(chosen["correlation"], saved["correlation"])

    def test_a_run_with_the_hook_trains_as_one_without(self):
        out = self.root / "runs" / "no-hook"
        run.main([str(self.experiment), "--output-dir", str(out)])
        keys = ("train_loss", "val_loss", "xcorr", "ema_xcorr")
        ours = json.loads((self.run_dir / "history.json").read_text())
        theirs = json.loads((out / "fold" / "seed-1" / "history.json").read_text())
        for a, b in zip(ours, theirs, strict=True):
            for key in keys:
                self.assertAlmostEqual(a[key], b[key], places=5)

    def test_snippets_are_a_window_of_the_scored_span(self):
        payload = snippets(0, self.scored[0][1], n_clips=4, seconds=5.0)
        self.assertEqual(payload["epoch"], 0)
        (clip,) = payload["clips"]
        self.assertEqual(len(clip["time"]), 300)
        self.assertEqual(len(clip["truth"]), len(clip["signal"]))
        times = self.scored[0][1][0].times
        mid = (times[0] + times[-1]) / 2
        self.assertLess(abs(np.mean(clip["time"]) - mid), 0.1)
        inside = all(
            clip["time"][0] <= t <= clip["time"][-1] for t in clip["truth_inhale"]
        )
        self.assertTrue(inside)
        self.assertGreater(len(clip["truth_inhale"]), 10)  # 3 Hz over 5 s
        json.dumps(payload)

    def test_snippets_take_one_clip_per_recording_and_keep_inside_a_tail(self):
        (clip,) = self.scored[0][1]
        tail = clip.__class__(**{**clip.__dict__, "span": (0.75, 1.0)})
        payload = snippets(0, [tail, tail], n_clips=4, seconds=30.0)
        self.assertEqual(len(payload["clips"]), 1)
        start = payload["clips"][0]["time"][0]
        self.assertGreaterEqual(start, clip.times[int(len(clip.times) * 0.75)])

    def test_predict_writes_a_trace_per_video_without_a_cache(self):
        out = self.root / "predictions" / "cold"
        main(
            [
                "--checkpoint",
                str(self.checkpoint),
                "--clips",
                str(self.list),
                "--out",
                str(out),
                "--device",
                "cpu",
                "--amp",
                "off",
                "--workers",
                "1",
            ]
        )
        written = sorted(p.name for p in out.glob(f"*{SUFFIX}"))
        self.assertEqual(len(written), 3)
        table = pd.read_parquet(out / f"video_face_1_part_1{SUFFIX}")
        self.assertEqual(list(table.columns), ["time", "signal", "onset", "inhale"])
        self.assertEqual(len(table), 1200)
        self.assertAlmostEqual(float(table["signal"].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(table["signal"].std(ddof=0)), 1.0, places=3)
        self.assertTrue((out / SETTINGS).is_file())

    def test_predict_needs_the_checkpoints_recipe(self):
        other = self.root / "other.toml"
        other.write_text(
            self.list.read_text().replace(
                "[preprocess]", "[preprocess]\nselect_fs = 30"
            )
        )
        with self.assertRaisesRegex(SystemExit, "select_fs 30.0 \\(trained with 60"):
            main(
                [
                    "--checkpoint",
                    str(self.checkpoint),
                    "--clips",
                    str(other),
                    "--out",
                    str(self.root / "predictions" / "x"),
                    "--device",
                    "cpu",
                ]
            )

    def test_a_folder_of_another_checkpoints_predictions_is_refused(self):
        out = self.root / "predictions" / "theirs"
        out.mkdir(parents=True)
        (out / SETTINGS).write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
        with self.assertRaisesRegex(SystemExit, "another checkpoint"):
            main(
                [
                    "--checkpoint",
                    str(self.checkpoint),
                    "--clips",
                    str(self.list),
                    "--out",
                    str(out),
                    "--device",
                    "cpu",
                    "--amp",
                    "off",
                ]
            )

    def test_a_missing_thermistor_is_never_needed(self):
        clip_list = load(ClipList, self.list, check_files=False)
        clips = select_clips([clip_list], ["3"])
        self.assertEqual([c.group for c in clips], ["3"])
        absent = clip_list.model_copy()
        moved = self.root / "data" / "thermistor_3_part_1.parquet"
        moved.rename(moved.with_suffix(".away"))
        try:
            (clip,) = select_clips([absent], ["3"])
            self.assertIsNone(clip.thermistor)
        finally:
            moved.with_suffix(".away").rename(moved)
        with self.assertRaisesRegex(SystemExit, "in none of the clip lists"):
            select_clips([clip_list], ["9"])


if __name__ == "__main__":
    unittest.main()
