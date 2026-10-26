"""Training's ``on_scored`` hook, ``zephyr predict``, and the validation snippets the
app cuts from the hook, on a tiny real run (ffmpeg, CPU, a few optimiser steps)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from _synthetic import HAS_FFMPEG, make_dataset, write_experiment

from zephyr import run
from zephyr.annotate import open_state, prepare_frames, save_boxes
from zephyr.clips import main as scan_main
from zephyr.config import ClipList, TrainParams, load
from zephyr.evaluate import evaluate_groups
from zephyr.gui.worker import snippets
from zephyr.predict import (
    SETTINGS,
    SUFFIX,
    crop_size,
    main,
    read_checkpoint,
    select_clips,
)
from zephyr.preprocess import main as preprocess
from zephyr.train import settle_head_threshold

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

    def predict(self, clips: Path, out: Path) -> None:
        main(
            ["--checkpoint", str(self.checkpoint), "--clips", str(clips)]
            + ["--out", str(out), "--device", "cpu", "--amp", "off", "--workers", "1"]
        )

    def test_a_checkpoint_says_its_crop_and_predict_refuses_another(self):
        self.assertEqual(crop_size(read_checkpoint(self.checkpoint)), (32, 32))
        other = self.root / "wrong-crop.toml"
        other.write_text(
            self.list.read_text().replace("16, 8, 32, 32", "16, 8, 24, 24")
        )
        with self.assertRaisesRegex(SystemExit, "trained on 32x32 crops"):
            self.predict(other, self.root / "predictions" / "wrong")

    def test_videos_from_any_folder_are_listed_cropped_and_predicted(self):
        folder = self.root / "elsewhere"
        folder.mkdir()
        for name in ("video_face_1_part_1.mp4", "video_face_1_part_1.parquet"):
            (folder / name).write_bytes((self.root / "data" / name).read_bytes())
        out = self.root / "predictions" / "videos"
        path = out / "videos.toml"
        scan_main(
            ["scan", str(folder), "--glob", "*.mp4", "-o", str(path)]
            + ["--checkpoint", str(self.checkpoint)]
        )
        listed = load(ClipList, path)
        self.assertEqual(listed.preprocess.target_size, (64, 48))
        (clip,) = listed.resolve()
        self.assertEqual(clip.group, "video_face_1_part_1")  # each video its own
        self.assertFalse(clip.labelled)
        frames = self.root / "frames"
        prepare_frames(listed.resolve(), frames)
        state = open_state(path, frames, checkpoint=self.checkpoint)
        self.assertEqual((state.box_size, state.locked), (32, True))
        state.resize(16)
        state.set_target(32, 24)
        self.assertEqual((state.box_size, state.target_size), (32, (64, 48)))
        state.place(32, 24)
        save_boxes(path, state)
        self.predict(path, out)
        self.assertTrue((out / f"video_face_1_part_1{SUFFIX}").is_file())

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

    def test_restart_predicts_every_clip_again_even_for_another_checkpoint(self):
        out = self.root / "predictions" / "again"
        out.mkdir(parents=True)
        (out / SETTINGS).write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
        stale = out / f"video_face_1_part_1{SUFFIX}"
        stale.write_text("old")
        main(
            ["--checkpoint", str(self.checkpoint), "--clips", str(self.list)]
            + ["--out", str(out), "--device", "cpu", "--amp", "off"]
            + ["--workers", "1", "--restart", "--groups", "1"]
        )
        self.assertEqual(len(pd.read_parquet(stale)), 1200)  # predicted again
        self.assertNotEqual(
            json.loads((out / SETTINGS).read_text())["checkpoint_sha256"], "0" * 64
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


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class HeadThresholdTests(unittest.TestCase):
    """A run that sweeps the onset threshold on its validation clips, and the commands
    that read it back."""

    GRID = "head_thresholds = [0.05, 0.3, 0.6]\n"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.list = make_dataset(cls.root)
        preprocess(
            [str(cls.list), "--cache", str(cls.root / "cache"), "--workers", "1"]
        )
        plain = write_experiment(cls.root, cls.list, name="plain")
        swept = write_experiment(cls.root, cls.list, name="swept")
        swept.write_text(swept.read_text() + cls.GRID)
        headless = write_experiment(cls.root, cls.list, name="headless")
        headless.write_text(headless.read_text() + cls.GRID + "w_onset = 0.0\n")
        for experiment in (plain, swept, headless):
            run.main([str(experiment)])
        cls.plain = cls.root / "runs" / "plain" / "fold" / "seed-1"
        cls.swept = cls.root / "runs" / "swept" / "fold" / "seed-1"
        cls.headless = cls.root / "runs" / "headless" / "fold" / "seed-1"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _evaluation(self, run_dir: Path) -> dict:
        return json.loads((run_dir / "evaluation.json").read_text())

    def test_without_a_grid_nothing_is_chosen_and_the_default_scores(self):
        self.assertNotIn("head_threshold", read_checkpoint(self.plain / "best.pt"))
        self.assertEqual(
            self._evaluation(self.plain)["head_threshold"],
            {"value": 0.5, "source": "default"},
        )

    def test_the_checkpoint_keeps_the_threshold_chosen_on_validation(self):
        chosen = read_checkpoint(self.swept / "best.pt")["head_threshold"]
        self.assertIn(chosen["value"], (0.05, 0.3, 0.6))
        self.assertEqual(chosen["chosen_on"], "validation")
        self.assertEqual(chosen["val_clips"], 1)
        self.assertEqual(len(chosen["sweep"]), 3)
        self.assertEqual(
            self._evaluation(self.swept)["head_threshold"],
            {"value": chosen["value"], "source": "checkpoint"},
        )

    def test_without_an_onset_loss_no_threshold_is_chosen_or_used(self):
        self.assertNotIn("head_threshold", read_checkpoint(self.headless / "best.pt"))
        evaluation = self._evaluation(self.headless)
        self.assertIsNone(evaluation["head_threshold"])
        self.assertNotIn("head_inhale_f1", evaluation["clips"][0])

    def test_a_fold_without_validation_clips_keeps_the_default(self):
        path = self.root / "no-val.pt"
        path.write_bytes((self.swept / "best.pt").read_bytes())
        chosen = settle_head_threshold(
            path,
            None,
            TrainParams(head_thresholds=(0.3,)),
            torch.device("cpu"),
            None,
        )
        self.assertEqual((chosen["value"], chosen["chosen_on"]), (0.5, None))
        self.assertEqual(read_checkpoint(path)["head_threshold"], chosen)

    def test_a_flag_overrides_the_checkpoint_and_says_so(self):
        clip_list = load(ClipList, self.list, check_files=False)
        held_out = [c for c in clip_list.resolve() if c.group == "3"]
        result = evaluate_groups(
            [self.swept / "best.pt"],
            {"held_out": held_out},
            self.root / "cache",
            clip_list.preprocess,
            device="cpu",
            amp="off",
            window=512,
            frame_chunk=64,
            threshold=0.9,
        )
        self.assertEqual(result["head_threshold"], {"value": 0.9, "source": "flag"})

    def test_predict_records_the_threshold_and_refuses_another_in_its_folder(self):
        out = self.root / "predictions" / "swept"
        argv = ["--checkpoint", str(self.swept / "best.pt"), "--clips", str(self.list)]
        argv += ["--out", str(out), "--device", "cpu", "--amp", "off"]
        argv += ["--workers", "1", "--groups", "3"]
        main(argv)
        own = read_checkpoint(self.swept / "best.pt")["head_threshold"]["value"]
        self.assertEqual(
            json.loads((out / SETTINGS).read_text())["head_threshold"],
            {"value": own, "source": "checkpoint"},
        )
        with self.assertRaisesRegex(SystemExit, "another onset threshold"):
            main([*argv, "--head-threshold", "0.9"])


if __name__ == "__main__":
    unittest.main()
