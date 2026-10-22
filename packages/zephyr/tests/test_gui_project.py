"""The app's headless parts: the project folder and running commands as local jobs."""

import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from _clipfiles import touch_clip, write_list
from _synthetic import HAS_FFMPEG, make_dataset, write_experiment

from zephyr.config import Derive, Experiment, PreprocessParams, load
from zephyr.gui.jobs import Job
from zephyr.gui.project import (
    CACHE_BYTES_PER_FRAME,
    DONE,
    FOLDERS,
    PARTLY,
    TODO,
    Project,
    _scored,
    cache_need,
    human_bytes,
    is_smoke,
    preview_scan,
    run_label,
    run_status,
    split_groups,
)
from zephyr.preprocess import select_anchor_indices


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        videos = [touch_clip(self.root / "data", g, 1) for g in (1, 2, 3, 4)]
        self.list = write_list(self.root / "clips.toml", videos)
        (self.root / "notes.toml").write_text('title = "not a clip list"\n')

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_project_is_a_folder_with_a_clip_list_or_its_layout(self):
        self.assertTrue(Project.exists(self.root))  # clips.toml
        self.assertTrue(Project.exists(self.list))
        empty = self.root / "empty"
        empty.mkdir()
        self.assertFalse(Project.exists(empty))
        Project.open(empty)
        self.assertTrue(Project.exists(empty))

    def test_a_folder_gets_the_layout_and_its_clip_lists(self):
        project = Project.open(self.root)
        self.assertTrue(all((self.root / name).is_dir() for name in FOLDERS))
        self.assertEqual(project.clip_lists(), [self.list.resolve()])
        self.assertEqual(project.clip_list, self.list.resolve())
        self.assertEqual(project.groups(), ["1", "2", "3", "4"])

    def test_a_clip_list_opens_its_folder_on_that_list(self):
        other = write_list(
            self.root / "more.toml", [touch_clip(self.root / "data", 5, 1)]
        )
        project = Project.open(other)
        self.assertEqual(project.root, self.root.resolve())
        self.assertEqual(project.clip_list, other.resolve())

    def test_a_list_that_does_not_load_is_listed_with_why(self):
        broken = self.root / "broken.toml"
        broken.write_text(
            '[preprocess]\ntarget_size = [10, 10]\n[[clip]]\nvideo = "x.mp4"\nbox = [0, 0, 32, 32]\n'
        )
        project = Project.open(self.root)
        self.assertIn(broken.resolve(), project.clip_lists())
        self.assertRegex(project.errors[broken.resolve()], "cannot derive group")

    def test_a_new_experiment_is_written_validated_and_never_overwritten(self):
        project = Project.open(self.root)
        path = project.new_experiment(
            "held-out-4",
            train=["1", "2"],
            val=["3"],
            test={"test": ["4"]},
            seeds=(1, 2),
            train_params={"epochs": 5},
            machine={"device": "cpu", "amp": None},
        )
        experiment = load(Experiment, path)
        (fold,) = experiment.fold
        self.assertEqual(fold.train[0].groups, ["1", "2"])
        self.assertEqual(fold.val.groups, ["3"])
        self.assertEqual(fold.test["test"].groups, ["4"])
        self.assertEqual(fold.train_params.epochs, 5)
        self.assertEqual(
            experiment.output_dir, (self.root / "runs" / "held-out-4").resolve()
        )
        self.assertEqual(experiment.features_dir, (self.root / "cache").resolve())
        self.assertNotIn("amp", tomllib.loads(path.read_text()))
        self.assertEqual(
            project.run_dirs(path),
            [experiment.output_dir / "held-out-4" / f"seed-{s}" for s in (1, 2)],
        )
        with self.assertRaisesRegex(ValueError, "never edited"):
            project.new_experiment("held-out-4", train=["1"], test={"test": ["4"]})

    def test_an_experiment_the_fold_checks_refuse_is_not_left_behind(self):
        project = Project.open(self.root)
        with self.assertRaisesRegex(ValueError, "also a training clip"):
            project.new_experiment("leaky", train=["1", "2"], test={"test": ["2"]})
        self.assertFalse((self.root / "experiments" / "leaky.toml").exists())
        with self.assertRaisesRegex(ValueError, "plain file name"):
            project.new_experiment("a b", train=["1"], test={"test": ["2"]})

    def test_runs_are_the_folders_holding_a_history_or_checkpoint(self):
        project = Project.open(self.root)
        run = self.root / "runs" / "e" / "fold" / "seed-1"
        run.mkdir(parents=True)
        (run / "history.json").write_text("[]")
        (self.root / "runs" / "e" / "empty").mkdir()
        self.assertEqual(project.runs(), [run.resolve()])
        self.assertEqual(project.checkpoints(), [])


class StepTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.videos = [touch_clip(self.root / "data", g, 1) for g in (1, 2)]

    def states(self, project) -> dict[str, str]:
        return {title: step.state for title, step in project.steps().items()}

    def test_an_empty_project_has_everything_to_do(self):
        states = self.states(Project.open(self.root))
        self.assertEqual(states.pop("Project"), DONE)
        self.assertEqual(set(states.values()), {TODO})

    def test_boxes_and_breathing_follow_the_clip_list(self):
        path = write_list(self.root / "clips.toml", self.videos)
        text = path.read_text()
        path.write_text(text.replace("box = [10, 10, 32, 32]\n", "", 1))
        project = Project.open(self.root)
        states = self.states(project)
        self.assertEqual((states["Clips"], states["Boxes"]), (DONE, PARTLY))
        self.assertEqual(states["Breathing"], TODO)  # the built-in default, unreviewed
        self.assertIn("1 of 2 clip(s) boxed", project.steps()["Boxes"].detail)
        path.write_text(
            text.replace(
                "[[clip]]", '[events]\nmethod = "peak_detection"\n\n[[clip]]', 1
            )
        )
        self.assertEqual(self.states(project)["Boxes"], DONE)
        self.assertEqual(self.states(project)["Breathing"], PARTLY)  # named
        named = path.read_text()
        path.write_text(named.replace('.mp4"\n', '.mp4"\nreviewed = true\n'))
        self.assertEqual(self.states(project)["Breathing"], DONE)
        path.write_text(named.replace('.mp4"\n', '.mp4"\nselection = "include"\n'))
        self.assertEqual(self.states(project)["Breathing"], DONE)  # marked, unreviewed

    def test_train_results_and_predict_follow_the_folders(self):
        write_list(self.root / "clips.toml", self.videos)
        project = Project.open(self.root)
        smoke = self.root / "runs" / "e" / "smoke" / "fold" / "seed-1"
        smoke.mkdir(parents=True)
        (smoke / "best.pt").touch()
        (smoke / "evaluation.json").write_text("{}")
        states = self.states(project)
        self.assertEqual((states["Train"], states["Results"]), (TODO, TODO))
        project.new_experiment("e", train=["1"], test={"test": ["2"]})
        self.assertEqual(self.states(project)["Train"], PARTLY)
        run = self.root / "runs" / "e" / "fold" / "seed-1"
        run.mkdir(parents=True)
        (run / "best.pt").touch()
        self.assertEqual(self.states(project)["Train"], DONE)
        (run / "evaluation.json").write_text("{}")
        self.assertEqual(self.states(project)["Results"], DONE)
        out = self.root / "predictions" / "first"
        out.mkdir()
        (out / "predictions.json").write_text("{}")
        self.assertEqual(self.states(project)["Predict"], TODO)
        (out / "video.prediction.parquet").touch()
        self.assertEqual(self.states(project)["Predict"], DONE)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        videos = [touch_clip(self.root / "data", g, 1) for g in (1, 2)]
        write_list(self.root / "clips.toml", videos)
        self.project = Project.open(self.root)
        self.experiment = self.project.new_experiment(
            "e", train=["1"], test={"test": ["2"]}, seeds=(3,)
        )

    def test_a_run_is_traced_to_its_experiment_fold_seed_and_smoke(self):
        (real,) = self.project.run_dirs(self.experiment)
        (smoke,) = self.project.run_dirs(self.experiment, smoke=True)
        self.assertEqual(self.project.run_of(real), (self.experiment, "e", 3, False))
        self.assertEqual(self.project.run_of(smoke), (self.experiment, "e", 3, True))
        self.assertIsNone(self.project.run_of(self.root / "runs" / "other"))

    def test_the_status_says_how_far_a_run_got(self):
        (run,) = self.project.run_dirs(self.experiment)
        self.assertEqual(run_status(run), "")  # nothing there yet
        run.mkdir(parents=True)
        rows = [{"epoch": 0}, {"epoch": 1, "since_best": 0}, {"epoch": 2}]
        (run / "history.json").write_text(json.dumps(rows))
        fold = {"fold": {"train_params": {"epochs": 200}}}
        (run / "config.json").write_text(json.dumps(fold))
        (run / "best.pt").touch()
        self.assertEqual(
            run_status(run), "3 of 200 epochs, best.pt from epoch 2, not scored"
        )
        (run / "evaluation.json").write_text("{}")
        self.assertTrue(run_status(run).endswith(", scored"))

    def test_runs_are_named_by_fold_and_seed(self):
        (real,) = self.project.run_dirs(self.experiment)
        (smoke,) = self.project.run_dirs(self.experiment, smoke=True)
        runs = self.project.runs_dir
        self.assertEqual(run_label(real, runs), "e · seed 3")
        self.assertEqual(run_label(smoke, runs), "e · seed 3 · smoke")
        self.assertEqual(run_label(runs / "x" / "f" / "seed-1", runs), "x / f · seed 1")
        self.assertEqual(run_label(Path("/else/run"), runs), "/else/run")
        self.assertEqual((is_smoke(real, runs), is_smoke(smoke, runs)), (False, True))

    def test_a_scored_trace_comes_with_its_sensor_and_both_inhales(self):
        t = np.arange(0, 20, 1 / 60)
        signal = np.sin(2 * np.pi * 2 * t)
        trace = self.root / "feat-x.parquet"
        pd.DataFrame({"time": t + 100, "signal": signal}).to_parquet(trace)
        target = self.root / "target.parquet"
        pd.DataFrame({"signal": signal[:-10]}).to_parquet(target)
        events = self.root / "events.npz"
        np.savez(events, onset_times=[101.0, 101.5], offset_times=[101.2, 101.7])
        entry = SimpleNamespace(target=target, events=events)
        found = _scored(trace, entry)
        self.assertEqual(len(found["time"]), len(t) - 10)  # cut to the sensor's
        self.assertEqual(list(found["truth_inhale"]), [101.0, 101.5])
        self.assertGreater(len(found["inhale"]), 30)  # about 2 Hz, from DSP
        self.assertIsNone(_scored(self.root / "none.parquet", entry))
        self.assertIsNone(self.project.scored_trace(self.root / "runs" / "x", "c"))


class CacheNeedTests(unittest.TestCase):
    def test_the_space_is_frames_times_crop_pixels_of_clips_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = [touch_clip(root / "data", g, 1) for g in (1, 2, 3)]
            listed = write_list(root / "clips.toml", videos, box=(10, 10, 32, 32))
            for video in videos[:2]:  # the third's timestamps stay unreadable
                frame = pd.DataFrame({"Time": np.arange(1000) / 200})  # a 200 Hz camera
                frame.to_parquet(video.with_suffix(".parquet"))
            need = cache_need([listed], root / "cache")
            self.assertEqual((need.todo, need.cached, need.unknown), (3, 0, 1))
            kept = len(
                select_anchor_indices(
                    np.arange(1000) / 200,
                    PreprocessParams(target_size=(100, 80)).select_fs,
                )
            )
            self.assertLess(kept, 1000)  # only the frames on the select_fs grid
            self.assertEqual(need.bytes, 2 * kept * (32 * 32 + CACHE_BYTES_PER_FRAME))
        self.assertEqual(human_bytes(1.3e9), "1.3 GB")
        self.assertEqual(human_bytes(512), "512 bytes")


class PreviewTests(unittest.TestCase):
    def test_a_scan_preview_lists_sessions_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = [touch_clip(root, g, 1) for g in (1, 2)]
            (root / "video_face_other.mp4").touch()
            found = preview_scan(root, "*.mp4", Derive())
            self.assertEqual([f.video for f in found[:2]], videos)
            self.assertEqual([f.group for f in found], ["1", "2", None])
            self.assertEqual([f.timestamps for f in found], [True, True, False])
            self.assertEqual(
                found[0].sensor, (root / "thermistor_1_part_1.parquet").exists()
            )
            self.assertEqual(preview_scan(root / "missing", "*.mp4", Derive()), [])


class SplitTests(unittest.TestCase):
    GROUPS = tuple(f"s{k}" for k in range(10))

    def test_shares_are_rounded_and_every_group_lands_once(self):
        train, val, test = split_groups(self.GROUPS, (60, 20, 20), seed=3)
        self.assertEqual((len(train), len(val), len(test)), (6, 2, 2))
        self.assertEqual(sorted(train + val + test), list(self.GROUPS))
        self.assertEqual(train, sorted(train))  # in the groups' order

    def test_a_draw_is_its_seed(self):
        self.assertEqual(
            split_groups(self.GROUPS, (6, 2, 2), 1),
            split_groups(self.GROUPS, (60, 20, 20), 1),
        )
        self.assertNotEqual(
            split_groups(self.GROUPS, (60, 20, 20), 1),
            split_groups(self.GROUPS, (60, 20, 20), 2),
        )

    def test_each_side_asked_for_gets_a_group(self):
        train, val, test = split_groups(self.GROUPS[:3], (90, 5, 5))
        self.assertEqual((len(train), len(val), len(test)), (1, 1, 1))
        train, val, test = split_groups(self.GROUPS[:2], (80, 0, 20))
        self.assertEqual((len(train), val, len(test)), (1, [], 1))

    def test_too_few_groups_or_no_train_or_test_is_refused(self):
        for groups, shares in (
            (self.GROUPS[:2], (60, 20, 20)),
            (self.GROUPS, (0, 50, 50)),
            (self.GROUPS, (50, 50, 0)),
        ):
            with self.assertRaises(ValueError):
                split_groups(groups, shares)


class WithoutTheUiExtraTests(unittest.TestCase):
    def test_the_headless_parts_import_and_the_app_says_what_to_install(self):
        code = (
            "import sys; sys.modules['PySide6'] = sys.modules['pyqtgraph'] = None; "
            "import zephyr.annotate, zephyr.annotate_events, zephyr.predict; "
            "import zephyr.gui.project, zephyr.gui.jobs, zephyr.gui.worker; "
            "import zephyr.gui; zephyr.gui.main([])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("needs the `ui` extra", result.stderr)


class JobTests(unittest.TestCase):
    def test_a_command_runs_in_a_child_process_and_reports_its_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Job("clips", ["check", "missing.toml"], Path(tmp)).start()
            self.assertEqual(job.command_line(), "zephyr clips check missing.toml")
            code = job.wait(120)
            messages = job.poll()
        self.assertNotEqual(code, 0)
        self.assertEqual(messages[-1], ("exit", code))
        log = "\n".join(m for kind, m in messages if kind == "log")
        self.assertIn("missing.toml", log)
        self.assertEqual(job.poll(), [])

    def test_only_zephyr_commands_run(self):
        with self.assertRaisesRegex(ValueError, "does not run"):
            Job("rm", ["-rf", "/"], Path("."))

    def test_cancel_stops_the_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Job("preprocess", ["--help"], Path(tmp))
            job.start()
            job.cancel()
            job.wait(120)
            self.assertFalse(job.running)
        self.assertIsNotNone(job.returncode)

    @unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
    def test_a_run_started_as_a_job_sends_its_snippets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = make_dataset(root)
            experiment = write_experiment(root, clips)
            prep = Job("preprocess", [str(clips), "--cache", "cache"], root).start()
            self.assertEqual(prep.wait(300), 0, prep.poll())
            job = Job(
                "run", [str(experiment)], root, snippets={"clips": 2, "seconds": 4}
            ).start()
            self.assertEqual(job.wait(600), 0)
            messages = job.poll()
        sent = [m for kind, m in messages if kind == "snippets"]
        self.assertEqual([s["epoch"] for s in sent], [0, 1])
        self.assertEqual(len(sent[0]["clips"][0]["time"]), 240)
        self.assertTrue(any("best held-out" in m for k, m in messages if k == "log"))


if __name__ == "__main__":
    unittest.main()
