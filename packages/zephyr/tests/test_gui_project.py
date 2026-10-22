"""The app's headless parts: the project folder and running commands as local jobs."""

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from _clipfiles import touch_clip, write_list
from _synthetic import HAS_FFMPEG, make_dataset, write_experiment

from zephyr.config import Experiment, load
from zephyr.gui.jobs import Job
from zephyr.gui.project import FOLDERS, Project


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        videos = [touch_clip(self.root / "data", g, 1) for g in (1, 2, 3, 4)]
        self.list = write_list(self.root / "clips.toml", videos)
        (self.root / "notes.toml").write_text('title = "not a clip list"\n')

    def tearDown(self):
        self._tmp.cleanup()

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
