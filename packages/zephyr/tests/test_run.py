import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from zephyr import run
from zephyr.config import Experiment, TrainParams, load, unchecked_files
from zephyr.evaluate import trace_name, write_trace
from zephyr.run import choose_machine, needs_training, run_fold
from zephyr.train import snapshot_path


def _args(**flags) -> argparse.Namespace:
    return argparse.Namespace(
        **{"device": None, "amp": None, "num_workers": None} | flags
    )


class ChooseMachineTests(unittest.TestCase):
    def test_settings_an_experiment_leaves_out_fall_back_to_defaults(self):
        # An experiment without num_workers loads it as None, not as a missing key.
        machine = choose_machine(
            _args(), {"device": "cpu", "amp": None, "num_workers": None}
        )
        self.assertEqual(machine.num_workers, 4)
        self.assertEqual(machine.amp, "bf16")
        self.assertEqual(machine.device, "cpu")

    def test_flag_beats_experiment_beats_default(self):
        experiment = {"device": "cpu", "amp": "fp16", "num_workers": 2}
        self.assertEqual(choose_machine(_args(), experiment).num_workers, 2)
        flagged = choose_machine(_args(num_workers=0, amp="off"), experiment)
        self.assertEqual(flagged.num_workers, 0)
        self.assertEqual(flagged.amp, "off")

    def test_no_settings_at_all_fall_back_to_defaults(self):
        self.assertEqual(choose_machine(_args(device="cpu"), {}).num_workers, 4)


class SnapshotTests(unittest.TestCase):
    def test_every_nth_epoch_is_kept_under_its_one_based_number(self):
        run = Path("run")
        kept = [snapshot_path(run, e, 50) for e in range(200)]
        self.assertEqual(
            [p.name for p in kept if p is not None],
            ["epoch-050.pt", "epoch-100.pt", "epoch-150.pt", "epoch-200.pt"],
        )

    def test_off_by_default(self):
        self.assertEqual(TrainParams().save_every, 0)
        self.assertIsNone(snapshot_path(Path("run"), 49, 0))


class NeedsTrainingTests(unittest.TestCase):
    def _run(self, d: str, *names: str) -> Path:
        for name in names:
            (Path(d) / name).touch()
        return Path(d)

    def test_a_new_run_trains(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(needs_training(Path(d), resume=False))

    def test_a_run_stopped_part_way_continues_only_when_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            run = self._run(d, "best.pt", "last.pt")
            self.assertTrue(needs_training(run, resume=True))
            self.assertFalse(needs_training(run, resume=False))

    def test_a_finished_run_is_only_scored_again(self):
        with tempfile.TemporaryDirectory() as d:
            run = self._run(d, "best.pt", "last.pt", "evaluation.json")
            self.assertFalse(needs_training(run, resume=True))


class AgainTests(unittest.TestCase):
    """``--restart`` and ``--evaluate-only``, with training and scoring mocked."""

    def run_fold(self, run_dir: Path, **flags):
        fold = mock.MagicMock(init_from=None)
        with (
            mock.patch("zephyr.run.train_fold") as train,
            mock.patch("zephyr.run.evaluate_groups") as score,
        ):
            run_fold(
                fold,
                1,
                run_dir,
                run_dir,
                mock.MagicMock(),
                runner=mock.MagicMock(),
                **flags,
            )
        return train.called, score.called

    def test_restart_deletes_the_run_and_trains_it_again(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "seed-1"
            run.mkdir()
            for name in ("best.pt", "last.pt", "evaluation.json"):
                (run / name).touch()
            self.assertEqual(self.run_fold(run), (False, True))  # finished: scored
            self.assertEqual(self.run_fold(run, restart=True), (True, True))
            self.assertFalse((run / "evaluation.json").exists())

    def test_evaluate_only_scores_a_stopped_run_and_never_trains(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with self.assertRaisesRegex(SystemExit, "no best.pt"):
                self.run_fold(run, evaluate_only=True)
            (run / "best.pt").touch()
            (run / "last.pt").touch()
            self.assertEqual(self.run_fold(run, evaluate_only=True), (False, True))


class TraceTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(Experiment.model_fields["save_traces"].default)

    def test_a_trace_is_written_under_the_entry_prefix_and_reads_back(self):
        class Entry:
            stem = Path("cache/clip_span1_FaceCamera-0123456789")

        times = np.arange(5) / 60
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "traces"
            path = write_trace(folder, Entry(), times, times * 2, None)
            self.assertEqual(path.name, trace_name(Entry()))
            self.assertEqual(path.name, "feat-clip_span1_FaceCamera-0123456789.parquet")
            saved = pd.read_parquet(path)
            self.assertEqual(list(saved.columns), ["time", "signal"])  # no head
            np.testing.assert_array_equal(saved["time"], times)
            np.testing.assert_allclose(saved["signal"], times * 2, rtol=1e-6)
            self.assertEqual(saved["signal"].dtype, np.float32)
            onset = np.linspace(0, 1, 5)
            saved = pd.read_parquet(write_trace(folder, Entry(), times, times, onset))
            np.testing.assert_allclose(saved["onset"], onset, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()


def _remote_experiment(root: Path) -> Path:
    """An experiment over a list whose videos are not on this machine."""
    clips = "\n".join(
        f'[[clip]]\nvideo = "clips/v{g}.mp4"\ntimestamps = "clips/v{g}.parquet"\n'
        f'thermistor = "clips/t{g}.parquet"\ngroup = "{g}"\nbox = [0, 0, 32, 32]\n'
        for g in (1, 2)
    )
    (root / "clips.toml").write_text(f"[preprocess]\ntarget_size = [64, 48]\n\n{clips}")
    path = root / "experiment.toml"
    path.write_text(
        'seeds = [1]\noutput_dir = "runs"\nfeatures_dir = "cache"\n'
        '[[fold]]\nname = "fold"\n[[fold.train]]\nclips = "clips.toml"\n'
        'groups = ["1"]\n[fold.test]\nheld_out = { clips = "clips.toml", '
        'groups = ["2"] }\n'
    )
    return path


class UncheckedFilesTests(unittest.TestCase):
    def test_lists_load_without_their_videos_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _remote_experiment(Path(tmp))
            with self.assertRaisesRegex(ValueError, "do not exist"):
                load(Experiment, path)
            with unchecked_files():
                (fold,) = load(Experiment, path).fold
                self.assertEqual([c.group for c in fold.train_clips()[0]], ["1"])
            with self.assertRaisesRegex(ValueError, "do not exist"):
                fold.train_clips()

    def test_run_still_refuses_clips_missing_from_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _remote_experiment(Path(tmp))
            with self.assertRaisesRegex(SystemExit, "not in the feature cache"):
                run.main([str(path), "--no-check-files", "--device", "cpu"])
