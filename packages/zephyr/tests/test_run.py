import argparse
import unittest
from pathlib import Path

from zephyr.config import TrainParams
from zephyr.run import choose_machine
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


if __name__ == "__main__":
    unittest.main()
