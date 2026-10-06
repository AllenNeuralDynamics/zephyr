import argparse
import unittest

from zephyr.run import choose_machine


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

    def test_bare_fold_has_no_experiment_settings(self):
        self.assertEqual(choose_machine(_args(device="cpu"), {}).num_workers, 4)


if __name__ == "__main__":
    unittest.main()
