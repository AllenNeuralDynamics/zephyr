"""Checkpoint selection (``train.Selection``): which scoring is the best and when to stop."""

import unittest

from zephyr.train import Selection

# small-gray-sched-v2's validation xcorr, every 4 epochs: the live weights until the
# averaged ones were ready (epoch 31), then the averaged ones.
LIVE = [0.783, 0.798, 0.826, 0.815, 0.799, 0.800, 0.826]
EMA = [0.779, 0.805, 0.818, 0.823]


class SelectionTests(unittest.TestCase):
    def test_the_averaged_weights_are_not_held_to_a_live_best(self):
        s = Selection()
        for x in LIVE:
            s.offer(x, ema=False)
        self.assertEqual((s.best, s.since_best), (0.826, 4))
        self.assertEqual([s.offer(x, ema=True) for x in EMA], [True] * 4)
        self.assertEqual((s.best, s.since_best, s.on_ema), (0.823, 0, True))

    def test_averaged_scorings_after_the_first_compete_with_each_other(self):
        s = Selection()
        self.assertTrue(s.offer(0.80, ema=True))
        self.assertFalse(s.offer(0.79, ema=True))
        self.assertEqual((s.best, s.since_best), (0.80, 1))

    def test_a_resumed_selection_keeps_its_averaged_best(self):
        s = Selection(best=0.80, since_best=2, on_ema=True)
        self.assertFalse(s.offer(0.79, ema=True))
        self.assertEqual((s.best, s.since_best), (0.80, 3))

    def test_without_averaged_weights_only_the_live_best_counts(self):
        s = Selection()
        self.assertEqual(
            [s.offer(x, ema=False) for x in (0.5, 0.7, 0.6, 0.65)],
            [True, True, False, False],
        )
        self.assertEqual((s.best, s.since_best), (0.7, 2))


if __name__ == "__main__":
    unittest.main()
