import unittest

import numpy as np

from zephyr.dataset import mixture_weights


class MixtureWeightsTests(unittest.TestCase):
    def test_one_list_is_proportional_to_offsets(self):
        offsets = np.array([300.0, 100.0, 50.0])
        np.testing.assert_array_equal(
            mixture_weights(offsets, [0, 0, 0], [1.0]), offsets / offsets.sum()
        )

    def test_lists_get_their_share_and_clips_stay_proportional_within(self):
        offsets = np.array([300.0, 100.0, 50.0, 50.0])
        weights = mixture_weights(offsets, [0, 0, 1, 2], [0.5, 0.25, 0.25])
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[:2].sum(), 0.5)
        self.assertAlmostEqual(weights[0] / weights[1], 3.0)
        self.assertAlmostEqual(weights[2], 0.25)
        self.assertAlmostEqual(weights[3], 0.25)

    def test_weights_are_relative(self):
        offsets = np.array([1.0, 1.0, 1.0])
        a = mixture_weights(offsets, [0, 1, 2], [1, 1, 2])
        b = mixture_weights(offsets, [0, 1, 2], [0.25, 0.25, 0.5])
        np.testing.assert_allclose(a, b)

    def test_a_list_with_no_usable_clip_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no clip long enough"):
            mixture_weights(np.array([1.0, 1.0]), [0, 0], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
