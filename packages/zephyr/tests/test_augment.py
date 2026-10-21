import unittest

import numpy as np

from zephyr.augment import AugmentConfig, apply_spatial, blur, zoom


def _dot(size=21, at=(10, 14)) -> np.ndarray:
    """One frame, one channel: a bright dot on a dark field."""
    block = np.zeros((1, 1, size, size), np.float32)
    block[0, 0, at[0], at[1]] = 255.0
    return block


def _centre(block: np.ndarray) -> tuple[float, float]:
    image = block[0, 0]
    ys, xs = np.indices(image.shape)
    return float((ys * image).sum() / image.sum()), float(
        (xs * image).sum() / image.sum()
    )


class ZoomTests(unittest.TestCase):
    def test_a_factor_of_one_leaves_the_block_alone(self):
        block = (
            np.random.default_rng(0).uniform(0, 255, (3, 2, 21, 21)).astype(np.float32)
        )
        np.testing.assert_allclose(zoom(block, 1.0), block, atol=1e-3)

    def test_zoom_moves_a_dot_away_from_the_centre(self):
        y, x = _centre(zoom(_dot(at=(10, 14)), 1.5))
        self.assertAlmostEqual(y, 10.0, places=1)
        self.assertAlmostEqual(x, 16.0, places=1)

    def test_edges_are_repeated_not_zero(self):
        block = np.full((1, 1, 21, 21), 90.0, np.float32)
        np.testing.assert_allclose(zoom(block, 0.9), block, atol=1e-3)


class RotationTests(unittest.TestCase):
    def test_a_quarter_turn_moves_a_dot_round_the_centre(self):
        # 4 px right of the centre (10, 10); cv2 turns counter-clockwise on screen
        y, x = _centre(zoom(_dot(at=(10, 14)), 1.0, angle=90.0))
        self.assertAlmostEqual(y, 6.0, places=1)
        self.assertAlmostEqual(x, 10.0, places=1)


class BlurTests(unittest.TestCase):
    def test_blur_keeps_the_mean_and_spreads_a_dot(self):
        frames = _dot()[:, 0]
        out = blur(frames, 1.0)
        self.assertAlmostEqual(float(out.sum()), 255.0, places=2)
        self.assertLess(float(out.max()), 255.0 / 4)
        np.testing.assert_array_equal(blur(frames, 0.0), frames)

    def test_apply_spatial_blurs_the_gray_frames(self):
        block = (
            np.random.default_rng(3).uniform(0, 255, (2, 1, 21, 21)).astype(np.float32)
        )
        out = apply_spatial(
            block.copy(), np.random.default_rng(4), AugmentConfig(blur=2.0)
        )
        self.assertLess(float(out.std()), float(block.std()))


class ApplySpatialTests(unittest.TestCase):
    def _block(self) -> np.ndarray:
        return (
            np.random.default_rng(1).uniform(0, 255, (4, 1, 21, 21)).astype(np.float32)
        )

    def test_off_by_default_and_draws_nothing_extra(self):
        base = AugmentConfig(shift_px=4, noise=2.0)
        again = AugmentConfig(shift_px=4, noise=2.0, scale=0.0, rotation=0.0, blur=0.0)
        a = apply_spatial(self._block(), np.random.default_rng(5), base)
        b = apply_spatial(self._block(), np.random.default_rng(5), again)
        np.testing.assert_array_equal(a, b)

    def test_enabled_by_scale_alone(self):
        self.assertFalse(AugmentConfig().enabled)
        self.assertTrue(AugmentConfig(scale=0.1).enabled)
        self.assertTrue(AugmentConfig(rotation=5.0).enabled)
        self.assertTrue(AugmentConfig(blur=0.5).enabled)

    def test_scale_changes_the_block_and_keeps_its_shape(self):
        out = apply_spatial(
            self._block(), np.random.default_rng(2), AugmentConfig(scale=0.2)
        )
        self.assertEqual(out.shape, (4, 1, 21, 21))
        self.assertFalse(np.allclose(out, self._block()))


if __name__ == "__main__":
    unittest.main()
