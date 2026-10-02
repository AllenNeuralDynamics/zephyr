import unittest

import numpy as np

from zephyr.baselines import pixel

FS = 60.0
T = np.arange(1800) / FS
BREATH = np.sin(2 * np.pi * 3.0 * T)


def video(rng):
    frames = 100 + rng.normal(0, 1.0, (len(T), 16, 16))
    frames[:, 4:8, 4:8] += 10 * BREATH[:, None, None]
    return frames.astype(np.float32)


class PixelTests(unittest.TestCase):
    def test_pixel_pca_finds_breathing_component(self):
        trace = pixel.pixel_pca(video(np.random.default_rng(0)), FS)
        self.assertGreater(abs(np.corrcoef(trace, BREATH)[0, 1]), 0.9)

    def test_snr_weighted_finds_breathing_pixels(self):
        trace = pixel.snr_weighted(video(np.random.default_rng(1)), FS)
        self.assertGreater(abs(np.corrcoef(trace, BREATH)[0, 1]), 0.9)

    def test_flow_projection_integrates_velocity(self):
        rng = np.random.default_rng(2)
        velocity = np.cos(2 * np.pi * 3.0 * T)  # displacement ~ sin
        flow_x = 0.6 * velocity[:, None, None] + rng.normal(0, 0.05, (len(T), 8, 8))
        flow_y = 0.8 * velocity[:, None, None] + rng.normal(0, 0.05, (len(T), 8, 8))
        trace = pixel.flow_projection(flow_x, flow_y, FS)
        self.assertGreater(abs(np.corrcoef(trace, BREATH)[0, 1]), 0.9)

    def test_outputs_have_one_sample_per_frame(self):
        frames = video(np.random.default_rng(3))
        self.assertEqual(pixel.pixel_pca(frames, FS).shape, (len(T),))
        self.assertEqual(pixel.snr_weighted(frames, FS).shape, (len(T),))
