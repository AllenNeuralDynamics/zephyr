import unittest

import numpy as np

from zephyr.benchmarks import pixel
from zephyr.benchmarks.common import bandpass

FS = 60.0
T = np.arange(1800) / FS
BREATH = np.sin(2 * np.pi * 3.0 * T)


def video(rng, noise=1.0):
    frames = 100 + rng.normal(0, noise, (len(T), 16, 16))
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

    def test_motion_energy_follows_the_speed_of_movement(self):
        # rectified: a sinusoidal breath moves fastest twice a cycle, so the energy follows
        # |d breath / dt|, not the breath itself (quiet video: frame noise under the motion)
        trace = pixel.motion_energy(video(np.random.default_rng(4), noise=0.1), FS)
        speed = np.abs(np.diff(BREATH, prepend=BREATH[0]))
        self.assertGreater(
            np.corrcoef(trace, bandpass(speed - speed.mean(), FS))[0, 1], 0.9
        )
        self.assertLess(abs(np.corrcoef(trace, BREATH)[0, 1]), 0.2)

    def test_outputs_have_one_sample_per_frame(self):
        frames = video(np.random.default_rng(3))
        self.assertEqual(pixel.pixel_pca(frames, FS).shape, (len(T),))
        self.assertEqual(pixel.snr_weighted(frames, FS).shape, (len(T),))
        self.assertEqual(pixel.motion_energy(frames, FS).shape, (len(T),))
