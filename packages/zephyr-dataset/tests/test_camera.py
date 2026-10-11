import unittest

import numpy as np
from _corpusfakes import camera_frame

from zephyr.corpus import camera


class CameraTests(unittest.TestCase):
    def test_a_clean_camera_measures_clean(self):
        measured = camera.measure(camera_frame())
        self.assertAlmostEqual(measured["rate_hz"], 240.0, places=3)
        self.assertTrue(camera.clean(measured, video_frames=2400))
        self.assertFalse(camera.clean(measured, video_frames=2401))

    def test_the_rate_is_measured_not_assumed(self):
        for hz in (30.0, 60.0, 240.0):
            measured = camera.measure(camera_frame(hz=hz))
            self.assertAlmostEqual(measured["rate_hz"], hz, places=3)
            self.assertTrue(camera.clean(measured))

    def test_a_dropped_frame_is_a_drop_and_a_jump_in_both_clocks(self):
        for hz in (60.0, 240.0):
            measured = camera.measure(camera_frame(drop=10, hz=hz))
            self.assertEqual(measured["dropped_frames"], 1)
            self.assertEqual(measured["ref_jumps"], 1)
            self.assertEqual(measured["cam_jumps"], 1)
            self.assertFalse(camera.clean(measured))

    def test_a_spike_and_rollback_is_repaired_both_rows(self):
        for hz in (60.0, 240.0):
            broken = camera_frame(rollback=300, hz=hz)
            before = camera.measure(broken)
            self.assertEqual(before["ref_backwards"], 1)
            self.assertGreater(before["max_jitter_s"], camera.CLOCK_JITTER_S)
            fixed, repaired = camera.fix_rollbacks(broken)
            self.assertEqual(repaired, 2)
            np.testing.assert_allclose(
                fixed["ReferenceTime"], camera_frame(hz=hz)["ReferenceTime"], atol=1e-9
            )
            self.assertTrue(camera.clean(camera.measure(fixed)))

    def test_nothing_to_repair_returns_the_same_values(self):
        frame = camera_frame()
        fixed, repaired = camera.fix_rollbacks(frame)
        self.assertEqual(repaired, 0)
        self.assertTrue(fixed.equals(frame))


if __name__ == "__main__":
    unittest.main()
