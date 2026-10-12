"""``rois``: the size prior, the ROI square, and the helper with a stand-in detector."""

import io
import unittest
import uuid

import numpy as np
from PIL import Image

from zephyr.corpus import rois, storage
from zephyr.corpus.config import Layout

GATE = rois.Gate((100, 200), (80, 160), (0, 720), (0, 540), 0.1)


def boxes(*rows):
    return np.array(rows, dtype=float)


class PickTests(unittest.TestCase):
    def test_a_box_of_the_wrong_size_loses_to_a_worse_one_of_the_right_size(self):
        found = boxes([0, 0, 600, 500], [300, 200, 440, 300], [10, 10, 40, 40])
        box, score = rois.pick(found, np.array([0.9, 0.3, 0.8]), GATE)
        self.assertEqual(list(box), [300, 200, 440, 300])
        self.assertEqual(score, 0.3)

    def test_nothing_qualifying_is_none(self):
        found = boxes([300, 200, 440, 300])
        self.assertIsNone(rois.pick(found, np.array([0.05]), GATE))
        small = boxes([0, 0, 10, 10])
        self.assertIsNone(rois.pick(small, np.array([0.9]), GATE))

    def test_a_box_outside_the_centre_window_is_none(self):
        right = rois.Gate((100, 200), (80, 160), (400, 700), (200, 400), 0.1)
        ear = boxes([20, 200, 160, 300])
        self.assertIsNone(rois.pick(ear, np.array([0.9]), right))
        self.assertIsNotNone(
            rois.pick(boxes([420, 220, 560, 320]), np.array([0.9]), right)
        )

    def test_the_square_is_offset_and_kept_inside_the_image(self):
        shape = (720, 540)
        self.assertEqual(
            rois.square([300, 200, 400, 300], 96, (10, -5), shape), (312, 197)
        )
        self.assertEqual(rois.square([0, 0, 100, 100], 96, (10, -5), shape), (12, 0))
        self.assertEqual(
            rois.square([650, 480, 750, 580], 96, (0, 0), shape), (624, 444)
        )
        with self.assertRaises(ValueError):
            rois.square([0, 0, 1, 1], 96, (0, 0), (50, 540))


class TemplateTests(unittest.TestCase):
    def test_the_example_is_found_where_it_was_pasted(self):
        rng = np.random.default_rng(0)
        patch = rng.integers(0, 255, (30, 40), dtype=np.uint8)
        scene = rng.integers(0, 60, (200, 300), dtype=np.uint8)
        scene[90:120, 150:190] = patch
        found = rois.Template([Image.fromarray(patch)], blur=0)(
            [Image.fromarray(scene)]
        )
        box, score = rois.pick(
            *found[0], rois.Gate((30, 50), (20, 40), (0, 300), (0, 200), 0)
        )
        self.assertEqual(list(box), [150, 90, 190, 120])
        self.assertGreater(score, 0.99)


def png(shade: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (720, 540), shade).save(buffer, format="PNG")
    return buffer.getvalue()


class ProposeTests(unittest.TestCase):
    def setUp(self):
        self.layout = Layout(f"memory://rois-{uuid.uuid4().hex[:8]}")
        for i, name in enumerate(["a_1", "b_2", "c_3"]):
            storage.write_bytes(self.layout.thumbnail(name, "FaceCamera"), png(40 * i))
        storage.write_bytes(self.layout.thumbnail("a_1", "SideCamera"), png(0))

    @staticmethod
    def detector(images):
        # a snout-sized box in every image but the dark one
        return [
            (
                boxes([300, 200, 440, 300]),
                np.array([0.5 if np.asarray(im).mean() else 0.01]),
            )
            for im in images
        ]

    def propose(self, **kwargs):
        args = {"camera": "FaceCamera", "gate": GATE, "offset": (10, 0), "size": 96}
        args |= kwargs
        return rois.propose(self.layout, self.detector, **args)

    def test_one_row_per_thumbnail_of_the_camera(self):
        table, images = self.propose()
        self.assertEqual(list(table["session_id"]), ["a_1", "b_2", "c_3"])
        self.assertEqual(list(table["status"]), ["none", "ok", "ok"])
        self.assertEqual(table.loc[1, ["x0", "y0"]].tolist(), [332, 202])
        self.assertEqual(set(images), {"a_1", "b_2", "c_3"})

    def test_a_sample_is_random_but_repeatable(self):
        first, _ = self.propose(sample=2, seed=1)
        again, _ = self.propose(sample=2, seed=1)
        self.assertEqual(len(first), 2)
        self.assertEqual(list(first["session_id"]), list(again["session_id"]))

    def test_the_preview_is_written(self):
        import tempfile
        from pathlib import Path

        table, images = self.propose()
        with tempfile.TemporaryDirectory() as folder:
            rois.preview(images, table, 96, Path(folder) / "p.png")
            self.assertTrue((Path(folder) / "p.png").stat().st_size > 0)

    def test_skipped_sessions_are_left_out(self):
        table, _ = self.propose(skip={"b_2"})
        self.assertEqual(list(table["session_id"]), ["a_1", "c_3"])

    def test_no_thumbnails_stops(self):
        empty = Layout(f"memory://rois-{uuid.uuid4().hex[:8]}")
        with self.assertRaises(SystemExit):
            rois.propose(empty, self.detector, "FaceCamera", GATE, (0, 0), 96)


if __name__ == "__main__":
    unittest.main()
