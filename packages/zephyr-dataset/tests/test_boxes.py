"""``boxes``: template-matched boxes for the face clips, a hand-placed one, and publishing."""

import io
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import tomlkit
from PIL import Image

from zephyr.annotate import frame_path
from zephyr.config import ClipList, load
from zephyr.corpus import boxes, rois, storage
from zephyr.corpus.config import DatasetSpec, Layout

PATCH = (520, 260, 590, 325)  # where the "snout" is in the thumbnails that have one


def thumbnail(snout: bool, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60, (540, 720), dtype=np.uint8)
    if snout:
        x0, y0, x1, y1 = PATCH
        image[y0:y1, x0:x1] = np.random.default_rng(99).integers(
            0, 255, (y1 - y0, x1 - x0)
        )
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class BoxTests(unittest.TestCase):
    def setUp(self):
        self.layout = Layout(f"memory://boxes-{uuid.uuid4().hex[:8]}")
        self.spec = DatasetSpec.model_construct(root=str(self.layout.root))
        self.sessions = {"a": True, "b": True, "c": False}
        doc = tomlkit.document()
        doc.add("preprocess", {"target_size": [360, 270]})
        doc.add("events", {"method": "peak_detection"})
        entries = tomlkit.aot()
        for i, (sid, snout) in enumerate(self.sessions.items()):
            for camera in ("FaceCamera", "SideCamera"):
                storage.write_bytes(
                    self.layout.thumbnail(sid, camera),
                    thumbnail(snout and camera[0] == "F", i),
                )
                for span in ("span1", "span2"):
                    stem = f"clips/{sid}_{span}_{camera}"
                    entries.append(
                        {
                            "video": f"{stem}.mp4",
                            "timestamps": f"{stem}.parquet",
                            "thermistor": f"clips/{sid}_{span}_thermistor.parquet",
                            "group": sid,
                        }
                    )
        doc.add("clip", entries)
        storage.write_text(self.layout.clip_list, tomlkit.dumps(doc))
        self.work = Path(tempfile.mkdtemp())
        saved = list(rois.EXAMPLES)
        rois.EXAMPLES[:] = [("a", PATCH)]  # the snout of session a
        self.addCleanup(rois.EXAMPLES.__setitem__, slice(None), saved)

    def test_match_then_hand_placed_then_published(self):
        boxes.prepare(self.spec, self.work, "FaceCamera", 96, 0.58)
        face = load(ClipList, self.work / "face.toml", check_files=False).resolve()
        self.assertEqual(len(face), 6)  # the face clips only
        self.assertTrue(all("FaceCamera" in c.video.name for c in face))
        by_session = {}
        for c in face:
            by_session.setdefault(c.group, []).append(c.box)
        self.assertEqual(by_session["c"], [None, None])  # nothing to match: for the UI
        for sid in "ab":  # both spans share the box, centred on the snout
            self.assertEqual(by_session[sid][0], by_session[sid][1])
            x, y, w, h = by_session[sid][0]
            self.assertEqual((w, h), (96, 96))
            self.assertAlmostEqual(x + 48, (PATCH[0] + PATCH[2]) / 4, delta=2)
            self.assertAlmostEqual(y + 48, (PATCH[1] + PATCH[3]) / 4, delta=2)
        for c in face:  # a frame per clip for the UI, at the thumbnail's size
            self.assertEqual(
                np.load(frame_path(self.work / "frames", c.video)).shape, (540, 720)
            )
        # a box placed by hand in the UI, then published into the dataset's clips.toml
        write = tomlkit.parse((self.work / "face.toml").read_text(encoding="utf-8"))
        for e in write["clip"]:
            if e["group"] == "c":
                e["box"] = [10, 20, 96, 96]
        (self.work / "face.toml").write_text(tomlkit.dumps(write), encoding="utf-8")
        boxes.publish(self.spec, self.work, "FaceCamera")
        listed = tomlkit.parse(self.layout.clip_list.read_text(encoding="utf-8"))[
            "clip"
        ]
        boxed = {
            (e["group"], "FaceCamera" in e["video"]): list(e["box"])
            for e in listed
            if "box" in e
        }
        self.assertEqual(boxed[("c", True)], [10, 20, 96, 96])
        self.assertEqual(
            len([e for e in listed if "box" in e]), 6
        )  # never the side camera
        self.assertFalse(any("box" in e for e in listed if "SideCamera" in e["video"]))

    def test_a_published_box_is_not_overwritten_by_the_match(self):
        boxes.prepare(self.spec, self.work, "FaceCamera", 96, 0.58)
        write = tomlkit.parse((self.work / "face.toml").read_text(encoding="utf-8"))
        for e in write["clip"]:
            if e["group"] == "a":
                e["box"] = [1, 2, 96, 96]
        (self.work / "face.toml").write_text(tomlkit.dumps(write), encoding="utf-8")
        boxes.publish(self.spec, self.work, "FaceCamera")
        boxes.prepare(self.spec, self.work, "FaceCamera", 96, 0.58)  # a re-run
        face = load(ClipList, self.work / "face.toml", check_files=False).resolve()
        self.assertEqual({c.box for c in face if c.group == "a"}, {(1, 2, 96, 96)})


if __name__ == "__main__":
    unittest.main()
