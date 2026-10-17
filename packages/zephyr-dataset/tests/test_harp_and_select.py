import unittest
from dataclasses import fields
from itertools import pairwise
from pathlib import Path

import numpy as np
from _corpusfakes import harp_bytes

from zephyr.config import load
from zephyr.corpus import harp
from zephyr.corpus.config import DatasetSpec, Qc
from zephyr.corpus.select import draw_clips
from zephyr.session_selection import Settings

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class HarpTests(unittest.TestCase):
    def test_messages_round_trip(self):
        times = 1000 + np.arange(5) * 0.004
        frame = harp.read_messages(harp_bytes(times, [1, 2, 3, 4, 5]))
        np.testing.assert_allclose(frame["Time"], times, atol=32e-6)
        self.assertEqual(frame["value"].tolist(), [1, 2, 3, 4, 5])

    def test_register_files_are_found_by_address_in_natural_order(self):
        names = ["d/Dev_32_10.bin", "d/Dev_320.bin", "d/Dev_32_2.bin", "d/Dev_32.bin"]
        self.assertEqual(
            harp.chunks(names, "Dev", 32),
            ["d/Dev_32.bin", "d/Dev_32_2.bin", "d/Dev_32_10.bin"],
        )

    def test_write_times_are_the_writes_only(self):
        times = [10.0, 10.5, 11.0]
        data = b"".join(
            harp_bytes([t], [0], harp.OPERATION_CONTROL, kind)
            for t, kind in zip(times, [1, 2, 0x08 | 2])  # read, write, failed write
        )
        np.testing.assert_allclose(harp.write_times(data), [10.5, 11.0], atol=32e-6)
        self.assertEqual(len(harp.write_times(b"")), 0)

    def test_address_comes_from_device_yml(self):
        yml = "registers:\n  RawVoltage: {address: 32, type: U16}\n"
        self.assertEqual(harp.address(yml, "RawVoltage"), 32)
        with self.assertRaises(KeyError):
            harp.address(yml, "Missing")


class DrawClipsTests(unittest.TestCase):
    def test_clips_do_not_overlap_and_lie_in_the_spans(self):
        spans = [(0.0, 400.0), (500.0, 1400.0)]
        for sid in ("a", "b", "c", "d"):
            clips = draw_clips(sid, spans, 300.0, 3, seed=1)
            starts = sorted(c["start"] for c in clips.values())
            self.assertEqual(list(clips), ["span1", "span2", "span3"])
            self.assertTrue(all(b - a >= 300.0 for a, b in pairwise(starts)))
            for c in clips.values():
                self.assertTrue(
                    any(a <= c["start"] and c["end"] <= b for a, b in spans)
                )

    def test_the_draw_is_seeded_by_session(self):
        spans = [(0.0, 2000.0)]
        self.assertEqual(
            draw_clips("x", spans, 300.0, 2, 7), draw_clips("x", spans, 300.0, 2, 7)
        )
        self.assertNotEqual(
            draw_clips("x", spans, 300.0, 2, 7), draw_clips("y", spans, 300.0, 2, 7)
        )

    def test_none_when_the_clips_do_not_fit(self):
        self.assertIsNone(draw_clips("x", [(0.0, 500.0)], 300.0, 2, 1))
        self.assertIsNone(draw_clips("x", [(0.0, 200.0)], 300.0, 1, 1))


class SpecTests(unittest.TestCase):
    def test_qc_names_every_session_selection_setting(self):
        self.assertEqual(set(Qc.model_fields), {f.name for f in fields(Settings)})

    def test_the_corpus_spec_loads(self):
        spec = load(DatasetSpec, CONFIGS / "vr-foraging-sniff.toml")
        self.assertEqual(spec.select.clips_per_session, 2)
        self.assertEqual(spec.qc.settings().artifact_pad_s, 2.0)


if __name__ == "__main__":
    unittest.main()
