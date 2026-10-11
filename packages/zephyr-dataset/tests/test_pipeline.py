"""The whole pipeline on an in-memory bucket: inventory, fetch, lists, qc, select."""

import gzip
import io
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import tomlkit
from _corpusfakes import camera_frame, harp_bytes

from zephyr.annotate_events import TuneState
from zephyr.corpus import fetch, inventory, lists, qc, select, videos
from zephyr.corpus.config import DatasetSpec, Layout, Review

FS_HZ = 250.0
DEVICE_YML = "registers:\n  RawVoltage: {address: 32, type: U16}\n"


def breathing(t, seed):
    rng = np.random.default_rng(seed)
    depth = 1 + 0.15 * np.sin(2 * np.pi * 0.13 * t + seed)
    return (
        2000 + 300 * depth * np.sin(2 * np.pi * 5.0 * t) + rng.standard_normal(len(t))
    )


def flat(t, seed):
    return 2000 + 2 * np.random.default_rng(seed).standard_normal(len(t))


SESSIONS = {  # session_id: (raw folder name, signal, cameras)
    "100_2026-01-01_10-00-00": (
        "behavior_100_2026-01-01_10-00-00",
        breathing,
        {"FaceCamera": camera_frame(drop=100), "SideCamera": camera_frame()},
    ),
    "behavior_101_2026-01-02_10-00-00": (
        "101_2026-01-02_10-00-00",
        breathing,
        {"FaceCamera": camera_frame()},  # no SideCamera: never selected
    ),
    "102_2026-01-03_10-00-00": (
        "102_2026-01-03_10-00-00",
        flat,
        {"FaceCamera": camera_frame(), "SideCamera": camera_frame()},
    ),
    "103_2026-01-04_10-00-00": (
        "103_2026-01-04_10-00-00",
        breathing,
        {  # cameras need not share a rate
            "FaceCamera": camera_frame(),
            "SideCamera": camera_frame(n=600, rollback=200, hz=60.0),
        },
    ),
}
SUBJECTS = {  # the metadata's subject; the last session was corrected after naming
    "100_2026-01-01_10-00-00": "100",
    "behavior_101_2026-01-02_10-00-00": "101",
    "102_2026-01-03_10-00-00": "102",
    "103_2026-01-04_10-00-00": "900",
}


def fake_bucket():
    fs = fsspec.filesystem("memory")
    bucket = f"bucket-{uuid.uuid4().hex[:8]}"  # memory filesystems share one store
    records = []
    t = np.arange(0, 40.0, 1 / FS_HZ) + 5000.0
    for k, (sid, (raw, signal, cameras)) in enumerate(SESSIONS.items()):
        harp_dir = f"{bucket}/{raw}/behavior/SniffDetector.harp"
        values = np.round(signal(t - t[0], k)).astype(np.uint16)
        half = len(t) // 2
        fs.pipe(f"{harp_dir}/device.yml", DEVICE_YML.encode())
        fs.pipe(f"{harp_dir}/SniffDetector_32.bin", harp_bytes(t[:half], values[:half]))
        fs.pipe(
            f"{harp_dir}/SniffDetector_32_1.bin", harp_bytes(t[half:], values[half:])
        )
        fs.pipe(f"{harp_dir}/SniffDetector_33.bin", harp_bytes([5000.0], [1], 33))
        for name, frame in cameras.items():
            folder = f"{bucket}/{raw}/behavior-videos/{name}"
            fs.pipe(f"{folder}/video.mp4", b"x")
            fs.pipe(f"{folder}/metadata.csv", frame.to_csv(index=False).encode())
        session = pd.DataFrame(
            {
                "session_id": [sid],
                "subject_id": [SUBJECTS[sid]],
                "date": ["2026-01-01"],
                "rig": [json.dumps({"rig_name": "R1"})],
                "curriculum_stage_name": ["s"],
                "dataset_version": ["0.6.1" if k else "0.5.0"],
            }
        )
        buffer = io.BytesIO()
        session.to_parquet(buffer)
        fs.pipe(f"{bucket}/derived_{k}/session.parquet", buffer.getvalue())
        records.append(record(f"derived_{k}", bucket, "1.0", f"2026-01-0{k + 1}", raw))
    first, last = SESSIONS["100_2026-01-01_10-00-00"][0], records[3]
    # an older packaging of the first session, superseded by the one above
    records.append(record("old", bucket, "0.9", "2025-12-01", first))
    # a newer packaging of the last session, made after the spec's `until`
    raw = last["data_description"]["source_data"]
    records.append(record("later", bucket, "2.0", "2026-03-01", raw[0]))
    return fs, bucket, records


def record(name, bucket, version, created, source) -> dict:
    """A DocDB record as the aggregate query returns it."""
    return {
        "name": name,
        "location": f"s3://{bucket}/{name}",
        "data_description": {
            "creation_time": f"{created}T12:00:00Z",
            "source_data": [source],
        },
        "processing": {
            "data_processes": [
                {
                    "name": "primary-nwb-packaging-vr-foraging",
                    "output_parameters": {"packaging_version": version},
                }
            ]
        },
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fs, bucket, docdb = fake_bucket()
        self.spec = DatasetSpec.model_validate(
            {
                "root": "data",
                "source": {"bucket": bucket},
                "query": {
                    "until": "2026-02-01",
                    "min_dataset_version": "0.6.0",
                    "include_sessions": ["100_2026-01-01_10-00-00"],
                },
                "preprocess": {"target_size": [360, 270]},
                "events": {"method": "peak_detection"},
                "qc": {
                    "window_s": 2.0,
                    "min_span_s": 6.0,
                    "max_outside": 0.1,
                    "max_breaks": 0.05,
                    "outlier_z": 3.0,
                    "min_snr_gap_db": 3.0,
                    "snr_ceiling_db": 60.0,
                    "min_amplitude": 20.0,
                    "artifact_z": 8.0,
                    "artifact_pad_s": 0.5,
                },
                "select": {"clip_s": 5.0, "clips_per_session": 2, "seed": 3},
            },
            context={"base": Path(self.tmp.name)},
        )
        self.layout = Layout(self.spec.root)
        self.layout.root.mkdir()
        with gzip.open(self.layout.docdb, "wt", encoding="utf-8") as handle:
            json.dump(
                {"queried": "2026-02-02T00:00:00+00:00", "records": docdb}, handle
            )

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        inventory.build(self.spec, workers=2, fs=self.fs)
        fetch.run(self.spec, workers=2, fs=self.fs)
        lists.run(self.spec)
        qc.run(self.spec)
        return select.run(self.spec, workers=2, fs=self.fs)

    def test_builds_the_dataset_folder(self):
        results = self.build()
        table = pd.read_parquet(self.layout.inventory)
        self.assertEqual(sorted(table["asset"]), [f"derived_{k}" for k in range(4)])
        self.assertEqual(self.layout.sessions(), sorted(SESSIONS))
        raw = pd.read_parquet(self.layout.thermistor("102_2026-01-03_10-00-00"))
        self.assertEqual(list(raw.columns), ["Time", "Signal"])
        self.assertEqual(len(raw), 40 * FS_HZ)
        self.assertTrue(raw["Time"].is_monotonic_increasing)
        files = {p.name for p in self.layout.root.iterdir()}
        self.assertEqual(
            files,
            {
                "docdb.json.gz",
                "inventory.parquet",
                "thermistor",
                "sessions.toml",
                "qc.parquet",
                "results.json",
            },
        )
        by_id = {s["session_id"]: s for s in results["sessions"]}
        self.assertEqual(by_id["102_2026-01-03_10-00-00"]["selection"], "exclude")
        self.assertTrue(by_id["102_2026-01-03_10-00-00"]["reason"])  # a dead sensor
        # passed: the three breathing sessions; the second has no SideCamera
        self.assertEqual(len(results["passed"]), 3)
        self.assertEqual(
            results["selected"], ["100_2026-01-01_10-00-00", "103_2026-01-04_10-00-00"]
        )
        chosen = by_id["100_2026-01-01_10-00-00"]
        self.assertEqual(sorted(chosen["clips"]), ["span1", "span2"])
        self.assertEqual(sorted(chosen["cameras"]), ["FaceCamera", "SideCamera"])
        self.assertTrue(chosen["cameras"]["SideCamera"].endswith("/SideCamera/"))
        lonely = by_id["behavior_101_2026-01-02_10-00-00"]
        self.assertEqual(list(lonely["cameras"]), ["FaceCamera"])
        self.assertIsNone(lonely["clips"])
        self.assertEqual(json.loads(self.layout.results.read_text()), results)

    def test_videos_repairs_rollbacks_and_reports_what_stays_broken(self):
        self.build()

        def frames(url):  # what ffprobe would say: the video's own frame count
            return 600 if "103_" in url and "SideCamera" in url else 2400

        table = videos.run(self.spec, workers=2, fs=self.fs, frames=frames)
        self.assertEqual(len(table), 4)  # two selected sessions x two cameras
        row = table.set_index(["session_id", "camera"])
        repaired = row.loc[("103_2026-01-04_10-00-00", "SideCamera")]
        self.assertEqual(repaired["before_ref_backwards"], 1)
        self.assertEqual(repaired["repaired_rows"], 2)  # the spike and the rollback
        self.assertTrue(repaired["clean"])
        self.assertAlmostEqual(repaired["after_rate_hz"], 60.0, places=3)
        dropped = row.loc[("100_2026-01-01_10-00-00", "FaceCamera")]
        self.assertEqual(dropped["after_dropped_frames"], 1)
        self.assertFalse(dropped["clean"])  # and 2399 rows against 2400 video frames
        fixed = pd.read_parquet(
            self.layout.timestamps("103_2026-01-04_10-00-00", "SideCamera")
        )
        self.assertTrue(fixed["ReferenceTime"].is_monotonic_increasing)
        np.testing.assert_allclose(
            fixed["ReferenceTime"].to_numpy(), 5000.0 + np.arange(600) / 60.0
        )
        again = videos.run(self.spec, workers=2, fs=self.fs, frames=lambda url: 1 / 0)
        pd.testing.assert_frame_equal(again, table)  # resumable: nothing redone
        raw = pd.read_parquet(
            self.layout.camera_metadata("103_2026-01-04_10-00-00", "SideCamera")
        )
        self.assertFalse(raw["ReferenceTime"].is_monotonic_increasing)  # unmodified
        # a changed repair re-runs from the cache alone: no S3, no ffprobe
        shutil.rmtree(self.layout.video_dir)
        self.layout.cameras.unlink()
        redone = videos.run(self.spec, workers=2, fs=None, frames=lambda url: 1 / 0)
        pd.testing.assert_frame_equal(redone, table)
        # select then drops sessions whose cameras are not clean, saying why
        results = select.run(self.spec, workers=2, fs=self.fs)
        self.assertEqual(results["selected"], ["103_2026-01-04_10-00-00"])
        self.assertEqual(results["counts"]["dropped_for_cameras"], 1)
        dropped = {s["session_id"]: s for s in results["sessions"]}[
            "100_2026-01-01_10-00-00"
        ]
        self.assertIsNone(dropped["clips"])
        self.assertTrue(any("FaceCamera" in p for p in dropped["dropped"]))
        self.assertTrue(any("dropped frames" in p for p in dropped["dropped"]))

    def test_the_list_groups_by_subject(self):
        self.build()
        state = TuneState(self.layout.clips)
        self.assertTrue(all(e.clip.video is None for e in state.entries))
        groups = {e.clip.source.stem: e.clip.group for e in state.entries}
        self.assertEqual(groups["behavior_101_2026-01-02_10-00-00"], "101")
        self.assertEqual(groups["100_2026-01-01_10-00-00"], "100")
        # a corrected session: its metadata, not its name, gives the subject
        self.assertEqual(groups["103_2026-01-04_10-00-00"], "900")

    def test_reviews_survive_a_rerun_and_decide_the_selection(self):
        self.build()
        doc = tomlkit.parse(self.layout.clips.read_text())
        for clip in doc["clip"]:
            if "103_" in str(clip["thermistor"]):
                clip["selection"] = "exclude"
                clip["reviewed"] = True
        self.layout.clips.write_text(tomlkit.dumps(doc))
        results = self.build()  # every step again
        self.assertEqual(results["selected"], ["100_2026-01-01_10-00-00"])
        doc = tomlkit.parse(self.layout.clips.read_text())
        self.assertEqual(len(doc["clip"]), len(SESSIONS))
        reviewed = [c for c in doc["clip"] if c.get("reviewed")]
        self.assertEqual([str(c["selection"]) for c in reviewed], ["exclude"])

    def test_reviews_in_the_spec_are_written_in_and_kept_by_qc(self):
        review = Review(session_id="103_2026-01-04_10-00-00", selection="exclude")
        self.spec = self.spec.model_copy(update={"review": [review]})
        results = self.build()
        self.assertEqual(results["selected"], ["100_2026-01-01_10-00-00"])
        by_id = {s["session_id"]: s for s in results["sessions"]}
        self.assertTrue(by_id["103_2026-01-04_10-00-00"]["reviewed"])
        self.assertEqual(by_id["103_2026-01-04_10-00-00"]["suggested"], "include")
        self.assertTrue(by_id["103_2026-01-04_10-00-00"]["included"])  # QC's spans kept

    def test_the_saved_query_and_until_fix_the_inventory(self):
        table = inventory.build(self.spec, workers=2, fs=self.fs)
        self.assertEqual(set(table["queried"]), {"2026-02-02T00:00:00+00:00"})
        self.assertEqual(sorted(table["asset"]), [f"derived_{k}" for k in range(4)])
        spec = self.spec.model_copy(
            update={"query": self.spec.query.model_copy(update={"until": None})}
        )
        table = inventory.build(spec, workers=2, fs=self.fs)
        self.assertIn("later", set(table["asset"]))  # the newer packaging wins
        self.assertNotIn("derived_3", set(table["asset"]))

    def test_the_query_drops_old_versions_unless_included(self):
        table = inventory.build(self.spec, workers=2, fs=self.fs)
        kept = inventory.wanted(self.spec, table)["session_id"].tolist()
        self.assertIn("100_2026-01-01_10-00-00", kept)  # 0.5.0, but included
        spec = self.spec.model_copy(
            update={
                "query": self.spec.query.model_copy(update={"include_sessions": []})
            }
        )
        self.assertNotIn(
            "100_2026-01-01_10-00-00",
            inventory.wanted(spec, table)["session_id"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
