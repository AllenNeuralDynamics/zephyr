import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zephyr import baseline
from zephyr.baselines import timing


class SummariseTests(unittest.TestCase):
    def test_summary_statistics(self):
        s = timing.summarise([1.0, 2.0, 3.0, 6.0])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean_s"], 3.0)
        self.assertAlmostEqual(s["median_s"], 2.5)
        self.assertAlmostEqual(s["min_s"], 1.0)
        self.assertAlmostEqual(s["max_s"], 6.0)
        self.assertGreater(s["sd_s"], 0.0)

    def test_single_measurement_has_no_spread(self):
        self.assertIsNone(timing.summarise([2.0])["sd_s"])


class PickClipsTests(unittest.TestCase):
    def _entries(self, n):
        return [SimpleNamespace(clip_id=f"c{i:02d}") for i in range(n)]

    def test_picks_evenly_spaced_clips_in_order(self):
        picked = timing.pick_clips(self._entries(24), 8)
        self.assertEqual(len(picked), 8)
        keys = [e.clip_id for e in picked]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 8)

    def test_asking_for_more_than_exist_returns_all(self):
        self.assertEqual(len(timing.pick_clips(self._entries(3), 8)), 3)


class TimeMethodTests(unittest.TestCase):
    def test_warmup_clip_is_not_counted(self):
        calls = []
        entries = [SimpleNamespace(clip_id=f"c{i}") for i in range(4)]
        result = timing.time_method(lambda e: calls.append(e.clip_id), entries)
        self.assertEqual(calls[0], "c0")  # warm-up ran...
        self.assertEqual(len(calls), 5)  # ...on top of the 4 timed clips
        self.assertEqual(
            [c["clip_id"] for c in result["clips"]], ["c0", "c1", "c2", "c3"]
        )
        self.assertEqual(result["summary"]["n"], 4)


class WarmupClipTests(unittest.TestCase):
    def test_warmup_clip_outside_the_timed_set_is_never_timed(self):
        calls = []
        timed = [SimpleNamespace(clip_id=f"c{i}") for i in range(3)]
        warmup = SimpleNamespace(clip_id="warm")
        result = timing.time_method(
            lambda e: calls.append(e.clip_id), timed, warmup=warmup
        )
        self.assertEqual(calls, ["warm", "c0", "c1", "c2"])
        self.assertEqual([c["clip_id"] for c in result["clips"]], ["c0", "c1", "c2"])

    def test_pick_warmup_returns_a_clip_not_in_the_timed_set(self):
        entries = [SimpleNamespace(session_idx=i, part=1) for i in range(10)]
        timed = timing.pick_clips(entries, 4)
        warmup = timing.pick_warmup(entries, timed)
        self.assertNotIn(warmup, timed)


class WarmFileCacheTests(unittest.TestCase):
    def test_reads_every_distinct_feature_file_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, size in (("a.npy", 1000), ("b.npy", 5000)):
                path = Path(tmp) / name
                path.write_bytes(b"x" * size)
                paths.append(path)
            entries = [SimpleNamespace(features=p) for p in paths + paths[:1]]
            self.assertEqual(timing.warm_file_cache(entries), 6000)


class TimingTableTests(unittest.TestCase):
    def test_collect_includes_timing_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "timing.json").write_text(
                json.dumps(
                    {
                        "n_clips": 8,
                        "clip_seconds": 300.0,
                        "methods": [
                            {
                                "method": "physnet",
                                "device": "cuda",
                                "params_m": 0.77,
                                "summary": {"mean_s": 3.0, "sd_s": 0.1, "n": 8},
                            }
                        ],
                    }
                )
            )
            baseline.collect(root, runs_dir=None)
            text = (root / "results.md").read_text(encoding="utf-8")
        self.assertIn("Inference time per clip", text)
        self.assertIn("physnet", text)
        self.assertIn("3.00", text)
