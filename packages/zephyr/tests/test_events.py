import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from _clipfiles import touch_clip, write_list

from zephyr import events
from zephyr.config import ClipList, load
from zephyr.dataset import WindowDataset
from zephyr.evaluation import score_clip
from zephyr.features import ClipEntry
from zephyr.targets import load_target

FS = 250.0
GRID_FS = 60.0


def mixed_breathing(duration_s=40.0):
    """Deep 2 Hz breaths and shallow 3 Hz sniffs, alternating every 10 s.

    Returns the thermistor frame and the true peak and trough times.
    """
    t = np.arange(0, duration_s, 1 / FS)
    v = np.zeros_like(t)
    peaks, troughs = [], []
    for k, start in enumerate(np.arange(0, duration_s, 10.0)):
        deep = k % 2 == 0
        rate, amplitude = (2.0, 5.0) if deep else (3.0, 0.3)
        inside = (t >= start) & (t < start + 10.0)
        v[inside] = amplitude * np.sin(2 * np.pi * rate * (t[inside] - start))
        cycles = np.arange(0, 10.0, 1 / rate) + start
        peaks += list(cycles + 0.25 / rate)
        troughs += list(cycles + 0.75 / rate)
    frame = pd.DataFrame({"Time": t, "Signal": v})
    return frame, np.array(peaks), np.array(troughs)


def recall(found, truth, tolerance_s=0.02, margin_s=1.0):
    """Share of true events (away from the ends and segment joins) found."""
    inner = truth[(truth > margin_s) & (truth < truth.max() - margin_s)]
    inner = inner[np.abs((inner % 10.0) - 5.0) < 5.0 - margin_s]
    hit = (
        [np.min(np.abs(found - x)) <= tolerance_s for x in inner] if len(found) else []
    )
    return float(np.mean(hit)) if len(hit) else 0.0


class DetectorTests(unittest.TestCase):
    def test_peak_detection_matches_the_grid_rule_off_the_grid(self):
        """Same events as the 60 Hz detection the target used, each within a
        frame of it, but at the thermistor's own resolution."""
        frame, peaks, _ = mixed_breathing()
        t = frame["Time"].to_numpy()
        grid = t[0] + np.arange(int((t[-1] - t[0]) * GRID_FS) + 1) / GRID_FS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermistor.parquet"
            frame.to_parquet(path)
            target = load_target(path, grid)
        found = events.detect("peak_detection", frame)
        self.assertEqual(len(found.inhale), len(target.onset_times))
        self.assertLess(np.abs(found.inhale - target.onset_times).max(), 1 / GRID_FS)
        deep = peaks[(peaks // 10.0) % 2 == 0]
        self.assertGreater(recall(found.inhale, deep, tolerance_s=0.004), 0.95)

    def test_one_global_threshold_loses_shallow_breaths(self):
        frame, peaks, _ = mixed_breathing()
        shallow = peaks[(peaks // 10.0) % 2 == 1]
        found = events.detect("peak_detection", frame)
        self.assertLess(recall(found.inhale, shallow, tolerance_s=0.03), 0.5)

    def test_local_detectors_find_deep_and_shallow_breaths(self):
        frame, peaks, troughs = mixed_breathing()
        for method in ("adaptive_prominence", "hysteresis_cycles"):
            with self.subTest(method=method):
                found = events.detect(method, frame)
                self.assertGreater(recall(found.inhale, peaks), 0.95)
                self.assertGreater(recall(found.exhale, troughs), 0.95)

    def test_hysteresis_cycles_alternate(self):
        frame, _, _ = mixed_breathing()
        found = events.detect("hysteresis_cycles", frame)
        kinds = np.concatenate(
            [np.ones(len(found.inhale)), -np.ones(len(found.exhale))]
        )
        order = kinds[np.argsort(np.concatenate([found.inhale, found.exhale]))]
        self.assertTrue(np.all(order[1:] != order[:-1]))

    def test_unknown_detector_or_parameter_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown detector"):
            events.detector_params("nope")
        with self.assertRaises(ValueError):
            events.detector_params("peak_detection", {"prominence": 0.2})


class IdentityTests(unittest.TestCase):
    def test_identity_follows_the_method_and_its_parameters(self):
        default = events.identity("peak_detection", {})
        self.assertEqual(default["params"]["prominence_frac"], 0.1)
        self.assertEqual(
            events.identity("peak_detection", {"prominence_frac": 0.2})["params"][
                "prominence_frac"
            ],
            0.2,
        )
        self.assertNotEqual(default, events.identity("adaptive_prominence", {}))


class ClipListEventsTests(unittest.TestCase):
    def test_unknown_detector_in_a_clip_list_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = touch_clip(Path(tmp) / "data", 1, 1)
            path = write_list(
                Path(tmp) / "list.toml", [video], extra='[events]\nmethod = "nope"'
            )
            with self.assertRaisesRegex(ValueError, "unknown detector"):
                load(ClipList, path)

    def test_events_section_reaches_every_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = touch_clip(Path(tmp) / "data", 1, 1)
            path = write_list(
                Path(tmp) / "list.toml",
                [video],
                extra='[events]\nmethod = "hysteresis_cycles"',
            )
            (clip,) = load(ClipList, path).resolve()
            self.assertEqual(clip.events.method, "hysteresis_cycles")

    def test_a_clips_own_events_beat_the_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            videos = [touch_clip(Path(tmp) / "data", g, 1) for g in (1, 2)]
            path = write_list(
                Path(tmp) / "list.toml",
                videos,
                extra='[events]\nmethod = "hysteresis_cycles"',
            )
            own = (
                'events = { method = "adaptive_prominence", '
                "params = { prominence_frac = 0.3 } }\n"
            )
            path.write_text(
                path.read_text().replace(
                    "box = [10, 10, 32, 32]\n", "box = [10, 10, 32, 32]\n" + own, 1
                )
            )
            mine, inherited = load(ClipList, path).resolve()
            self.assertEqual(mine.events.method, "adaptive_prominence")
            self.assertEqual(mine.events.params, {"prominence_frac": 0.3})
            self.assertEqual(inherited.events.method, "hysteresis_cycles")

    def test_excluded_spans_reach_the_event_set_and_the_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = touch_clip(Path(tmp) / "data", 1, 1)
            path = write_list(Path(tmp) / "list.toml", [video])
            mixed_breathing(5.0)[0].to_parquet(
                video.parent / "thermistor_1_part_1.parquet"
            )
            plain = load(ClipList, path).resolve()[0]
            path.write_text(
                path.read_text().replace(
                    "box = [10, 10, 32, 32]\n",
                    "box = [10, 10, 32, 32]\nexcluded = [[3.0, 2.0], [0.5, 1.0]]\n",
                )
            )
            with self.assertRaisesRegex(ValueError, "start < end"):
                load(ClipList, path)
            path.write_text(path.read_text().replace("[3.0, 2.0]", "[2.0, 3.0]"))
            marked = load(ClipList, path).resolve()[0]
            self.assertEqual(marked.excluded, ((0.5, 1.0), (2.0, 3.0)))
            found = marked.events.for_clip(marked)
            np.testing.assert_array_equal(found.excluded, [[0.5, 1.0], [2.0, 3.0]])
            self.assertEqual(len(plain.events.for_clip(plain).excluded), 0)
            self.assertNotIn("excluded", plain.events.identity(plain))
            self.assertEqual(
                marked.events.identity(marked)["excluded"], [[0.5, 1.0], [2.0, 3.0]]
            )

    def test_an_unknown_clip_detector_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = touch_clip(Path(tmp) / "data", 1, 1)
            path = write_list(Path(tmp) / "list.toml", [video])
            path.write_text(
                path.read_text().replace(
                    "box = [10, 10, 32, 32]\n",
                    'box = [10, 10, 32, 32]\nevents = { method = "nope" }\n',
                )
            )
            with self.assertRaisesRegex(ValueError, "unknown detector"):
                load(ClipList, path)


class ExcludedScoringTests(unittest.TestCase):
    def test_events_inside_excluded_spans_do_not_count(self):
        t = np.arange(0, 20, 1 / GRID_FS)
        signal = np.sin(2 * np.pi * 2 * t)
        truth = pd.DataFrame({"Time": t, "Signal": signal})
        onsets = np.arange(0.125, 20, 0.5)
        wrong = np.where(onsets < 10, onsets, onsets + 0.2)  # bad in the second half
        kwargs = {"truth_onset_times_s": onsets, "predicted_onset_times_s": wrong}
        scored = score_clip(truth, truth, **kwargs)
        masked = score_clip(truth, truth, excluded_s=[(9.9, 20.0)], **kwargs)
        self.assertLess(scored.inhale_f1, 0.6)
        self.assertAlmostEqual(masked.inhale_f1, 1.0)


class ExcludedWindowTests(unittest.TestCase):
    def test_no_window_overlaps_an_excluded_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = int(120 * GRID_FS)
            times = np.arange(n) / GRID_FS
            np.save(root / "features.npy", np.zeros((n, 4, 4, 4), np.uint8))
            np.save(root / "times.npy", times)
            pd.DataFrame({"time": times, "signal": np.zeros(n)}).to_parquet(
                root / "t.parquet"
            )
            np.savez(
                root / "events.npz",
                onset_times=np.arange(0, 120, 0.5),
                offset_times=np.arange(0.25, 120, 0.5),
                excluded=np.array([[30.0, 90.0]]),
            )
            entry = ClipEntry(
                clip_id="fake/clip",
                recording="fake#1",
                n_frames=n,
                n_output=n,
                features=root / "features.npy",
                frame_times=root / "times.npy",
                baselines=root / "base",
                times=root / "times.npy",
                target=root / "t.parquet",
                events=root / "events.npz",
            )
            kwargs = {
                "window": 60,
                "mean": np.zeros(4),
                "std": np.ones(4),
                "select_fs": GRID_FS,
                "output_fs": GRID_FS,
                "motion_tau_s": 1 / GRID_FS,
            }
            random = WindowDataset([[entry]], length=500, **kwargs)
            gridded = WindowDataset([[entry]], stride=60, **kwargs)
            for dataset in (random, gridded):
                for i in range(len(dataset)):
                    _, start, _ = dataset.draw(i)
                    lo, hi = start / GRID_FS, (start + dataset.extent(1.0)) / GRID_FS
                    self.assertFalse(lo <= 90.0 and hi >= 30.0, (lo, hi))


if __name__ == "__main__":
    unittest.main()
