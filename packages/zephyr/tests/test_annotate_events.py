import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from _clipfiles import write_list

from zephyr.annotate_events import METRICS, TuneState, mean_metrics
from zephyr.config import ClipList, load

FS = 250.0
RATE_HZ = 2.0


def make_video(root: Path, group: int, duration_s: float = 20.0) -> Path:
    """A clip breathing at 2 Hz: peaks at 0.125 + k/2 s, troughs 0.25 s later."""
    folder = root / "clips"
    folder.mkdir(exist_ok=True)
    t = np.arange(0, duration_s, 1 / FS)
    pd.DataFrame({"Time": t, "Signal": np.sin(2 * np.pi * RATE_HZ * t)}).to_parquet(
        folder / f"thermistor_{group}_part_1.parquet"
    )
    video = folder / f"video_face_{group}_part_1.mp4"
    video.touch()
    (folder / f"video_face_{group}_part_1.parquet").touch()
    return video


class TuneStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.videos = [make_video(self.root, 1), make_video(self.root, 2)]
        self.path = write_list(
            self.root / "list.toml", self.videos, extra="# keep this comment"
        )
        self.state = TuneState(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_starts_from_the_lists_detector(self):
        self.assertEqual(self.state.describe(0), ("peak_detection", ""))
        self.assertEqual(self.state.dirty, set())
        inhale = self.state.times("inhale")
        self.assertGreater(len(inhale), 30)
        self.assertLess(np.abs(inhale % 0.5 - 0.125).max(), 0.01)

    def test_apply_changes_only_the_chosen_clips(self):
        self.state.apply([1], "adaptive_prominence", {"prominence_frac": 0.3})
        self.assertEqual(self.state.dirty, {1})
        self.assertEqual(self.state.describe(0), ("peak_detection", ""))
        self.assertEqual(
            self.state.describe(1), ("adaptive_prominence", "prominence_frac=0.3")
        )

    def test_a_bad_method_or_parameter_changes_nothing(self):
        for method, params in (("nope", {}), ("peak_detection", {"bogus": 1.0})):
            with self.assertRaises(ValueError):
                self.state.apply([0, 1], method, params)
        self.assertEqual(self.state.dirty, set())

    def test_apply_reports_progress(self):
        seen = []
        self.state.apply(
            [0, 1], "hysteresis_cycles", None, lambda done, total: seen.append(done)
        )
        self.assertEqual(seen, [1, 2])

    def test_save_writes_events_next_to_the_box_and_keeps_the_rest(self):
        self.state.apply([1], "adaptive_prominence", {"prominence_frac": 0.3})
        self.assertEqual(self.state.save(), 1)
        self.assertIn("# keep this comment", self.path.read_text())
        first, second = load(ClipList, self.path).resolve()
        self.assertEqual(first.events.method, "peak_detection")
        self.assertEqual(second.events.method, "adaptive_prominence")
        self.assertEqual(second.events.params["prominence_frac"], 0.3)
        self.assertEqual(second.events.params["window_s"], 2.0)  # written in full
        self.assertEqual(second.box, (10, 10, 32, 32))
        self.assertEqual(self.state.dirty, set())
        again = TuneState(self.path)
        self.assertEqual(
            again.describe(1), ("adaptive_prominence", "prominence_frac=0.3")
        )

    def test_add_and_remove_videos_reach_the_file_on_save(self):
        extra = make_video(self.root, 3)
        self.assertEqual(self.state.add_videos([extra, extra]), [extra.resolve()])
        self.assertEqual(len(self.state.clips), 3)
        self.state.remove([0])
        self.assertTrue(self.state.modified)
        self.state.save()
        names = [c.video.name for c in load(ClipList, self.path).resolve()]
        self.assertEqual(names, [self.videos[1].name, extra.name])
        self.assertFalse(self.state.modified)

    def test_a_rejected_video_is_saved_and_left_out_of_everything_downstream(self):
        self.state.set_rejected([1])
        self.assertEqual(self.state.dirty, {1})
        self.assertEqual(self.state.accepted(), [0])
        self.state.save()
        self.assertEqual(self.path.read_text().count("rejected = true"), 1)
        clip_list = load(ClipList, self.path)
        self.assertEqual(
            [c.video.name for c in clip_list.resolve()], [self.videos[0].name]
        )
        both = clip_list.resolve(include_rejected=True)
        self.assertEqual(len(both), 2)
        again = TuneState(self.path)  # the tuner still lists it, marked
        self.assertEqual([e.rejected for e in again.entries], [False, True])
        again.set_rejected([1], False)
        again.save()
        self.assertNotIn("rejected", self.path.read_text())
        self.assertEqual(len(load(ClipList, self.path).resolve()), 2)

    def test_at_least_one_video_must_stay_in_use(self):
        self.state.set_rejected([0])
        with self.assertRaisesRegex(ValueError, "at least one video"):
            self.state.set_rejected([1])
        with self.assertRaisesRegex(ValueError, "at least one video"):
            self.state.remove([1])
        self.assertEqual(self.state.accepted(), [1])
        self.state.path.write_text(
            self.state.path.read_text().replace(
                "box = [10, 10, 32, 32]\n", "box = [10, 10, 32, 32]\nrejected = true\n"
            )
        )
        with self.assertRaisesRegex(ValueError, "every clip of the list is rejected"):
            load(ClipList, self.path)

    def test_excluded_spans_are_marked_undone_saved_and_reloaded(self):
        self.state.exclude(9.0, 3.0)  # either order
        self.state.exclude(12.0, 14.0)
        self.assertEqual(self.state.spans().tolist(), [[3.0, 9.0], [12.0, 14.0]])
        self.assertTrue(self.state.delete_span(13.0))
        self.assertFalse(self.state.delete_span(100.0))
        self.assertTrue(self.state.undo_span())
        self.assertEqual(len(self.state.spans()), 2)
        with self.assertRaisesRegex(ValueError, "before its end"):
            self.state.exclude(5.0, 5.0)
        self.assertEqual(self.state.spans(1).tolist(), [])  # per clip
        self.state.save()
        first, second = load(ClipList, self.path).resolve()
        self.assertEqual(first.excluded, ((3.0, 9.0), (12.0, 14.0)))
        self.assertEqual(second.excluded, ())
        self.state.delete_span(5.0)
        self.state.delete_span(13.0)
        self.state.save()
        self.assertNotIn("excluded", self.path.read_text())

    def test_spans_are_edited_and_removed_by_index(self):
        for start, end in ((3.0, 4.0), (8.0, 9.0), (12.0, 13.0)):
            self.state.exclude(start, end)
        self.state.update_span(1, 7.5, 9.5)
        self.assertEqual(
            self.state.spans().tolist(), [[3.0, 4.0], [7.5, 9.5], [12.0, 13.0]]
        )
        self.state.update_span(0, 15.0, 16.0)  # moved past the others: order restored
        self.assertEqual(self.state.spans()[:, 0].tolist(), [7.5, 12.0, 15.0])
        with self.assertRaisesRegex(ValueError, "before its end"):
            self.state.update_span(0, 9.0, 8.0)
        self.assertEqual(len(self.state.spans()), 3)
        self.assertFalse(self.state.remove_spans([]))
        self.assertTrue(self.state.remove_spans([0, 2]))
        self.assertEqual(self.state.spans().tolist(), [[12.0, 13.0]])
        self.assertTrue(self.state.undo_span())  # the removal is one change
        self.assertEqual(len(self.state.spans()), 3)

    def test_metrics_ignore_events_inside_excluded_spans(self):
        before = self.state.breath_stats(0)
        self.assertEqual(self.state.clip_metrics(0)["spans"], 0)
        self.state.exclude(0.0, 10.0)
        after = self.state.breath_stats(0)
        self.assertEqual(self.state.clip_metrics(0)["spans"], 1)
        self.assertLess(len(after["rates"]), len(before["rates"]) - 15)
        self.assertAlmostEqual(float(np.median(after["ie"])), 0.25, places=2)
        self.state.exclude(0.0, 30.0)
        self.assertEqual(len(self.state.breath_stats(0)["rates"]), 0)

    def test_a_video_without_a_thermistor_is_refused(self):
        lone = self.root / "clips" / "video_face_9_part_1.mp4"
        lone.touch()
        with self.assertRaisesRegex(ValueError, "no thermistor"):
            self.state.add_videos([lone])
        self.assertEqual(len(self.state.clips), 2)

    def test_the_last_clip_cannot_be_removed(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.state.remove([0, 1])

    def test_breath_stats_durations_and_same_kind_neighbours(self):
        stats = self.state.breath_stats(band=(1.0, 15.0))
        self.assertEqual((stats["ii_breaks"], stats["ee_breaks"]), (0, 0))
        self.assertAlmostEqual(float(np.median(stats["ie"])), 0.25, places=2)
        self.assertAlmostEqual(float(np.median(stats["ei"])), 0.25, places=2)
        self.assertAlmostEqual(stats["mean"], RATE_HZ, places=2)
        self.assertEqual(stats["outside"], 0)
        slow = self.state.breath_stats(band=(2.5, 15.0))
        self.assertEqual(slow["outside"], len(slow["rates"]))

    def test_clip_metrics_and_their_mean_over_clips(self):
        first, second = (self.state.clip_metrics(i) for i in (0, 1))
        self.assertEqual(set(first), set(METRICS))
        mean = mean_metrics([first, {**second, "ee_breaks": first["ee_breaks"] + 2}])
        self.assertAlmostEqual(mean["ee_breaks"], first["ee_breaks"] + 1)
        self.assertAlmostEqual(mean["ie_median"], 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
