import math
import unittest

from zephyr.config import ResolvedClip
from zephyr.evaluate import build_result, refuse_leaked, summarise_rows


def _row(clip, recording, corr, f1, **extra):
    return {
        "clip_id": clip,
        "recording": recording,
        "correlation": corr,
        "inhale_f1": f1,
        "exhale_f1": f1,
        "kl_ibi": 0.5,
    } | extra


class BuildResultTests(unittest.TestCase):
    def test_recordings_groups_and_composite(self):
        groups = {
            "new": [_row("a", "r1", 0.2, 0.4), _row("b", "r1", 0.4, 0.6)],
            "known": [_row("c", "r2", 0.6, 0.8)],
        }
        result = build_result(groups)
        self.assertEqual(len(result["per_recording"]), 2)
        self.assertAlmostEqual(result["per_recording"][0]["correlation"], 0.3)
        self.assertEqual(set(result["groups"]), {"new", "known", "all"})
        self.assertAlmostEqual(result["groups"]["new"]["summary"]["correlation"], 0.3)
        self.assertAlmostEqual(result["groups"]["all"]["summary"]["correlation"], 0.45)
        summary = result["summary"]
        expected = (
            0.5 * summary["inhale_f1"]
            + 0.2 * summary["exhale_f1"]
            + 0.2 * max(0.0, summary["correlation"])
            + 0.1 * math.exp(-summary["kl_ibi"])
        )
        self.assertAlmostEqual(result["composite"], expected)

    def test_head_metric_is_summarised_only_when_present(self):
        self.assertNotIn("head_inhale_f1", summarise_rows([_row("a", "r", 0.1, 0.1)]))
        rows = [_row("a", "r", 0.1, 0.1, head_inhale_f1=0.9)]
        self.assertAlmostEqual(summarise_rows(rows)["head_inhale_f1"], 0.9)

    def test_all_cannot_be_a_group_name(self):
        with self.assertRaises(ValueError):
            build_result({"all": [_row("a", "r", 0.1, 0.1)]})


def _clip(folder: str, group: str) -> ResolvedClip:
    from pathlib import Path

    video = Path("/data") / folder / f"video_{group}.mp4"
    return ResolvedClip(
        video=video, timestamps=video, thermistor=video, group=group, box=None
    )


class RefuseLeakedTests(unittest.TestCase):
    def setUp(self):
        self.trained = _clip("train", "1")
        self.state = {
            "train_videos": [str(self.trained.video)],
            "train_recordings": [self.trained.recording_id],
        }

    def test_refuses_a_trained_video(self):
        with self.assertRaisesRegex(SystemExit, "trained on this video"):
            refuse_leaked([self.state], [self.trained])

    def test_refuses_another_clip_of_a_trained_recording(self):
        sibling = _clip("train", "1").model_copy(
            update={"video": self.trained.video.with_name("video_1b.mp4")}
        )
        with self.assertRaisesRegex(SystemExit, "its recording"):
            refuse_leaked([self.state], [sibling])

    def test_same_group_in_another_folder_is_fine(self):
        refuse_leaked([self.state], [_clip("test", "1")])

    def test_old_checkpoint_without_a_record_warns_instead(self):
        refuse_leaked([{}], [self.trained])

    def test_refuses_what_the_init_from_parent_trained_on(self):
        parent = _clip("parent", "7")
        fine_tuned = {
            "train_videos": [str(self.trained.video)],
            "train_recordings": [self.trained.recording_id],
            "inherited_videos": [str(parent.video)],
            "inherited_recordings": [parent.recording_id],
        }
        with self.assertRaisesRegex(SystemExit, "trained on this video"):
            refuse_leaked([fine_tuned], [parent])


if __name__ == "__main__":
    unittest.main()
