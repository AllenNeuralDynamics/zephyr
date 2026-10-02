import math
import unittest

from zephyr.evaluate import build_result


def _row(clip, session, corr, f1):
    return {
        "clip_id": clip,
        "session_idx": session,
        "part": 1,
        "correlation": corr,
        "inhale_f1": f1,
        "exhale_f1": f1,
        "kl_ibi": 0.5,
    }


class BuildResultTests(unittest.TestCase):
    def test_sessions_strata_and_composite(self):
        rows = [_row("a", 1, 0.2, 0.4), _row("b", 1, 0.4, 0.6), _row("c", 2, 0.6, 0.8)]
        strata = {"new_animals": [1], "known_animals_new_date": [2]}
        result = build_result(rows, {1, 2}, strata)
        self.assertEqual(result["sessions"], [1, 2])
        self.assertEqual(len(result["per_session"]), 2)
        self.assertAlmostEqual(result["per_session"][0]["correlation"], 0.3)
        self.assertEqual(
            set(result["strata"]), {"new_animals", "known_animals_new_date", "all"}
        )
        self.assertAlmostEqual(
            result["strata"]["new_animals"]["summary"]["correlation"], 0.3
        )
        summary = result["summary"]
        expected = (
            0.5 * summary["inhale_f1"]
            + 0.2 * summary["exhale_f1"]
            + 0.2 * max(0.0, summary["correlation"])
            + 0.1 * math.exp(-summary["kl_ibi"])
        )
        self.assertAlmostEqual(result["composite"], expected)

    def test_no_strata_gives_empty_strata(self):
        result = build_result([_row("a", 3, 0.1, 0.1)], {3}, {})
        self.assertEqual(result["strata"], {})
