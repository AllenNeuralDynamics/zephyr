import json
import tempfile
import unittest
from pathlib import Path

from zephyr.benchmarks import collect

METRICS = {"correlation": 0.1, "inhale_f1": 0.1, "exhale_f1": 0.1, "kl_ibi": 1.0}


def _evaluation(corr: float, **extra) -> dict:
    summary = METRICS | {"correlation": corr} | extra
    return {"groups": {"all": {"recordings": ["r"], "summary": summary}}}


class CollectTests(unittest.TestCase):
    def _write(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_collect_tables_every_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root / "pixel" / "pca" / "blind" / "evaluation.json", _evaluation(0.1)
            )
            self._write(
                root / "facemap" / "motion" / "evaluation.json", _evaluation(0.3)
            )
            rows = collect.collect(root, runs_dir=None)
            by = {(r["method"], r["variant"]): r for r in rows}
            self.assertAlmostEqual(by[("pixel-pca", "blind")]["correlation_mean"], 0.1)
            self.assertAlmostEqual(by[("facemap", "motion")]["correlation_mean"], 0.3)
            self.assertTrue((root / "results.md").exists())

    def test_collect_aggregates_run_seeds_labels_archs_and_adds_head_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            for fold, arch, head in (("net", "tscan", None), ("zeph", "zephyr", 0.9)):
                for seed, corr in ((1, 0.4), (2, 0.6)):
                    run = runs / fold / f"seed-{seed}"
                    extra = {} if head is None else {"head_inhale_f1": head}
                    self._write(run / "evaluation.json", _evaluation(corr, **extra))
                    self._write(
                        run / "config.json",
                        {"fold": {"train_params": {"arch": arch}}, "seed": seed},
                    )
            rows = collect.collect(root / "out", runs)
            by = {(r["method"], r["variant"]): r for r in rows}
            self.assertAlmostEqual(by[("tscan", "net")]["correlation_mean"], 0.5)
            self.assertEqual(by[("zephyr", "zeph")]["n_seeds"], 2)
            head = by[("zephyr (onset head)", "zeph")]
            self.assertAlmostEqual(head["inhale_f1_mean"], 0.9)
            self.assertIsNone(head["correlation_mean"])
            self.assertNotIn(("zephyr (onset head)", "net"), by)
            text = (root / "out" / "results.md").read_text(encoding="utf-8")
            self.assertIn("Best per method", text)
