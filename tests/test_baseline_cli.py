import json
import tempfile
import unittest
from pathlib import Path

from zephyr import baseline
from zephyr.benchmark import DEFAULT_CONFIG, load_config


class NetCommandTests(unittest.TestCase):
    def test_tscan_command(self):
        config = baseline.net_config(
            load_config(DEFAULT_CONFIG), "tscan", Path("out"), [17]
        )
        job = config.jobs()[0]
        command = baseline.net_train_command(
            config, job, "tscan", lr=1e-3, resume=False
        )
        self.assertEqual(command[command.index("--arch") + 1], "tscan")
        self.assertEqual(command[command.index("--channels") + 1], "gray")
        self.assertEqual(command[command.index("--w-onset") + 1], "0.0")
        self.assertEqual(command[command.index("--time-stretch") + 1], "1")
        self.assertEqual(command[command.index("--lr") + 1], "0.001")
        self.assertEqual(command[command.index("--val-fraction") + 1], "0")
        self.assertTrue(str(config.run_dir(job)).startswith(str(Path("out") / "tscan")))

    def test_physnet_command_uses_short_window_and_single_scale(self):
        config = baseline.net_config(
            load_config(DEFAULT_CONFIG), "physnet", Path("out"), [17]
        )
        command = baseline.net_train_command(
            config, config.jobs()[0], "physnet", lr=None, resume=False
        )
        self.assertEqual(command[command.index("--window") + 1], "128")
        self.assertEqual(command[command.index("--scales") + 1], "1")
        self.assertNotIn("--lr", command)


class CollectTests(unittest.TestCase):
    def _write(self, path: Path, corr: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "correlation": corr,
            "inhale_f1": 0.1,
            "exhale_f1": 0.1,
            "kl_ibi": 1.0,
        }
        path.write_text(
            json.dumps(
                {
                    "summary_by_session": summary,
                    "strata": {"all": {"sessions": [1], "summary": summary}},
                }
            )
        )

    def test_collect_tables_every_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "pixel" / "pca" / "blind" / "evaluation.json", 0.1)
            self._write(root / "facemap" / "motion" / "evaluation.json", 0.3)
            self._write(root / "tscan" / "runs" / "a" / "evaluation.json", 0.4)
            self._write(root / "tscan" / "runs" / "b" / "evaluation.json", 0.6)
            rows = baseline.collect(root, zephyr_results=None)
            by = {(r["method"], r["variant"]): r for r in rows}
            self.assertAlmostEqual(by[("pixel-pca", "blind")]["correlation_mean"], 0.1)
            self.assertAlmostEqual(by[("tscan", "-")]["correlation_mean"], 0.5)
            self.assertEqual(by[("tscan", "-")]["n_seeds"], 2)
            self.assertTrue((root / "results.md").exists())


class HeadAndBestTests(unittest.TestCase):
    def _zephyr(self, root: Path) -> Path:
        results = root / "zephyr" / "results.json"
        results.parent.mkdir(parents=True)
        metrics = {
            f"{m}_{s}": v
            for m, v in (
                ("correlation", 0.9),
                ("inhale_f1", 0.8),
                ("exhale_f1", 0.85),
                ("kl_ibi", 0.05),
            )
            for s in ("mean", "sd")
        }
        results.write_text(
            json.dumps(
                {
                    "summary_across_seeds": [
                        {
                            "representation": "gray+diff",
                            "objective": "multitask",
                            "stratum": "all",
                            "n_seeds": 2,
                        }
                        | metrics
                    ]
                }
            )
        )
        for seed, f1 in ((1, 0.90), (2, 0.94)):
            run = root / "zephyr" / "runs" / f"gray-diff__multitask__seed-{seed}"
            run.mkdir(parents=True)
            (run / "head_evaluation.json").write_text(
                json.dumps({"strata": {"all": {"summary": {"head_inhale_f1": f1}}}})
            )
        return results

    def test_collect_adds_onset_head_rows_and_best_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = baseline.collect(root / "out", self._zephyr(root))
            head = [r for r in rows if r["method"] == "zephyr (onset head)"]
            self.assertEqual(len(head), 1)
            self.assertEqual(head[0]["variant"], "gray+diff/multitask")
            self.assertAlmostEqual(head[0]["inhale_f1_mean"], 0.92)
            self.assertEqual(head[0]["n_seeds"], 2)
            self.assertIsNone(head[0]["correlation_mean"])
            text = (root / "out" / "results.md").read_text(encoding="utf-8")
            self.assertIn("Best per method", text)
            self.assertIn("0.920", text)
