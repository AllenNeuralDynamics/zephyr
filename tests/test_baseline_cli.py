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
