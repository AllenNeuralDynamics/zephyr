import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from zephyr.benchmarks import report

REPRESENTATIONS = ("gray", "gray+diff", "gray+flow", "gray+diff+flow")
SEEDS = (17, 42, 101, 202, 314)


def _write_run(root: Path, representation: str, objective: str, seed: int) -> None:
    fold = f"{representation.replace('+', '-')}__{objective}"
    run = root / fold / f"seed-{seed}"
    run.mkdir(parents=True)
    multitask = objective == "multitask"
    channels = representation
    (run / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "fold": {
                    "train_params": {
                        "channels": channels,
                        "w_onset": 0.5 if multitask else 0.0,
                    }
                },
            }
        )
    )
    summary = {"correlation": 0.8, "inhale_f1": 0.85}
    if multitask:
        summary["head_inhale_f1"] = 0.9
    groups = {
        name: {"recordings": [], "summary": summary}
        for name in ("new_animals", "known_animals_new_date", "all")
    }
    (run / "evaluation.json").write_text(json.dumps({"groups": groups}))
    history = [{"epoch": e, "train_loss": 1.0 / (e + 1)} for e in range(30)]
    (run / "history.json").write_text(json.dumps(history))


class ReportTests(unittest.TestCase):
    def test_report_is_built_from_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rep, obj, seed in product(
                REPRESENTATIONS, ("signal", "multitask"), SEEDS
            ):
                _write_run(root / "runs", rep, obj, seed)
            runs = report.Runs(root / "runs", root / "out")
            self.assertEqual(len(runs.jobs()), 40)
            self.assertEqual(runs.seeds, list(SEEDS))
            report.build_report(runs)
            for name in ("final-report-metrics.png", "final-report-summary.md"):
                self.assertTrue((root / "out" / name).exists(), name)
            results = json.loads((root / "out/final-report-results.json").read_text())
            head = [r for r in results["per_network"] if "head_inhale_f1" in r]
            self.assertEqual(len(head), 40)
