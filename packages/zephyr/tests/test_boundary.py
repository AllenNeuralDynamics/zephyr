"""zephyr runs only the zephyr network and never depends on zephyr-benchmarks."""

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.evaluate import load_checkpoint
from zephyr.model import BreathingNet

SRC = Path(__file__).resolve().parents[1] / "src" / "zephyr"
MODULES = sorted(
    "zephyr." + ".".join(p.relative_to(SRC).with_suffix("").parts)
    for p in SRC.rglob("*.py")
    if p.name not in ("__init__.py", "_version.py")
)


class BoundaryTests(unittest.TestCase):
    def test_no_module_imports_the_benchmarks(self):
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    if node.level:
                        names += [alias.name for alias in node.names]
                for name in names:
                    if name == "benchmarks" or name.startswith("zephyr.benchmarks"):
                        offenders.append(f"{path.name}: {name}")
        self.assertEqual(offenders, [])

    def test_no_module_names_a_comparison_network(self):
        pattern = re.compile(r"ts-?can|physnet", re.IGNORECASE)
        offenders = [
            p.name
            for p in sorted(SRC.rglob("*.py"))
            if pattern.search(p.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_importing_every_module_leaves_the_benchmarks_unloaded(self):
        code = (
            "import importlib, sys\n"
            f"for name in {MODULES!r}:\n"
            "    importlib.import_module(name)\n"
            "print(sorted(m for m in sys.modules if m.startswith('zephyr.benchmarks')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


class LoadCheckpointTests(unittest.TestCase):
    def _save(self, root: Path, **extra) -> Path:
        channels = ChannelSet.parse("gray")
        state = {
            "model": BreathingNet(channels=channels).state_dict(),
            "channels": list(channels.names),
            "feature_config": {"channel_names": list(CHANNEL_NAMES)},
            "mean": np.zeros(len(CHANNEL_NAMES)),
            "std": np.ones(len(CHANNEL_NAMES)),
        } | extra
        path = root / "best.pt"
        torch.save(state, path)
        return path

    def test_loads_zephyr_checkpoints_with_or_without_an_arch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in ({}, {"arch": "zephyr", "arch_kwargs": {}}):
                path = self._save(Path(tmp), **extra)
                model, *_ = load_checkpoint(path, torch.device("cpu"))
                self.assertIsInstance(model, BreathingNet)

    def test_refuses_any_other_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._save(Path(tmp), arch="other", arch_kwargs={"x": 1})
            with self.assertRaisesRegex(SystemExit, "only zephyr checkpoints"):
                load_checkpoint(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
