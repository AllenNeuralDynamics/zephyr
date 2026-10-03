import subprocess
import sys
import unittest


class ModuleEntrypointTests(unittest.TestCase):
    """``zephyr benchmark`` launches ``python -m zephyr.train`` / ``zephyr.evaluate``.

    A module without a ``__main__`` block runs nothing and exits 0, which looks
    exactly like a successful job.
    """

    def _usage(self, module: str) -> str:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_train_runs_as_a_module(self):
        self.assertIn("usage", self._usage("zephyr.train").lower())

    def test_evaluate_runs_as_a_module(self):
        self.assertIn("usage", self._usage("zephyr.evaluate").lower())
