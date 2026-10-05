import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "zephyr"
LIMIT = 10


def docstring_lines(path: Path) -> int:
    """Lines in a module docstring, not counting a trailing CLI usage block."""
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=True)
    if not doc:
        return 0
    lines = doc.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "CLI" and set(lines[i + 1].strip()) == {"-"}:
            lines = lines[:i]
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return len(lines)


class ModuleDocstringTests(unittest.TestCase):
    def test_module_docstrings_are_short(self):
        counts = {
            str(p.relative_to(SRC)): docstring_lines(p)
            for p in sorted(SRC.rglob("*.py"))
            if p.name != "_version.py"
        }
        too_long = {name: n for name, n in counts.items() if n > LIMIT}
        self.assertEqual(too_long, {}, f"module docstrings over {LIMIT} lines")


if __name__ == "__main__":
    unittest.main()
