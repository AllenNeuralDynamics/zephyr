"""A project: a folder holding a dataset's clip lists and everything made from them.

Its clip lists are the ``*.toml`` files in it; beside them ``experiments/`` (each written
once, never edited), ``frames/`` (one frame per clip, for boxes), ``cache/`` (the
feature cache), ``runs/<experiment>/`` and ``predictions/<name>/``. Needs no Qt.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit

from ..config import FOLD_NAME, ClipList, Experiment, load

FOLDERS = ("experiments", "frames", "cache", "runs", "predictions")
RUN_FILES = ("history.json", "best.pt", "evaluation.json")


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


@dataclass
class Project:
    """The project folder *root* and the clip list being worked on."""

    root: Path
    clip_list: Path | None = None
    errors: dict[Path, str] = field(default_factory=dict)
    """Clip lists in the folder that do not load, and why."""

    @classmethod
    def open(cls, path: Path) -> "Project":
        """The project of *path*: a folder, or a clip list (the project is its
        folder). The folder's layout is created if it is missing."""
        path = Path(path).resolve()
        root = path.parent if path.suffix == ".toml" else path
        root.mkdir(parents=True, exist_ok=True)
        for name in FOLDERS:
            (root / name).mkdir(exist_ok=True)
        project = cls(root)
        lists = project.clip_lists()
        if path.suffix == ".toml":
            project.clip_list = path
        elif lists:
            project.clip_list = lists[0]
        return project

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    def clip_lists(self) -> list[Path]:
        """The clip lists in the folder (files with a ``[[clip]]`` table), sorted;
        those that do not load are listed too, with why in :attr:`errors`."""
        found = []
        self.errors = {}
        for path in sorted(self.root.glob("*.toml")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not re.search(r"^\s*\[\[clip\]\]", text, re.MULTILINE):
                continue
            found.append(path)
            try:
                load(ClipList, path, check_files=False)
            except ValueError as exc:
                self.errors[path] = str(exc)
        return found

    def load_list(self, path: Path | None = None) -> ClipList:
        """The clip list *path* (default: the project's), its files not checked."""
        path = path or self.clip_list
        if path is None:
            raise ValueError("the project has no clip list yet")
        return load(ClipList, path, check_files=False)

    def groups(self, path: Path | None = None) -> list[str]:
        """The groups of a clip list's clips in use, in file order."""
        return list(dict.fromkeys(c.group for c in self.load_list(path).resolve()))

    def experiments(self) -> list[Path]:
        return sorted(self.experiments_dir.glob("*.toml"))

    def runs(self) -> list[Path]:
        """Every run folder under ``runs/`` (one holding a history, a checkpoint or
        an evaluation), newest first."""
        found = {
            path.parent for name in RUN_FILES for path in self.runs_dir.rglob(name)
        }
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def checkpoints(self) -> list[Path]:
        return [run / "best.pt" for run in self.runs() if (run / "best.pt").is_file()]

    def run_dirs(self, experiment: Path, smoke: bool = False) -> list[Path]:
        """Where *experiment*'s (fold, seed) runs train, as ``zephyr run`` puts them."""
        loaded = load(Experiment, experiment)
        output = loaded.output_dir / "smoke" if smoke else loaded.output_dir
        return [
            output.resolve() / fold.name / f"seed-{seed}"
            for fold in loaded.fold
            for seed in loaded.seeds
        ]

    def new_experiment(
        self,
        name: str,
        *,
        train: list[str],
        test: dict[str, list[str]],
        val: list[str] | None = None,
        seeds: tuple[int, ...] = (1,),
        train_params: dict | None = None,
        machine: dict | None = None,
        clip_list: Path | None = None,
    ) -> Path:
        """Write ``experiments/<name>.toml``: one fold training on the *train*
        groups of the clip list, early-stopping on *val* and scoring each *test*
        group. It is validated before it is written, and never overwrites one."""
        if not FOLD_NAME.match(name):
            raise ValueError(
                f"{name!r} must be a plain file name (letters, digits, '.', '_', '-')"
            )
        path = self.experiments_dir / f"{name}.toml"
        if path.exists():
            raise ValueError(
                f"{path.name} exists; an experiment is never edited, so give the new "
                "one another name"
            )
        source = clip_list or self.clip_list
        if source is None:
            raise ValueError("the project has no clip list yet")
        clips = _relative(source, self.experiments_dir)
        doc = tomlkit.document()
        doc.add(tomlkit.comment(f"Experiment {name}, written by the zephyr app."))
        doc.add("seeds", list(seeds))
        doc.add("output_dir", _relative(self.runs_dir / name, self.experiments_dir))
        doc.add("features_dir", _relative(self.cache_dir, self.experiments_dir))
        for key, value in (machine or {}).items():
            if value is not None:
                doc.add(key, value)
        fold = tomlkit.table()
        fold.add("name", name)
        sources = tomlkit.aot()
        entry = tomlkit.table()
        entry.add("clips", clips)
        entry.add("groups", list(train))
        sources.append(entry)
        fold.add("train", sources)
        if val:
            fold.add("val", {"clips": clips, "groups": list(val)})
        tests = tomlkit.table()
        for group_name, groups in test.items():
            inline = tomlkit.inline_table()
            inline.update({"clips": clips, "groups": list(groups)})
            tests.add(group_name, inline)
        fold.add("test", tests)
        if train_params:
            fold.add("train_params", dict(train_params))
        folds = tomlkit.aot()
        folds.append(fold)
        doc.add("fold", folds)
        text = tomlkit.dumps(doc)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        try:
            load(Experiment, path)
        except ValueError:
            path.unlink()
            raise
        return path
