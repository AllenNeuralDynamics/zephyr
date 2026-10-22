"""A project: a folder holding a dataset's clip lists and everything made from them.

Its clip lists are the ``*.toml`` files in it; beside them ``experiments/`` (each written
once, never edited), ``frames/`` (one frame per clip, for boxes), ``cache/`` (the
feature cache), ``runs/<experiment>/`` and ``predictions/<name>/``. Needs no Qt.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit

from .. import features
from ..clips import natural_key
from ..config import FOLD_NAME, ClipList, Derive, Experiment, load

FOLDERS = ("experiments", "frames", "cache", "runs", "predictions")
RUN_FILES = ("history.json", "best.pt", "evaluation.json")
DONE, PARTLY, TODO = "done", "partly", "todo"


@dataclass(frozen=True)
class Step:
    """How far one step of the project is: :data:`DONE`, :data:`PARTLY` or
    :data:`TODO`, and what that rests on."""

    state: str
    detail: str


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def run_status(run: Path) -> str:
    """Where a run folder stands: epochs logged of those planned, the epoch
    ``best.pt`` was kept from, and whether it is scored (empty for no run yet)."""
    run = Path(run)
    try:
        rows = json.loads((run / "history.json").read_text())
    except (OSError, ValueError):
        rows = []
    if not rows and not (run / "best.pt").is_file():
        return ""
    try:
        config = json.loads((run / "config.json").read_text())
        planned = f" of {config['fold']['train_params']['epochs']}"
    except (OSError, ValueError, KeyError, TypeError):
        planned = ""
    parts = [f"{len(rows)}{planned} epochs"]
    best = [r["epoch"] + 1 for r in rows if r.get("since_best") == 0]
    if best:
        parts.append(f"best.pt from epoch {best[-1]}")
    scored = (run / "evaluation.json").is_file()
    parts.append("scored" if scored else "not scored")
    return ", ".join(parts)


def run_label(run: Path, runs_dir: Path) -> str:
    """A run's name for a list: ``fold · seed N``, the experiment first when the fold
    is named otherwise, and ``· smoke`` for a smoke run; a run outside *runs_dir* is
    its path."""
    run = Path(run)
    try:
        parts = run.resolve().relative_to(Path(runs_dir).resolve()).parts
    except ValueError:
        return run.as_posix()
    if len(parts) < 3 or not parts[-1].startswith("seed-"):
        return "/".join(parts)
    experiment, *middle, fold, seed = parts
    smoke = middle == ["smoke"]
    name = fold if fold == experiment else f"{experiment} / {fold}"
    return f"{name} · seed {seed[5:]}" + (" · smoke" if smoke else "")


def is_smoke(run: Path, runs_dir: Path) -> bool:
    try:
        return "smoke" in Path(run).resolve().relative_to(runs_dir.resolve()).parts
    except ValueError:
        return False


@dataclass(frozen=True)
class Found:
    """A video a scan would list: its group (None where the pattern misses) and
    whether its timestamps and breathing files exist."""

    video: Path
    group: str | None
    timestamps: bool
    sensor: bool


def preview_scan(folder: Path, glob: str, derive: Derive) -> list[Found]:
    """What ``clips scan`` would list from *folder*, without writing anything."""
    folder = Path(folder)
    if not glob or not folder.is_dir():
        return []
    found = []
    for video in sorted(folder.glob(glob), key=natural_key):
        fields = derive.fields(video)
        if fields is None:
            found.append(Found(video, None, False, False))
            continue
        found.append(
            Found(
                video,
                fields["group"],
                (video.parent / derive.timestamps.format(**fields)).is_file(),
                (video.parent / derive.thermistor.format(**fields)).is_file(),
            )
        )
    return found


def _scored(path: Path, entry) -> dict | None:
    from ..evaluate import truth_events
    from ..quality_control.metrics import find_events

    if not path.is_file():
        return None
    trace = pd.read_parquet(path)
    time = trace["time"].to_numpy()
    signal = trace["signal"].to_numpy()
    onset = trace["onset"].to_numpy() if "onset" in trace else None
    out = {
        "time": time,
        "signal": signal,
        "inhale": time[find_events(time, signal, onset).inhale],
        "onset": onset,
        "entry": entry,
        "truth": None,
        "truth_inhale": None,
    }
    if entry.target is not None and entry.target.is_file():
        truth = pd.read_parquet(entry.target)["signal"].to_numpy()
        n = min(len(truth), len(time))
        out |= {"time": time[:n], "signal": signal[:n], "truth": truth[:n]}
    if entry.events is not None and entry.events.is_file():
        out["truth_inhale"] = truth_events(entry).inhale
    return out


@dataclass(frozen=True)
class CacheNeed:
    """What preprocessing would add to a cache: *bytes* for the *todo* clips not
    cached yet (*unknown* of them without a readable timestamps file, not counted),
    beside the *cached* ones."""

    bytes: int
    todo: int
    cached: int
    unknown: int


CACHE_BYTES_PER_FRAME = 20
"""Besides its crop, what each frame adds to an entry (its time, the output grid,
the target), in bytes."""


def cache_need(clip_lists: list[Path], cache_dir: Path) -> CacheNeed:
    """The space preprocessing *clip_lists* into *cache_dir* would take: per clip
    not cached, the frames it keeps (its timestamps' frames nearest the recipe's
    ``select_fs`` grid, as preprocessing picks them) times its crop's pixels, one
    byte each, as the cache stores them (:mod:`zephyr.features`)."""
    from ..preprocess import select_anchor_indices

    total = todo = cached = unknown = 0
    for path in clip_lists:
        clip_list = load(ClipList, path, check_files=False)
        for clip in clip_list.resolve():
            if clip.video is None or clip.box is None:
                continue  # preprocessing refuses it
            if features.lookup(clip, clip_list.preprocess, cache_dir) is not None:
                cached += 1
                continue
            todo += 1
            try:
                times = pd.read_parquet(clip.timestamps, columns=["Time"])
                frames = len(
                    select_anchor_indices(
                        times["Time"].to_numpy(), clip_list.preprocess.select_fs
                    )
                )
            except (OSError, ValueError, KeyError):
                unknown += 1
                continue
            width, height = clip.box[2], clip.box[3]
            total += frames * (width * height + CACHE_BYTES_PER_FRAME)
    return CacheNeed(total, todo, cached, unknown)


def human_bytes(count: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if count < 1000 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "bytes" else f"{count:.1f} {unit}"
        count /= 1000
    return ""


def split_groups(
    groups: list[str], proportions: tuple[float, float, float], seed: int = 1
) -> tuple[list[str], list[str], list[str]]:
    """Draw *groups* (sessions) into train, validation and test in *proportions*
    (any scale; a zero validation share means none), with *seed*. Counts are rounded
    by largest remainder, and train and test (and validation, if asked for) get at
    least one group each; each side keeps the groups' order."""
    shares = np.asarray(proportions, float)
    if shares.shape != (3,) or (shares < 0).any() or shares[[0, 2]].min() <= 0:
        raise ValueError("train and test need a share above 0, validation 0 or more")
    wanted = shares > 0
    if len(groups) < wanted.sum():
        raise ValueError(
            f"{len(groups)} group(s) cannot fill {wanted.sum()} sides; add sessions "
            "or drop validation"
        )
    exact = shares / shares.sum() * len(groups)
    counts = np.floor(exact).astype(int)
    for k in np.argsort(-(exact - counts), kind="stable")[: len(groups) - counts.sum()]:
        counts[k] += 1
    for k in np.flatnonzero(wanted & (counts == 0)):
        counts[k] += 1
        counts[np.argmax(counts)] -= 1
    order = np.random.default_rng(seed).permutation(len(groups))
    bounds = np.cumsum(counts)[:-1]
    sides = [sorted(part) for part in np.split(order, bounds)]
    return tuple([groups[i] for i in side] for side in sides)


@dataclass
class Project:
    """The project folder *root* and the clip list being worked on."""

    root: Path
    clip_list: Path | None = None
    errors: dict[Path, str] = field(default_factory=dict)
    """Clip lists in the folder that do not load, and why."""

    @staticmethod
    def exists(path: Path) -> bool:
        """Whether *path* (a folder, or a clip list in one) is a project already: it
        has a clip list or any of the project's folders. Opening one that is not
        makes it one, so the app asks first."""
        path = Path(path)
        root = path.parent if path.suffix == ".toml" else path
        if path.suffix == ".toml" or any((root / name).is_dir() for name in FOLDERS):
            return True
        return root.is_dir() and bool(Project(root).clip_lists())

    @classmethod
    def open(cls, path: Path, *, create: bool = False) -> "Project":
        """The project of *path*: a folder, or a clip list (the project is its
        folder). Nothing is written to open one; with *create* (a new project) the
        folder and its layout are made. Otherwise each folder appears when the step
        that fills it first runs (a dataset's own folder opened to check its list
        stays as it was)."""
        path = Path(path).resolve()
        root = path.parent if path.suffix == ".toml" else path
        if create:
            root.mkdir(parents=True, exist_ok=True)
            for name in FOLDERS:
                (root / name).mkdir(exist_ok=True)
        elif not root.is_dir():
            raise FileNotFoundError(f"{root} does not exist")
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

    def run_of(self, run: Path) -> tuple[Path, str, int, bool] | None:
        """The experiment, fold, seed and smoke flag *run* was trained for, from
        the project's experiments (None for a run none of them names)."""
        run = Path(run).resolve()
        for experiment in self.experiments():
            for smoke in (False, True):
                try:
                    dirs = self.run_dirs(experiment, smoke=smoke)
                except ValueError:
                    break
                if run in dirs:
                    return experiment, run.parent.name, int(run.name[5:]), smoke
        return None

    def scored_trace(self, run: Path, clip_id: str) -> dict | None:
        """What *run* was scored on for its test clip *clip_id*, to draw: the trace
        it saved (``traces/``, with ``--save-traces``), the sensor's, both sets of
        inhales and the clip's cache entry; None for a run of no experiment here, a
        clip not cached, or no trace saved."""
        from ..evaluate import TRACES, trace_name  # imports torch: only when drawn

        found = self.run_of(run)
        if found is None:
            return None
        experiment = load(Experiment, found[0])
        for path in dict.fromkeys(
            s.clips
            for fold in experiment.fold
            for s in [*fold.train, *fold.test.values()]
        ):
            clip_list = load(ClipList, path, check_files=False)
            for clip in clip_list.resolve():
                if clip.video is None or clip.box is None:
                    continue
                entry = features.lookup(
                    clip, clip_list.preprocess, experiment.features_dir
                )
                if entry is not None and entry.clip_id == clip_id:
                    return _scored(Path(run) / TRACES / trace_name(entry), entry)
        return None

    def steps(self) -> dict[str, Step]:
        """Each step's state, read from the folder alone (keyed by page title)."""
        out = {"Project": Step(DONE, str(self.root))}
        out.update(self._list_steps())
        real = [
            r for r in self.runs() if "smoke" not in r.relative_to(self.runs_dir).parts
        ]
        trained = [r for r in real if (r / "best.pt").is_file()]
        experiments = self.experiments()
        out["Train"] = Step(
            DONE if trained else PARTLY if experiments or real else TODO,
            f"{len(experiments)} experiment(s), {len(trained)} trained run(s)",
        )
        scored = [r for r in real if (r / "evaluation.json").is_file()]
        out["Results"] = Step(
            DONE if scored else TODO, f"{len(scored)} evaluated run(s)"
        )
        folders = [
            p.parent
            for p in self.predictions_dir.glob("*/predictions.json")
            if any(p.parent.glob("*.prediction.parquet"))
        ]
        out["Predict"] = Step(
            DONE if folders else TODO, f"{len(folders)} folder(s) of predictions"
        )
        return out

    def _list_steps(self) -> dict[str, Step]:
        """Clips (the list loads), Boxes (every clip in use has one) and Breathing
        (a detector is named, by ``[events]`` or each clip, and every sensed clip in
        use is decided: reviewed, or marked include or exclude; named or some decided
        is partly done)."""
        if self.clip_list is None:
            nothing = Step(TODO, "no clip list yet")
            return {"Clips": nothing, "Boxes": nothing, "Breathing": nothing}
        try:
            clip_list = self.load_list()
        except ValueError as exc:
            broken = Step(TODO, f"{self.clip_list.name} does not load")
            return {
                "Clips": Step(PARTLY, str(exc)),
                "Boxes": broken,
                "Breathing": broken,
            }
        clips = clip_list.resolve()
        sensed = [c for c in clips if c.labelled]
        steps = {"Clips": Step(DONE, f"{len(clips)} clip(s), {len(sensed)} sensed")}
        boxed = sum(c.box is not None for c in clips)
        steps["Boxes"] = Step(
            TODO if not boxed else DONE if boxed == len(clips) else PARTLY,
            f"{boxed} of {len(clips)} clip(s) boxed",
        )
        in_use = [c for c in clip_list.clip if c.selection != "exclude"]
        named = "events" in clip_list.model_fields_set or all(
            c.events is not None for c in in_use
        )
        reviewed = sum(
            c.reviewed or c.selection is not None
            for c in in_use
            if c.thermistor is not False
        )
        if not sensed:
            state = TODO
        elif named and reviewed >= len(sensed):
            state = DONE
        else:
            state = PARTLY if named or reviewed else TODO
        method = sensed[0].events.method if sensed else "-"
        steps["Breathing"] = Step(
            state,
            f"{method}{'' if named else ' (the built-in default)'}, "
            f"{reviewed} of {len(sensed)} sensed clip(s) decided (reviewed, or "
            "marked include or exclude)",
        )
        return steps

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
