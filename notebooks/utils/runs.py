"""The benchmark's runs, a local copy of their test clips, and what is computed from them.

Which runs are compared, and under which labels, is a results manifest of
``zephyr-benchmarks`` (``configs/results/``, :data:`MANIFEST`). What the notebook needs
of a run (its checkpoint and scores) is downloaded once into ``cache/runs/``
(the bucket is public: read unsigned, no credentials); traces on the test clips are
the runs' own ``traces/`` where saved, else predicted here (inference only), and the tables of :mod:`zephyr.benchmarks.analysis` built from them,
each cached the first time. The clips come from a local copy of the dataset and its
feature cache (:mod:`.test_sets`).
"""

import functools
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zephyr import features
from zephyr.benchmarks import analysis, baselines, common, timing
from zephyr.benchmarks.config import ReportedRun, ResultsManifest
from zephyr.benchmarks.runner import load_checkpoint
from zephyr.config import ClipList, TrainParams, load
from zephyr.evaluate import (
    HEAD_THRESHOLD,
    TRACES,
    aggregate_recordings,
    choose_head_threshold,
    head_event_indices,
    head_inhale_counts,
    trace_name,
)
from zephyr.infer import predict_clip

from . import test_sets
from .test_sets import CACHE

REPO = Path(__file__).resolve().parents[2]
MANIFEST_ENV = "ZEPHYR_RESULTS"
MANIFEST = Path(
    os.environ.get(
        MANIFEST_ENV,
        REPO / "packages/zephyr-benchmarks/configs/results/corpus-v3.toml",
    )
)
"""The results manifest the notebook reports, unless ``ZEPHYR_RESULTS`` names another."""
RESULTS: ResultsManifest = load(ResultsManifest, MANIFEST)
S3_RUNS = RESULTS.root.rstrip("/")
"""Where the manifest's runs were uploaded."""
LOCAL_RUNS = CACHE / "runs"
PREDICTIONS = CACHE / "walkthrough"
HEAD_ENV = "ZEPHYR_HEAD_THRESHOLD"
HEAD_MODE = os.environ.get(HEAD_ENV, "validation")
"""``validation`` (the default, what the manuscript reports): each run's onset threshold
is chosen on its own validation clips (:func:`head_threshold`), as training now does;
any other value (e.g. ``default``) keeps the scorer's 0.5."""
DERIVED = PREDICTIONS / "head-validation" if HEAD_MODE == "validation" else PREDICTIONS
"""Where the scores, tables and errors of a run whose onset threshold was chosen on
validation are cached, apart from those at the scorer's (:func:`_derived`)."""
DEFAULT_ROOT = Path("C:/Data/zephyr-local-v2")
"""The local copy of the dataset and its cache, unless ``ZEPHYR_TEST_ROOT`` names one."""
METHODS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.method}
"""Every compared method, in the order tables list them."""
VARIANTS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.variant}
"""The first method trained differently, compared with it alone."""
LEARNING_CURVE: dict[str, ReportedRun] = {r.label: r for r in RESULTS.learning_curve}
"""The first method trained on parts of its training set."""
ALL = METHODS | VARIANTS | LEARNING_CURVE


def cv_name(label: str, fold: str) -> str:
    """The name of a cross-validation method's run on one fold."""
    return f"{label} | {fold}"


CV_VARIANTS: dict[str, ReportedRun] = (
    {r.label: r for r in RESULTS.cross_validation.variant}
    if RESULTS.cross_validation is not None
    else {}
)
"""The first method trained differently on each cross-validation fold."""
CV_RUNS: dict[str, ReportedRun] = (
    {
        cv_name(m.label, f): m.model_copy(update={"fold": f})
        for f in RESULTS.cross_validation.folds
        for m in (*RESULTS.cross_validation.method, *CV_VARIANTS.values())
    }
    if RESULTS.cross_validation is not None
    else {}
)
"""Each cross-validation method and variant on each fold (:func:`cv_name`)."""
RUNS = ALL | CV_RUNS
"""Every run by name: :data:`ALL` and :data:`CV_RUNS`."""
REFERENCE = RESULTS.method[0].label
"""The method every other is compared with (Zephyr)."""
SINGLE = cv_name(REFERENCE, RESULTS.cross_validation.folds[0])
"""The trained network the single-network analyses use (example clips, error cases):
the reference on the first cross-validation fold."""


def url(m: ReportedRun) -> str:
    return f"{S3_RUNS}/{m.run}/"


PIXEL_BIN = 2
"""The pixel methods' spatial binning, as the benchmark ran them."""


def root() -> Path:
    """The local dataset copy, named as the original so its cache entries are found."""
    path = Path(os.environ.get(test_sets.ROOT_ENV, DEFAULT_ROOT))
    test_sets.use_copy(path)
    return path


@functools.cache
def _s3():
    import fsspec

    if S3_RUNS.startswith("s3://"):
        return fsspec.filesystem("s3", anon=True)  # read only, and the bucket is public
    return fsspec.filesystem("file")


def _fetch(key: str) -> Path | None:
    """``<run>/<path>`` from the runs' root into :data:`LOCAL_RUNS`, once; ``None`` if
    absent (the run has not finished)."""
    local = LOCAL_RUNS / key
    if local.exists():
        return local
    remote = f"{S3_RUNS.removeprefix('s3://')}/{key}"
    if not _s3().exists(remote):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    _s3().get(remote, str(part), recursive=_s3().isdir(remote))
    part.replace(local)
    return local


def dataset_file(name: str, url: str) -> Path:
    """A file of the dataset (*url*: its root, the spec's ``root``): the local copy's
    when it has one, else downloaded once into ``cache/dataset/``."""
    local = root() / "dataset" / name
    if local.is_file():
        return local
    cached = CACHE / "dataset" / name
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        _s3().get(f"{url.removeprefix('s3://').rstrip('/')}/{name}", str(cached))
    return cached


def folder(m: ReportedRun) -> str:
    """A run's folder under the root: ``<run>/<fold>``."""
    return f"{m.run}/{m.fold or RESULTS.fold}"


@functools.cache
def seeds(folder: str) -> tuple[str, ...]:
    """The seed folders a network run has finished (a ``best.pt`` and its scores), in
    its *folder* (``<run>/<fold>``, :func:`folder`)."""
    local = sorted(
        p.parent.name for p in LOCAL_RUNS.glob(f"{folder}/*/evaluation.json")
    )
    try:
        found = _s3().glob(
            f"{S3_RUNS.removeprefix('s3://')}/{folder}/seed-*/evaluation.json"
        )
    except Exception:  # noqa: BLE001 -- offline: whatever was downloaded before
        return tuple(local)
    return tuple(sorted({Path(p).parent.name for p in found} | set(local)))


def evaluation(name: str, seed: str | None = None) -> dict | None:
    """The run's own ``evaluation.json`` (scored on Code Ocean), or ``None`` if the
    run has not finished."""
    return _evaluation(RUNS[name], seed)


def _evaluation(m: ReportedRun, seed: str | None = None) -> dict | None:
    if m.kind == "network":
        found = seeds(folder(m))
        if not found:
            return None
        key = f"{folder(m)}/{seed or found[0]}/evaluation.json"
    else:
        key = f"{folder(m)}/{m.kind}/{m.variant}/evaluation.json"
    path = _fetch(key)
    return json.loads(path.read_text()) if path else None


def training(name: str) -> pd.DataFrame:
    """The sessions a network run trained on (its first seed's ``train_videos.json``),
    with their ``subject`` and ``rig`` from the split."""
    m = RUNS[name]
    path = _fetch(f"{folder(m)}/{seeds(folder(m))[0]}/train_videos.json")
    stems = [Path(v).stem for v in json.loads(path.read_text())]
    sessions = sorted({stem.rsplit("_span", 1)[0] for stem in stems})
    folds = pd.read_parquet(root() / "dataset" / "raw" / "folds.parquet").set_index(
        "session_id"
    )
    return pd.DataFrame(
        {
            "session": sessions,
            "subject": folds.loc[sessions, "subject_id"].astype(str).to_numpy(),
            "rig": folds.loc[sessions, "rig"].to_numpy(),
        }
    )


def available(names) -> list[str]:
    """The methods of *names* whose run has finished."""
    return [n for n in names if evaluation(n) is not None]


def runs_table(names=ALL) -> pd.DataFrame:
    """One row per method: its S3 folder, seeds finished, and whether it has a head."""
    rows = []
    for name in names:
        m = RUNS[name]
        done = (
            seeds(folder(m))
            if m.kind == "network"
            else ("—",)
            if evaluation(name)
            else ()
        )
        rows.append(
            {
                "method": name,
                "run": m.run + (f" ({m.kind}/{m.variant})" if m.variant else ""),
                "S3": url(m),
                "seeds": ", ".join(done) if done else "not finished",
                "inhale events": "onset head" if m.head else "DSP on the trace",
            }
        )
    return pd.DataFrame(rows)


# --- scores, re-scored here by zephyr's scorer -----------------------------------------


def _subjects() -> dict[str, str]:
    folds = pd.read_parquet(root() / "dataset" / "raw" / "folds.parquet")
    return dict(zip(folds["session_id"], folds["subject_id"], strict=True))


def scores(names=ALL, level: str = "clips") -> pd.DataFrame:
    """Per-clip (``clips``) or per-recording (``per_recording``) scores of every
    finished method in *names* on its test clips (:func:`rescored`); one ``inhale``
    column holds the F1 each method reports (the head's if it has one)."""
    return _scores([(name, name) for name in names], level)


def cv_scores(level: str = "per_recording", variants: bool = False) -> pd.DataFrame:
    """:func:`scores` of every cross-validation method (or, with *variants*, every
    variant) on every fold of the manifest's ``[cross_validation]``, with a ``fold``
    column; the ``set`` is the fold's own."""
    cv = RESULTS.cross_validation
    if cv is None:
        return pd.DataFrame()
    frames = [
        _scores([(m.label, cv_name(m.label, f))], level).assign(fold=f)
        for f in cv.folds
        for m in (cv.variant if variants else cv.method)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _derived(name: str) -> Path:
    """Where run *name*'s scores and tables are cached: :data:`DERIVED` when its onset
    threshold is not the scorer's, else with every other run's."""
    return DERIVED if head_threshold(name) != HEAD_THRESHOLD else PREDICTIONS


def _tag(name: str) -> str:
    """What run *name*'s cached files are named by: its run, fold and variant (never
    its label, which a manifest may point at another run)."""
    m = RUNS[name]
    tag = f"{m.run}-{m.fold or RESULTS.fold}-{m.variant.replace('/', '-') or 'net'}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


def _scores_path(name: str, folder: Path | None = None) -> Path:
    return (folder or _derived(name)) / f"scores-{_tag(name)}.parquet"


def rescored(name: str) -> pd.DataFrame:
    """Per-clip scores of run *name* on the test clips its ``evaluation.json`` lists
    (its ``clip_id``, ``recording`` and ``group``), scored again from the cached
    predictions by zephyr's scorer (:func:`zephyr.benchmarks.analysis.score_prediction`),
    so that every run is scored by the current rule whenever it was trained; cached. At
    another onset threshold than the scorer's, the scores made at it are kept and only
    the head's inhale F1 is scored again (:func:`zephyr.evaluate.head_inhale_counts`)."""
    path = _scores_path(name)
    if path.is_file():
        return pd.read_parquet(path)
    base = _scores_path(name, PREDICTIONS)
    if path != base and base.is_file():
        out = pd.read_parquet(base)
        threshold = head_threshold(name)
        f1 = []
        for clip_id in out["clip_id"]:
            stem = clip_id.split("/")[-1]
            p = prediction(name, stem)
            f1.append(
                head_inhale_counts(
                    truth_entry(stem), p["times"], p["onset"], threshold
                )["f1"]
            )
        out["head_inhale_f1"] = f1
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
        return out
    rows = []
    for c in evaluation(name)["clips"]:
        if c["group"] not in SETS:
            continue
        stem = c["clip_id"].split("/")[-1]
        score = analysis.score_prediction(
            truth_entry(stem), prediction(name, stem), RUNS[name].head
        )
        keys = {k: c[k] for k in ("clip_id", "recording", "group")}
        rows.append(keys | score)
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def _level(clips: pd.DataFrame, level: str) -> pd.DataFrame:
    """Clip rows as they are (``clips``), or averaged per recording within each group
    (``per_recording``), as ``evaluate`` does."""
    if level == "clips":
        return clips
    rows = []
    for group, part in clips.groupby("group", sort=False):
        rows += [
            r | {"group": group} for r in aggregate_recordings(part.to_dict("records"))
        ]
    return pd.DataFrame(rows)


def _scores(named: list[tuple[str, str]], level: str) -> pd.DataFrame:
    """*named*: ``(label, run name)`` pairs; the label goes in the ``method`` column."""
    subjects = _subjects()
    frames = []
    for label, name in named:
        if evaluation(name) is None:
            continue
        m = RUNS[name]
        seed = seeds(folder(m))[0] if m.kind == "network" else "—"
        df = _level(rescored(name), level).assign(method=label, seed=seed, head=m.head)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["session"] = df["recording"].str.split("#").str[-1]
    df["subject"] = df["session"].map(subjects)
    df["set"] = df["group"].map(SETS)
    head_f1 = df.get("head_inhale_f1", np.nan)
    df["inhale"] = np.where(df["head"], head_f1, df["inhale_f1"]).astype(float)
    df = df.drop(columns="head")
    if level == "clips":
        df["stem"] = df["clip_id"].str.split("/").str[-1]
    return df


SETS = {"test_a": "A: new animals", "test_b": "B: new sessions", "test_c": "C: 359 Hz"}
METRICS = {
    "correlation": "Correlation",
    "inhale": "Inhale F1",
    "exhale_f1": "Exhale F1 (DSP)",
    "kl_ibi": "KL-IBI (lower is better)",
}


# --- the test clips and predictions on them -------------------------------------------


@functools.cache
def clips() -> dict[str, tuple]:
    """Clip stem -> ``(resolved clip, recipe)`` over the test lists of :func:`root`."""
    return test_sets._clips(root())


@functools.cache
def _listed() -> dict[str, tuple]:
    """Clip stem -> ``(resolved clip, recipe)`` over the dataset copy's clip list."""
    clip_list = load(ClipList, root() / "dataset" / "clips.toml", check_files=False)
    return {c.source.stem: (c, clip_list.preprocess) for c in clip_list.resolve()}


def clip(stem: str) -> tuple:
    """``(resolved clip, recipe)`` of any clip of the dataset copy."""
    return clips().get(stem) or _listed()[stem]


@functools.cache
def clip_sets() -> dict[str, str]:
    """Clip stem -> the test set whose list holds it."""
    out = {}
    for path in sorted((root() / "dataset").glob("full_test_*.toml")):
        clip_list = load(ClipList, path, check_files=False)
        name = SETS["test_" + path.stem[-1]]
        out |= {c.source.stem: name for c in clip_list.resolve()}
    return out


def _span(stem: str) -> str:
    """A clip's recording span (``<session>_span<k>``): its stem without the camera."""
    return stem.rsplit("_", 1)[0]


@functools.cache
def _spans() -> dict[str, str]:
    """Recording span -> the stem of the dataset copy's clip of it."""
    return {_span(s): s for s in _all_entries()}


def is_local(stem: str) -> bool:
    """Whether the dataset copy holds clip *stem* itself (its frames, not only its span)."""
    return stem in clips() or stem in _all_entries()


def truth_entry(stem: str) -> features.ClipEntry:
    """The entry that holds clip *stem*'s truth: its own, or, for a clip of another
    camera than the copy's, the copy's clip of the same recording span (the same
    thermistor, events and clock; never its frames)."""
    if is_local(stem):
        return entry(stem)
    local = _spans().get(_span(stem))
    if local is None:
        raise KeyError(f"no clip of {_span(stem)} in the dataset copy")
    return entry(local)


def entry(stem: str) -> features.ClipEntry:
    """The cache entry of any clip: a test clip of the published split, else any clip of
    the dataset copy (the cross-validation folds test others)."""
    if stem not in clips():
        return _all_entries()[stem]
    clip, recipe = clips()[stem]
    found = features.lookup(clip, recipe, root() / "cache" / test_sets.RECIPE)
    if found is None:
        raise FileNotFoundError(f"no cache entry for {stem} under {root()}")
    return found


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp(dev: torch.device) -> torch.dtype | None:
    if dev.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@functools.cache
def network(name: str) -> tuple:
    """``(model, mean, std, window, frame_chunk)`` of a network's first seed."""
    m = RUNS[name]
    path = _fetch(f"{folder(m)}/{seeds(folder(m))[0]}/best.pt")
    model, mean, std, state = load_checkpoint(path, device())
    params = state.get("params") or {}
    return (
        model,
        mean,
        std,
        int(params.get("infer_window", 1024)),
        int(params.get("frame_chunk", 256)),
    )


def _saved(
    name: str, e: features.ClipEntry, stem: str | None = None
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """``(trace, onset)`` as the run saved it when it scored *e* (a network's
    ``traces/``, the baselines' ``traces/``), downloaded; ``None`` if it saved none.
    A network's trace of clip *stem* (another camera's clip of *e*'s span) is put on
    *e*'s output grid by its own times."""
    m = RUNS[name]
    if m.kind == "network":
        traces = f"{folder(m)}/{seeds(folder(m))[0]}/{TRACES}"
        # The clip's own trace first: a run that scored several cameras of a span (one
        # network on all of them) saved one per camera, and *e* is the copy's clip of
        # the span, another camera's.
        path = (
            _fetch(f"{traces}/{_trace_files(traces)[stem]}")
            if stem is not None and stem in _trace_files(traces)
            else None
        )
        if path is None:
            path = _fetch(f"{traces}/{trace_name(e)}")
        if path is None and Path(e.stem).name.rsplit("-", 1)[0] in _trace_files(traces):
            # a run on another recipe (e.g. select_fs) names its traces by its own entry
            path = _fetch(
                f"{traces}/{_trace_files(traces)[Path(e.stem).name.rsplit('-', 1)[0]]}"
            )
        if path is None:
            return None
        s = pd.read_parquet(path)
        signal = s["signal"].to_numpy()
        onset = s["onset"].to_numpy() if "onset" in s else None
        times = np.load(e.times)
        if "time" in s and not np.array_equal(s["time"].to_numpy(), times[: len(s)]):
            at = s["time"].to_numpy()
            signal = np.interp(times, at, signal)
            onset = None if onset is None else np.interp(times, at, onset)
        return signal, onset
    if m.kind == "facemap":
        path = _fetch(
            f"{folder(m)}/facemap/{m.variant}/traces/{baselines.trace_key(e)}.npy"
        )
        return None if path is None else (np.load(path), None)
    method = m.variant.split("/")[0]
    path = _fetch(f"{folder(m)}/pixel/{method}/traces/{baselines.trace_key(e)}.npy")
    return None if path is None else (_calibrated(name, np.load(path)), None)


@functools.cache
def _trace_files(traces: str) -> dict[str, str]:
    """Clip stem -> trace file name in a network run's *traces* folder (its
    ``feat-<stem>-<key>.parquet``), listed once."""
    try:
        names = [
            Path(p).name for p in _s3().ls(f"{S3_RUNS.removeprefix('s3://')}/{traces}")
        ]
    except FileNotFoundError:
        names = [p.name for p in (LOCAL_RUNS / traces).glob("feat-*.parquet")]
    return {
        n.removeprefix("feat-").removesuffix(".parquet").rsplit("-", 1)[0]: n
        for n in names
        if n.startswith("feat-") and n.endswith(".parquet")
    }


def _calibrated(name: str, raw: np.ndarray) -> np.ndarray:
    """A pixel method's output-grid trace as its variant reports it."""
    trace = common.blind_polarity(raw)
    if RUNS[name].variant.endswith("/calibrated"):
        fitted = evaluation(name)
        trace = fitted["sign"] * common.shift(trace, fitted["lag_samples"])
    return trace


def _predicted(
    name: str, e: features.ClipEntry
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(trace, onset)`` predicted here, for a run that saved no traces."""
    m = RUNS[name]
    if m.kind == "pixel":
        method = m.variant.split("/")[0]
        raw = baselines.pixel_trace(method, e, common.FS, PIXEL_BIN)
        return _calibrated(name, common.to_output_grid(e, raw)), None
    if m.kind != "network":
        raise FileNotFoundError(f"{name}: no saved trace for {e.stem.name}")
    model, mean, std, window, chunk = network(name)
    dev = device()
    return predict_clip(
        model,
        e,
        mean,
        std,
        window=window,
        device=dev,
        frame_chunk=chunk,
        amp_dtype=_amp(dev),
    )


def prediction(name: str, stem: str) -> dict:
    """Method *name*'s trace on clip *stem* (z-scored, on the clip's 60 Hz grid), its
    onset probability (zeros without a head) and the target. The trace the run saved
    is read from its download (:func:`_fetch`), never copied; only a trace predicted
    here, for a run that saved none, is cached (``cache/walkthrough/``), and a saved
    one always wins over it. A clip of another camera than the copy's has only the
    run's saved trace (on the grid of the copy's clip of its span); it is never
    predicted here, from frames of another camera."""
    e = truth_entry(stem)
    m = RUNS[name]
    path = _prediction_path(name, stem)
    found = _saved(name, e, stem)
    if found is None and not is_local(stem):
        raise FileNotFoundError(f"{name}: no saved trace for {stem}")
    if found is None and path.is_file():
        with np.load(path) as p:
            out = {k: p[k] for k in p.files}
    else:
        signal, onset = found if found is not None else _predicted(name, e)
        times = np.load(e.times)
        n = min(len(times), len(signal))
        signal = signal[:n].astype(np.float32)
        signal = (signal - signal.mean()) / (signal.std() or 1.0)
        onset = (
            np.zeros(n, np.float32) if onset is None else onset[:n].astype(np.float32)
        )
        out = {"signal": signal, "onset": onset, "times": times[:n]}
        if found is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".tmp.npz")
            np.savez(part, **out)
            part.replace(path)
        elif path.is_file():
            path.unlink()  # a local prediction the run's own trace replaces
    if m.head:
        out["threshold"] = head_threshold(name)
        peaks = head_event_indices(out["onset"], out["times"], out["threshold"])
        out["head"] = out["times"][peaks]
    return out


def _run_dir(m: ReportedRun) -> str:
    return f"{folder(m)}/{seeds(folder(m))[0]}"


def _validation_recipe_is_local(name: str) -> bool:
    """Whether run *name* read its validation clips with the local copy's recipe, and
    the copy holds those clips (not another camera's), so they can be predicted from
    its feature cache."""
    run = _run_dir(RUNS[name])
    config = json.loads(_fetch(f"{run}/config.json").read_text())
    theirs = config.get("val_clip_list", {}).get("preprocess", {})
    ours = load(
        ClipList, root() / "dataset" / "clips.toml", check_files=False
    ).preprocess
    ours = json.loads(ours.model_dump_json())
    if not theirs or any(ours.get(k) != v for k, v in theirs.items() if k in ours):
        return False
    clips_ = json.loads(_fetch(f"{run}/best_per_clip.json").read_text())
    return all(is_local(c["clip_id"].split("/")[-1]) for c in clips_)


def validation_prediction(name: str, stem: str) -> dict:
    """Run *name*'s trace and onset probability on one of its validation clips,
    predicted here (inference only) and cached."""
    m = RUNS[name]
    path = PREDICTIONS / "validation" / m.run / (m.fold or "published") / f"{stem}.npz"
    if not path.is_file():
        e = entry(stem)
        signal, onset = _predicted(name, e)
        times = np.load(e.times)
        n = min(len(times), len(signal))
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".tmp.npz")
        np.savez(part, times=times[:n], signal=signal[:n], onset=onset[:n])
        part.replace(path)
    with np.load(path) as p:
        return {k: p[k] for k in p.files}


def head_thresholds() -> dict[str, dict]:
    """Every onset threshold chosen so far (:func:`head_threshold`), by run
    (:func:`_tag`)."""
    path = DERIVED / "head-thresholds.json"
    return json.loads(path.read_text()) if path.is_file() else {}


@functools.cache
def head_threshold(name: str) -> float:
    """The onset threshold of run *name*: the scorer's, unless :data:`HEAD_MODE` is
    ``validation``, then the one training would have chosen on the run's own
    validation clips (:func:`zephyr.evaluate.choose_head_threshold` over
    :data:`zephyr.benchmarks.analysis.HEAD_THRESHOLDS`); cached with its sweep. A
    checkpoint that holds its own (chosen in training, as its ``evaluation.json``
    records) keeps it; a run whose validation clips used another recipe than the local
    copy keeps the scorer's, and says so."""
    if HEAD_MODE != "validation" or not RUNS[name].head:
        return HEAD_THRESHOLD
    chosen = head_thresholds()
    key = _tag(name)
    if key not in chosen:
        own = ((evaluation(name) or {}).get("head_threshold")) or {}
        if own.get("source") == "checkpoint":
            chosen[key] = {
                "value": own["value"],
                "chosen_on": "training",
                "rule": "the checkpoint's own, chosen in training on its validation clips",
            }
        elif not _validation_recipe_is_local(name):
            chosen[key] = {
                "value": HEAD_THRESHOLD,
                "chosen_on": None,
                "rule": "the default: its validation clips are not in the dataset copy",
            }
        else:
            run = _run_dir(RUNS[name])
            clips = json.loads(_fetch(f"{run}/best_per_clip.json").read_text())
            stems = [c["clip_id"].split("/")[-1] for c in clips]
            on_val = []
            for s in stems:
                p = validation_prediction(name, s)
                on_val.append((entry(s), p["times"], p["onset"]))
            chosen[key] = choose_head_threshold(
                on_val, analysis.HEAD_THRESHOLDS, TrainParams().head_threshold_f1_tie
            )
        DERIVED.mkdir(parents=True, exist_ok=True)
        (DERIVED / "head-thresholds.json").write_text(json.dumps(chosen, indent=1))
    return float(chosen[key]["value"])


def _prediction_path(name: str, stem: str) -> Path:
    m = RUNS[name]
    kind = m.variant.replace("/", "-") or "net"
    return PREDICTIONS / m.run / m.fold / kind / f"{stem}.npz"


def truth(stem: str) -> dict:
    """:func:`zephyr.benchmarks.analysis.truth` of a test clip."""
    return analysis.truth(truth_entry(stem))


def gray(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """``(frame_times, frames)``: the clip's gray crops, memory-mapped ``(T, 1, H, W)``."""
    e = entry(stem)
    return np.load(e.frame_times), np.load(e.frames, mmap_mode="r")


def thumbnail(stem: str) -> np.ndarray | None:
    """The session's whole first frame, if the dataset copy has its thumbnail."""
    from PIL import Image

    session = stem.rsplit("_span", 1)[0]
    path = root() / "dataset" / "thumbnails" / f"{session}_FaceCamera.png"
    return np.asarray(Image.open(path).convert("L")) if path.is_file() else None


def box(stem: str) -> tuple[float, float, float, float]:
    """The clip's crop box in the pixels of its :func:`thumbnail` (the clip list's box
    is in the frame the recipe resizes to, ``target_size``)."""
    clip_, recipe = clip(stem)
    image = thumbnail(stem)
    scale = 1.0 if image is None else image.shape[1] / recipe.target_size[0]
    return tuple(v * scale for v in clip_.box)


# --- wall time ------------------------------------------------------------------------


def timed() -> set[str]:
    """The methods :func:`wall_time` has timed already."""
    path = PREDICTIONS / "timing.json"
    return set(json.loads(path.read_text())) if path.is_file() else set()


def wall_time(names, n_clips: int = 8) -> pd.DataFrame:
    """Inference seconds per clip of each method (:mod:`zephyr.benchmarks.timing`: the
    same clips for all, one untimed warm-up clip, files read into the OS cache first).
    Networks on the GPU at each one's own inference window; pixel and Facemap on the
    CPU. Cached in ``cache/walkthrough/timing.json``; delete it to time again."""
    path = PREDICTIONS / "timing.json"
    done = json.loads(path.read_text()) if path.is_file() else {}
    entries = [entry(s) for s in clips()]
    timed = timing.pick_clips(entries, n_clips)
    warm = timing.pick_warmup(entries, timed)
    timing.warm_file_cache([*timed, warm])
    for name in names:
        if name in done:
            continue
        m = RUNS[name]
        if m.kind == "network":
            model = network(name)[0]
            dev = device()

            def infer(e, name=name, dev=dev):
                _predicted(name, e)  # timed: always inference, never a download
                if dev.type == "cuda":
                    torch.cuda.synchronize()

            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result = timing.time_method(infer, timed, warm)
            result["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
            result["peak_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
            )
            result["device"] = timing.hardware()["gpu"] or "cpu"
        elif m.kind == "pixel":
            method = m.variant.split("/")[0]
            result = timing.time_method(
                timing.pixel_method(method, common.FS, PIXEL_BIN), timed, warm
            )
            result["device"] = "cpu (numpy)"
        else:
            fitted = evaluation(name)
            bases = _fetch(f"{folder(m)}/facemap/bases.npz")
            result = timing.time_method(
                timing.facemap_method(
                    m.variant,
                    bases,
                    n_components=fitted["n_components"],
                    lags=np.asarray(fitted["lags"]),
                    seed=0,
                ),
                timed,
                warm,
            )
            result["device"] = "cpu (numpy)"
        result["video_s"] = [len(np.load(e.times)) / timing.CLIP_FS for e in timed]
        result["when"] = time.strftime("%Y-%m-%d %H:%M")
        done[name] = result
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=2))
    rows = []
    for name in names:
        r = done[name]
        seconds = np.array([c["seconds"] for c in r["clips"]])
        video = np.array(r["video_s"])
        rows.append(
            {
                "method": name,
                "device": r["device"],
                "parameters (M)": r.get("params_m"),
                "s per clip (median)": float(np.median(seconds)),
                "s per min of video": float(np.median(seconds / video * 60)),
                "x real time": float(np.median(video / seconds)),
                "frames per s": float(np.median(video * timing.CLIP_FS / seconds)),
                "peak GPU memory (GB)": r.get("peak_gb"),
            }
        )
    return pd.DataFrame(rows)


# --- the analysis tables, per method, cached ------------------------------------------


def _tables_paths(name: str) -> dict[str, Path]:
    return {
        k: _derived(name) / f"analysis-{_tag(name)}-{k}.parquet"
        for k in analysis.TABLES
    }


def done(name: str) -> bool:
    """Whether :func:`tables` of *name* are cached."""
    return all(p.is_file() for p in _tables_paths(name).values())


def test_clips(name: str) -> dict[str, str]:
    """Clip stem -> test set of the clips run *name* was scored on: the published test
    lists, or, for a cross-validation run, the clips its own ``evaluation.json`` lists."""
    if name not in CV_RUNS:
        return clip_sets()
    result = evaluation(name)
    return {
        c["clip_id"].split("/")[-1]: SETS[c["group"]]
        for c in result["clips"]
        if c["group"] in SETS
    }


def tables(
    name: str, progress: Callable[[Iterable], Iterable] = iter
) -> dict[str, pd.DataFrame]:
    """:func:`zephyr.benchmarks.analysis.clip_tables` of one method over every test
    clip, labelled by method, clip and set; cached. *progress* wraps the loop over clips
    (e.g. a progress bar); predictions not cached yet are made here."""
    paths = _tables_paths(name)
    if done(name):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    parts: dict[str, list] = {k: [] for k in analysis.TABLES}
    sets = test_clips(name)
    for stem in progress(list(sets)):
        label = {"method": name, "stem": stem, "set": sets[stem]}
        found = analysis.clip_tables(
            prediction(name, stem), truth(stem), RUNS[name].head
        )
        for k, v in found.items():
            parts[k].append(v.assign(**label))
    out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    _derived(name).mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(paths[k])
    return out


def errors(name: str, stem: str) -> pd.DataFrame:
    """:func:`zephyr.benchmarks.analysis.errors` of one method on one test clip."""
    frame_t, frames = gray(stem)
    found = analysis.errors(
        prediction(name, stem), truth(stem), RUNS[name].head, frame_t, frames
    )
    return found.assign(stem=stem, method=name)


def set_errors(name: str, set_name: str) -> pd.DataFrame:
    """:func:`errors` of every clip of one test set (``SETS`` value), cached in
    ``cache/walkthrough``; delete the file to recompute."""
    path = _derived(name) / f"errors-{_tag(name)}-{set_name.split(':')[0]}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    stems = sorted(s for s, v in test_clips(name).items() if v == set_name)
    found = pd.concat([errors(name, s) for s in stems], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    found.to_parquet(path)
    return found


@functools.cache
def _all_entries() -> dict[str, features.ClipEntry]:
    """Every clip of the dataset copy with a cache entry, train and validation too."""
    clip_list = load(ClipList, root() / "dataset" / "clips.toml", check_files=False)
    cache = root() / "cache" / test_sets.RECIPE
    out = {}
    for c in clip_list.resolve():
        found = features.lookup(c, clip_list.preprocess, cache)
        if found is not None:
            out[c.source.stem] = found
    return out


def split_stems() -> dict[str, list[str]]:
    """Clip stems of the training and validation sets and of each test set."""
    folds = pd.read_parquet(root() / "dataset" / "raw" / "folds.parquet")
    sets = dict(zip(folds["session_id"], folds["set"], strict=True))
    names = {"train": "Train", "val": "Validation"}
    out: dict[str, list[str]] = {"Train": [], "Validation": []}
    for stem in _all_entries():
        s = names.get(sets.get(stem.rsplit("_span", 1)[0]))
        if s:
            out[s].append(stem)
    for stem, name in clip_sets().items():
        out.setdefault(name, []).append(stem)
    return out


def rate_distribution() -> pd.DataFrame:
    """:func:`zephyr.benchmarks.analysis.rate_distribution` of every split, from each
    clip's cached thermistor events."""
    entries = _all_entries()
    return analysis.rate_distribution(
        {
            label: [np.load(entries[s].events)["onset_times"] for s in stems]
            for label, stems in split_stems().items()
        }
    )


def breath_rates() -> dict[str, np.ndarray]:
    """Every thermistor breath's rate (1 / interval to the next inhale, Hz) in the
    training and validation sets and in each test set, from each clip's cached events."""
    entries = _all_entries()
    return {
        label: np.concatenate(
            [1.0 / np.diff(np.load(entries[s].events)["onset_times"]) for s in stems]
        )
        for label, stems in split_stems().items()
        if stems
    }


def animal_breath_rates() -> dict[str, np.ndarray]:
    """Every thermistor breath's rate (Hz) per animal, over the clips of every session
    the corpus assigns to a set, from each clip's cached events."""
    folds = pd.read_parquet(root() / "dataset" / "raw" / "folds.parquet")
    used = folds[folds["set"].notna()]
    subject = dict(zip(used["session_id"], used["subject_id"].astype(str), strict=True))
    out: dict[str, list[np.ndarray]] = {}
    for stem, e in _all_entries().items():
        animal = subject.get(stem.rsplit("_span", 1)[0])
        if animal is not None:
            out.setdefault(animal, []).append(
                1.0 / np.diff(np.load(e.events)["onset_times"])
            )
    return {k: np.concatenate(v) for k, v in out.items()}


def frame_embeddings(stem: str, seconds: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """The reference network's frame embeddings over the first *seconds* of a clip
    (:func:`zephyr.benchmarks.analysis.frame_embeddings`); cached."""
    path = PREDICTIONS / "embeddings" / f"{stem}-{seconds:g}s.npz"
    if not path.is_file():
        model, mean, std, _, chunk = network(REFERENCE)
        times, emb = analysis.frame_embeddings(
            model, mean, std, entry(stem), seconds, device(), chunk
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, times=times, embeddings=emb)
    s = np.load(path)
    return s["times"], s["embeddings"]


# --- a run's own held-out clips, made here ------------------------------------------

REMOTE_DATASET = "/data/zephyr-dataset"
"""Where the capsule mounts the dataset: the root a run's clip lists name."""
DATASET_BUCKET = "aind-scratch-data/vr-foraging/zephyr-dataset"


def held_out_entries(
    name: str, spans: Iterable[str]
) -> list[tuple[features.ClipEntry, tuple[int, ...], tuple[int, int]]]:
    """Network run *name*'s own held-out clips (test set A of its ``config.json``) of
    the recording *spans*, made on this machine: their files downloaded under
    ``root()/views`` (which stands for the capsule's dataset, so cache keys match its),
    and their crops made by zephyr's preprocessing with the list's own recipe and boxes
    into ``root()/cache``. Each comes with its box and the frame size (``target_size``)
    the box is on."""
    from zephyr.preprocess import preprocess_clips

    m = RUNS[name]
    config = json.loads(
        _fetch(f"{folder(m)}/{seeds(folder(m))[0]}/config.json").read_text()
    )
    listed = config["test_clip_lists"]["test_a"]
    wanted = set(spans)
    local = root() / "views"
    mapping = f"{local.as_posix()}={REMOTE_DATASET}"
    known = os.environ.get("ZEPHYR_PATH_MAP", "")
    if mapping not in known.split(";"):
        os.environ["ZEPHYR_PATH_MAP"] = ";".join(p for p in (known, mapping) if p)

    def here(path: str) -> str:
        return path.replace(REMOTE_DATASET, local.as_posix())

    clips = [
        c for c in listed["clip"] if Path(c["video"]).stem.rsplit("_", 1)[0] in wanted
    ]
    for c in clips:
        for key in ("video", "timestamps", "thermistor"):
            if not isinstance(c.get(key), str):
                continue
            target = Path(here(c[key]))
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                key_s3 = c[key].replace(REMOTE_DATASET, DATASET_BUCKET)
                _s3().get(key_s3, str(part))
                part.replace(target)
            c[key] = target.as_posix()
    clip_list = ClipList.model_validate({**listed, "clip": clips})
    resolved = clip_list.resolve()
    size = clip_list.preprocess.target_size
    camera = Path(resolved[0].video).stem.rsplit("_", 1)[1]
    cache = root() / "cache" / f"features-{size[0]}x{size[1]}-box96x96-60hz-{camera}"
    preprocess_clips(resolved, clip_list.preprocess, cache, workers=4)
    return [
        (features.lookup(c, clip_list.preprocess, cache), tuple(c.box), tuple(size))
        for c in resolved
    ]


def clip_scores(name: str) -> pd.DataFrame:
    """Every clip a network run scored, as its own ``evaluation.json`` gives it: the
    ``set`` (its group), ``stem``, ``session`` (the recording), ``camera`` (after the
    span in the stem), ``correlation`` and ``inhale`` (the head's inhale F1 at the
    checkpoint's own onset threshold)."""
    found = evaluation(name)
    rows = pd.DataFrame(found["clips"])
    stem = rows["clip_id"].map(lambda c: Path(c).name)
    return pd.DataFrame(
        {
            "set": rows["group"],
            "stem": stem,
            "session": rows["recording"].str.rsplit("#", n=1).str[1],
            "camera": stem.str.split(r"_span\d+_", regex=True).str[1],
            "correlation": rows["correlation"],
            "inhale": rows["head_inhale_f1"],
        }
    )


DF_ROOT = Path(os.environ.get("ZEPHYR_DF_ROOT", "C:/Data/zephyr-df-dataset"))
"""The local copy of zephyr-df-dataset: its held-out list (``df_test_a.toml``, the
dataset's entries of animal 811021) and the crops of those clips in ``cache/``."""
REMOTE_DF = "/data/zephyr-df-dataset"


@functools.cache
def df_entries() -> dict[str, tuple[features.ClipEntry, tuple[int, ...]]]:
    """Each held-out DF clip's cache entry and crop box (on the 360 x 270 frame), by
    stem; the copy named as the capsule names the dataset, so cache keys match."""
    mapping = f"{DF_ROOT.as_posix()}={REMOTE_DF}"
    known = os.environ.get("ZEPHYR_PATH_MAP", "")
    if mapping not in known.split(";"):
        os.environ["ZEPHYR_PATH_MAP"] = ";".join(p for p in (known, mapping) if p)
    clip_list = load(ClipList, DF_ROOT / "df_test_a.toml")
    out = {}
    for c in clip_list.resolve():
        entry = features.lookup(c, clip_list.preprocess, DF_ROOT / "cache")
        if entry is not None:
            out[Path(str(c.video)).stem] = (entry, tuple(c.box))
    return out
