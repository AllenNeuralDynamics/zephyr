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
from zephyr.config import ClipList, load
from zephyr.evaluate import TRACES, head_event_indices, trace_name
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
DEFAULT_ROOT = Path("C:/Data/zephyr-local-v2")
"""The local copy of the dataset and its cache, unless ``ZEPHYR_TEST_ROOT`` names one."""
METHODS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.method}
"""Every compared method, in the order tables list them."""
VARIANTS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.variant}
"""The first method trained differently, compared with it alone."""
ALL = METHODS | VARIANTS
REFERENCE = RESULTS.method[0].label
"""The method every other is compared with (Zephyr)."""


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


@functools.cache
def seeds(run: str) -> tuple[str, ...]:
    """The seed folders a network run has finished (a ``best.pt`` and its scores)."""
    local = sorted(
        p.parent.name
        for p in LOCAL_RUNS.glob(f"{run}/{RESULTS.fold}/*/evaluation.json")
    )
    try:
        found = _s3().glob(
            f"{S3_RUNS.removeprefix('s3://')}/{run}/{RESULTS.fold}/seed-*/evaluation.json"
        )
    except Exception:  # noqa: BLE001 -- offline: whatever was downloaded before
        return tuple(local)
    return tuple(sorted({Path(p).parent.name for p in found} | set(local)))


def evaluation(name: str, seed: str | None = None) -> dict | None:
    """The run's own ``evaluation.json`` (scored on Code Ocean), or ``None`` if the
    run has not finished."""
    m = ALL[name]
    if m.kind == "network":
        found = seeds(m.run)
        if not found:
            return None
        key = f"{m.run}/{RESULTS.fold}/{seed or found[0]}/evaluation.json"
    else:
        key = f"{m.run}/{m.kind}/{m.variant}/evaluation.json"
    path = _fetch(key)
    return json.loads(path.read_text()) if path else None


def available(names) -> list[str]:
    """The methods of *names* whose run has finished."""
    return [n for n in names if evaluation(n) is not None]


def runs_table(names=ALL) -> pd.DataFrame:
    """One row per method: its S3 folder, seeds finished, and whether it has a head."""
    rows = []
    for name in names:
        m = ALL[name]
        done = (
            seeds(m.run) if m.kind == "network" else ("—",) if evaluation(name) else ()
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


# --- scores, as the runs scored themselves -------------------------------------------


def _subjects() -> dict[str, str]:
    folds = pd.read_parquet(root() / "dataset" / "folds.parquet")
    return dict(zip(folds["session_id"], folds["subject_id"], strict=True))


def scores(names=ALL, level: str = "clips") -> pd.DataFrame:
    """Per-clip (``clips``) or per-recording (``per_recording``) scores of every seed of
    every finished method in *names*, from their own ``evaluation.json``; one ``inhale``
    column holds the F1 each method reports (the head's if it has one)."""
    subjects = _subjects()
    frames = []
    for name in names:
        m = ALL[name]
        for seed in seeds(m.run) if m.kind == "network" else (None,):
            result = evaluation(name, seed)
            if result is None:
                continue
            df = pd.DataFrame(result[level]).assign(method=name, seed=seed or "—")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["session"] = df["recording"].str.split("#").str[-1]
    df["subject"] = df["session"].map(subjects)
    df["set"] = df["group"].map(SETS)
    head = df["method"].map(lambda n: ALL[n].head)
    df["inhale"] = np.where(head, df.get("head_inhale_f1"), df["inhale_f1"])
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
def clip_sets() -> dict[str, str]:
    """Clip stem -> the test set whose list holds it."""
    out = {}
    for path in sorted((root() / "dataset").glob("full_test_*.toml")):
        clip_list = load(ClipList, path, check_files=False)
        name = SETS["test_" + path.stem[-1]]
        out |= {c.source.stem: name for c in clip_list.resolve()}
    return out


def entry(stem: str) -> features.ClipEntry:
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
    m = ALL[name]
    path = _fetch(f"{m.run}/{RESULTS.fold}/{seeds(m.run)[0]}/best.pt")
    model, mean, std, state = load_checkpoint(path, device())
    params = state.get("params") or {}
    return (
        model,
        mean,
        std,
        int(params.get("infer_window", 1024)),
        int(params.get("frame_chunk", 256)),
    )


def _infer_network(name: str, e: features.ClipEntry) -> tuple[np.ndarray, np.ndarray]:
    """The trace the run scored, from its ``traces/`` when it saved them (``zephyr run
    --save-traces``), else predicted here."""
    m = ALL[name]
    saved = _fetch(f"{m.run}/{RESULTS.fold}/{seeds(m.run)[0]}/{TRACES}/{trace_name(e)}")
    if saved is not None:
        with np.load(saved) as s:
            return s["signal"], (s["onset"] if s["onset"].size else None)
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


def _infer_baseline(name: str, e: features.ClipEntry) -> np.ndarray:
    m = ALL[name]
    if m.kind == "facemap":
        key = baselines.trace_key(e)
        path = _fetch(f"{m.run}/facemap/{m.variant}/traces/{key}.npy")
        return np.load(path)
    method, variant = m.variant.split("/")
    saved = _fetch(f"{m.run}/pixel/{method}/traces/{baselines.trace_key(e)}.npy")
    raw = (
        np.load(saved)
        if saved is not None
        else common.to_output_grid(
            e, baselines.pixel_trace(method, e, common.FS, PIXEL_BIN)
        )
    )
    trace = common.blind_polarity(raw)
    if variant == "calibrated":
        fitted = evaluation(name)
        trace = fitted["sign"] * common.shift(trace, fitted["lag_samples"])
    return trace


def prediction(name: str, stem: str) -> dict:
    """Method *name*'s trace on clip *stem* (z-scored, on the clip's 60 Hz grid), its
    onset probability (zeros without a head) and the target; cached the first time."""
    path = (
        PREDICTIONS
        / ALL[name].run
        / (ALL[name].variant.replace("/", "-") or "net")
        / f"{stem}.npz"
    )
    if not path.is_file():
        e = entry(stem)
        if ALL[name].kind == "network":
            signal, onset = _infer_network(name, e)
        else:
            signal, onset = _infer_baseline(name, e), None
        times = np.load(e.times)
        n = min(len(times), len(signal))
        signal = signal[:n].astype(np.float32)
        signal = (signal - signal.mean()) / (signal.std() or 1.0)
        onset = (
            np.zeros(n, np.float32) if onset is None else onset[:n].astype(np.float32)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".tmp.npz")
        np.savez(part, signal=signal, onset=onset, times=times[:n])
        part.replace(path)
    p = np.load(path)
    out = {k: p[k] for k in p.files}
    if ALL[name].head:
        out["head"] = out["times"][head_event_indices(out["onset"], out["times"])]
    return out


def cached(name: str, stem: str) -> bool:
    m = ALL[name]
    return (
        PREDICTIONS / m.run / (m.variant.replace("/", "-") or "net") / f"{stem}.npz"
    ).is_file()


def truth(stem: str) -> dict:
    """:func:`zephyr.benchmarks.analysis.truth` of a test clip."""
    return analysis.truth(entry(stem))


def gray(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """``(frame_times, frames)``: the clip's gray crops, memory-mapped ``(T, 1, H, W)``."""
    e = entry(stem)
    return np.load(e.frame_times), np.load(e.group_file("gray"), mmap_mode="r")


def thumbnail(stem: str) -> np.ndarray | None:
    """The session's whole first frame, if the dataset copy has its thumbnail."""
    from PIL import Image

    session = stem.rsplit("_span", 1)[0]
    path = root() / "dataset" / "thumbnails" / f"{session}_FaceCamera.png"
    return np.asarray(Image.open(path).convert("L")) if path.is_file() else None


def box(stem: str) -> tuple[float, float, float, float]:
    """The clip's crop box in the pixels of its :func:`thumbnail` (the clip list's box
    is in the frame the recipe resizes to, ``target_size``)."""
    clip, recipe = clips()[stem]
    image = thumbnail(stem)
    scale = 1.0 if image is None else image.shape[1] / recipe.target_size[0]
    return tuple(v * scale for v in clip.box)


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
        m = ALL[name]
        if m.kind == "network":
            model = network(name)[0]
            dev = device()

            def infer(e, name=name, dev=dev):
                _infer_network(name, e)
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
            bases = _fetch(f"{m.run}/facemap/bases.npz")
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
                "peak GPU memory (GB)": r.get("peak_gb"),
            }
        )
    return pd.DataFrame(rows)


# --- the analysis tables, per method, cached ------------------------------------------


def _tables_paths(name: str) -> dict[str, Path]:
    tag = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    return {k: PREDICTIONS / f"analysis-{tag}-{k}.parquet" for k in analysis.TABLES}


def done(name: str) -> bool:
    """Whether :func:`tables` of *name* are cached."""
    return all(p.is_file() for p in _tables_paths(name).values())


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
    for stem in progress(list(clips())):
        label = {"method": name, "stem": stem, "set": clip_sets()[stem]}
        found = analysis.clip_tables(
            prediction(name, stem), truth(stem), ALL[name].head
        )
        for k, v in found.items():
            parts[k].append(v.assign(**label))
    out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(paths[k])
    return out


def errors(name: str, stem: str) -> pd.DataFrame:
    """:func:`zephyr.benchmarks.analysis.errors` of one method on one test clip."""
    frame_t, frames = gray(stem)
    found = analysis.errors(
        prediction(name, stem), truth(stem), ALL[name].head, frame_t, frames
    )
    return found.assign(stem=stem, method=name)


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
    folds = pd.read_parquet(root() / "dataset" / "folds.parquet")
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
