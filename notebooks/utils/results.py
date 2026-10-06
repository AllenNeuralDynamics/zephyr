"""Locate and load what the notebooks plot. Nothing here trains or writes results."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from zephyr import features, video
from zephyr.benchmarks.common import blind_polarity
from zephyr.benchmarks.runner import load_checkpoint
from zephyr.channels import CHANNEL_NAMES, decode_flow
from zephyr.config import ClipList, load
from zephyr.evaluate import head_event_indices
from zephyr.evaluation import score_clip
from zephyr.infer import predict_clip
from zephyr.signal import (
    BREATHING_SIGNAL_COLUMN,
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    detect_inhalation_events,
    resample_uniform,
)

from . import style

ROOT: Path = Path(__file__).resolve().parents[2]
CACHE: Path = ROOT / "notebooks" / "cache"
CLIP_LISTS: dict[str, Path] = {
    "new_animals": ROOT
    / "packages/zephyr-benchmarks/configs/clips/face_test_new_animals.toml",
    "known_animals_new_date": ROOT
    / "packages/zephyr-benchmarks/configs/clips/face_test_known_animals_new_date.toml",
}
FEATURES: Path = ROOT / "data/features-v2"
BASELINES: Path = ROOT / "benchmarks/baselines-v1"
ABLATION: Path = ROOT / "benchmarks/input-objective-v1/runs"
CHECKPOINTS: dict[str, Path] = {
    "Zephyr": ROOT
    / "runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask"
    / "seed-42/best.pt",
    "TS-CAN": BASELINES / "tscan/runs/gray__signal__seed-42/best.pt",
    "PhysNet": BASELINES / "physnet/runs/gray__signal__seed-42/best.pt",
}
REFERENCE_RUN: Path = (
    ROOT / "runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask"
)
NO_STRETCH_RUN: Path = ROOT / "runs/stretch-ablation/benchmark-no-stretch"
NO_STRETCH: str = "Zephyr, no stretch"
STRETCH_RUNS: dict[str, Path] = {
    "Zephyr": REFERENCE_RUN / "seed-42",
    NO_STRETCH: NO_STRETCH_RUN / "seed-17",
}
"""The two networks of the stretch ablation: the benchmark Zephyr and its no-stretch twin."""
PIXEL_TRACES: Path = BASELINES / "pixel/flow/traces/test"
FACEMAP_TRACES: Path = ROOT / "runs/facemap-traces/facemap/both/traces"
OCCLUSION: Path = ROOT / "runs/thermistor-quality/occlusion_60s_f1.npz"

INPUTS: dict[str, tuple[Path, str]] = {
    "feature cache": (
        FEATURES,
        "zephyr preprocess <clip list> --cache data/features-v2",
    ),
    "zephyr checkpoint": (CHECKPOINTS["Zephyr"], "zephyr run <experiment>.toml"),
    "TS-CAN checkpoint": (
        CHECKPOINTS["TS-CAN"],
        "zephyr-benchmarks run <experiment>.toml",
    ),
    "PhysNet checkpoint": (
        CHECKPOINTS["PhysNet"],
        "zephyr-benchmarks run <experiment>.toml",
    ),
    "pixel traces": (PIXEL_TRACES, "zephyr-benchmarks pixel <fold>.toml --cache <dir>"),
    "Facemap traces": (
        FACEMAP_TRACES,
        "zephyr-benchmarks facemap <fold>.toml --cache <dir>",
    ),
    "baseline results": (
        BASELINES / "pixel",
        "zephyr-benchmarks pixel|facemap <fold>.toml",
    ),
    "no-stretch checkpoint": (
        STRETCH_RUNS[NO_STRETCH] / "best.pt",
        "zephyr-benchmarks run configs/experiments/stretch-ablation.toml",
    ),
    "ablation runs": (ABLATION, "zephyr-benchmarks run <experiment>.toml"),
    "occlusion maps": (
        OCCLUSION,
        "zephyr-benchmarks occlusion --checkpoint <best.pt> ...",
    ),
}

STRATA: tuple[str, ...] = ("new_animals", "known_animals_new_date")


def missing() -> dict[str, str]:
    """Inputs that do not exist, each with the command that produces it."""
    return {name: cmd for name, (path, cmd) in INPUTS.items() if not path.exists()}


@dataclass(frozen=True)
class TestClip:
    entry: features.ClipEntry
    stratum: str
    box: tuple[int, int, int, int]
    video: Path
    target_size: tuple[int, int]


def clips() -> dict[str, TestClip]:
    """Every test clip by name, with its stratum and crop box."""
    out: dict[str, TestClip] = {}
    for stratum, path in CLIP_LISTS.items():
        clip_list = load(ClipList, path)
        resolved = clip_list.resolve()
        entries = features.require(resolved, clip_list.preprocess, FEATURES)
        for clip, entry in zip(resolved, entries, strict=True):
            out[short_name(entry)] = TestClip(
                entry, stratum, clip.box, clip.video, clip_list.preprocess.target_size
            )
    return out


def short_name(entry: features.ClipEntry) -> str:
    """``test/video_face_6_part_1`` -> ``face_6_part_1``."""
    return entry.clip_id.removeprefix("test/video_")


def raw_frame(clip: TestClip, t_s: float) -> NDArray:
    """The full video frame at time *t_s*, scaled to the working frame size."""
    return video.decode_window(
        clip.video, scale_to=clip.target_size, start_s=t_s, dur_s=0.1
    )[0]


def channel_frames(entry: features.ClipEntry, t_s: float) -> dict[str, NDArray]:
    """What the network sees at time *t_s*: each channel in physical units."""
    index = int(np.searchsorted(np.load(entry.frame_times), t_s))
    stack = np.load(entry.features, mmap_mode="r")[index]
    physical: dict[str, Callable[[NDArray], NDArray]] = {
        "gray": lambda x: x.astype(float),
        "diff": lambda x: x.astype(float) - 128.0,
        "flow_x": decode_flow,
        "flow_y": decode_flow,
    }
    return {n: physical[n](p) for n, p in zip(CHANNEL_NAMES, stack, strict=True)}


def truth(entry: features.ClipEntry) -> pd.DataFrame:
    """The thermistor trace, columns ``Time`` and ``Signal``."""
    return pd.read_parquet(entry.thermistor)


def _zscore(x: NDArray) -> NDArray:
    return (x - x.mean()) / x.std()


def _infer(method: str, entry: features.ClipEntry) -> tuple[NDArray, NDArray]:
    """Signal and onset-head probability of one network on one clip (CPU)."""
    device = torch.device("cpu")
    model, mean, std, _ = load_checkpoint(CHECKPOINTS[method], device)
    return predict_clip(model, entry, mean, std, device=device, amp_dtype=None)


def method_traces(entry: features.ClipEntry) -> pd.DataFrame:
    """One z-scored breathing trace per method on the clip's 60 Hz output grid.

    Networks run inference only and the result is cached, so a clip is
    computed once.  Also holds Zephyr's onset-head probability as ``onset``.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{short_name(entry)}.parquet"
    if cache.exists():
        print(f"Cache loaded from {cache}")
        return pd.read_parquet(cache)
    print(f"No cache found in {cache}. Calculating from scratch.")
    n = entry.n_output
    key = Path(entry.features).stem
    old_key = f"test_{short_name(entry).removeprefix('face_')}"
    zephyr_signal, onset = _infer("Zephyr", entry)
    traces: dict[str, NDArray] = {
        TIME_COLUMN: np.load(entry.times)[:n],
        "Pixel": blind_polarity(np.load(PIXEL_TRACES / f"{old_key}.npy")),
        "Facemap": np.load(FACEMAP_TRACES / f"{key}.npy"),
        "TS-CAN": _infer("TS-CAN", entry)[0],
        "PhysNet": _infer("PhysNet", entry)[0],
        "Zephyr": zephyr_signal,
    }
    frame = pd.DataFrame(
        {k: v if k == TIME_COLUMN else _zscore(v[:n]) for k, v in traces.items()}
    )
    frame["onset"] = onset[:n]
    frame.to_parquet(cache)
    return frame


def no_stretch_traces(entry: features.ClipEntry) -> pd.DataFrame:
    """The no-stretch network's z-scored trace and onset probability on a clip.

    Inference only (CPU), cached per clip. Columns: ``Time``, ``Zephyr, no stretch``,
    ``onset``.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{short_name(entry)}-no-stretch.parquet"
    if cache.exists():
        print(f"Cache loaded from {cache}")
        return pd.read_parquet(cache)
    print(f"No cache found in {cache}. Calculating from scratch.")
    device = torch.device("cpu")
    model, mean, std, _ = load_checkpoint(STRETCH_RUNS[NO_STRETCH] / "best.pt", device)
    signal, onset = predict_clip(model, entry, mean, std, device=device, amp_dtype=None)
    n = entry.n_output
    frame = pd.DataFrame(
        {TIME_COLUMN: np.load(entry.times)[:n], NO_STRETCH: _zscore(signal[:n])}
    )
    frame["onset"] = onset[:n]
    frame.to_parquet(cache)
    return frame


def stretch_summary() -> pd.DataFrame:
    """Scores of the stretch-ablation networks over all 12 test recordings.

    One row per metric, one column per network, read from their ``evaluation.json``.
    """
    metrics = (*METRICS, "head_inhale_f1")
    columns = {
        name: json.loads((run / "evaluation.json").read_text())["groups"]["all"][
            "summary"
        ]
        for name, run in STRETCH_RUNS.items()
    }
    return pd.DataFrame(
        {name: {m: s[m] for m in metrics} for name, s in columns.items()}
    )


def events(times: NDArray, signal: NDArray) -> tuple[NDArray, NDArray]:
    """Inhale and exhale onset times (s) the scorer detects in a trace."""
    trace = pd.DataFrame({TIME_COLUMN: times, BREATHING_SIGNAL_COLUMN: signal})
    uniform = resample_uniform(trace)
    inhale, exhale = detect_inhalation_events(
        uniform[BREATHING_SIGNAL_COLUMN].to_numpy(), CANONICAL_BREATHING_SAMPLING_RATE
    )
    t = uniform[TIME_COLUMN].to_numpy()
    return t[inhale], t[exhale]


def method_scores(entry: features.ClipEntry, traces: pd.DataFrame) -> pd.DataFrame:
    """Scores of every method on this clip, as the benchmark scores them.

    ``head_inhale_f1`` is Zephyr's inhale F1 when events come from its onset head
    rather than from the trace; it is empty for the other methods.
    """
    times = traces[TIME_COLUMN].to_numpy()
    reference = truth(entry)
    rows: dict[str, dict[str, float | None]] = {}
    for method in style.METHODS:
        predicted = pd.DataFrame(
            {TIME_COLUMN: times, BREATHING_SIGNAL_COLUMN: traces[method]}
        )
        rows[method] = score_clip(reference, predicted).to_dict()
        if method == "Zephyr":
            head_events = times[head_event_indices(traces["onset"].to_numpy(), times)]
            head = score_clip(reference, predicted, predicted_onset_times_s=head_events)
            rows[method]["head_inhale_f1"] = head.inhale_f1
    return pd.DataFrame(rows).T[[*METRICS, "head_inhale_f1"]]


def head_scores(run: str) -> pd.DataFrame:
    """Per-clip scores of one trained Zephyr network from the ablation runs, with
    events taken from the trace and from the onset head.

    Inference only (CPU); cached per network.
    """
    cache = CACHE / f"head-{run}.parquet"
    if cache.exists():
        print(f"Cache loaded from {cache}")
        return pd.read_parquet(cache)
    print(f"No cache found in {cache}. Calculating from scratch.")
    device = torch.device("cpu")
    model, mean, std, _ = load_checkpoint(ABLATION / run / "best.pt", device)
    rows: list[dict[str, object]] = []
    for name, clip in clips().items():
        entry = clip.entry
        signal, onset = predict_clip(
            model, entry, mean, std, device=device, amp_dtype=None
        )
        times = np.load(entry.times)[: entry.n_output]
        predicted = pd.DataFrame({TIME_COLUMN: times, BREATHING_SIGNAL_COLUMN: signal})
        reference = truth(entry)
        head_events = times[head_event_indices(onset, times)]
        trace = score_clip(reference, predicted)
        head = score_clip(reference, predicted, predicted_onset_times_s=head_events)
        rows.append(
            {
                "clip": name,
                "recording": entry.recording,
                "stratum": clip.stratum,
                "inhale_f1": trace.inhale_f1,
                "kl_ibi": trace.kl_ibi,
                "head_inhale_f1": head.inhale_f1,
                "head_kl_ibi": head.kl_ibi,
            }
        )
    CACHE.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(cache)
    return frame


def by_stratum(scores: pd.DataFrame, column: str) -> dict[str, float]:
    """Mean of a per-clip score: within each recording first, then across them."""
    per_recording = scores.groupby(["stratum", "recording"])[column].mean()
    return per_recording.groupby("stratum").mean().to_dict()


METRICS: tuple[str, ...] = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi")


def _stratum_rows(path: Path, **labels: object) -> list[dict[str, object]]:
    """One row per stratum of an ``evaluation.json``, plus the onset head's inhale
    F1 (``head_inhale_f1``) when a ``head_evaluation.json`` sits beside it."""
    strata = json.loads(path.read_text())["strata"]
    head_file = path.parent / "head_evaluation.json"
    head = json.loads(head_file.read_text())["strata"] if head_file.exists() else {}
    rows = []
    for stratum in STRATA:
        row = labels | {"stratum": stratum}
        row |= {m: strata[stratum]["summary"][m] for m in METRICS}
        if head:
            row["head_inhale_f1"] = head[stratum]["summary"]["head_inhale_f1"]
        rows.append(row)
    return rows


def ablation() -> pd.DataFrame:
    """Every ablation network: input set, objective, seed, stratum, scores."""
    rows: list[dict[str, object]] = []
    for run in sorted(ABLATION.iterdir()):
        inputs, objective, seed = run.name.split("__")
        labels = {
            "inputs": inputs,
            "objective": objective,
            "seed": int(seed.split("-")[1]),
        }
        rows += _stratum_rows(run / "evaluation.json", **labels)
    return pd.DataFrame(rows)


def benchmark() -> pd.DataFrame:
    """Every compared run: method, seed, stratum, scores (one row each).

    Zephyr's rows also carry the scores with events from the onset head
    (``head_*``); its KL-IBI is recomputed so both come from the same inference.
    """
    pixel = BASELINES / "pixel"
    sources: dict[str, list[Path]] = {
        "Pixel": [pixel / "flow/blind/evaluation.json"],
        "Facemap": [BASELINES / "facemap/both/evaluation.json"],
        "TS-CAN": sorted((BASELINES / "tscan/runs").glob("*/evaluation.json")),
        "PhysNet": sorted((BASELINES / "physnet/runs").glob("*/evaluation.json")),
        "Zephyr": sorted(ABLATION.glob("gray-diff-flow__multitask__*/evaluation.json")),
    }
    rows: list[dict[str, object]] = []
    for method, paths in sources.items():
        for path in paths:
            run = path.parent.name
            scores = head_scores(run) if method == "Zephyr" else None
            for row in _stratum_rows(path, method=method, run=run):
                if scores is not None:
                    row["kl_ibi"] = by_stratum(scores, "kl_ibi")[row["stratum"]]
                    row["head_kl_ibi"] = by_stratum(scores, "head_kl_ibi")[
                        row["stratum"]
                    ]
                rows.append(row)
    return pd.DataFrame(rows)


def mean_frames(seconds: float = 60.0) -> dict[str, NDArray]:
    """Mean gray crop of each clip over its first *seconds* (the occlusion window).

    Cached in one file, keyed by clip name (``face_6_part_1``).
    """
    cache = CACHE / "mean-frames.npz"
    if cache.exists():
        print(f"Cache loaded from {cache}")
        return dict(np.load(cache))
    print(f"No cache found in {cache}. Calculating from scratch.")
    out: dict[str, NDArray] = {}
    for name, clip in clips().items():
        gray = np.load(clip.entry.features, mmap_mode="r")[: round(seconds * 60) : 6, 0]
        out[name] = np.asarray(gray, dtype=float).mean(0)
    CACHE.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


def occlusion() -> dict[str, NDArray]:
    """Saved occlusion maps: per-clip drop in event F1, one array per channel."""
    return dict(np.load(OCCLUSION))
