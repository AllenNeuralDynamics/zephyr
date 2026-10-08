"""Locate and load what the notebooks plot. Nothing here trains or writes results."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.signal import welch

from zephyr import features, video
from zephyr.benchmarks.common import blind_polarity
from zephyr.benchmarks.runner import load_checkpoint
from zephyr.channels import CHANNEL_NAMES, decode_flow
from zephyr.config import ClipList, load
from zephyr.evaluate import head_event_indices, score_entries
from zephyr.evaluation import score_clip
from zephyr.infer import predict_clip
from zephyr.signal import (
    BREATHING_SIGNAL_COLUMN,
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    detect_inhalation_events,
    filter_sniff_signal,
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
CLIP_LISTS_30HZ: dict[str, Path] = {
    stratum: path.with_name(f"{path.stem}_30hz.toml")
    for stratum, path in CLIP_LISTS.items()
}
"""The test clip lists preprocessed with the CNN seeing 30 Hz frames."""
TRAIN_CLIPS: Path = ROOT / "packages/zephyr-benchmarks/configs/clips/face_train.toml"
OOD_CLIPS: dict[str, Path] = {
    "side_camera": ROOT
    / "packages/zephyr-benchmarks/configs/clips/ood_side_right.toml",
}
"""The out-of-distribution rig: a side view of the face, never trained on by Zephyr."""
FEATURES: Path = ROOT / "data/features-v2"
FEATURES_30HZ: Path = ROOT / "data/features-v2-30hz"
BASELINES: Path = ROOT / "benchmarks/baselines-v1"
ABLATION: Path = ROOT / "benchmarks/input-objective-v1/runs"
ZEPHYR_RUN: Path = (
    ROOT / "runs/rate-balance-pool2-both/benchmark-rate-balanced-pool2-both/seed-42"
)
"""The network every plot uses as Zephyr: rate-balanced, no stretch, pooled signal
branch (``signal_pool = 2``) with the onset head reading both streams."""
ZEPHYR_TAG: str = ZEPHYR_RUN.parent.name
"""Names Zephyr's caches, so those of another network are never read."""
ZEPHYR_30HZ_RUN: Path = (
    ROOT
    / "runs/rate-balance-pool2-both-30hz/benchmark-rate-balanced-pool2-both-30hz/seed-42"
)
"""The same network trained and run on 30 Hz frames (output still at 60 Hz)."""
REFERENCE_RUN: Path = (
    ROOT / "runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask"
)
"""The benchmark network, trained with time stretching: the stretch ablation's
baseline. Not :data:`ZEPHYR_RUN`, which was trained without stretching."""
CHECKPOINTS: dict[str, Path] = {
    "Zephyr": ZEPHYR_RUN / "best.pt",
    "TS-CAN": BASELINES / "tscan/runs/gray__signal__seed-42/best.pt",
    "PhysNet": BASELINES / "physnet/runs/gray__signal__seed-42/best.pt",
}
NO_STRETCH_RUN: Path = ROOT / "runs/stretch-ablation/benchmark-no-stretch"
STRETCH: str = "Zephyr, stretch"
NO_STRETCH: str = "Zephyr, no stretch"
STRETCH_RUNS: dict[str, Path] = {
    STRETCH: REFERENCE_RUN / "seed-42",
    NO_STRETCH: NO_STRETCH_RUN / "seed-17",
}
"""The two networks of the stretch ablation: Zephyr with stretch and its no-stretch twin."""
PIXEL_TRACES: Path = BASELINES / "pixel/flow/traces/test"
FACEMAP_TRACES: Path = ROOT / "runs/facemap-traces/facemap/both/traces"
OCCLUSION: Path = ZEPHYR_RUN.parents[1] / "occlusion_60s_f1.npz"
"""Occlusion maps of :data:`ZEPHYR_RUN`, next to its experiment's runs."""
OOD_RUNS: dict[str, Path] = {
    "Face only": REFERENCE_RUN / "seed-42",
    "Face + side 13, 14": ROOT
    / "runs/ood-finetune-hold15/ood-hold15-retrain-init/seed-42",
    "Side 14, 15 only": ROOT / "runs/ood-scratch/ood-scratch-hold13/seed-42",
    "Side 13, 14 only": ROOT / "runs/ood-scratch/ood-scratch-hold15/seed-42",
    "PhysNet (DSP), side 14, 15 only": ROOT
    / "runs/physnet-ood-scratch/physnet-ood-scratch-hold13/seed-42",
    "PhysNet (DSP), side 13, 14 only": ROOT
    / "runs/physnet-ood-scratch/physnet-ood-scratch-hold15/seed-42",
}
"""The out-of-distribution networks: the face benchmark network, that network
fine-tuned on two side sessions as well, two trained from scratch on two side
sessions each (a leave-one-session-out pair: held out 13, held out 15), and PhysNet
trained as that pair."""
OOD_NO_HEAD: frozenset[str] = frozenset(n for n in OOD_RUNS if n.startswith("PhysNet"))
"""Networks without an onset head: their inhale F1 is DSP on the trace."""

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
}

STRATA: tuple[str, ...] = ("new_animals", "known_animals_new_date")


OCCLUSION_COMMAND: str = (
    f"zephyr-benchmarks occlusion --checkpoint {ZEPHYR_RUN.relative_to(ROOT).as_posix()}"
    "/best.pt --clips <both test lists> --cache data/features-v2 "
    f"--out {OCCLUSION.relative_to(ROOT).as_posix()}"
)


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


def clips(
    lists: dict[str, Path] = CLIP_LISTS, cache: Path = FEATURES
) -> dict[str, TestClip]:
    """Every clip of *lists* (the test clips by default) by name, with its stratum
    (the list's key) and crop box; features are read from *cache*."""
    out: dict[str, TestClip] = {}
    for stratum, path in lists.items():
        clip_list = load(ClipList, path)
        resolved = clip_list.resolve()
        entries = features.require(resolved, clip_list.preprocess, cache)
        for clip, entry in zip(resolved, entries, strict=True):
            out[short_name(entry)] = TestClip(
                entry, stratum, clip.box, clip.video, clip_list.preprocess.target_size
            )
    return out


def train_entries() -> list[features.ClipEntry]:
    """Every training clip of the benchmark, preprocessed."""
    clip_list = load(ClipList, TRAIN_CLIPS)
    return features.require(clip_list.resolve(), clip_list.preprocess, FEATURES)


def recording_label(entry: features.ClipEntry) -> str:
    """``C:/.../data/test#6`` -> ``rec 6``."""
    return "rec " + entry.recording.rsplit("#", 1)[-1]


def short_name(entry: features.ClipEntry) -> str:
    """``test/video_face_6_part_1`` -> ``face_6_part_1``; other folders keep their
    name as a prefix (``train_face_6_part_1``), since names repeat across folders."""
    name = entry.clip_id.removeprefix("test/").replace("video_", "", 1)
    return name.replace("/", "_")


def raw_frame(clip: TestClip, t_s: float) -> NDArray:
    """The full video frame at time *t_s*, scaled to the working frame size."""
    return video.decode_window(
        clip.video, scale_to=clip.target_size, start_s=t_s, dur_s=0.1
    )[0]


def channel_frames(entry: features.ClipEntry, t_s: float) -> dict[str, NDArray]:
    """What the network sees at time *t_s*: each channel in physical units."""
    index = int(np.searchsorted(np.load(entry.frame_times), t_s))
    return channel_planes(np.load(entry.features, mmap_mode="r")[index])


def channel_planes(stack: NDArray) -> dict[str, NDArray]:
    """One stored ``(C, H, W)`` feature frame as each channel in physical units."""
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


def filtered_truth(entry: features.ClipEntry) -> pd.DataFrame:
    """The thermistor filtered as the training target is, still in ADC units."""
    frame = truth(entry)
    t = frame[TIME_COLUMN].to_numpy()
    fs = 1.0 / float(np.median(np.diff(t)))
    signal = filter_sniff_signal(frame[BREATHING_SIGNAL_COLUMN].to_numpy(float), fs)
    return pd.DataFrame({TIME_COLUMN: t, BREATHING_SIGNAL_COLUMN: signal})


def breathing_rate(entry: features.ClipEntry) -> float:
    """Mean breathing rate of a clip (Hz): the thermistor's inhalations per second."""
    frame = truth(entry)
    t = frame[TIME_COLUMN].to_numpy()
    inhales, _ = events(t, frame[BREATHING_SIGNAL_COLUMN].to_numpy())
    return len(inhales) / float(t[-1] - t[0])


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
    computed once. Also holds Zephyr's onset-head probability as ``onset``.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{short_name(entry)}.parquet"
    others = [m for m in style.METHODS if m != "Zephyr"]
    if cache.exists():
        print(f"Cache loaded from {cache}")
        frame = pd.read_parquet(cache, columns=[TIME_COLUMN, *others])
    else:
        print(f"No cache found in {cache}. Calculating from scratch.")
        n = entry.n_output
        key = Path(entry.features).stem
        old_key = f"test_{short_name(entry).removeprefix('face_')}"
        traces: dict[str, NDArray] = {
            TIME_COLUMN: np.load(entry.times)[:n],
            "Pixel": blind_polarity(np.load(PIXEL_TRACES / f"{old_key}.npy")),
            "Facemap": np.load(FACEMAP_TRACES / f"{key}.npy"),
            "TS-CAN": _infer("TS-CAN", entry)[0],
            "PhysNet": _infer("PhysNet", entry)[0],
        }
        frame = pd.DataFrame(
            {k: v if k == TIME_COLUMN else _zscore(v[:n]) for k, v in traces.items()}
        )
        frame.to_parquet(cache)
    zephyr = zephyr_outputs(entry).drop(columns=TIME_COLUMN)
    return pd.concat([frame, zephyr], axis=1)


def zephyr_outputs(
    entry: features.ClipEntry, run: Path = ZEPHYR_RUN, device: str = "cpu"
) -> pd.DataFrame:
    """Zephyr's z-scored trace and onset probability: ``Time``, ``Zephyr``, ``onset``.

    Inference only (CPU unless *device* says otherwise, in fp32 either way), cached
    per clip and per network (*run*, by its experiment name).
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{short_name(entry)}-zephyr-{run.parent.name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    print(f"No cache found in {cache}. Calculating from scratch.")
    torch_device = torch.device(device)
    model, mean, std, _ = load_checkpoint(run / "best.pt", torch_device)
    signal, onset = predict_clip(
        model, entry, mean, std, device=torch_device, amp_dtype=None
    )
    n = entry.n_output
    frame = pd.DataFrame(
        {TIME_COLUMN: np.load(entry.times)[:n], "Zephyr": _zscore(signal[:n])}
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
    Inhale F1 is the onset head's; exhale F1 has no head and is DSP. KL-IBI is left
    out: the files only hold it from DSP events.
    """
    metrics = {
        "correlation": "correlation",
        "head_inhale_f1": "inhale F1 (head)",
        "exhale_f1": "exhale F1 (DSP)",
    }
    columns = {
        name: json.loads((run / "evaluation.json").read_text())["groups"]["all"][
            "summary"
        ]
        for name, run in STRETCH_RUNS.items()
    }
    return pd.DataFrame(
        {
            name: {label: s[m] for m, label in metrics.items()}
            for name, s in columns.items()
        }
    )


def frame_rate_scores() -> pd.DataFrame:
    """Scores of Zephyr trained on 60 Hz frames and on 30 Hz frames (all 24 test clips).

    One row per metric, one column per network, from their ``evaluation.json``.
    Inhale F1 is the onset head's; exhale F1 has no head and is DSP.
    """
    metrics = {
        "correlation": "correlation",
        "head_inhale_f1": "inhale F1 (head)",
        "exhale_f1": "exhale F1 (DSP)",
    }
    runs = {"Zephyr, 60 Hz frames": ZEPHYR_RUN, "Zephyr, 30 Hz frames": ZEPHYR_30HZ_RUN}
    columns = {
        name: json.loads((run / "evaluation.json").read_text())["groups"]["all"][
            "summary"
        ]
        for name, run in runs.items()
    }
    return pd.DataFrame(
        {
            name: {label: s[m] for m, label in metrics.items()}
            for name, s in columns.items()
        }
    )


def frame_embeddings(
    entry: features.ClipEntry, run: Path = ZEPHYR_RUN, seconds: float = 60.0
) -> tuple[NDArray, NDArray]:
    """The CNN's per-frame embeddings over the first *seconds* of a clip.

    Returns ``(frame_times, embeddings)``, ``(T,)`` and ``(T, embed)``, before any
    resampling to the output grid: what the TCN's input is interpolated from.
    Inference only (CPU), cached per clip and per network.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{short_name(entry)}-embeddings-{run.parent.name}-{seconds:g}s.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["times"], stored["embeddings"]
    print(f"No cache found in {cache}. Calculating from scratch.")
    model, mean, std, _ = load_checkpoint(run / "best.pt", torch.device("cpu"))
    model.eval()
    frame_times = np.load(entry.frame_times)
    n = int(np.searchsorted(frame_times, frame_times[0] + seconds))
    channels = model.channels
    shape = (1, -1, 1, 1)
    mu, sd = (
        np.asarray(s, np.float32).reshape(shape) for s in channels.take_stats(mean, std)
    )
    stack = channels.take(np.asarray(np.load(entry.features, mmap_mode="r")[:n]))
    frames = torch.from_numpy((stack.astype(np.float32) - mu) / sd)
    with torch.no_grad():
        embeddings = model.encode(frames.unsqueeze(0), chunk=256)[0].numpy()
    np.savez(cache, times=frame_times[:n], embeddings=embeddings)
    return frame_times[:n], embeddings


def interpolation_r2(times: NDArray, x: NDArray) -> NDArray:
    """Per column of ``(T, D)`` *x*: R² of rebuilding each sample by linear
    interpolation between its two neighbours (at the real *times*).

    1 for a smooth signal, about -0.5 for white noise (the error of a midpoint
    between two independent samples is 1.5 times the variance).
    """
    x = x.reshape(len(x), -1)
    weight = ((times[1:-1] - times[:-2]) / (times[2:] - times[:-2]))[:, None]
    rebuilt = x[:-2] + (x[2:] - x[:-2]) * weight
    middle = x[1:-1]
    return 1 - ((middle - rebuilt) ** 2).mean(0) / middle.var(0)


def mean_spectrum(times: NDArray, x: NDArray) -> tuple[NDArray, NDArray]:
    """Welch power density of each z-scored column of *x*, averaged over columns
    and scaled to unit area, so signals of any amplitude compare by shape."""
    x = x.reshape(len(x), -1)
    fs = 1.0 / float(np.median(np.diff(times)))
    z = (x - x.mean(0)) / x.std(0)
    freq, power = welch(z, fs=fs, nperseg=int(4 * fs), axis=0)
    power = power.mean(1)
    return freq, power / np.trapezoid(power, freq)


def head_events(frame: pd.DataFrame) -> NDArray:
    """Inhale times (s) a network's onset head asserts; *frame* has ``Time`` and
    ``onset``. The only source of inhale events for a network that has a head."""
    times = frame[TIME_COLUMN].to_numpy()
    return times[head_event_indices(frame["onset"].to_numpy(), times)]


def events(times: NDArray, signal: NDArray) -> tuple[NDArray, NDArray]:
    """Inhale and exhale onset times (s) the scorer detects in a trace (DSP).

    For the thermistor, for methods without an onset head, and for exhalations;
    a network with a head takes its inhalations from :func:`head_events`.
    """
    trace = pd.DataFrame({TIME_COLUMN: times, BREATHING_SIGNAL_COLUMN: signal})
    uniform = resample_uniform(trace)
    inhale, exhale = detect_inhalation_events(
        uniform[BREATHING_SIGNAL_COLUMN].to_numpy(), CANONICAL_BREATHING_SAMPLING_RATE
    )
    t = uniform[TIME_COLUMN].to_numpy()
    return t[inhale], t[exhale]


def method_scores(entry: features.ClipEntry, traces: pd.DataFrame) -> pd.DataFrame:
    """Scores of every method on this clip, as the benchmark scores them.

    Zephyr's inhale F1 and KL-IBI use its onset head's events; methods without a
    head use DSP on their trace (column ``inhale events`` says which). Exhale F1
    is DSP for every method.
    """
    times = traces[TIME_COLUMN].to_numpy()
    reference = truth(entry)
    rows: dict[str, dict[str, object]] = {}
    for method in style.METHODS:
        predicted = pd.DataFrame(
            {TIME_COLUMN: times, BREATHING_SIGNAL_COLUMN: traces[method]}
        )
        if method in HEADED:
            row = score_clip(
                reference, predicted, predicted_onset_times_s=head_events(traces)
            ).to_dict()
        else:
            row = score_clip(reference, predicted).to_dict()
        row["inhale events"] = "head" if method in HEADED else "DSP"
        rows[method] = row
    return pd.DataFrame(rows).T[[*METRICS, "inhale events"]]


def mean_scores() -> pd.DataFrame:
    """:func:`method_scores` averaged over every test clip: one row per method, the
    mean of each metric over the clips (``inhale events`` says where they come from)."""
    per_clip = [
        method_scores(c.entry, method_traces(c.entry)) for c in clips().values()
    ]
    stacked = pd.concat(per_clip, keys=range(len(per_clip)))
    mean = stacked[list(METRICS)].astype(float).groupby(level=1, sort=False).mean()
    return mean.assign(**{"inhale events": per_clip[0]["inhale events"]})


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
HEADED: frozenset[str] = frozenset({"Zephyr"})
"""Compared methods that have an onset head; their inhale events come only from it."""


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


OOD_METRICS: tuple[str, ...] = ("correlation", "inhale_f1")
"""What the OOD figure shows. ``inhale_f1`` is the onset head's, or DSP on the trace
for the networks of :data:`OOD_NO_HEAD`."""


def _f1_column(network: str) -> str:
    """The scorer's column holding *network*'s inhale F1."""
    return "inhale_f1" if network in OOD_NO_HEAD else "head_inhale_f1"


def _face_rows(run: Path, network: str, side: set[str]) -> list[dict]:
    """Per-clip face scores the run's own evaluation wrote (its side clips, if any,
    are rescored by :func:`_side_rows`)."""
    return [
        {
            "network": network,
            "clip": c["clip_id"],
            "data": "Face, held out",
            "correlation": c["correlation"],
            "inhale_f1": c[_f1_column(network)],
        }
        for c in json.loads((run / "evaluation.json").read_text())["clips"]
        if c["clip_id"] not in side
    ]


def _side_rows(run: Path, network: str) -> list[dict]:
    """Every side clip scored with zephyr's per-clip scorer, labelled by whether
    *run* trained on it (from the videos it recorded training on).

    Trained-on clips are in-sample, a reference and never a score; ``evaluate``
    refuses them, so its scorer is called directly. Inference only (CPU), cached
    per run.
    """
    f1 = _f1_column(network)
    side = list(clips(OOD_CLIPS).values())
    cache = CACHE / f"ood-side-{run.parent.parent.name}-{run.parent.name}.parquet"
    if not cache.exists():
        CACHE.mkdir(exist_ok=True)
        device = torch.device("cpu")
        model, mean, std, _ = load_checkpoint(run / "best.pt", device)
        rows = score_entries(
            [(model, mean, std)],
            [c.entry for c in side],
            device,
            window=1024,
            frame_chunk=256,
            amp_dtype=None,
            head=network not in OOD_NO_HEAD,
        )
        frame = pd.DataFrame(rows)[["clip_id", "correlation", f1]]
        frame.rename(columns={"clip_id": "clip"}).to_parquet(cache)
    trained = {
        Path(v).resolve() for v in json.loads((run / "train_videos.json").read_text())
    }
    trained_ids = {c.entry.clip_id for c in side if c.video.resolve() in trained}
    frame = pd.read_parquet(cache).rename(columns={f1: "inhale_f1"})
    frame["data"] = [
        "Side, trained on" if c in trained_ids else "Side, held out"
        for c in frame["clip"]
    ]
    return frame.assign(network=network).to_dict("records")


def ood_scores() -> pd.DataFrame:
    """Correlation and inhale F1 of every network of :data:`OOD_RUNS`, per clip:
    ``network``, ``clip``, ``data`` (face or side, held out or trained on)."""
    side = {c.entry.clip_id for c in clips(OOD_CLIPS).values()}
    rows: list[dict] = []
    for network, run in OOD_RUNS.items():
        rows += _face_rows(run, network, side) + _side_rows(run, network)
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
