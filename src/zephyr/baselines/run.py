"""Run the pixel and Facemap baselines over every preprocessed clip and score them.

Layout under ``out_root`` (default ``benchmarks/baselines-v1``)::

    pixel/{flow,pca,snr}/traces/{split}/{clip_id}.npy   raw output-grid traces (cache)
    pixel/{method}/{blind,calibrated}/evaluation.json
    facemap/bases.npz
    facemap/{motion,movie,both}/evaluation.json

Each ``evaluation.json`` has the same shape as ``zephyr evaluate --out``, plus the
method's own fitted constants.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import build_result, load_test_strata, truth_frame, zscore
from ..evaluation import score_clip
from ..signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from . import common, facemap_ridge, pixel

PIXEL_METHODS = ("flow", "pca", "snr")
FACEMAP_VARIANTS = ("motion", "movie", "both")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def target_signal(entry) -> np.ndarray:
    return pd.read_parquet(entry.target)["signal"].to_numpy(np.float64)


def score_traces(traces, entries, *, packaged_root, split, split_manifest) -> dict:
    """Score output-grid *traces* (``clip_id -> array``) exactly as ``zephyr evaluate`` does."""
    rows = []
    for entry in sorted(entries, key=lambda e: (e.session_idx, e.part)):
        times = np.load(entry.times)
        signal = traces[entry.clip_id]
        n = min(len(times), len(signal))
        predicted = pd.DataFrame(
            {
                TIME_COLUMN: times[:n].astype(np.float64),
                BREATHING_SIGNAL_COLUMN: zscore(signal[:n]).astype(np.float64),
            }
        )
        truth = truth_frame(packaged_root, split, entry.session_idx, entry.part)
        rows.append(
            {
                "clip_id": entry.clip_id,
                "session_idx": entry.session_idx,
                "part": entry.part,
            }
            | score_clip(truth, predicted).to_dict()
        )
    return build_result(
        rows, {e.session_idx for e in entries}, load_test_strata(split_manifest)
    )


def pixel_trace(method: str, entry, fs: float, bin_factor: int) -> np.ndarray:
    """Selection-grid trace for one clip."""
    if method == "flow":
        return pixel.flow_projection(
            common.load_channel(entry, "flow_x", bin_factor=4),
            common.load_channel(entry, "flow_y", bin_factor=4),
            fs,
        )
    frames = common.load_channel(entry, "gray", bin_factor=bin_factor)
    if method == "pca":
        return pixel.pixel_pca(frames, fs)
    if method == "snr":
        return pixel.snr_weighted(frames, fs)
    raise ValueError(f"unknown pixel method {method!r}")


def cached_traces(method, entries, split, out_root, fs, bin_factor) -> dict:
    cache = out_root / "pixel" / method / "traces" / split
    cache.mkdir(parents=True, exist_ok=True)
    traces = {}
    for entry in entries:
        path = cache / f"{entry.clip_id}.npy"
        if not path.exists():
            trace = pixel_trace(method, entry, fs, bin_factor)
            np.save(path, common.to_output_grid(entry, trace))
        traces[entry.clip_id] = np.load(path)
        print(f"  {method} {split} {entry.clip_id}", flush=True)
    return traces


def run_pixel(
    train, test, *, methods, out_root, fs, output_fs, bin_factor, scoring
) -> None:
    for method in methods:
        raw_train = cached_traces(method, train, "train", out_root, fs, bin_factor)
        raw_test = cached_traces(method, test, "test", out_root, fs, bin_factor)
        blind_train = {k: common.blind_polarity(v) for k, v in raw_train.items()}
        blind_test = {k: common.blind_polarity(v) for k, v in raw_test.items()}

        pairs = []
        for entry in train:
            truth = target_signal(entry)
            n = min(len(truth), len(blind_train[entry.clip_id]))
            pairs.append((blind_train[entry.clip_id][:n], truth[:n]))
        positive = float(np.mean([np.corrcoef(p, t)[0, 1] > 0 for p, t in pairs]))
        sign, lag, train_r = common.fit_sign_lag(
            pairs, round(common.MAX_LAG_S * output_fs)
        )
        calibrated_test = {
            k: sign * common.shift(v, lag) for k, v in blind_test.items()
        }

        base = {"method": method, "band_hz": list(common.BREATH_BAND_HZ)}
        write_json(
            out_root / "pixel" / method / "blind" / "evaluation.json",
            score_traces(blind_test, test, **scoring)
            | base
            | {"variant": "blind", "train_positive_polarity_fraction": positive},
        )
        write_json(
            out_root / "pixel" / method / "calibrated" / "evaluation.json",
            score_traces(calibrated_test, test, **scoring)
            | base
            | {
                "variant": "calibrated",
                "sign": sign,
                "lag_samples": lag,
                "train_correlation": train_r,
            },
        )
        print(f"{method}: sign {sign:+d} lag {lag} train r {train_r:+.3f}", flush=True)


def run_facemap(
    train,
    test,
    *,
    out_root,
    n_components,
    samples_per_clip,
    bin_factor,
    max_lag,
    lag_step,
    seed,
    scoring,
) -> None:
    rng = np.random.default_rng(seed)
    samples = {"motion": [], "movie": []}
    for entry in train:
        frames = common.load_channel(entry, "gray", bin_factor=bin_factor)
        idx = np.sort(
            rng.choice(len(frames), min(samples_per_clip, len(frames)), replace=False)
        )
        samples["motion"].append(facemap_ridge.motion_matrix(frames)[idx])
        samples["movie"].append(facemap_ridge.movie_matrix(frames)[idx])
    bases = {k: facemap_ridge.fit_basis(v, n_components) for k, v in samples.items()}
    del samples
    (out_root / "facemap").mkdir(parents=True, exist_ok=True)
    np.savez(
        out_root / "facemap" / "bases.npz",
        **{
            f"{k}_{f}": getattr(b, f)
            for k, b in bases.items()
            for f in ("mean", "components")
        },
    )

    projections: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for split, entries in (("train", train), ("test", test)):
        for entry in entries:
            frames = common.load_channel(entry, "gray", bin_factor=bin_factor)
            matrices = {
                "motion": facemap_ridge.motion_matrix(frames),
                "movie": facemap_ridge.movie_matrix(frames),
            }
            projections[(split, entry.clip_id)] = {
                k: facemap_ridge.zscore_columns(
                    common.to_output_grid(entry, facemap_ridge.project(m, bases[k]))
                ).astype(np.float32)
                for k, m in matrices.items()
            }
            print(f"  facemap {split} {entry.clip_id}", flush=True)

    lags = np.arange(-max_lag, max_lag + 1, lag_step)

    def features(split, entry, variant):
        p = projections[(split, entry.clip_id)]
        if variant == "both":
            return np.concatenate([p["motion"], p["movie"]], axis=1)
        return p[variant]

    for variant in FACEMAP_VARIANTS:
        grams: dict[int, facemap_ridge.Gram] = {}
        for entry in train:
            x = facemap_ridge.lagged_design(features("train", entry, variant), lags)
            y = target_signal(entry)
            n = min(len(x), len(y))
            gram = facemap_ridge.Gram.of(x[:n], y[:n])
            grams[entry.session_idx] = (
                grams[entry.session_idx] + gram if entry.session_idx in grams else gram
            )
        alpha, table = facemap_ridge.select_alpha(grams)
        w = facemap_ridge.solve(facemap_ridge.total(grams), alpha)
        traces = {
            e.clip_id: facemap_ridge.lagged_design(features("test", e, variant), lags)
            @ w
            for e in test
        }
        write_json(
            out_root / "facemap" / variant / "evaluation.json",
            score_traces(traces, test, **scoring)
            | {
                "method": "facemap",
                "variant": variant,
                "alpha": alpha,
                "alpha_table": table,
                "n_features": len(w),
                "lags": lags.tolist(),
                "n_components": n_components,
                "bin_factor": bin_factor,
            },
        )
        print(f"facemap {variant}: alpha {alpha:.3g}", flush=True)
