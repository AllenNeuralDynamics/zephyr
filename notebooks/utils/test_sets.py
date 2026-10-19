"""Networks scored on the split's test sets, clip by clip, with what is known of each clip.

Reads a local folder (``ROOT_ENV``) laid out as ``LAYOUT``: copies of the dataset's lists
and tables, the cache entries of the scored clips, the checkpoints, and one ``zephyr
evaluate`` result per network. Image statistics, predictions and thumbnail embeddings
(DINOv2, the ``rois`` extra) are computed here (inference only) and cached in
``notebooks/cache/``; nothing is trained or written into the folder.
"""

import functools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from zephyr import features
from zephyr.config import PATH_MAP_ENV, ClipList, load
from zephyr.evaluate import head_event_indices, load_checkpoint
from zephyr.infer import predict_clip

from . import style

CACHE = Path(__file__).resolve().parents[1] / "cache"
"""``notebooks/cache/``: everything the notebooks compute once (ignored by git)."""
ROOT_ENV = "ZEPHYR_TEST_ROOT"
"""The local folder; the notebook offers it (or ``C:/Data/zephyr-local-v2``) by default."""
DATASET_ROOT = "/data/zephyr-dataset"
"""Where the dataset the copy was taken from lives (the cache keys name it)."""
LAYOUT = {
    "lists": "dataset/full_test_{a,b,c}.toml",
    "tables": "dataset/{folds,qc}.parquet",
    "cache": "cache/features-360x270-box96x96-60hz/",
    "runs": "runs/<network>/dev/seed-17/best.pt",
    "scores": "evaluations/<network>.json",
    "thumbnails": "dataset/thumbnails/<session>_FaceCamera.png",
}
GROUPS = {
    "full_test_a": "A: new animals",
    "full_test_b": "B: new sessions",
    "full_test_c": "C: 359 Hz",
}
METRICS = ["correlation", "inhale_f1", "exhale_f1", "kl_ibi", "head_inhale_f1"]
RECIPE = "features-360x270-box96x96-60hz"


def use_copy(root: Path) -> None:
    """Name *root*'s dataset copy as the original, so its cache entries are found."""
    os.environ[PATH_MAP_ENV] = f"{(root / 'dataset').as_posix()}={DATASET_ROOT}"


def networks(root: Path) -> dict[str, Path]:
    """Network label (run folder without its hash) -> its ``zephyr evaluate`` result."""
    return {
        p.stem.rsplit("-", 1)[0]: p
        for p in sorted((root / "evaluations").glob("*.json"))
    }


def scores(root: Path, names: list[str]) -> pd.DataFrame:
    """One row per network and clip of the test lists: the clip's scores, group,
    session and stem."""
    found = networks(root)
    frames = []
    for name in names:
        rows = pd.DataFrame(json.loads(found[name].read_text())["clips"])
        frames.append(rows.assign(network=name))
    df = pd.concat(frames, ignore_index=True)
    df["stem"] = df["clip_id"].str.split("/").str[-1]
    df["session"] = df["recording"].str.split("#").str[-1]
    df["set"] = df["group"].map(GROUPS)
    return df[df["stem"].isin(_clips(root))].reset_index(drop=True)


@functools.cache
def _clips(root: Path) -> dict[str, tuple]:
    """Clip stem -> (resolved clip, recipe) over the test lists."""
    out = {}
    for path in sorted((root / "dataset").glob("full_test_*.toml")):
        clip_list = load(ClipList, path, check_files=False)
        out |= {c.source.stem: (c, clip_list.preprocess) for c in clip_list.resolve()}
    return out


def _image_stats(gray: NDArray) -> dict[str, float]:
    """Brightness, contrast, frame-to-frame motion and sharpness of 300 frames."""
    idx = np.linspace(0, len(gray) - 2, 300).astype(int)
    a = gray[idx, 0].astype(np.float32)
    b = gray[idx + 1, 0].astype(np.float32)
    lap = (
        4 * a[:, 1:-1, 1:-1]
        - a[:, :-2, 1:-1]
        - a[:, 2:, 1:-1]
        - a[:, 1:-1, :-2]
        - a[:, 1:-1, 2:]
    )
    return {
        "brightness": float(a.mean()),
        "contrast": float(a.std(axis=(1, 2)).mean()),
        "motion": float(np.abs(b - a).mean()),
        "sharpness": float(np.abs(lap).mean()),
        "dark_frac": float((a <= 5).mean()),
        "bright_frac": float((a >= 250).mean()),
    }


def clip_info(root: Path) -> pd.DataFrame:
    """One row per test clip: animal, rig, date, thermistor QC, box, image statistics."""
    cached = CACHE / f"test_sets_clip_info-{root.name}.parquet"  # one per folder
    if cached.is_file():
        return pd.read_parquet(cached)
    use_copy(root)
    cache = root / "cache" / RECIPE
    rows = []
    for stem, (clip, recipe) in _clips(root).items():
        entry = features.lookup(clip, recipe, cache)
        if entry is None:
            continue
        sidecar = features.paths(entry.stem.parent, entry.stem.name)["sidecar"]
        stats = json.loads(sidecar.read_text())["stats"]
        gray = np.load(entry.group_file("gray"), mmap_mode="r")
        rows.append(
            {
                "stem": stem,
                "box_x": clip.box[0],
                "box_y": clip.box[1],
                "native_hz": stats["native_fs_hz"],
                "breathing_hz": stats["breathing_rate_hz"],
                "thermistor_amp": stats["target_scale_adc"],
            }
            | _image_stats(gray)
        )
    info = pd.DataFrame(rows)
    info["session"] = info["stem"].str.rsplit("_span", n=1).str[0]
    folds = pd.read_parquet(root / "dataset/folds.parquet")
    folds = folds[["session_id", "subject_id", "rig", "date"]]
    qc = pd.read_parquet(root / "dataset/qc.parquet")
    qc = qc[["session_id", "snr_db", "break_frac", "good_frac", "rate_cv"]]
    info = info.merge(folds, left_on="session", right_on="session_id", how="left")
    info = info.merge(qc, on="session_id", how="left").drop(columns="session_id")
    CACHE.mkdir(parents=True, exist_ok=True)
    info.to_parquet(cached)
    return info


@functools.cache
def _network(root: Path, name: str, device: str):
    """``(model, mean, std, device, amp)`` of network *name*, loaded once."""
    (run,) = (root / "runs").glob(f"{name}-*/dev/seed-17/best.pt")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, mean, std, _ = load_checkpoint(run, dev)
    return model, mean, std, dev, torch.bfloat16 if dev.type == "cuda" else None


def _entry(root: Path, stem: str):
    use_copy(root)
    clip, recipe = _clips(root)[stem]
    return features.lookup(clip, recipe, root / "cache" / RECIPE)


def prediction(root: Path, name: str, stem: str, device: str = "cuda") -> Path | None:
    """Network *name*'s prediction of clip *stem*, cached as ``<stem>.npz`` (``signal``,
    ``onset``, ``truth``, ``times``) and predicted only the first time; ``None`` for a
    clip with no cache entry or target."""
    path = CACHE / f"test_sets_predictions-{name}" / f"{stem}.npz"
    if path.is_file():
        return path
    entry = _entry(root, stem)
    if entry is None or not entry.has_target:
        return None
    model, mean, std, dev, amp = _network(root, name, device)
    signal, onset = predict_clip(model, entry, mean, std, device=dev, amp_dtype=amp)
    truth = pd.read_parquet(entry.target)["signal"].to_numpy(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".tmp.npz")
    np.savez(part, signal=signal, onset=onset, truth=truth, times=np.load(entry.times))
    part.replace(path)
    return path


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over recordings (clips averaged within their recording first), as zephyr
    reports, per network and set."""
    per_recording = df.groupby(["network", "set", "recording"])[METRICS].mean()
    return per_recording.groupby(["network", "set"]).mean().reset_index()


def missing(root: Path) -> list[str]:
    """What the folder lacks of ``LAYOUT``, empty when it is complete."""
    wanted = [root / "dataset" / f"full_test_{s}.toml" for s in "abc"]
    wanted += [
        root / "dataset/folds.parquet",
        root / "dataset/qc.parquet",
        root / "cache" / RECIPE,
    ]
    absent = [str(p) for p in wanted if not p.exists()]
    if not networks(root):
        absent.append(str(root / "evaluations/<network>.json"))
    return absent


COVARIATES = {
    "break_frac": "thermistor breaks",
    "good_frac": "thermistor good frac.",
    "snr_db": "thermistor SNR (dB)",
    "breathing_hz": "breathing rate (Hz)",
    "sharpness": "image sharpness",
    "brightness": "image brightness",
    "motion": "frame-to-frame motion",
}


def _jitter(n: int, seed: int = 0) -> NDArray:
    return np.random.default_rng(seed).uniform(-0.25, 0.25, n)


def by_video_figure(df: pd.DataFrame, names: list[str]):
    """Every clip's correlation and inhale F1, per set (rows) and network (x)."""
    sets = [s for s in GROUPS.values() if s in set(df["set"])]
    colours = dict(zip(names, style.series(len(names)), strict=True))
    fig, axes = style.figure(
        "double", 0.32 * len(sets), nrows=len(sets), ncols=2, squeeze=False
    )
    for row, name in enumerate(sets):
        for col, metric in enumerate(["correlation", "inhale_f1"]):
            ax = axes[row, col]
            for x, net in enumerate(names):
                v = df.loc[
                    (df["set"] == name) & (df["network"] == net), metric
                ].to_numpy()
                ax.plot(
                    x + _jitter(len(v)), v, "o", ms=1.6, alpha=0.5, color=colours[net]
                )
                ax.plot(
                    [x - 0.3, x + 0.3], [np.median(v)] * 2, color=style.POOLED, lw=1.2
                )
            ax.set_xticks(
                range(len(names)),
                names if row == len(sets) - 1 else [""] * len(names),
                rotation=20,
                ha="right",
            )
            ax.set_ylabel(f"{name}\n{metric}" if col == 0 else metric)
            ax.set_ylim(-0.6 if metric == "correlation" else -0.05, 1.02)
    return fig


def by_animal_figure(df: pd.DataFrame, info: pd.DataFrame, names: list[str]):
    """Median clip correlation of every animal (rows, worst first) for each network."""
    m = df.merge(info[["stem", "subject_id"]], on="stem")
    table = m.pivot_table(
        index=["set", "subject_id"],
        columns="network",
        values="correlation",
        aggfunc="median",
    )[names]
    table = table.loc[table.mean(axis=1).sort_values().index]
    fig, ax = style.figure("double", max(0.4, 0.022 * len(table)))
    image = ax.imshow(
        table.to_numpy(), aspect="auto", cmap=style.EMBEDDING_CMAP, vmin=-1, vmax=1
    )
    ax.set_yticks(
        range(len(table)), [f"{s[0]} {a}" for s, a in table.index], fontsize=5
    )
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="median clip correlation", shrink=0.6)
    return fig


def covariates_figure(df: pd.DataFrame, info: pd.DataFrame, network: str):
    """Clip correlation of one network against what is known of each clip."""
    m = df[df["network"] == network].merge(info, on=["stem", "session"])
    fig, axes = style.figure("double", 0.42, nrows=2, ncols=4)
    for ax, (key, label) in zip(axes.flat, COVARIATES.items(), strict=False):
        x = m[key].to_numpy(float)
        ax.plot(
            x, m["correlation"], "o", ms=1.5, alpha=0.5, color=style.color("Zephyr")
        )
        rho = pd.Series(x).corr(m["correlation"], method="spearman")
        ax.set_title(f"rho {rho:+.2f}", fontweight="normal")
        ax.set_xlabel(label)
        if key == "break_frac":
            ax.set_xscale("symlog", linthresh=1e-4)
    ax = axes.flat[-1]
    order = m.groupby("rig")["correlation"].median().sort_values().index
    ax.boxplot(
        [m.loc[m["rig"] == r, "correlation"] for r in order],
        tick_labels=list(order),
        showfliers=False,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=5)
    ax.set_xlabel("rig")
    for ax in axes[:, 0]:
        ax.set_ylabel("clip correlation")
    return fig


CASE_COLOURS = {
    "truth": style.TRUTH,
    "prediction": style.color("Zephyr"),
    "head": "#d55e00",
}


def clip_table(df: pd.DataFrame, info: pd.DataFrame, name: str) -> pd.DataFrame:
    """Network *name*'s clips (*df* from :func:`scores`) with each clip's animal, date,
    rig and set; ``r`` is its correlation as scored."""
    m = df[df["network"] == name].merge(info, on=["stem", "session"])
    m["date"] = pd.to_datetime(m["date"])
    return m.assign(r=m["correlation"])


def clip_traces(root: Path, name: str, stem: str) -> dict:
    """Everything recorded for clip *stem* and network *name*, on the session clock: the
    raw thermistor, the target, the prediction and onset head, the truth events and the
    gray frames."""
    entry = _entry(root, stem)
    p = np.load(prediction(root, name, stem))
    clip, _ = _clips(root)[stem]
    raw = pd.read_parquet(clip.thermistor)
    events = np.load(entry.events)
    times = p["times"]
    truth = p["truth"]
    return {
        "stem": stem,
        "times": times,
        "signal": p["signal"],
        "onset": p["onset"],
        "truth": (truth - truth.mean()) / (truth.std() or 1.0),
        "raw_t": raw["Time"].to_numpy(),
        "raw": raw["Signal"].to_numpy(),
        "inhales": events["onset_times"],
        "head": times[head_event_indices(p["onset"], times)],
        "frame_t": np.load(entry.frame_times),
        "gray": np.load(entry.group_file("gray"), mmap_mode="r"),
    }


def _window(t: NDArray, t0: float, t1: float) -> NDArray:
    return (t >= t0) & (t <= t1)


def _inhale_lines(ax, tr: dict, t0: float, t1: float) -> None:
    for t in tr["inhales"][_window(tr["inhales"], t0, t1)]:
        ax.axvline(t - tr["times"][0], color="0.75", lw=0.4, zorder=0)


def _traces_axes(ax, tr: dict, t0: float, t1: float) -> None:
    """Target and prediction over [t0, t1]."""
    w = _window(tr["times"], t0, t1)
    rel = tr["times"][w] - tr["times"][0]
    ax.plot(
        rel,
        tr["truth"][w],
        color=CASE_COLOURS["truth"],
        lw=0.9,
        label="thermistor (target)",
    )
    ax.plot(
        rel,
        tr["signal"][w],
        color=CASE_COLOURS["prediction"],
        lw=0.9,
        label="prediction",
    )
    _inhale_lines(ax, tr, t0, t1)
    ax.set_xlim(t0 - tr["times"][0], t1 - tr["times"][0])


def animal_options(m: pd.DataFrame) -> dict[str, str]:
    """Each animal of *m* (:func:`clip_table`), lowest median r first: label -> subject."""
    r = m.groupby("subject_id")["r"].median().sort_values()
    rigs = m.groupby("subject_id")["rig"].nunique()
    return {f"{s} (median r {r[s]:+.2f}, rigs: {rigs[s]})": s for s in r.index}


def animal_rig_figure(
    root: Path, name: str, m: pd.DataFrame, subject: str, seconds: float = 4.0
):
    """One excerpt per rig animal *subject* ran on (its clip there with the median r):
    thermistor and prediction."""
    d = m[m["subject_id"] == subject]
    rigs = sorted(d["rig"].unique())
    fig, axes = style.figure("double", 0.17 * len(rigs), nrows=len(rigs), squeeze=False)
    for ax, rig in zip(axes[:, 0], rigs, strict=True):
        on = d[d["rig"] == rig].sort_values("r")
        row = on.iloc[len(on) // 2]
        tr = clip_traces(root, name, row["stem"])
        mid = tr["times"][len(tr["times"]) // 2]
        _traces_axes(ax, tr, mid, mid + seconds)
        rs = ", ".join(f"{v:+.2f}" for v in on["r"])
        ax.set_title(
            f"rig {rig}: {row['date']:%Y-%m-%d}, r {row['r']:+.2f} "
            f"(r of its {len(on)} clips here: {rs})",
            fontweight="normal",
            loc="left",
            fontsize=6.5,
        )
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=5, ncol=2)
    axes[-1, 0].set_xlabel("time in clip (s)")
    return fig


def raw_figure(tr: dict, start: float, length: float, n_frames: int = 8):
    """Clip *tr* (:func:`clip_traces`) over [start, start + length] s from its start: gray
    frames, the raw thermistor, target against prediction, and the onset head."""
    t0 = tr["times"][0] + start
    t1 = t0 + length
    frames = [f"f{i}" for i in range(n_frames)]
    layout = [frames, ["raw"] * n_frames, ["trace"] * n_frames, ["head"] * n_frames]
    fig, ax = style.mosaic(layout, "double", 0.62, height_ratios=[0.9, 1, 1.4, 0.8])
    for name, t in zip(frames, np.linspace(t0, t1, n_frames), strict=True):
        k = int(np.clip(np.searchsorted(tr["frame_t"], t), 0, len(tr["gray"]) - 1))
        ax[name].imshow(tr["gray"][k, 0], cmap="gray", vmin=0, vmax=255)
        ax[name].set_axis_off()
        ax[name].set_title(
            f"{t - tr['times'][0]:.2f} s", fontsize=5, fontweight="normal"
        )
    w = _window(tr["raw_t"], t0, t1)
    ax["raw"].plot(
        tr["raw_t"][w] - tr["times"][0], tr["raw"][w], color=style.TRUTH, lw=0.6
    )
    _inhale_lines(ax["raw"], tr, t0, t1)
    ax["raw"].set(ylabel="thermistor (ADC)", xlim=(start, start + length))
    _traces_axes(ax["trace"], tr, t0, t1)
    ax["trace"].set(ylabel="z-score")
    ax["trace"].legend(loc="upper right", fontsize=5, ncol=2)
    w = _window(tr["times"], t0, t1)
    rel = tr["times"][w] - tr["times"][0]
    ax["head"].plot(rel, tr["onset"][w], color=CASE_COLOURS["prediction"], lw=0.7)
    _inhale_lines(ax["head"], tr, t0, t1)
    head = tr["head"][_window(tr["head"], t0, t1)] - tr["times"][0]
    ax["head"].plot(
        head, np.full(len(head), 1.02), "v", ms=2.5, color=CASE_COLOURS["head"]
    )
    ax["head"].set(ylabel="onset head", xlabel="time in clip (s)")
    ax["head"].set(xlim=(start, start + length), ylim=(0, 1.1))
    fig.suptitle(
        f"{tr['stem']} (grey: thermistor inhales, triangles: head events)",
        fontsize=6.5,
        fontweight="normal",
    )
    return fig


QC = {
    "session_break_frac": "alternation breaks (session)",
    "session_consistency_z": "event inconsistency z (session)",
    "session_good_frac": "good-window fraction (session)",
    "break_frac": "alternation breaks",
    "outside_frac": "breaths outside the rate band",
    "rate_cv": "breath rate CV",
    "amp_cv": "amplitude CV",
    "snr_db": "SNR (dB)",
    "amplitude": "amplitude (ADC)",
    "mean_rate": "breathing rate (Hz)",
}
"""The span QC's metrics (``zephyr.session_selection``), each clip's own unless marked
session (``qc.parquet``)."""


def _noise(x: NDArray) -> float:
    step = np.diff(x)
    return 1.4826 * float(np.median(np.abs(step - np.median(step)))) / np.sqrt(2)


def clip_qc(root: Path) -> pd.DataFrame:
    """The span QC's metrics over each test clip's own stretch of thermistor (1 s blocks
    for amplitude, the clip's inhales and exhales for the rest); cached."""
    cached = CACHE / f"test_sets_clip_qc-{root.name}.parquet"  # one per folder
    if cached.is_file():
        return pd.read_parquet(cached)
    rows = []
    for stem, (clip, _) in _clips(root).items():
        entry = _entry(root, stem)
        if entry is None or not entry.has_target:
            continue
        frames = np.load(entry.frame_times)
        t0, t1 = float(frames[0]), float(frames[-1])
        raw = pd.read_parquet(clip.thermistor)
        t = raw["Time"].to_numpy()
        x = raw["Signal"].to_numpy(float)[(t >= t0) & (t <= t1)]
        block = round(1 / float(np.median(np.diff(t))))
        lo, hi = np.percentile(
            x[: len(x) // block * block].reshape(-1, block), [5, 95], 1
        )
        amp = hi - lo
        events = np.load(entry.events)
        on, off = (
            events[k][_window(events[k], t0, t1)]
            for k in ("onset_times", "offset_times")
        )
        kind = np.r_[np.zeros(len(on)), np.ones(len(off))][np.argsort(np.r_[on, off])]
        rates = 1 / np.diff(np.sort(on))
        rows.append(
            {
                "stem": stem,
                "snr_db": 20 * np.log10(amp.mean() / _noise(x)),
                "amplitude": amp.mean(),
                "amp_cv": amp.std() / amp.mean(),
                "break_frac": np.mean(kind[1:] == kind[:-1]),
                "outside_frac": np.mean((rates < 2) | (rates > 15)),
                "rate_cv": rates.std() / rates.mean(),
                "mean_rate": rates.mean(),
            }
        )
    out = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cached)
    return out


def qc_table(root: Path, m: pd.DataFrame) -> pd.DataFrame:
    """*m* (:func:`clip_table`) with the :data:`QC` metrics."""
    qc = pd.read_parquet(root / "dataset/qc.parquet").drop_duplicates("session_id")
    qc = qc.set_index("session_id")
    m = m.drop(
        columns=["snr_db", "break_frac", "rate_cv", "good_frac"], errors="ignore"
    )
    m = m.merge(clip_qc(root), on="stem")
    for key in ("break_frac", "consistency_z", "good_frac"):
        m[f"session_{key}"] = m["session"].map(qc[key])
    return m


def qc_figure(q: pd.DataFrame):
    """Whether thermistor quality goes with the network's r (*q* from :func:`qc_table`):
    Spearman rho of each QC metric with r over clips, and r against amplitude."""
    fig, ax = style.mosaic([["rho", "scatter"]], "double", 0.36)
    rho = pd.Series(
        [q[[key, "r"]].corr("spearman").iloc[0, 1] for key in QC],
        index=list(QC.values()),
    )
    ax["rho"].barh(rho.index, rho, color=style.color("Zephyr"))
    ax["rho"].axvline(0, color="0.6", lw=0.5)
    ax["rho"].invert_yaxis()
    ax["rho"].tick_params(axis="y", labelsize=5.5)
    ax["rho"].set(xlabel="Spearman rho with r, over clips", xlim=(-0.6, 0.6))
    ax["scatter"].scatter(
        q["amplitude"], q["r"], s=3, alpha=0.6, color=style.color("Zephyr")
    )
    ax["scatter"].set(
        xscale="log", xlabel="thermistor amplitude (ADC)", ylabel="r as scored"
    )
    return fig


SPLIT = {
    "train": "train",
    "val": "validation",
    "test_a": GROUPS["full_test_a"],
    "test_b": GROUPS["full_test_b"],
    "test_c": GROUPS["full_test_c"],
}
"""``folds.parquet``'s sets, named as the scores name them."""
SPLIT_COLOURS = {
    "train": "0.6",
    "validation": "#cc79a7",
    GROUPS["full_test_a"]: "#d55e00",
    GROUPS["full_test_b"]: style.color("Zephyr"),
    GROUPS["full_test_c"]: "#009e73",
}

_traces = functools.cache(clip_traces)
""":func:`clip_traces`, kept, so moving a slider only redraws."""


def low_sessions(m: pd.DataFrame, below: float) -> pd.DataFrame:
    """The sessions of *m* (:func:`clip_table`) whose clips' mean r (as scored) is
    under *below*, worst first, with their clips worst first."""
    g = m.sort_values("r").groupby("session", sort=False)
    out = g.agg(
        subject_id=("subject_id", "first"),
        rig=("rig", "first"),
        set=("set", "first"),
        r=("r", "mean"),
        stems=("stem", list),
        clip_r=("r", list),
    )
    return out[out["r"] < below].sort_values("r").reset_index()


def low_figure(
    root: Path, name: str, low: pd.DataFrame, start: float, length: float, which: int
):
    """Each session of *low* (:func:`low_sessions`), worst first: its *which* clip (0 the
    worse) over [start, start + length] s of the clip: the frame at *start*, the raw
    thermistor, and target against prediction (grey lines: the thermistor's inhales)."""
    layout = []
    for i in range(len(low)):
        layout += [[f"img{i}", f"raw{i}"], [f"img{i}", f"tr{i}"]]
    fig, ax = style.mosaic(
        layout,
        "double",
        0.13 * len(low),
        width_ratios=[1, 9],
        height_ratios=[1, 1.4] * len(low),
    )
    for i, row in enumerate(low.itertuples()):
        k = min(which, len(row.stems) - 1)
        tr = _traces(root, name, row.stems[k])
        t0 = tr["times"][0] + start
        t1 = t0 + length
        f = int(np.clip(np.searchsorted(tr["frame_t"], t0), 0, len(tr["gray"]) - 1))
        ax[f"img{i}"].imshow(tr["gray"][f, 0], cmap="gray", vmin=0, vmax=255)
        ax[f"img{i}"].set_axis_off()
        w = _window(tr["raw_t"], t0, t1)
        raw = ax[f"raw{i}"]
        raw.plot(
            tr["raw_t"][w] - tr["times"][0], tr["raw"][w], color=style.TRUTH, lw=0.5
        )
        raw.set_xlim(start, start + length)
        raw.tick_params(labelbottom=False, labelsize=4.5)
        raw.set_title(
            f"{row.session}: {row.subject_id}, rig {row.rig}, {row.set} | clip "
            f"{k + 1} of {len(row.stems)}: r {row.clip_r[k]:+.2f} (session r "
            f"{row.r:+.2f})",
            fontweight="normal",
            loc="left",
            fontsize=5.5,
        )
        _traces_axes(ax[f"tr{i}"], tr, t0, t1)
        ax[f"tr{i}"].set_yticks([])
        if i < len(low) - 1:
            ax[f"tr{i}"].tick_params(labelbottom=False)
    ax["tr0"].legend(loc="upper right", fontsize=4.5, ncol=2)
    ax[f"tr{len(low) - 1}"].set_xlabel("time in clip (s)")
    return fig


VIEWS = {"face crop (what the network sees)": "crop", "whole frame": "frame"}
EMBEDDER = "facebook/dinov2-small"
"""A self-supervised image model whose features say how alike two images look."""


@functools.cache
def _boxes(root: Path) -> tuple[dict[str, tuple[int, ...]], tuple[int, int]]:
    """Session -> its face box, and the frame the boxes are in (``clips.toml``)."""
    clip_list = load(ClipList, root / "dataset/clips.toml", check_files=False)
    boxes = {c.group: tuple(c.box) for c in clip_list.resolve()}
    return boxes, tuple(clip_list.preprocess.target_size)


def thumbnail(root: Path, session: str, view: str):
    """The session's face camera thumbnail (``LAYOUT``), whole or cropped to its box."""
    from PIL import Image

    image = Image.open(root / "dataset/thumbnails" / f"{session}_FaceCamera.png")
    image = image.convert("RGB")
    if view == "crop":
        boxes, (w, _) = _boxes(root)
        x, y, bw, bh = (v * image.width / w for v in boxes[session])
        image = image.crop((x, y, x + bw, y + bh))
    return image


def embeddings(root: Path, view: str) -> pd.DataFrame:
    """Every session's thumbnail embedding (:data:`EMBEDDER`, unit length; rows by
    session), for *view* ``crop`` or ``frame``; cached."""
    cached = CACHE / f"test_sets_thumbnails-{root.name}-{view}.parquet"
    if cached.is_file():
        return pd.read_parquet(cached)
    from transformers import AutoImageProcessor, AutoModel

    sessions = sorted(_boxes(root)[0])
    processor = AutoImageProcessor.from_pretrained(EMBEDDER)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(EMBEDDER).to(dev).eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(sessions), 32):
            images = [thumbnail(root, s, view) for s in sessions[i : i + 32]]
            batch = processor(images=images, return_tensors="pt").to(dev)
            rows.append(model(**batch).pooler_output.float().cpu().numpy())
    x = np.concatenate(rows)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    out = pd.DataFrame(x, index=pd.Index(sessions, name="session"))
    out.columns = [f"e{j}" for j in out.columns]
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cached)
    return out


def nearest_train(root: Path, view: str, m: pd.DataFrame) -> pd.DataFrame:
    """Every session of the split: its set, animal and rig, its r (test sessions, mean
    over clips of *m*), the thumbnails' first two principal components, and the cosine
    similarity to the most alike training session (``nearest``, which, its animal), to
    the most alike of its own animal (``nearest_same``) and of another
    (``nearest_other``); a training session is not its own neighbour."""
    folds = pd.read_parquet(root / "dataset/folds.parquet").dropna(subset=["set"])
    folds = folds.set_index("session_id")
    e = embeddings(root, view).loc[folds.index].to_numpy()
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    train = (folds["set"] == "train").to_numpy()
    animal = folds["subject_id"].astype(str).to_numpy()
    to_train = np.where(train[None, :], sim, -np.inf)
    best = to_train.argmax(1)
    centred = e - e.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = centred @ vt[:2].T
    out = folds[["subject_id", "rig"]].assign(
        set=folds["set"].map(SPLIT),
        pc1=pcs[:, 0],
        pc2=pcs[:, 1],
        nearest=to_train[np.arange(len(e)), best],
        nearest_session=folds.index[best],
        nearest_animal=animal[best],
    )
    same = animal[:, None] == animal[None, :]
    for key, mask in (("nearest_same", same), ("nearest_other", ~same)):
        v = np.where(mask, to_train, -np.inf).max(1)
        out[key] = np.where(np.isfinite(v), v, np.nan)
    out["r"] = m.groupby("session")["r"].mean()
    return out.reset_index(names="session")


def similarity_figure(t: pd.DataFrame, below: float):
    """Whether the sessions the network does badly on look unlike its training sessions
    (*t* from :func:`nearest_train`): the thumbnails' map (first two principal
    components; ringed: test sessions under *below*), r against the similarity to the
    most alike training session, and that similarity by set and outcome."""
    from scipy.stats import spearmanr

    fig, ax = style.mosaic([["map", "r", "box"]], "double", 0.36)
    low = t["r"] < below
    for name, colour in SPLIT_COLOURS.items():
        d = t[t["set"] == name]
        ax["map"].scatter(d["pc1"], d["pc2"], s=3, color=colour, label=name, lw=0)
    ax["map"].scatter(
        t.loc[low, "pc1"],
        t.loc[low, "pc2"],
        s=14,
        facecolor="none",
        edgecolor="k",
        lw=0.5,
        label=f"r < {below:g}",
    )
    ax["map"].set(xlabel="PC 1", ylabel="PC 2", xticks=[], yticks=[])
    ax["map"].legend(fontsize=4.5, markerscale=1.5)
    test = t[t["r"].notna()]
    for name, colour in SPLIT_COLOURS.items():
        d = test[test["set"] == name]
        ax["r"].scatter(d["nearest"], d["r"], s=4, color=colour, lw=0)
    rho = spearmanr(test["nearest"], test["r"]).statistic
    ax["r"].axhline(below, color="0.6", lw=0.5, ls="--")
    ax["r"].set(
        xlabel="similarity to the most alike training session", ylabel="session r"
    )
    ax["r"].set_title(
        f"test sessions, Spearman rho {rho:+.2f}", fontweight="normal", fontsize=6.5
    )
    groups = {
        "train (to another)": t.loc[t["set"] == "train", "nearest"],
        "validation": t.loc[t["set"] == "validation", "nearest"],
        f"test, r >= {below:g}": test.loc[test["r"] >= below, "nearest"],
        f"test, r < {below:g}": test.loc[test["r"] < below, "nearest"],
    }
    for x, v in enumerate(groups.values()):
        ax["box"].plot(x + _jitter(len(v)), v, "o", ms=1.5, alpha=0.5, color="0.4")
        ax["box"].plot([x - 0.3, x + 0.3], [v.median()] * 2, color=style.POOLED, lw=1.2)
    ax["box"].set_xticks(
        range(len(groups)), list(groups), rotation=20, ha="right", fontsize=5.5
    )
    ax["box"].set_ylabel("similarity to the most alike training session")
    return fig


def _small(image, size: int = 96):
    """*image* no larger than *size* pixels a side, so a figure of many stays small."""
    image = image.copy()
    image.thumbnail((size, size))
    return image


def nearest_figure(
    root: Path, view: str, t: pd.DataFrame, low: pd.DataFrame, k: int = 4
):
    """Each session of *low* (:func:`low_sessions`), worst first: its thumbnail and the
    *k* most alike training thumbnails (*t* from :func:`nearest_train`) with their
    similarity, animal and rig (in red: the session's own animal)."""
    e = embeddings(root, view)
    train = t[t["set"] == "train"].set_index("session")
    vectors = e.loc[train.index].to_numpy()
    fig, axes = style.figure(
        "double", 0.155 * len(low), nrows=len(low), ncols=k + 1, squeeze=False
    )
    for row, s in zip(axes, low.itertuples(), strict=True):
        sim = vectors @ e.loc[s.session].to_numpy()
        order = np.argsort(sim)[::-1][:k]
        row[0].imshow(_small(thumbnail(root, s.session, view)), interpolation="nearest")
        row[0].set_title(
            f"test {s.set.split(':')[0]}: {s.subject_id} rig {s.rig}\nr {s.r:+.2f}",
            fontsize=5,
            fontweight="normal",
            color=SPLIT_COLOURS[s.set],
        )
        for ax, j in zip(row[1:], order, strict=True):
            n = train.index[j]
            ax.imshow(_small(thumbnail(root, n, view)), interpolation="nearest")
            same = str(train.loc[n, "subject_id"]) == str(s.subject_id)
            ax.set_title(
                f"{train.loc[n, 'subject_id']} rig {train.loc[n, 'rig']}\n"
                f"sim {sim[j]:.3f}",
                fontsize=5,
                fontweight="normal",
                color="#d55e00" if same else "k",
            )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axes[0, 0].set_ylabel("test session", fontsize=5.5)
    axes[0, 1].set_ylabel("most alike training", fontsize=5.5)
    return fig
