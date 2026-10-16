"""Networks scored on the split's test sets, clip by clip, with what is known of each clip.

Reads a local folder (``ROOT_ENV``) laid out as ``LAYOUT``: copies of the dataset's lists
and tables, the cache entries of the scored clips, the checkpoints, and one ``zephyr
evaluate`` result per network. Image statistics and prediction-to-truth lags are computed
here (inference only) and cached in ``notebooks/cache/``; nothing is trained or written
into the folder.
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
from .results import CACHE

ROOT_ENV = "ZEPHYR_TEST_ROOT"
"""The local folder; the notebook offers it (or ``C:/Data/zephyr-local``) as the default."""
DATASET_ROOT = "/data/zephyr-dataset"
"""Where the dataset the copy was taken from lives (the cache keys name it)."""
LAYOUT = {
    "lists": "dataset/full_test_{a,b,c}.toml",
    "tables": "dataset/{folds,qc}.parquet",
    "cache": "cache/features-360x270-box96x96-60hz/",
    "runs": "runs/<network>/dev/seed-17/best.pt",
    "scores": "evaluations/<network>.json",
}
GROUPS = {
    "full_test_a": "A: new animals",
    "full_test_b": "B: new sessions",
    "full_test_c": "C: 359 Hz",
}
METRICS = ["correlation", "inhale_f1", "exhale_f1", "kl_ibi", "head_inhale_f1"]
RECIPE = "features-360x270-box96x96-60hz"
MAX_LAG_S = 0.75


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
    """One row per network and clip: the clip's scores, group, session and stem."""
    found = networks(root)
    frames = []
    for name in names:
        rows = pd.DataFrame(json.loads(found[name].read_text())["clips"])
        frames.append(rows.assign(network=name))
    df = pd.concat(frames, ignore_index=True)
    df["stem"] = df["clip_id"].str.split("/").str[-1]
    df["session"] = df["recording"].str.split("#").str[-1]
    df["set"] = df["group"].map(GROUPS)
    return df


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


def best_lag(signal: NDArray, truth: NDArray, fs: float, max_lag_s: float = MAX_LAG_S):
    """``(lag_s, r, r_at_0)``: the lag maximising the correlation of *signal* shifted by
    it with *truth* (positive: the signal comes later than the truth), that correlation,
    and the correlation unshifted."""
    n = min(len(signal), len(truth))
    x = (signal[:n] - signal[:n].mean()) / (signal[:n].std() or 1.0)
    y = (truth[:n] - truth[:n].mean()) / (truth[:n].std() or 1.0)
    lags = np.arange(-round(max_lag_s * fs), round(max_lag_s * fs) + 1)
    r = np.array(
        [
            np.mean(x[max(0, k) :][: n - abs(k)] * y[max(0, -k) :][: n - abs(k)])
            for k in lags
        ]
    )
    best = int(np.argmax(r))
    return float(lags[best] / fs), float(r[best]), float(r[lags == 0][0])


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


def n_predicted(root: Path, name: str) -> tuple[int, int]:
    """How many of the test clips network *name* has predicted, of how many."""
    folder = CACHE / f"test_sets_predictions-{name}"
    stems = _clips(root)
    return sum((folder / f"{s}.npz").is_file() for s in stems), len(stems)


def predictions(root: Path, name: str, device: str = "cuda") -> Path:
    """Folder of network *name*'s prediction of every test clip (:func:`prediction`)."""
    stems = list(_clips(root))
    for i, stem in enumerate(stems, 1):
        prediction(root, name, stem, device)
        if i % 100 == 0 or i == len(stems):
            print(f"{name}: {i}/{len(stems)} clips predicted", flush=True)
    return CACHE / f"test_sets_predictions-{name}"


def lags(root: Path, name: str, device: str = "cuda") -> pd.DataFrame:
    """Every test clip's best prediction-to-truth lag for network *name*, from its
    cached predictions (:func:`predictions`)."""
    rows = []
    for path in sorted(predictions(root, name, device).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        p = np.load(path)
        fs = 1.0 / float(np.median(np.diff(p["times"])))
        lag, r, r0 = best_lag(p["signal"], p["truth"], fs)
        rows.append({"stem": path.stem, "lag_s": lag, "r_best": r, "r_at_0": r0})
    return pd.DataFrame(rows).assign(network=name)


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


FRAME_S = 1 / 60
"""One frame of the output grid: lags within one are no offset at all."""
CASE_COLOURS = {
    "truth": style.TRUTH,
    "prediction": style.color("Zephyr"),
    "shifted": "#d55e00",
}
CASES = [
    "fine (r >= 0.5)",
    "fails, no lag",
    "fails, lagged, rescued",
    "fails, lagged, not rescued",
]


def lag_table(
    lag: pd.DataFrame, info: pd.DataFrame, scores_df: pd.DataFrame
) -> pd.DataFrame:
    """*lag* (:func:`lags`) with each clip's animal, date, rig, set and failure case."""
    keep = scores_df[["stem", "network", "set"]]
    m = lag.merge(info, on="stem").merge(keep, on=["stem", "network"])
    m["lag_ms"] = m["lag_s"] * 1000
    m["date"] = pd.to_datetime(m["date"])
    lagged = m["lag_s"].abs() > 1.5 * FRAME_S
    fails = m["r_at_0"] < 0.5
    m["case"] = np.select([~fails, ~lagged, m["r_best"] >= 0.7], CASES[:3], CASES[3])
    return m


def lag_diagnosis_figure(m: pd.DataFrame):
    """Whether the prediction-to-truth lag explains the failures (one network, *m* from
    :func:`lag_table`): every clip's lag, what shifting recovers, the failures it
    explains, lag by animal, the lagged animals over time, and lag by rig."""
    layout = [["hist", "gain", "cases"], ["animal"] * 3, ["time", "time", "rig"]]
    fig, ax = style.mosaic(layout, "double", 1.05)
    band = {"color": "0.9", "zorder": 0}
    edge = 1.5 * FRAME_S * 1000
    bins = np.arange(-200, 201, FRAME_S * 1000) - FRAME_S * 500
    ax["hist"].hist(m["lag_ms"], bins=bins, color=style.color("Zephyr"))
    ax["hist"].axvspan(-edge, edge, **band)
    ax["hist"].set(
        xlabel="best lag (ms, - prediction earlier)", ylabel="clips", yscale="log"
    )
    dots = ax["gain"].scatter(
        m["r_at_0"],
        m["r_best"],
        c=m["lag_ms"].abs(),
        cmap="viridis",
        s=4,
        vmin=0,
        vmax=120,
    )
    ax["gain"].plot([-0.4, 1], [-0.4, 1], color="0.6", lw=0.5)
    fig.colorbar(dots, ax=ax["gain"], label="|lag| (ms)")
    ax["gain"].set(xlabel="r unshifted (as scored)", ylabel="r at best lag")
    counts = m["case"].value_counts().reindex(CASES).fillna(0)
    colours = ["0.75", "#d55e00", style.color("Zephyr"), "#e69f00"]
    ax["cases"].barh(counts.index, counts, color=colours)
    ax["cases"].invert_yaxis()
    ax["cases"].set(xlabel="clips")
    animals = m.groupby("subject_id")["lag_ms"].median().sort_values()
    sets = dict(
        zip(GROUPS.values(), ["#d55e00", style.color("Zephyr"), "#009e73"], strict=True)
    )
    for x, subject in enumerate(animals.index):
        d = m[m["subject_id"] == subject]
        colour = sets[d["set"].iloc[0]]
        ax["animal"].plot(
            x + _jitter(len(d), x), d["lag_ms"], "o", ms=1.8, alpha=0.6, color=colour
        )
    ax["animal"].axhspan(-edge, edge, **band)
    ax["animal"].set_xticks(
        range(len(animals)), list(animals.index), rotation=90, fontsize=4.5
    )
    ax["animal"].set(ylabel="best lag (ms)")
    for label, colour in sets.items():
        ax["animal"].plot([], [], "o", color=colour, label=label)
    ax["animal"].legend(loc="upper left", fontsize=5)
    off = animals[animals.abs() > edge].index
    animal_colours = dict(zip(off, style.series(len(off)), strict=True))
    for subject, colour in animal_colours.items():
        d = m[m["subject_id"] == subject].sort_values("date")
        ax["time"].plot(
            d["date"], d["lag_ms"], "o-", ms=2, lw=0.6, color=colour, label=subject
        )
    ax["time"].axhspan(-edge, edge, **band)
    ax["time"].set(ylabel="best lag (ms), lagged animals")
    ax["time"].legend(fontsize=4.5, ncol=4)
    _rig_axes(ax["rig"], m, edge, animal_colours)
    return fig


def _rig_axes(ax, m: pd.DataFrame, edge: float, animal_colours: dict) -> None:
    """Share of each rig's clips that are lagged, split by the animal of the lagged clips
    (lagged animals in their colours from the time panel, any other animal grey)."""
    lagged = m["lag_ms"].abs() > edge
    clips = m.groupby("rig").size()
    part = m[lagged].groupby(["rig", "subject_id"]).size().unstack(fill_value=0)
    part = part.reindex(clips.index, fill_value=0).div(clips, axis=0)
    rigs = part.sum(axis=1).sort_values().index
    order = [s for s in animal_colours if s in part] + [
        s for s in part if s not in animal_colours
    ]
    left = pd.Series(0.0, index=rigs)
    for subject in order:
        width = part.loc[rigs, subject]
        colour = animal_colours.get(subject, "0.75")
        ax.barh(rigs, width, left=left, color=colour, edgecolor="white", lw=0.3)
        left += width
    for y, rig in enumerate(rigs):
        ax.text(left[rig], y, f" {clips[rig]}", va="center", fontsize=4.5, color="0.4")
    ax.set(xlabel="share lagged, by animal")


def _label(row, what: str) -> str:
    return (
        f"{what}: {row.subject_id} {row.date:%Y-%m-%d}, lag {row.lag_ms:+.0f} ms, "
        f"r {row.r_at_0:+.2f} -> {row.r_best:+.2f}"
    )


def edge_cases(m: pd.DataFrame) -> dict[str, str]:
    """A few telling clips, chosen by rule from *m* (:func:`lag_table`): label -> stem."""
    m = m.assign(gain=m["r_best"] - m["r_at_0"])
    chosen = [
        ("aligned", m[(m["lag_s"] == 0) & (m["r_at_0"] > 0.9)].nlargest(1, "r_at_0")),
        ("lagged, rescued by the shift", m[m["case"] == CASES[2]].nlargest(1, "gain")),
    ]
    per_session = m.groupby(["subject_id", "date"])["lag_ms"].median().reset_index()
    counts = per_session.groupby("subject_id").size()
    per_session = per_session[per_session["subject_id"].isin(counts[counts >= 4].index)]
    if len(per_session):
        ends = per_session.sort_values("date").groupby("subject_id")["lag_ms"]
        drifting = (ends.last() - ends.first()).abs().idxmax()  # first vs last session
        d = m[m["subject_id"] == drifting].sort_values("date")
        chosen += [
            ("same animal, first session", d.iloc[:1]),
            ("same animal, last", d.iloc[-1:]),
        ]
    chosen += [
        ("lagged, not rescued", m[m["case"] == CASES[3]].nsmallest(1, "r_best")),
        ("fails without a lag", m[m["case"] == CASES[1]].nsmallest(1, "r_at_0")),
    ]
    return {_label(r, what): r.stem for what, rows in chosen for r in rows.itertuples()}


def clip_traces(root: Path, name: str, stem: str) -> dict:
    """Everything recorded for clip *stem* and network *name*, on the session clock: the
    raw thermistor, the target, the prediction and onset head, the truth events, the
    gray frames, and the best lag."""
    entry = _entry(root, stem)
    p = np.load(prediction(root, name, stem))
    clip, _ = _clips(root)[stem]
    raw = pd.read_parquet(clip.thermistor)
    events = np.load(entry.events)
    times = p["times"]
    fs = 1.0 / float(np.median(np.diff(times)))
    lag, r, r0 = best_lag(p["signal"], p["truth"], fs)
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
        "lag_s": lag,
        "r_best": r,
        "r_at_0": r0,
    }


def _window(t: NDArray, t0: float, t1: float) -> NDArray:
    return (t >= t0) & (t <= t1)


def _inhale_lines(ax, tr: dict, t0: float, t1: float) -> None:
    for t in tr["inhales"][_window(tr["inhales"], t0, t1)]:
        ax.axvline(t - tr["times"][0], color="0.75", lw=0.4, zorder=0)


def _traces_axes(ax, tr: dict, t0: float, t1: float) -> None:
    """Target, prediction and the prediction moved by its best lag, over [t0, t1]."""
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
    if tr["lag_s"] != 0:
        moved = f"prediction moved {-tr['lag_s'] * 1000:+.0f} ms"
        ax.plot(
            rel - tr["lag_s"],
            tr["signal"][w],
            color=CASE_COLOURS["shifted"],
            lw=0.8,
            ls="--",
            label=moved,
        )
    _inhale_lines(ax, tr, t0, t1)
    ax.set_xlim(t0 - tr["times"][0], t1 - tr["times"][0])


def edge_case_figure(
    root: Path, name: str, cases: dict[str, str], seconds: float = 4.0
):
    """One excerpt of each case: thermistor, prediction, and the prediction moved by its
    best lag (grey lines: the thermistor's inhales)."""
    fig, axes = style.figure(
        "double", 0.17 * len(cases), nrows=len(cases), squeeze=False
    )
    for ax, (what, stem) in zip(axes[:, 0], cases.items(), strict=True):
        tr = clip_traces(root, name, stem)
        mid = tr["times"][len(tr["times"]) // 2]
        _traces_axes(ax, tr, mid, mid + seconds)
        ax.set_title(what, fontweight="normal", loc="left", fontsize=6.5)
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=5, ncol=3)
    axes[-1, 0].set_xlabel("time in clip (s)")
    return fig


def animal_options(m: pd.DataFrame) -> dict[str, str]:
    """Each animal of *m* (:func:`lag_table`), most negative median lag first:
    label -> subject."""
    lag = m.groupby("subject_id")["lag_ms"].median().sort_values()
    rigs = m.groupby("subject_id")["rig"].nunique()
    return {f"{s} (median lag {lag[s]:+.0f} ms, rigs: {rigs[s]})": s for s in lag.index}


def animal_rig_figure(
    root: Path, name: str, m: pd.DataFrame, subject: str, seconds: float = 4.0
):
    """One excerpt per rig animal *subject* ran on (its clip there with the median lag):
    thermistor, prediction, and the prediction moved by its best lag."""
    d = m[m["subject_id"] == subject]
    rigs = sorted(d["rig"].unique())
    fig, axes = style.figure("double", 0.17 * len(rigs), nrows=len(rigs), squeeze=False)
    for ax, rig in zip(axes[:, 0], rigs, strict=True):
        on = d[d["rig"] == rig].sort_values("lag_ms")
        row = on.iloc[len(on) // 2]
        tr = clip_traces(root, name, row["stem"])
        mid = tr["times"][len(tr["times"]) // 2]
        _traces_axes(ax, tr, mid, mid + seconds)
        lags = ", ".join(f"{v:+.0f}" for v in on["lag_ms"])
        ax.set_title(
            f"rig {rig}: {row['date']:%Y-%m-%d}, r {row['r_at_0']:+.2f} -> "
            f"{row['r_best']:+.2f} (lags of its {len(on)} clips here: {lags} ms)",
            fontweight="normal",
            loc="left",
            fontsize=6.5,
        )
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=5, ncol=3)
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
    ax["trace"].legend(loc="upper right", fontsize=5, ncol=3)
    w = _window(tr["times"], t0, t1)
    rel = tr["times"][w] - tr["times"][0]
    ax["head"].plot(rel, tr["onset"][w], color=CASE_COLOURS["prediction"], lw=0.7)
    _inhale_lines(ax["head"], tr, t0, t1)
    head = tr["head"][_window(tr["head"], t0, t1)] - tr["times"][0]
    ax["head"].plot(
        head, np.full(len(head), 1.02), "v", ms=2.5, color=CASE_COLOURS["shifted"]
    )
    ax["head"].set(ylabel="onset head", xlabel="time in clip (s)")
    ax["head"].set(xlim=(start, start + length), ylim=(0, 1.1))
    fig.suptitle(
        f"{tr['stem']}: r {tr['r_at_0']:+.2f} as scored, {tr['r_best']:+.2f} at lag "
        f"{tr['lag_s'] * 1000:+.0f} ms (grey: thermistor inhales, triangles: head events)",
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
RHO = {
    "r as scored": ("r_at_0", "all"),
    "r at best lag": ("r_best", "all"),
    "r at best lag, no offset animals": ("r_best", "steady"),
    "|lag|": ("abs_lag", "all"),
}


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
    """*m* (:func:`lag_table`) with the :data:`QC` metrics, ``abs_lag`` and ``steady``
    (the clip's animal has no offset: median lag within 1.5 frames)."""
    qc = pd.read_parquet(root / "dataset/qc.parquet").drop_duplicates("session_id")
    qc = qc.set_index("session_id")
    m = m.drop(
        columns=["snr_db", "break_frac", "rate_cv", "good_frac"], errors="ignore"
    )
    m = m.merge(clip_qc(root), on="stem")
    for key in ("break_frac", "consistency_z", "good_frac"):
        m[f"session_{key}"] = m["session"].map(qc[key])
    m["abs_lag"] = m["lag_ms"].abs()
    offset = m.groupby("subject_id")["lag_ms"].transform("median").abs()
    m["steady"] = offset <= 1.5 * FRAME_S * 1000
    return m


def qc_figure(q: pd.DataFrame):
    """Whether thermistor quality goes with the network's r (*q* from :func:`qc_table`):
    Spearman rho of each QC metric with r, r at the best lag (with and without the
    animals with an offset) and |lag|; each animal's median amplitude against its median
    lag (offset animals named); r against amplitude, clip by clip."""
    fig, ax = style.mosaic([["rho", "rho", "amp", "scatter"]], "double", 0.42)
    rho = pd.DataFrame(
        {
            name: [
                q[q["steady"] | (which == "all")][[key, y]].corr("spearman").iloc[0, 1]
                for key in QC
            ]
            for name, (y, which) in RHO.items()
        },
        index=list(QC.values()),
    )
    ax["rho"].imshow(rho, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for (i, j), v in np.ndenumerate(rho.to_numpy()):
        ax["rho"].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5)
    ax["rho"].set_xticks(
        range(len(RHO)), list(RHO), rotation=20, ha="right", fontsize=5.5
    )
    ax["rho"].set_yticks(range(len(QC)), list(QC.values()), fontsize=5.5)
    ax["rho"].set_title("Spearman rho over clips", fontweight="normal", fontsize=6.5)
    groups = {"no offset": q["steady"], "offset animals": ~q["steady"]}
    colours = {"no offset": style.color("Zephyr"), "offset animals": "#d55e00"}
    animals = q.groupby("subject_id").agg(
        amplitude=("amplitude", "median"),
        lag_ms=("lag_ms", "median"),
        steady=("steady", "first"),
    )
    ax["amp"].axvspan(
        -1.5 * FRAME_S * 1000, 1.5 * FRAME_S * 1000, color="0.9", zorder=0
    )
    for subject, a in animals.iterrows():
        colour = colours["no offset" if a["steady"] else "offset animals"]
        ax["amp"].plot(a["lag_ms"], a["amplitude"], "o", ms=2.5, color=colour)
        if not a["steady"]:
            ax["amp"].annotate(
                subject,
                (a["lag_ms"], a["amplitude"]),
                fontsize=4.5,
                xytext=(2, 1),
                textcoords="offset points",
            )
    ax["amp"].set(
        yscale="log",
        xlabel="median lag (ms)",
        ylabel="median thermistor amplitude (ADC)",
    )
    ax["amp"].set_title("one dot per animal", fontweight="normal", fontsize=6.5)
    for k, g in groups.items():
        ax["scatter"].scatter(
            q.loc[g, "amplitude"],
            q.loc[g, "r_at_0"],
            s=3,
            alpha=0.6,
            color=colours[k],
            label=k,
        )
    ax["scatter"].set(
        xscale="log", xlabel="thermistor amplitude (ADC)", ylabel="r as scored"
    )
    ax["scatter"].legend(fontsize=5)
    return fig
