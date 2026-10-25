import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    import tomllib
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from utils import architecture, corpus_flow, plots, quality, runs, style, values
    from zephyr.benchmarks import analysis
    from zephyr.corpus.config import DatasetSpec

    style.use_style()

    FIGURES = Path(
        os.environ.get(
            "ZEPHYR_MANUSCRIPT_FIGURES", runs.REPO / "manuscript" / "figures"
        )
    )
    LAYOUT = tomllib.loads((FIGURES / "layout.toml").read_text(encoding="utf8"))
    # Quality-control reports of the reference checkpoint (Fig. 6).
    QC = Path(
        os.environ.get(
            "ZEPHYR_MANUSCRIPT_QC",
            runs.REPO / "notebooks/cache/quality-control/full-gray-sched-v3-seed-17",
        )
    )


@app.cell
def intro():
    mo.md(f"""
    # Manuscript figure panels

    Draws each data panel of the manuscript's figures at its printed size, as laid out
    in `{FIGURES / "layout.toml"}`, and writes it as `<figure>/<panel>.svg` beside the
    figure's `.typ` file, which embeds the panels as vectors and adds their letters.
    Runs come from the results manifest and clips from the local dataset copy, as for
    `walkthrough.py`; nothing is trained. Set `ZEPHYR_MANUSCRIPT_FIGURES` to write
    elsewhere.
    """)
    return


@app.function
def size(figure: str, name: str) -> tuple[float, float]:
    """A panel's printed width and height (mm), from the layout."""
    spec = LAYOUT[figure]["panels"][name]
    return spec["width_mm"], spec["height_mm"]


@app.cell
def example_clip():
    # A typical new-animal clip of the single network (runs.SINGLE): among its test-A
    # clips with frames in the dataset copy, the one whose trace correlation is nearest
    # their median.
    _c = runs.scores([runs.SINGLE]).query("set == @runs.SETS['test_a']")
    _c = _c[_c["stem"].map(runs.is_local)].set_index("stem")["correlation"]
    stem = (_c - _c.median()).abs().idxmin()
    truth = runs.truth(stem)
    zephyr = runs.prediction(runs.SINGLE, stem)
    origin = float(zephyr["times"][0])
    # The excerpt of it shown in Fig. 1e: start (s from the clip's start) and length.
    excerpt = (269.0, 8.0)
    return excerpt, origin, stem, truth, zephyr


@app.cell
def fig1bc_md():
    mo.md("""
    ## Fig. 1b, c: sniffing in the task

    One session of a new animal. b: the thermistor's breathing rate over the whole
    session (its events found as the corpus finds them) and speed around odor onset,
    apart for sites where the animal stopped and was rewarded, stopped and was not
    (unrewarded), or ran past (skipped: the task saw no stop), and breathing rate
    against speed; c: a minute of one clip, predicted by the fold that holds the animal
    out, with licks, rewards and odor sites. The behaviour comes from the session's processed
    asset (the same clock as the thermistor).
    """)
    return


@app.function
def task_window(
    sites: pd.DataFrame, t0: float, t1: float, window: float
) -> tuple[tuple[int, int], float]:
    """``(score, start)`` of the *window* in ``[t0, t1]`` holding the most kinds of
    whole odor sites, then the most sites (the earliest such, of equals)."""

    def score(start: float) -> tuple[int, int]:
        whole = sites[(sites["odor"] >= start) & (sites["leave"] <= start + window)]
        return whole["kind"].nunique(), len(whole)

    start = max(np.arange(t0, t1 - window, 0.5), key=score)
    return score(start), float(start)


@app.cell
def fig1bc_data(dataset, spec):
    from zephyr.config import ClipList as _ClipList
    from zephyr.config import load as _load
    from zephyr.events import detect as _detect

    # Its correlation (0.855) and head inhale F1 (0.981) are nearest test A's medians
    # (0.850, 0.977) among sessions of dataset version 1.0 or later (cv folds, seed 17,
    # 2026-10-23); animal 863680 is new in cv_2.
    _asset = "863680_2026-09-08_19-35-38_processed_2026-09-23_03-47-07"
    _session = _asset.split("_processed")[0]
    _name = runs.cv_name(runs.REFERENCE, "cv_2")

    def _table(name: str) -> pd.DataFrame:
        return pd.read_parquet(
            runs.dataset_file(f"{_asset}/{name}.parquet", "s3://aind-open-data")
        )

    _speed = _table("position_velocity")["velocity"]
    _licks = _table("licks")
    lick_times = _licks.index[_licks["is_lick_onset"]].to_numpy()
    _all = _table("sites")
    _all = _all[(_all["site_label"] == "RewardSite") & _all["odor_onset_time"].notna()]
    odor_sites = pd.DataFrame(
        {
            "odor": _all["odor_onset_time"],
            "leave": _all["stop_time"],
            "kind": [
                plots.site_kind(bool(c), bool(r))
                for c, r in zip(_all["has_choice"], _all["has_reward"], strict=True)
            ],
            "reward": _all["reward_onset_time"],
        }
    )
    task_clips = []
    for _span in (1, 2):
        _stem = f"{_session}_span{_span}_FaceCamera"
        _p, _t = runs.prediction(_name, _stem), runs.truth(_stem)
        _g = _p["times"]
        task_clips.append(
            pd.DataFrame(
                {
                    "time": _g,
                    "speed": np.interp(_g, _speed.index.to_numpy(), _speed.to_numpy()),
                    "video": analysis.breathing_rate(_p["head"], _g),
                    "thermistor": analysis.breathing_rate(_t["inhale"], _g),
                }
            )
        )
    # b: the thermistor over the whole session, its events found as the corpus finds
    # them (the clip list's detector), on a 60-Hz grid
    _raw = pd.read_parquet(
        runs.dataset_file(f"raw/thermistor/{_session}.parquet", spec.root)
    )
    _events = _load(_ClipList, dataset / "clips.toml", check_files=False).events
    _inhale = _detect(_events.method, _raw, params=_events.params).inhale
    _g = np.arange(_raw["Time"].iloc[0], _raw["Time"].iloc[-1], 1 / 60)
    task_session = pd.DataFrame(
        {
            "time": _g,
            "speed": np.interp(_g, _speed.index.to_numpy(), _speed.to_numpy()),
            "rate": analysis.breathing_rate(_inhale, _g),
        }
    )
    task_thermistor_hz = 1 / float(np.median(np.diff(_raw["Time"])))
    _lags = np.arange(-2.0, 5.0 + 1e-9, 1 / 60)
    _inside = odor_sites[
        odor_sites["odor"].between(_g[0] - _lags[0], _g[-1] - _lags[-1])
    ]
    _parts = []
    for (_site, _signal), _group in (
        ((k, s), g) for k, g in _inside.groupby("kind") for s in ("rate", "speed")
    ):
        _x = analysis.event_aligned(
            _g, task_session[_signal].to_numpy(), _group["odor"].to_numpy(), _lags
        )
        _parts.append(
            pd.DataFrame(
                {
                    "site": _site,
                    "signal": _signal,
                    "lag": _lags,
                    "mean": np.nanmean(_x, 0),
                    "sem": np.nanstd(_x, 0) / np.sqrt(len(_x)),
                    "n": len(_x),
                }
            )
        )
    task_aligned = pd.concat(_parts, ignore_index=True)
    return (
        lick_times,
        odor_sites,
        task_aligned,
        task_clips,
        task_session,
        task_thermistor_hz,
    )


@app.cell
def fig1b(task_aligned, task_session, task_thermistor_hz):
    fig1b = plots.task_summary_panel(
        *size("fig1", "b"), task_aligned, task_session, task_thermistor_hz
    )
    plots.save_panel(fig1b, FIGURES / "fig1" / "b.svg")
    _counts = task_aligned.groupby(["site", "signal"])["n"].first()
    v_fig1 = {
        f"sites_{s.lower()}": values.count(_counts[(s, "rate")]) for s in style.SITES
    }
    mo.vstack([fig1b, mo.ui.table(_counts.reset_index(), selection=None)])
    return (v_fig1,)


@app.cell
def fig1c(lick_times, odor_sites, task_clips):
    _window = 60.0
    # the clip whose best window holds the most kinds of site, then the most sites
    _found = []
    for _c in task_clips:
        _t = _c["time"]
        _found.append((*task_window(odor_sites, _t.iloc[0], _t.iloc[-1], _window), _c))
    _, _start, _clip = max(_found, key=lambda found: found[0])
    _ex = _clip[(_clip["time"] >= _start) & (_clip["time"] <= _start + _window)]
    _sites = odor_sites[
        (odor_sites["leave"] > _start) & (odor_sites["odor"] < _start + _window)
    ]
    fig1c = plots.task_stretch_panel(
        *size("fig1", "c"),
        _ex.assign(time=_ex["time"] - _start),
        _sites.assign(odor=_sites["odor"] - _start, leave=_sites["leave"] - _start),
        lick_times - _start,
        odor_sites["reward"].dropna().to_numpy() - _start,
    )
    plots.save_panel(fig1c, FIGURES / "fig1" / "c.svg")
    fig1c
    return


@app.cell
def fig1d_md():
    mo.md("""
    ## Fig. 1d: the network

    Read off the loaded checkpoint, so the layer sizes are the model's own; the input
    is three consecutive crops of the same clip.
    """)
    return


@app.cell
def fig1d(stem, truth):
    _frame_t, _frames = runs.gray(stem)
    _i = int(np.searchsorted(_frame_t, truth["inhale"][len(truth["inhale"]) // 2]))
    _net = runs.network(runs.SINGLE)[0]
    _hz = runs.clip(stem)[1].output_fs
    _svg = architecture.network_svg(
        _net,
        [np.asarray(_frames[j, 0]) for j in (_i - 1, _i, _i + 1)],
        *size("fig1", "d"),
        _hz,
    )
    (FIGURES / "fig1" / "d.svg").write_text(_svg, encoding="utf8")
    mo.Html(_svg)
    return


@app.cell
def fig1e_md():
    mo.md("""
    ## Fig. 1e: an example prediction

    The clip's face frame with the crop the network sees, then eight seconds of the
    same test-A clip: the network's trace
    over the thermistor and its onset probability, with the thermistor's inhales
    (black) and exhales (grey) and the head's inhales (blue).
    """)
    return


@app.cell
def fig1e(excerpt, origin, stem, truth, zephyr):
    _t0, _window = excerpt
    fig1e = plots.prediction_panel(
        *size("fig1", "e"),
        truth,
        zephyr,
        origin,
        _t0,
        _window,
        runs.thumbnail(stem),
        runs.box(stem),
    )
    plots.save_panel(fig1e, FIGURES / "fig1" / "e.svg")
    fig1e
    return


@app.cell
def fig2_md():
    mo.md("""
    ## Fig. 2: the corpus

    From the local dataset copy (`ZEPHYR_TEST_ROOT`/dataset) and the corpus spec:
    the selection flow, the breathing-rate mix and the split.
    """)
    return


@app.cell
def fig2_data():
    dataset = runs.root() / "dataset"
    spec = DatasetSpec.model_validate(
        tomllib.loads(
            (
                runs.REPO / "packages/zephyr-dataset/configs/vr-foraging-sniff.toml"
            ).read_text(encoding="utf8")
        )
    )
    folds = pd.read_parquet(dataset / "raw" / "folds.parquet")
    return dataset, folds, spec


@app.cell
def fig2a(dataset, spec):
    # From the dataset version the corpus starts at, the hand-picked ephys sessions
    # joining it there.
    _steps = corpus_flow.shorten(corpus_flow.flow(dataset, spec)[1:], sets=False)
    _steps[0]["name"] = "recorded"
    fig2a = corpus_flow.flow_panel(*size("fig2", "a"), _steps)
    plots.save_panel(fig2a, FIGURES / "fig2" / "a.svg")
    # recorded, passed the thermistor rules, passed the clip and camera rules, corpus
    v_selection = {
        k: values.count(s["sessions"])
        for k, s in zip(
            ("recorded", "thermistor", "clips", "corpus"), _steps, strict=True
        )
    }
    fig2a
    return (v_selection,)


@app.cell
def optional_cameras(dataset, folds, spec):
    # Videos of the optional cameras missing from corpus sessions, which keep their
    # required cameras: never recorded in the session, or left out on review (the
    # camera not on the animal).
    _corpus = set(folds.loc[folds["set"].notna(), "session_id"])
    _cameras = pd.read_parquet(dataset / "raw" / "cameras.parquet")
    _clips = pd.read_parquet(dataset / "raw" / "clips.parquet")
    _dropped = spec.dropped_cameras()
    v_cameras_missing = {}
    for _camera in spec.source.optional_cameras:
        _recorded = set(_cameras.loc[_cameras["camera"] == _camera, "session_id"])
        _cut = set(_clips.loc[_clips["camera"] == _camera, "session_id"])
        _reviewed = {s for s in _corpus & _recorded if (s, _camera) in _dropped}
        assert (_corpus & _recorded) - _cut == _reviewed, _camera
        _key = _camera.removesuffix("Camera").lower()
        v_cameras_missing[f"{_key}_not_recorded"] = values.count(
            len(_corpus - _recorded)
        )
        v_cameras_missing[f"{_key}_reviewed"] = values.count(len(_reviewed))
    v_cameras_missing
    return (v_cameras_missing,)


@app.cell
def fig2b(folds):
    # Every thermistor breath's rate over the corpus, and per animal.
    _rates = runs.animal_breath_rates()
    fig2b = plots.corpus_rate_panel(*size("fig2", "b"), _rates, step=0.5)
    plots.save_panel(fig2b, FIGURES / "fig2" / "b.svg")
    _r = np.concatenate([np.asarray(r) for r in _rates.values()])
    _r = _r[(_r >= 2) & (_r <= 15)]
    v_rates = {
        "breaths": values.count(len(_r)),
        "animals": values.count(len(_rates)),
        "slow": values.pct((_r < 4).mean()),
        "medium": values.pct(((_r >= 4) & (_r < 8)).mean()),
        "fast": values.pct((_r >= 8).mean()),
    }
    fig2b
    return (v_rates,)


@app.cell
def fig2c(folds):
    _used = folds[folds["set"].notna()]
    fig2c = plots.corpus_animals_panel(*size("fig2", "c"), _used)
    plots.save_panel(fig2c, FIGURES / "fig2" / "c.svg")
    _dates = pd.to_datetime(_used["date"])
    _per = (
        _used.assign(date=_dates)
        .groupby("subject_id")
        .agg(
            sessions=("session_id", "nunique"),
            days=("date", lambda d: (d.max() - d.min()).days),
            rigs=("rig", "nunique"),
        )
    )
    _clip_s = 600  # two 300-s clips per session
    v_corpus = {
        "sessions": values.count(_used["session_id"].nunique()),
        "animals": values.count(_used["subject_id"].nunique()),
        "rigs": values.count(_used["rig"].nunique()),
        "months": values.count(_used["month"].nunique()),
        "hours": values.count(_used["session_id"].nunique() * _clip_s / 3600),
        "median_sessions": f"{_per['sessions'].median():g}",
        "median_days": f"{_per['days'].median():g}",
        "multi_rig": values.count((_per["rigs"] > 1).sum()),
    }
    fig2c
    return (v_corpus,)


@app.cell
def fig3_md():
    mo.md("""
    ## Fig. 3: the benchmark

    Every method on test set A (new animals), as each run scored itself, and the
    walkthrough's clip-by-clip tables (onset lags, rate windows) for the networks.
    """)
    return


@app.cell
def fig3_methods():
    order = [
        "Pixel (motion)",
        "Pixel (PCA)",
        "Pixel (SNR)",
        "Facemap",
        "TS-CAN",
        "RhythmFormer",
        "PhysFormer",
        "PhysNet",
        "Zephyr",
    ]
    methods = runs.available(order)
    # The unit of comparison is an animal on one rig (Methods, Statistics): every
    # score is averaged within a unit, and units are what is counted and resampled.
    _folds = pd.read_parquet(runs.root() / "dataset" / "raw" / "folds.parquet")
    _rig = dict(zip(_folds["session_id"], _folds["rig"], strict=True))

    def with_unit(table: pd.DataFrame) -> pd.DataFrame:
        """*table* (a ``session`` and ``subject`` per row) with its ``rig`` and
        ``unit``."""
        rig = table["session"].map(_rig)
        return table.assign(rig=rig, unit=table["subject"].astype(str) + "@" + rig)

    recordings = with_unit(runs.scores(methods, "per_recording"))
    test_a = runs.SETS["test_a"]
    return methods, recordings, test_a, with_unit


@app.cell
def fig3_folds(methods, recordings, with_unit):
    # Every summary is over folds (Methods, Statistics): the five cross-validation folds
    # for the methods retrained on them, the published split ("published", one split,
    # no interval) for the others.
    cv_folds = runs.RESULTS.cross_validation.folds
    _cv = with_unit(runs.cv_scores("per_recording"))
    cv_methods = [m for m in methods if m in set(_cv["method"])]
    by_fold = pd.concat(
        [
            _cv[_cv["method"].isin(cv_methods)],
            recordings[~recordings["method"].isin(cv_methods)].assign(fold="published"),
        ],
        ignore_index=True,
    )

    def fold_tables(names, kinds=("breaths", "false_positives", "lags", "windows")):
        """The analysis tables of each method on each of its folds, labelled by the
        method and a ``fold`` column."""
        out = {k: [] for k in kinds}
        for n in names:
            for f in cv_folds if n in cv_methods else [None]:
                t = runs.tables(runs.cv_name(n, f) if f else n)
                for k in kinds:
                    out[k].append(t[k].assign(method=n, fold=f or "published"))
        return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}

    return by_fold, cv_methods, fold_tables


@app.cell
def fig3a(cv_methods, methods):
    # One clip of an animal held out by fold cv_0 and by the published split, so the
    # methods without cross-validation runs (published split, for now) can be shown:
    # the clip whose correlation, averaged over the methods, is nearest the median of
    # such clips; 6 s of it where the thermistor's breathing rate changes most.
    _fold = runs.RUNS[runs.SINGLE].fold
    _names = {
        n: runs.cv_name(n, _fold) if n in cv_methods else n
        for n in methods
        if n in cv_methods or n in ("PhysFormer", "RhythmFormer")
    }
    _all = runs.scores(list(_names.values())).query("set == @runs.SETS['test_a']")
    _all = _all[_all["stem"].map(runs.is_local)]
    _held = _all.groupby("stem")["method"].nunique() == len(_names)
    _mean = _all[_all["stem"].isin(_held[_held].index)].groupby("stem")["correlation"]
    _mean = _mean.mean()
    _stem = (_mean - _mean.median()).abs().idxmin()
    _truth = runs.truth(_stem)
    _preds = {n: runs.prediction(c, _stem) for n, c in _names.items()}
    _origin = float(_preds["Zephyr"]["times"][0])
    _inhales = _truth["inhale"] - _origin

    def _change(t0: float) -> int:
        first = np.count_nonzero((_inhales >= t0) & (_inhales < t0 + 3.0))
        second = np.count_nonzero((_inhales >= t0 + 3.0) & (_inhales < t0 + 6.0))
        return abs(second - first)

    _t0 = float(max(np.arange(0.0, _inhales[-1] - 6.0, 0.5), key=_change))
    fig3a = plots.trace_stack_panel(
        *size("fig3", "a"), _preds, _truth, _origin, _t0, 6.0
    )
    plots.save_panel(fig3a, FIGURES / "fig3" / "a.svg")
    mo.vstack([mo.md(f"{_stem}, from {_t0:.1f} s"), fig3a])
    return


@app.cell
def fig3b(by_fold, methods, test_a):
    # Inference time per clip (walkthrough harness): networks on this machine's GPU,
    # the pixel methods and Facemap in numpy on its CPU. Inhale F1 on new animals per
    # fold (dots) and its mean over folds with the 95% interval.
    timing = runs.wall_time(methods).set_index("method")
    on_cpu = [m for m in methods if not timing.loc[m, "device"].startswith("NVIDIA")]
    _a = by_fold[by_fold["set"] == test_a]
    fig3b = plots.accuracy_time_panel(
        *size("fig3", "b"),
        analysis.per_fold(_a, "inhale", by=["method"]),
        analysis.fold_scores(_a, "inhale", by=["method"]),
        timing["s per clip (median)"],
        methods,
        on_cpu=on_cpu,
        label_offsets={
            "Zephyr": (-4, 6),
            "TS-CAN": (3, -7),
            "PhysNet": (4, 3),
            "Pixel (SNR)": (3, 6),
        },
    )
    plots.save_panel(fig3b, FIGURES / "fig3" / "b.svg")
    fig3b
    return on_cpu, timing


@app.cell
def fig3cde(fold_tables, test_a):
    # The networks' clip-by-clip tables on new animals, per fold. c: F1 by rate, mean
    # over folds with its interval; d, e: every fold's new-animal breaths and windows
    # pooled (each animal is new in one fold only).
    nets = ["TS-CAN", "RhythmFormer", "PhysFormer", "PhysNet", "Zephyr"]
    whole = {k: v.query("set == @test_a") for k, v in fold_tables(nets).items()}
    _by_rate = analysis.f1_by_rate(
        whole["breaths"], whole["false_positives"], analysis.RATE_BINS_HZ[1:]
    )
    _rate = analysis.across_folds(_by_rate, "f1", by=["method", "bin"]).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    fig3c = plots.rate_band_panel(
        *size("fig3", "c"),
        _rate,
        nets[::-1],
        legend=False,  # colours as in b, where every method is named
    )
    plots.save_panel(fig3c, FIGURES / "fig3" / "c.svg")
    _cv = {k: v[v["fold"] != "published"] for k, v in whole.items()}
    _zephyr = _cv["lags"][_cv["lags"]["method"] == "Zephyr"]
    fig3d = plots.lag_rate_panel(*size("fig3", "d"), _zephyr)
    plots.save_panel(fig3d, FIGURES / "fig3" / "d.svg")
    fig3e = plots.bland_altman_panel(*size("fig3", "e"), _cv["windows"], "Zephyr")
    plots.save_panel(fig3e, FIGURES / "fig3" / "e.svg")

    _f1 = _rate.set_index(["method", "bin"])["mean"]
    _breaths = _cv["breaths"][_cv["breaths"]["method"] == "Zephyr"]
    _in_band = _breaths["rate_hz"].between(2, 15, inclusive="left")
    _pub = whole["breaths"][
        (whole["breaths"]["fold"] == "published")
        & (whole["breaths"]["method"] == "PhysFormer")
    ]
    _ms = _zephyr.assign(ms=1000 * _zephyr["lag_s"], method="Zephyr")

    def _lag_ci(lags):
        """The mean lag of each fold, then their mean and 95% t-interval over folds."""
        per_fold = lags.groupby(["method", "fold"], as_index=False)["ms"].mean()
        return analysis.across_folds(per_fold, "ms").iloc[0]

    _lag = _lag_ci(_ms)

    def _lag_at(low: float, high: float) -> str:
        return values.signed(
            _lag_ci(_ms[_ms["rate_hz"].between(low, high, "left")])["mean"]
        )

    _w = _cv["windows"][_cv["windows"]["method"] == "Zephyr"]
    _d = 60 * (_w["pred_hz"] - _w["true_hz"])  # breaths per minute
    _clips = runs.cv_scores("clips").query("method == 'Zephyr' and set == @test_a")
    _middle = _f1.loc["Zephyr"].loc[["6-8", "8-10", "10-12"]]
    v_rate = {
        "f1_slow": values.num(_f1.loc[("Zephyr", "2-3")]),
        "f1_middle": values.span(_middle.min(), _middle.max()),
        "f1_fast": values.num(_f1.loc[("Zephyr", "12-15")]),
        "physnet_f1_slow": values.num(_f1.loc[("PhysNet", "2-3")]),
        "physnet_f1_fast": values.num(_f1.loc[("PhysNet", "12-15")]),
        "physformer_f1_slow": values.num(_f1.loc[("PhysFormer", "2-3")]),
        "breaths": values.count(_in_band.sum()),
        "breaths_published": values.count(
            _pub["rate_hz"].between(2, 15, inclusive="left").sum()
        ),
        "lags": values.count(len(_zephyr)),
        "lag": values.num(_lag["mean"], 1),
        "lag_ci": values.span(_lag["low"], _lag["high"], 1),
        "lag_slow": _lag_at(2, 3),
        "lag_fast": _lag_at(11, 12),
        "windows": values.count(len(_w)),
        "mae": values.num(_d.abs().mean(), 1),
        "bias": values.signed(_d.mean()),
        "limits_low": values.signed(_d.mean() - 1.96 * _d.std()),
        "limits_high": values.signed(_d.mean() + 1.96 * _d.std()),
        # the error relative to the thermistor's rate, and the windows within 10% of it
        "mape": values.num((_d.abs() / (60 * _w["true_hz"])).mean() * 100, 1),
        "within_10": values.pct((_d.abs() / (60 * _w["true_hz"]) <= 0.10).mean()),
        "clips": values.count(len(_clips)),
        "sessions": values.count(_clips["session"].nunique()),
        "animals": values.count(_clips["subject"].nunique()),
    }
    mo.vstack([fig3c, fig3d, fig3e, mo.ui.table(_rate.round(3), selection=None)])
    return nets, v_rate, whole


@app.cell
def figs5(fold_tables, test_a):
    # Supplementary Fig. 5: why slow breaths score lower. Zephyr's breaths on new
    # animals per rate bin and fold: found within 25 ms, 50 ms and 10% of the breath's
    # cycle; the spread of its lags in ms and as a share of the cycle; breaths found
    # late (beyond 25 ms) against not found. Mean over folds with its interval.
    _t = fold_tables(["Zephyr"], ("breaths", "lags"))
    _t = {k: v.query("set == @test_a and fold != 'published'") for k, v in _t.items()}
    _timing = analysis.timing_by_rate(
        _t["lags"], _t["breaths"], analysis.RATE_BINS_HZ[1:]
    )
    _measures = [
        "within 25 ms",
        "within 50 ms",
        "within cycle",
        "iqr_ms",
        "iqr_cycle",
        "near",
        "absent",
    ]
    slow_timing = pd.concat(
        analysis.across_folds(
            _timing.rename(columns={m: "value"}), "value", by=["method", "bin"]
        ).assign(measure=m)
        for m in _measures
    )
    _panels = {
        "a": (
            {
                "within 25 ms": "Within 25 ms",
                "within 50 ms": "Within 50 ms",
                "within cycle": "Within 10% of cycle",
            },
            "Breaths found (%)",
            100.0,
        ),
        "b": ({"iqr_ms": "Timing spread"}, "Timing spread (ms)", 1.0),
        "c": ({"iqr_cycle": "Timing spread"}, "Timing spread (% of cycle)", 1.0),
        "d": (
            {"near": "Found 25–100 ms off", "absent": "Not found"},
            "Breaths (%)",
            100.0,
        ),
    }
    _figs = []
    for _panel, (_m, _label, _scale) in _panels.items():
        _fig = plots.rate_lines_panel(
            *size("figs5", _panel),
            slow_timing[slow_timing["measure"].isin(list(_m))],
            _m,
            _label,
            scale=_scale,
        )
        plots.save_panel(_fig, FIGURES / "figs5" / f"{_panel}.svg")
        _figs.append(_fig)
    _at = slow_timing.set_index(["measure", "bin"])["mean"]
    v_slow = {
        "within_25_slow": values.pct(_at[("within 25 ms", "2-3")]),
        "within_50_slow": values.pct(_at[("within 50 ms", "2-3")]),
        "within_cycle_slow": values.pct(_at[("within cycle", "2-3")]),
        "within_cycle_fast": values.pct(_at[("within cycle", "10-12")]),
        "iqr_ms_slow": values.num(_at[("iqr_ms", "2-3")], 0),
        "iqr_ms_mid": values.num(_at[("iqr_ms", "6-8")], 0),
        "iqr_cycle_slow": values.num(_at[("iqr_cycle", "2-3")], 0),
        "iqr_cycle_mid": values.num(_at[("iqr_cycle", "6-8")], 0),
        "near_slow": values.pct(_at[("near", "2-3")]),
        "absent_slow": values.pct(_at[("absent", "2-3")], 1),
    }
    mo.vstack([mo.hstack(_figs), mo.md(f"{v_slow}")])
    return (v_slow,)


@app.cell
def table1(
    by_fold, cv_methods, fold_tables, methods, on_cpu, recordings, test_a, timing
):
    # Table 1 (benchmark, new animals) and Supplementary Table (new sessions of known
    # animals, test set B): per method, each score averaged within each unit, then over
    # the units of a fold, then over folds (mean and 95% interval; methods without
    # cross-validation runs, the published split alone). Rate metrics from 10-s
    # windows. Written as CSV for main.typ.
    _windows_all = fold_tables(methods, ("windows",))["windows"]

    def _cell(row, digits):
        mean = 0.0 if abs(row["mean"]) < 0.5 * 10**-digits else row["mean"]
        if pd.isna(row["low"]):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ({row['low']:.{digits}f}–{row['high']:.{digits}f})"

    def _benchmark(set_name: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
        """The table's rows, each column's summary over folds, and the recordings."""
        recs = by_fold[by_fold["set"] == set_name]
        windows = _windows_all[_windows_all["set"] == set_name].copy()
        unit = recs.drop_duplicates(["fold", "session"]).set_index(["fold", "session"])[
            "unit"
        ]
        windows["unit"] = [
            unit.get((f, st.rsplit("_span", 1)[0]))
            for f, st in zip(windows["fold"], windows["stem"], strict=True)
        ]
        rates = analysis.rate_agreement(
            windows.dropna(subset=["unit"]), by=["method", "fold", "unit"]
        ).reset_index()
        rates[["MAE (Hz)", "RMSE (Hz)"]] *= 60  # breaths per minute
        columns = {
            "MAE (Hz)": ("MAE", 1, rates),
            "RMSE (Hz)": ("RMSE", 1, rates),
            "MAPE (%)": ("MAPE", 1, rates),
            "r": ("rate_r", 2, rates),
            "correlation": ("trace_r", 2, recs),
            "inhale": ("inhale_f1", 2, recs),
            "exhale_f1": ("exhale_f1", 2, recs),
        }
        summary = {
            c: analysis.fold_scores(t, c, by=["method"]).set_index("method")
            for c, (_, _, t) in columns.items()
        }
        rows = []
        for m in methods:
            row = {
                "method": plots.DISPLAY.get(m, m),
                "splits": str(int(summary["inhale"].loc[m, "folds"])),
                "seconds": f"{timing.loc[m, 's per clip (median)']:.2f}"
                + ("*" if m in on_cpu else ""),
            }
            for c, (name, digits, _) in columns.items():
                row[name] = _cell(summary[c].loc[m], digits).replace("-0.", "−0.")
            rows.append(row)
        return pd.DataFrame(rows), summary, recs

    benchmark, _summary, _a = _benchmark(test_a)
    benchmark_b, _summary_b, _b = _benchmark(runs.SETS["test_b"])
    (FIGURES.parent / "tables").mkdir(exist_ok=True)
    benchmark.to_csv(FIGURES.parent / "tables" / "benchmark.csv", index=False)
    benchmark_b.to_csv(FIGURES.parent / "tables" / "benchmark_b.csv", index=False)
    benchmark_diff = {
        c: _summary[c][["mean", "low", "high", "minus ref", "diff low", "diff high"]]
        for c in ("inhale", "correlation")
    }
    benchmark_b_diff = {
        c: _summary_b[c][["mean", "low", "high", "minus ref", "diff low", "diff high"]]
        for c in ("inhale", "correlation")
    }

    def _counts(recs: pd.DataFrame) -> dict:
        ref = recs[recs["method"] == runs.REFERENCE]
        return {
            "folds": int(ref["fold"].nunique()),
            "animals": int(ref["subject"].nunique()),
            "units": int(ref["unit"].nunique()),
        }

    cv_counts = {**_counts(_a), "cv methods": cv_methods}
    cv_counts_b = _counts(_b)

    def _ours_minus(summary: pd.DataFrame, other: str, digits: int) -> tuple[str, str]:
        """The reference minus *other* on the same resamples, and its interval."""
        row = summary.loc[other]
        return values.num(-row["minus ref"], digits), values.span(
            -row["diff high"], -row["diff low"], digits
        )

    _ref, _f1, _r = runs.REFERENCE, _summary["inhale"], _summary["correlation"]
    _pixels = ["Pixel (motion)", "Pixel (PCA)", "Pixel (SNR)"]
    _others = ["TS-CAN", "RhythmFormer", "PhysFormer"]
    _dsp = analysis.fold_scores(_a, "inhale_f1", by=["method"]).set_index("method")
    _published = analysis.fold_scores(
        recordings[recordings["set"] == test_a].assign(fold="published"),
        "correlation",
        by=["method"],
    ).set_index("method")
    _pub_a = recordings[(recordings["set"] == test_a) & (recordings["method"] == _ref)]
    v_fig3 = {
        "folds": values.count(cv_counts["folds"]),
        "published_units": values.count(_pub_a["unit"].nunique()),
        "published_animals": values.count(_pub_a["subject"].nunique()),
        "units_a": values.count(cv_counts["units"]),
        "animals_a": values.count(cv_counts["animals"]),
        "units_b": values.count(cv_counts_b["units"]),
        "animals_b": values.count(cv_counts_b["animals"]),
        "motion_r": values.num(_r.loc["Pixel (motion)", "mean"]),
        "pixel_r_max": values.num(_r.loc[["Pixel (PCA)", "Pixel (SNR)"], "mean"].max()),
        "pixel_f1": values.span(
            _f1.loc[_pixels, "mean"].min(), _f1.loc[_pixels, "mean"].max()
        ),
        "facemap_r": values.num(_r.loc["Facemap", "mean"]),
        "facemap_f1": values.num(_f1.loc["Facemap", "mean"]),
        "f1": values.num(_f1.loc[_ref, "mean"]),
        "f1_ci": values.span(_f1.loc[_ref, "low"], _f1.loc[_ref, "high"]),
        "physnet_f1": values.num(_f1.loc["PhysNet", "mean"]),
        "r": values.num(_r.loc[_ref, "mean"]),
        "r_ci": values.span(_r.loc[_ref, "low"], _r.loc[_ref, "high"]),
        "physnet_r": values.num(_r.loc["PhysNet", "mean"]),
        "others_r": values.span(
            _r.loc[_others, "mean"].min(), _r.loc[_others, "mean"].max()
        ),
        "published_r": values.num(_published.loc[_ref, "mean"]),
        "published_physnet_r": values.num(_published.loc["PhysNet", "mean"]),
        "dsp_f1": values.num(_dsp.loc[_ref, "mean"]),
        "exhale_f1": values.num(_summary["exhale_f1"].loc[_ref, "mean"]),
        "mae_units": values.num(_summary["MAE (Hz)"].loc[_ref, "mean"], 1),
        "seconds": values.num(timing.loc[_ref, "s per clip (median)"]),
        "physnet_seconds": values.num(timing.loc["PhysNet", "s per clip (median)"]),
        # the share of PhysNet's inference time Zephyr saves, and the transformers' time
        # as a multiple of Zephyr's
        "physnet_saved": values.pct(
            1
            - timing.loc[_ref, "s per clip (median)"]
            / timing.loc["PhysNet", "s per clip (median)"]
        ),
        "transformer_times": values.num(
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].min()
            / timing.loc[_ref, "s per clip (median)"],
            0,
        ),
        "transformer_seconds": values.span(
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].min(),
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].max(),
            1,
        ),
        "r_b": values.num(_summary_b["correlation"].loc[_ref, "mean"]),
        "f1_b": values.num(_summary_b["inhale"].loc[_ref, "mean"]),
        "physnet_r_b": values.num(_summary_b["correlation"].loc["PhysNet", "mean"]),
    }
    v_fig3["physnet_f1_diff"], v_fig3["physnet_f1_diff_ci"] = _ours_minus(
        _f1, "PhysNet", 3
    )
    v_fig3["physnet_r_diff"], v_fig3["physnet_r_diff_ci"] = _ours_minus(
        _r, "PhysNet", 2
    )
    mo.vstack(
        [
            mo.md(f"Test A: {cv_counts}"),
            mo.ui.table(benchmark, selection=None),
            *(mo.ui.table(v.round(3).reset_index()) for v in benchmark_diff.values()),
            mo.md(f"Test B: {cv_counts_b}"),
            mo.ui.table(benchmark_b, selection=None),
            *(mo.ui.table(v.round(3).reset_index()) for v in benchmark_b_diff.values()),
        ]
    )
    return benchmark_b_diff, benchmark_diff, cv_counts, cv_counts_b, v_fig3


@app.cell
def fig4_md():
    mo.md("""
    ## Fig. 4: generalization and its limits

    New animals against new sessions and animal by animal; F1 as the onset tolerance is
    relaxed; a clip of near misses and the clip at the 10th percentile of F1 (e, the
    curve against training size, waits for its runs).
    """)
    return


@app.cell
def fig4ab(by_fold, test_a):
    # a: new animals against new sessions, mean over folds with its interval.
    # b: every new-animal unit of the five folds (each animal is new in one fold).
    _shown = ["Facemap", "TS-CAN", "RhythmFormer", "PhysFormer", "PhysNet", "Zephyr"]
    _sets = [test_a, runs.SETS["test_b"]]
    _ab = by_fold[by_fold["set"].isin(_sets)]
    fig4a = plots.sets_panel(
        *size("fig4", "a"),
        pd.concat(
            analysis.fold_scores(_ab, m, by=["set", "method"]).assign(metric=m)
            for m in ("correlation", "inhale")
        ),
        _shown,
        ["correlation", "inhale"],
        _sets,
        labels=False,
    )
    plots.save_panel(fig4a, FIGURES / "fig4" / "a.svg")
    fig4b = plots.animals_panel(
        *size("fig4", "b"),
        by_fold[by_fold["fold"] != "published"],
        ["Zephyr"],
        "inhale",
        test_a,
        unit="unit",
        unit_label="Animal × rig",
        by_animal="subject",
    )
    plots.save_panel(fig4b, FIGURES / "fig4" / "b.svg")
    mo.hstack([fig4a, fig4b])
    return


@app.cell
def fig4e(whole):
    # e: F1 and the errors left (misses and false inhalations) as the tolerance is
    # relaxed, per fold, then over folds.
    _nets = ["Zephyr"]
    _mine = {k: v[v["method"].isin(_nets)] for k, v in whole.items()}
    _curve = analysis.tolerance_curve(
        _mine["lags"],
        _mine["breaths"],
        _mine["false_positives"],
        np.arange(0, 100.1, 2.5),
    )
    _by = ["method", "tolerance_ms"]
    tolerance = analysis.across_folds(_curve, "f1", by=_by).merge(
        _curve.groupby(_by)["errors"].mean().reset_index(), on=_by
    )
    # Misses and false inhalations apart, each over folds with its interval.
    for _k in ("misses", "false"):
        _apart = analysis.across_folds(_curve, _k, by=_by).rename(
            columns={"mean": _k, "low": f"{_k} low", "high": f"{_k} high"}
        )
        tolerance = tolerance.merge(_apart.drop(columns="folds"), on=_by)
    fig4c = plots.tolerance_panel(*size("fig4", "c"), tolerance, _nets)
    plots.save_panel(fig4c, FIGURES / "fig4" / "c.svg")
    _at = tolerance.set_index("tolerance_ms")
    v_tolerance = {
        "f1_12": values.num(_at.loc[12.5, "mean"]),
        "f1_25": values.num(_at.loc[25.0, "mean"]),
        "f1_100": values.num(_at.loc[100.0, "mean"]),
        "errors_25": values.pct(_at.loc[25.0, "errors"], 1),
        "errors_100": values.pct(_at.loc[100.0, "errors"], 1),
        "breaths": values.count(len(_mine["breaths"])),
    }
    fig4c
    return tolerance, v_tolerance


@app.cell
def fig5a(test_a, with_unit):
    # Fig. 5a: Zephyr trained on fewer animals (all their sessions), and on fewer
    # sessions of all its animals (open), on the cross-validation fold those runs were
    # trained on, against Zephyr on that fold; scored per unit on its new animals (test
    # A), the same units for every run. The 2-animal draws exist on the published split
    # only: they are reported apart, flagged, and not drawn.
    _size = runs.available(runs.LEARNING_CURVE)

    def _cv_fold(name):
        fold = runs.RUNS[name].fold or ""
        return next(
            (
                f
                for f in runs.RESULTS.cross_validation.folds
                if fold.startswith(f + "-")
            ),
            None,
        )

    _on_fold = [n for n in _size if _cv_fold(n)]
    _ref = runs.cv_name(runs.REFERENCE, _cv_fold(_on_fold[0]))
    _names = [_ref, *_on_fold]
    _published = [runs.REFERENCE, *(n for n in _size if not _cv_fold(n))]

    def _curve(names, reference):
        trained = {n: runs.training(n) for n in names}
        every = trained[reference]
        recs = with_unit(runs.scores(names, "per_recording"))
        rows = []
        for metric in ["inhale", "correlation"]:
            # One run at a time: each has its own known units.
            ci = pd.concat(
                analysis.bootstrap(
                    recs[(recs["set"] == test_a) & (recs["method"] == n)].assign(
                        set="New animals"
                    ),
                    metric,
                    reference=n,
                    unit="unit",
                )
                for n in names
            )
            for _, r in ci.iterrows():
                tr = trained[r["method"]]
                rows.append(
                    {
                        "group": "New animals",
                        "metric": metric,
                        "method": r["method"],
                        "animals": tr["subject"].nunique(),
                        "sessions": len(tr),
                        "full": len(tr)
                        == every["subject"].isin(set(tr["subject"])).sum(),
                        "units": r["units"],
                        **{k: r[k] for k in ("mean", "low", "high")},
                    }
                )
        return pd.DataFrame(rows), recs, trained

    learning, _recs, _trained = _curve(_names, _ref)
    _drawn, _, _ = _curve(_published, runs.REFERENCE)
    # The text's numbers: runs found by what they trained on, not by their labels.
    _f1 = learning[learning["metric"] == "inhale"].set_index("method")
    _r = learning[learning["metric"] == "correlation"].set_index("method")
    _full = _f1[_f1["full"]].sort_values("animals")
    _smallest = _full.index[0]  # fewest animals
    _partial = _f1[~_f1["full"]].index[0]  # all animals, a part of their sessions
    _d = _drawn[(_drawn["metric"] == "inhale") & (_drawn["animals"] == 2)]
    _d = _d.set_index("method").sort_index()
    # The fewest animals against all, paired on the units both were scored on.
    _pair = _recs[(_recs["set"] == test_a) & _recs["method"].isin([_ref, _smallest])]
    _shared = set.intersection(
        *(set(_pair.loc[_pair["method"] == n, "unit"]) for n in (_ref, _smallest))
    )
    _diff = (
        analysis.bootstrap(
            _pair[_pair["unit"].isin(_shared)], "inhale", reference=_ref, unit="unit"
        )
        .set_index("method")
        .loc[_smallest]
    )
    _nested = _full[_full.index != _ref]
    v_size = {
        "animals": ", ".join(values.count(n) for n in _nested["animals"].iloc[:-1])
        + " and "
        + values.count(_nested["animals"].iloc[-1]),
        "sessions": ", ".join(values.count(n) for n in _nested["sessions"].iloc[:-1])
        + " and "
        + values.count(_nested["sessions"].iloc[-1]),
        "all_animals": values.count(_f1.loc[_ref, "animals"]),
        "all_sessions": values.count(_f1.loc[_ref, "sessions"]),
        "nested_f1": ", ".join(
            values.num(x) for x in _f1.loc[_nested.index, "mean"].iloc[:-1]
        )
        + " and "
        + values.num(_f1.loc[_nested.index[-1], "mean"]),
        "fewest": values.count(_f1.loc[_smallest, "animals"]),
        "fewest_f1": values.num(_f1.loc[_smallest, "mean"]),
        "fewest_r": values.num(_r.loc[_smallest, "mean"]),
        "f1": values.num(_f1.loc[_ref, "mean"]),
        "r": values.num(_r.loc[_ref, "mean"]),
        "diff": values.num(_diff[f"{_ref} minus"]),
        "diff_ci": values.span(_diff["diff low"], _diff["diff high"]),
        "partial_sessions": values.count(_f1.loc[_partial, "sessions"]),
        "partial_f1": values.num(_f1.loc[_partial, "mean"]),
        "partial_r": values.num(_r.loc[_partial, "mean"]),
        "draw_sessions": values.span(_d["sessions"].min(), _d["sessions"].max(), 0),
        "draws_f1": ", ".join(values.num(x) for x in _d["mean"].iloc[:-1])
        + " and "
        + values.num(_d["mean"].iloc[-1]),
        "fold": _ref.split(" | ")[-1],
        "units": values.count(_f1.loc[_ref, "units"]),
    }
    # Rig coverage (Methods): new animals (test A) by whether the run trained on their
    # rig. Unpaired: the units in each group differ.
    _tested = _recs[_recs["set"] == test_a]
    _per_unit = (
        _tested.join(analysis.familiarity(_tested, _trained))
        .groupby(["method", "set", "rig_trained", "unit"], as_index=False)[
            ["inhale", "correlation"]
        ]
        .mean()
    )
    rig_coverage = _per_unit.groupby(["method", "rig_trained"], as_index=False).agg(
        units=("unit", "size"), inhale=("inhale", "mean"), r=("correlation", "mean")
    )
    fig5a = plots.learning_curve_panel(
        *size("fig5", "a"),
        learning,
        {"New animals": style.color(test_a)},
    )
    plots.save_panel(fig5a, FIGURES / "fig5" / "a.svg")
    mo.vstack(
        [
            fig5a,
            mo.ui.table(learning.round(3), selection=None),
            mo.md("New animals (test A), by whether the rig was in training"),
            mo.ui.table(rig_coverage.round(3), selection=None),
        ]
    )
    return (v_size,)


@app.cell
def fig4df(test_a):
    # Two test-A clips, neither an outlier: among clips whose correlation is at least
    # the median, the one with most near misses (predictions 25-100 ms off; d), around
    # its densest near misses; and, among the lowest tenth of clips by correlation, the
    # 2-s stretch in which every inhalation is found (no error, no excluded span, at
    # least 4 true inhalations) from the least confident onset output: the lowest
    # median onset probability at the head's events (f).
    _one = runs.SINGLE
    _clips = (
        runs.scores([_one])
        .query("set == @test_a")
        .loc[lambda d: d["stem"].map(runs.is_local)]
        .sort_values("inhale")
        .set_index("stem")
    )
    _lags = runs.tables(_one)["lags"].query("set == @test_a")
    _far = (
        _lags[_lags["lag_s"].abs() > plots.TOLERANCE_MS / 1000].groupby("stem").size()
    )
    _good = _clips.index[_clips["correlation"] >= _clips["correlation"].median()]
    _near = _far.reindex(_good).idxmax()
    _low = _clips[_clips["correlation"] <= _clips["correlation"].quantile(0.1)].drop(
        _near, errors="ignore"
    )

    def _quiet(stem):
        """A clip's fully found 2-s stretch with the lowest median onset probability at
        the head's events: ``(median, centre, event probabilities, stretch r)``."""
        p, t = runs.prediction(_one, stem), runs.truth(stem)
        err = runs.errors(_one, stem)["time"].to_numpy()
        ex = np.asarray(t["excluded"]).reshape(-1, 2)
        pt = p["times"]
        heights = p["onset"][np.searchsorted(pt, p["head"]).clip(0, len(pt) - 1)]
        target = np.interp(pt, t["times"], t["signal"])
        best = (np.inf, None, None, None)
        for c in np.arange(pt[0] + 1.0, pt[-1] - 1.0, 0.25):
            clean = (
                not np.any(np.abs(err - c) < 1.0)
                and not np.any((ex[:, 0] < c + 1) & (ex[:, 1] > c - 1))
                and np.sum(np.abs(t["inhale"] - c) < 1.0) >= 4
            )
            h = heights[np.abs(p["head"] - c) < 1.0]
            if clean and len(h) and np.median(h) < best[0]:
                inside = (pt >= c - 1) & (pt < c + 1)
                r = np.corrcoef(p["signal"][inside], target[inside])[0, 1]
                best = (float(np.median(h)), float(c), h, float(r))
        return best

    _quietest = {s: _quiet(s) for s in _low.index}
    _upper = min(_quietest, key=lambda s: _quietest[s][0])
    _cases = {"d": _clips.loc[_near], "e": _clips.loc[_upper]}
    low_r = {
        "clips": len(_low),
        "below r": float(_clips["correlation"].quantile(0.1)),
        "median F1": float(_low["inhale"].median()),
        "share F1 >= 0.9": float((_low["inhale"] >= 0.9).mean()),
        "spearman r, F1": float(
            _clips[["correlation", "inhale"]].corr("spearman").iloc[0, 1]
        ),
    }
    _shown = []
    for _panel, _row in _cases.items():
        _stem = _row.name
        _errors = runs.errors(_one, _stem)
        _p = runs.prediction(_one, _stem)
        _origin = float(_p["times"][0])
        _truth = runs.truth(_stem)
        if _panel == "d":
            _t = _errors.loc[_errors["near_miss"], "time"].to_numpy()
            _at = float(_t[np.argmax([np.sum(np.abs(_t - x) < 1.5) for x in _t])])
        else:
            _median_p, _at, _peaks, local_r = _quietest[_stem]
        _frame_t, _frames = runs.gray(_stem)
        _picked = dict.fromkeys(
            min(int(np.searchsorted(_frame_t, _at + dt)), len(_frames) - 1)
            for dt in np.linspace(-0.4, 0.4, 5)
        )
        _fig = plots.failure_panel(
            *size("fig4", _panel),
            _truth,
            _p,
            _errors,
            _origin,
            _at - _origin,
            [(_frame_t[i] - _origin, np.asarray(_frames[i, 0])) for i in _picked],
            half=1.0,
            legend=_panel == "d",
        )
        plots.save_panel(_fig, FIGURES / "fig4" / f"{_panel}.svg")
        _shown += [
            mo.md(
                f"**{_panel}**: {_stem}, inhale F1 {_row['inhale']:.3f}, "
                f"r {_row['correlation']:.3f}"
            ),
            _fig,
        ]
    v_clips = {
        "clips": values.count(len(_clips)),
        "spearman": values.num(low_r["spearman r, F1"]),
        "low_clips": values.count(low_r["clips"]),
        "low_r": values.num(low_r["below r"]),
        "low_f1": values.num(low_r["median F1"]),
        "low_f1_quartile": values.num(_low["inhale"].quantile(0.75)),
        "near_r": values.num(_cases["d"]["correlation"]),
        "near_f1": values.num(_cases["d"]["inhale"]),
        "quiet_r": values.num(_cases["e"]["correlation"]),
        "quiet_stretch_r": values.num(local_r),
        "quiet_peaks": values.span(_peaks.min(), _peaks.max()),
        "quiet_peak": values.num(_median_p),
        "threshold": values.num(runs.head_threshold(_one)),
    }
    mo.vstack(
        [
            mo.md(
                f"Lowest tenth by correlation: {low_r}; f's stretch r {local_r:.2f}, "
                f"onset at its {len(_peaks)} events {_peaks.min():.2f}-{_peaks.max():.2f} "
                f"(median {_median_p:.2f})"
            ),
            *_shown,
        ]
    )
    return (v_clips,)


@app.cell
def fig6_md():
    mo.md(f"""
    ## Fig. 6: judging predictions without a thermistor

    The quality-control reports of the reference checkpoint on test sets A and B, and
    its reference (onset calibration on the validation animals), from
    `{QC}`; made with `zephyr quality-control reference` and `run`.
    """)
    return


@app.cell
def fig6_data():
    qc_videos, qc_windows = quality.tables(str(QC / "test"))
    mo.stop(qc_videos.empty, mo.md("No quality-control reports yet."))
    _sets = runs.clip_sets()
    qc_videos = qc_videos.assign(set=qc_videos["stem"].map(_sets))
    qc_windows = qc_windows.assign(set=qc_windows["stem"].map(_sets))
    judged = {m: b for m, b in quality.JUDGED.items() if m in qc_videos}
    truth_f1 = "truth_inhale_f1"
    return judged, qc_videos, qc_windows, truth_f1


@app.cell
def fig6a(test_a):
    # Fig. 6a: what our network's errors on new animals are, per fold, then over folds.
    # The share that are near misses, then, among the rest, the share in fast breathing,
    # face motion and on shallow breaths, against the share of all true breaths with
    # each (motion and size are quantiles of each clip's own breaths, so those
    # baselines are fixed).
    _folds = runs.RESULTS.cross_validation.folds
    _names = {f: runs.cv_name(runs.REFERENCE, f) for f in _folds}
    _errors = pd.concat(
        runs.set_errors(n, test_a).assign(fold=f) for f, n in _names.items()
    )
    _breaths = pd.concat(
        runs.tables(n)["breaths"].query("set == @test_a").assign(fold=f)
        for f, n in _names.items()
    )
    _profile = analysis.error_profile(_errors)
    error_profile = analysis.across_folds(_profile, "share", by=["kind", "row"]).merge(
        _profile.groupby(["kind", "row"])["n"].sum().reset_index(), on=["kind", "row"]
    )
    _fast = _breaths.groupby("fold")["rate_hz"].apply(lambda r: (r > 8).mean())
    _baseline = {
        "fast breathing (> 8 Hz)": float(_fast.mean()),
        "face motion (top 10%)": 0.10,
        "shallow breath (bottom 25%)": 0.25,
    }
    fig6a = plots.error_profile_panel(*size("fig6", "a"), error_profile, _baseline)
    plots.save_panel(fig6a, FIGURES / "fig6" / "a.svg")
    _p = error_profile.set_index(["kind", "row"])
    _kinds = {"misses": "missed (FN)", "false": "false (FP)"}
    _near, _fast = "near miss", "fast breathing (> 8 Hz)"
    _motion, _shallow = "face motion (top 10%)", "shallow breath (bottom 25%)"

    def _range(row: str) -> str:
        shares = [100 * _p.loc[(k, row), "mean"] for k in _kinds.values()]
        return values.span(min(shares), max(shares), 0)

    v_errors = {
        **{k: values.count(_p.loc[(v, _near), "n"]) for k, v in _kinds.items()},
        **{
            f"{k}_near": values.pct(_p.loc[(v, _near), "mean"])
            for k, v in _kinds.items()
        },
        **{
            f"{k}_gross": values.count(_p.loc[(v, _fast), "n"])
            for k, v in _kinds.items()
        },
        "fast_misses": values.pct(_p.loc[(_kinds["misses"], _fast), "mean"]),
        "fast_breaths": values.pct(_baseline[_fast]),
        "motion": _range(_motion),
        "shallow": _range(_shallow),
    }
    mo.vstack(
        [
            fig6a,
            mo.ui.table(error_profile.round(3), selection=None),
            mo.md(f"Baselines: {_baseline}"),
        ]
    )
    return error_profile, v_errors


@app.cell
def fig6bc(judged, qc_videos, qc_windows, test_a, truth_f1):
    # Fig. 6b-c on new animals (test set A) only: the five measures that rank its
    # videos best, and the windows the three best at ranking windows keep.
    _videos = qc_videos[qc_videos["set"] == test_a]
    _windows = qc_windows[qc_windows["set"] == test_a]
    qc_ranks = {
        "Videos": analysis.quality_agreement(_videos, judged, truth_f1),
        "Windows": analysis.quality_agreement(_windows, judged, truth_f1),
    }
    _top = qc_ranks["Videos"].nlargest(5, "oriented")
    fig6b = plots.qc_agreement_panel(*size("fig6", "b"), {"Videos": _top})
    _best = qc_ranks["Windows"].sort_values("oriented", ascending=False)["measure"]
    _fractions = tuple(np.linspace(0.05, 1, 20))
    _checks = [c for c in _windows if c.startswith("ok_")]
    _w = _windows.dropna(subset=[truth_f1])
    _pass = _w[_w[_checks].fillna(True).astype(bool).all(axis=1)]
    fig6c = plots.qc_retention_panel(
        *size("fig6", "c"),
        {
            m: analysis.retention(_w, m, truth_f1, judged[m], _fractions)
            for m in _best.head(3)
        },
        analysis.retention(_w, truth_f1, truth_f1, "higher", _fractions),
        (len(_pass) / len(_w), float(_pass[truth_f1].mean())),
    )
    qc_kept = {
        "windows": len(_w),
        "pass every check": len(_pass) / len(_w),
        "F1 all": float(_w[truth_f1].mean()),
        "F1 passing": float(_pass[truth_f1].mean()),
        "F1 failing": float(_w.drop(_pass.index)[truth_f1].mean()),
    }
    v_qc = {
        "videos": values.count(len(_videos)),
        "spearman": values.span(_top["oriented"].min(), _top["oriented"].max()),
        "windows": values.count(qc_kept["windows"]),
        "pass": values.pct(qc_kept["pass every check"]),
        "f1_pass": values.num(qc_kept["F1 passing"]),
        "f1_fail": values.num(qc_kept["F1 failing"]),
        "f1_all": values.num(qc_kept["F1 all"]),
    }
    mo.vstack(
        [
            mo.hstack([fig6b, fig6c]),
            mo.md(f"{qc_kept}"),
            mo.ui.table(qc_ranks["Videos"].round(3), selection=None),
            mo.ui.table(qc_ranks["Windows"].round(3), selection=None),
        ]
    )
    return (v_qc,)


@app.cell
def fig6d():
    _reference = quality.reference(str(QC / "reference.json"))
    mo.stop(_reference is None, mo.md("No reference yet."))
    fig6d = plots.calibration_panel(*size("fig6", "d"), _reference)
    v_calibration = {"ece": values.num(_reference["onset_calibration"]["ece"], 3)}
    fig6d
    return (v_calibration,)


@app.cell
def tables_md():
    mo.md("""
    ## Supplementary tables

    Test set C (the 359-Hz animals), the sets of each cross-validation fold, the
    comparison networks' settings (from each run's own `config.json`); written as CSV
    for main.typ.
    """)
    return


@app.cell
def supplementary_tables(methods, recordings, spec):
    # Test set C of the published split, per method: each score averaged within each
    # animal x rig unit, then over the units (two animals: no interval).
    _c = recordings[recordings["set"] == runs.SETS["test_c"]]
    _rows = []
    for _m in methods:
        _u = (
            _c[_c["method"] == _m]
            .groupby("unit")[["correlation", "inhale", "exhale_f1"]]
            .mean()
        )
        _rows.append(
            {
                "method": plots.DISPLAY.get(_m, _m),
                "trace_r": f"{_u['correlation'].mean():.2f}",
                "inhale_f1": f"{_u['inhale'].mean():.2f}",
                "exhale_f1": f"{_u['exhale_f1'].mean():.2f}",
            }
        )
    test_c = pd.DataFrame(_rows)
    _tables = FIGURES.parent / "tables"
    test_c.to_csv(_tables / "test_c.csv", index=False)
    test_c_size = {
        "animals": int(_c["subject"].nunique()),
        "units": int(_c["unit"].nunique()),
        "sessions": int(_c["session"].nunique()),
    }

    # The sets of each cross-validation fold (cross_validation.parquet), with a check
    # that each fold's test A animals are those its runs were scored on.
    _cv = pd.read_parquet(runs.dataset_file("cross_validation.parquet", spec.root))
    _names = {
        "train": "Training",
        "val": "Validation",
        "test_a": "Test A",
        "test_b": "Test B",
        "test_c": "Test C",
    }
    _rows, _checked = [], {}
    _scored = runs.cv_scores("per_recording")
    for _f in runs.RESULTS.cross_validation.folds:
        _col = _cv[f"fold_{_f.split('_')[-1]}"]
        for _set, _label in _names.items():
            _in = _cv[_col == _set]
            _rows.append(
                {
                    "fold": _f,
                    "set": _label,
                    "animals": _in["subject_id"].nunique(),
                    "sessions": len(_in),
                }
            )
        _a = _scored[
            (_scored["fold"] == _f)
            & (_scored["method"] == runs.REFERENCE)
            & (_scored["set"] == runs.SETS["test_a"])
        ]
        _want = set(_cv.loc[_col == "test_a", "subject_id"].astype(str))
        _checked[_f] = set(_a["subject"].astype(str)) == _want
    fold_sizes = pd.DataFrame(_rows).pivot(
        index="set", columns="fold", values=["animals", "sessions"]
    )
    _wide = pd.DataFrame(index=list(_names.values()))
    for _f in runs.RESULTS.cross_validation.folds:
        _wide[_f] = [
            f"{fold_sizes.loc[_l, ('animals', _f)]} / {fold_sizes.loc[_l, ('sessions', _f)]}"
            for _l in _wide.index
        ]
    _wide.rename_axis("set").reset_index().to_csv(_tables / "cv_folds.csv", index=False)

    # The networks' settings, from each run's config.json, with their size and
    # inference time per 5-min clip on the GPU.
    _nets = ["Zephyr", "TS-CAN", "PhysNet", "PhysFormer", "RhythmFormer"]
    _gpu = runs.wall_time(_nets).set_index("method")
    _rows = []
    for _m in _nets:
        _run = runs.RUNS[_m]
        _path = runs._fetch(
            f"{runs.folder(_run)}/{runs.seeds(runs.folder(_run))[0]}/config.json"
        )
        _tp = json.loads(_path.read_text())["fold"]["train_params"]
        _rows.append(
            {
                "method": _m,
                "window": _tp["window"],
                "batch": _tp["batch_size"],
                "steps": _tp["steps_per_epoch"],
                "lr": _tp["lr"],
                "infer_window": _tp["infer_window"],
                "scales": "/".join(map(str, _tp["scales"])),
                "params_m": f"{_gpu.loc[_m, 'parameters (M)']:.2f}",
                "gpu_s": f"{_gpu.loc[_m, 's per clip (median)']:.2f}",
            }
        )
    comparison_nets = pd.DataFrame(_rows)
    comparison_nets.to_csv(_tables / "comparison_nets.csv", index=False)
    mo.vstack(
        [
            mo.md(f"Test C: {test_c_size}; folds match their runs: {_checked}"),
            mo.ui.table(test_c, selection=None),
            mo.ui.table(_wide.reset_index(), selection=None),
            mo.ui.table(comparison_nets, selection=None),
        ]
    )
    return comparison_nets, fold_sizes, test_c, test_c_size


@app.cell
def figs1(by_fold, test_a):
    # Supplementary Fig. 1: why the unit is an animal on one rig. Our network on new
    # animals (every fold's test A, each animal new once): how far apart two units
    # score when they share the animal but not the rig, and when they share the rig
    # but not the animal, with intervals from resampling animals.
    _a = by_fold[(by_fold["set"] == test_a) & (by_fold["method"] == runs.REFERENCE)]
    _a = _a[_a["fold"] != "published"]
    _pairs = {m: analysis.rig_animal_pairs(_a, m) for m in ("inhale", "correlation")}
    _summaries = {m: analysis.rig_animal_bootstrap(p) for m, p in _pairs.items()}
    figs1a = plots.rig_animal_panel(
        *size("figs1", "a"),
        {m: (_pairs[m], _summaries[m]) for m in _pairs},
        runs.REFERENCE,
    )
    plots.save_panel(figs1a, FIGURES / "figs1" / "a.svg")
    rig_animal = pd.concat(
        [v.assign(metric=k) for k, v in _summaries.items()], ignore_index=True
    )
    mo.vstack([figs1a, mo.ui.table(rig_animal.round(4), selection=None)])
    return (rig_animal,)


@app.cell
def figs2(by_fold, test_a, with_unit):
    # Supplementary Fig. 2: our network fed 30-Hz video (every other frame), still
    # predicting on the 60-Hz grid, against the same network on 60-Hz video, on every
    # fold. a: the difference per test set, paired by fold; b: inhale F1 on new animals
    # by breathing rate (each fold's breaths pooled, then over folds).
    _ref = by_fold[
        (by_fold["method"] == runs.REFERENCE) & (by_fold["fold"] != "published")
    ]
    _recs = pd.concat(
        [_ref, with_unit(runs.cv_scores(variants=True))], ignore_index=True
    )
    _name = next(iter(runs.CV_VARIANTS))
    _folds, _summaries = [], []
    for _m in ("correlation", "inhale", "exhale_f1"):
        _f = analysis.per_fold(_recs, _m)
        _p = _f[_f["method"] == _name].merge(
            _f[_f["method"] == runs.REFERENCE],
            on=["set", "fold"],
            suffixes=("", "_ref"),
        )
        _p = _p.assign(value=_p["value"] - _p["value_ref"], metric=_m)
        _folds.append(_p[["metric", "set", "fold", "value"]])
        _summaries.append(
            analysis.across_folds(_p, "value", by=["set"]).assign(metric=_m)
        )
    input_rate_folds = pd.concat(_folds, ignore_index=True)
    input_rate = pd.concat(_summaries, ignore_index=True)
    figs2a = plots.fold_difference_panel(
        *size("figs2", "a"), input_rate_folds, input_rate, _name, runs.REFERENCE
    )
    plots.save_panel(figs2a, FIGURES / "figs2" / "a.svg")
    _tables = {"breaths": [], "false_positives": []}
    for _n in (runs.REFERENCE, _name):
        for _f in runs.RESULTS.cross_validation.folds:
            _t = runs.tables(runs.cv_name(_n, _f))
            for _k, _parts in _tables.items():
                _parts.append(_t[_k].query("set == @test_a").assign(method=_n, fold=_f))
    _by_rate = analysis.f1_by_rate(
        pd.concat(_tables["breaths"], ignore_index=True),
        pd.concat(_tables["false_positives"], ignore_index=True),
        analysis.RATE_BINS_HZ[1:],
    )
    input_rate_by_rate = analysis.across_folds(
        _by_rate, "f1", by=["method", "bin"]
    ).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    figs2b = plots.rate_band_panel(
        *size("figs2", "b"),
        input_rate_by_rate,
        [runs.REFERENCE, _name],
        ylim=(0.8, 1.0),
    )
    plots.save_panel(figs2b, FIGURES / "figs2" / "b.svg")
    mo.vstack(
        [
            figs2a,
            figs2b,
            mo.ui.table(input_rate.round(4), selection=None),
            mo.ui.table(input_rate_by_rate.round(3), selection=None),
        ]
    )
    return input_rate, input_rate_by_rate


@app.cell
def figs3(test_a, with_unit):
    # Supplementary Fig. 3 and Table S5: our network trained on each camera's video, on
    # the cross-validation folds every camera has so far, scored on the recording spans
    # every camera has, all against the same thermistor events (the face clips').
    _runs = {
        "Face camera": runs.REFERENCE,
        "Side camera": "Zephyr, side camera",
        "Front camera": "Zephyr, front camera",
    }
    _cameras = list(_runs)
    _folds = [
        f
        for f in runs.RESULTS.cross_validation.folds
        if all(runs.evaluation(runs.cv_name(r, f)) is not None for r in _runs.values())
    ]
    _sets = [test_a, runs.SETS["test_b"]]
    _clips = pd.concat(
        [runs.cv_scores("clips"), runs.cv_scores("clips", variants=True)],
        ignore_index=True,
    )
    _clips = _clips[
        _clips["fold"].isin(_folds)
        & _clips["set"].isin(_sets)
        & _clips["method"].isin(_runs.values())
    ]
    _clips = with_unit(
        _clips.assign(
            method=_clips["method"].map({r: c for c, r in _runs.items()}),
            span=_clips["stem"].map(runs._span),
        )
    )
    # Only the spans every camera has, in each fold.
    _n = _clips.groupby(["fold", "span"])["method"].nunique()
    _shared = set(_n[_n == len(_cameras)].index)
    camera_clips = _clips[
        [k in _shared for k in zip(_clips["fold"], _clips["span"], strict=True)]
    ]
    # Fig. 5b: each camera on new animals and new sessions; table: every score per set.
    _ci = {
        m: analysis.bootstrap(camera_clips, m, reference=_cameras[0], unit="unit")
        for m in ("correlation", "inhale", "exhale_f1")
    }
    plots.save_panel(
        plots.sets_panel(
            *size("fig5", "b"),
            pd.concat(_ci[m].assign(metric=m) for m in ("correlation", "inhale")),
            _cameras,
            ["correlation", "inhale"],
            _sets,
            ylabel="Mean over units",
            labels=False,
            legend="lower right",
        ),
        FIGURES / "fig5" / "b.svg",
    )
    # a: each new-animal unit, ranked by the face camera, as Fig. 4b.
    figs3a = plots.animals_panel(
        *size("figs3", "a"),
        camera_clips,
        _cameras,
        "inhale",
        test_a,
        unit="unit",
        unit_label="Animal × rig",
    )
    plots.save_panel(figs3a, FIGURES / "figs3" / "a.svg")
    # b: inhale F1 on new animals by breathing rate, over the shared spans.
    _tables = {k: [] for k in ("breaths", "false_positives", "windows")}
    for _camera, _run in _runs.items():
        for _f in _folds:
            _t = runs.tables(runs.cv_name(_run, _f))
            for _k, _parts in _tables.items():
                _x = _t[_k].assign(method=_camera, fold=_f)
                _keep = [(_f, runs._span(s)) in _shared for s in _x["stem"].astype(str)]
                _parts.append(_x[_keep])
    _tables = {k: pd.concat(v, ignore_index=True) for k, v in _tables.items()}
    _by_rate = analysis.f1_by_rate(
        _tables["breaths"].query("set == @test_a"),
        _tables["false_positives"].query("set == @test_a"),
        analysis.RATE_BINS_HZ[1:],
    )
    camera_by_rate = analysis.across_folds(_by_rate, "f1", by=["method", "bin"]).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    figs3b = plots.rate_band_panel(*size("figs3", "b"), camera_by_rate, _cameras)
    plots.save_panel(figs3b, FIGURES / "figs3" / "b.svg")
    # c: every new-animal clip, each other camera against the face camera.
    _wide = (
        camera_clips[camera_clips["set"] == test_a]
        .pivot_table(index=["fold", "span"], columns="method", values="inhale")
        .reset_index()
    )
    figs3c = plots.paired_scatter_panel(
        *size("figs3", "c"), _wide, _cameras[0], _cameras[1:], "inhale"
    )
    plots.save_panel(figs3c, FIGURES / "figs3" / "c.svg")
    # Table S5: per set and camera, every score over units (mean, 95% bootstrap
    # interval), with the breathing-rate MAE of 10-s windows (breaths per minute).
    _w = _tables["windows"]
    _unit = camera_clips.drop_duplicates("stem").set_index("stem")["unit"]
    _w = _w.assign(unit=_w["stem"].map(_unit)).dropna(subset=["unit"])
    _mae = analysis.rate_agreement(_w, by=["method", "set", "unit"]).reset_index()
    _mae["mae"] = 60 * _mae["MAE (Hz)"]
    _ci["mae"] = analysis.bootstrap(_mae, "mae", reference=_cameras[0], unit="unit")

    def _cell(row, digits: int) -> str:
        return (
            f"{values.num(row['mean'], digits)} "
            f"({values.span(row['low'], row['high'], digits)})"
        )

    _rows = []
    for _set in _sets:
        for _camera in _cameras:
            _r = {
                m: t.set_index(["set", "method"]).loc[(_set, _camera)]
                for m, t in _ci.items()
            }
            _rows.append(
                {
                    "set": _set.split(":")[0],
                    "camera": _camera.split()[0],
                    "units": values.count(_r["inhale"]["units"]),
                    "trace_r": _cell(_r["correlation"], 2),
                    "inhale_f1": _cell(_r["inhale"], 2),
                    "exhale_f1": _cell(_r["exhale_f1"], 2),
                    "mae": _cell(_r["mae"], 1),
                }
            )
    cameras_table = pd.DataFrame(_rows)
    cameras_table.to_csv(FIGURES.parent / "tables" / "cameras.csv", index=False)
    _a = camera_clips[camera_clips["set"] == test_a]
    _b = camera_clips[camera_clips["set"] == runs.SETS["test_b"]]
    _on_a = {m: t.set_index(["set", "method"]) for m, t in _ci.items()}
    v_cameras = {
        **{
            f"{k}_{c.split()[0].lower()}": values.num(_on_a[m].loc[(test_a, c), "mean"])
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for c in _cameras
        },
        "folds": values.count(len(_folds)),
        "clips_a": values.count(len(_a) / len(_cameras)),
        "clips_b": values.count(len(_b) / len(_cameras)),
        "units_a": values.count(_a["unit"].nunique()),
        "units_b": values.count(_b["unit"].nunique()),
        "animals_a": values.count(_a["subject"].nunique()),
        "breaths": values.count(
            (_tables["breaths"].query("set == @test_a")["method"] == _cameras[0]).sum()
        ),
    }
    mo.vstack(
        [
            mo.hstack([figs3a, figs3b, figs3c]),
            mo.ui.table(cameras_table, selection=None),
            mo.md(f"{v_cameras}"),
        ]
    )
    return (v_cameras,)


@app.cell
def figs4():
    # Supplementary Fig. 4: what the three cameras see, in a representative subset of
    # sessions, its three cameras side by side with the crop box each network reads.
    # Face: the dataset's thumbnails and clip list; side and front: the first frames and
    # boxes their box lists were placed on.
    from PIL import Image

    from zephyr.config import ClipList, load

    _shown = [
        "behavior_810750_2025-11-05_20-47-56",
        "behavior_828418_2025-11-12_23-52-42",
        "behavior_811026_2025-11-25_21-25-01",
        "behavior_815104_2025-11-26_22-46-47",
        "behavior_828425_2025-11-28_22-26-13",
        "behavior_828425_2025-12-19_16-43-15",
        "behavior_806527_2026-01-23_19-16-56",
        "807093_2026-02-10_17-02-30",
        "841310_2026-02-19_21-04-00",
        "828424_2026-02-21_00-33-08",
        "828415_2026-02-28_00-20-42",
        "823993_2026-03-02_21-02-53",
        "863680_2026-06-26_16-20-33",
        "863698_2026-07-22_16-58-02",
        "864848_2026-09-18_19-51-33",
        "879294_2026-09-29_20-38-40",
    ]
    _lists = {
        "Face camera": (runs.root() / "dataset" / "clips.toml", None),
        "Side camera": (
            runs.root() / "boxes-side" / "side_camera_snout.toml",
            runs.root() / "boxes-side" / "frames",
        ),
        "Front camera": (
            runs.root() / "boxes-front" / "front_camera_snout.toml",
            runs.root() / "boxes-front" / "frames",
        ),
    }

    def _boxes(path):
        """First span's box of each session, in the list's preprocessed frame."""
        _list = load(ClipList, path, check_files=False)
        _out = {}
        for _c in _list.resolve():
            _session, _span = _c.source.stem.rsplit("_span", 1)
            if _span.startswith("1_") and _c.box is not None:
                _out[_session] = (_c.source.stem, _c.box)
        return _out, _list.preprocess.target_size[0]

    _found = {camera: _boxes(path) for camera, (path, _) in _lists.items()}

    def _image(camera, session):
        _stem, _ = _found[camera][0][session]
        _frames = _lists[camera][1]
        if _frames is None:
            _path = runs.root() / "dataset" / "thumbnails" / f"{session}_FaceCamera.png"
            return np.asarray(Image.open(_path).convert("L"))
        return np.load(next(_frames.glob(f"frame_{_stem}-*.npy")))

    _sessions = []
    for _session in _shown:
        _views = {}
        for _camera in _lists:
            _img = _image(_camera, _session)
            _, (_x, _y, _w, _h) = _found[_camera][0][_session]
            _scale = _img.shape[1] / _found[_camera][1] / 2  # shown at half size
            _views[_camera] = (
                _img[::2, ::2],
                (_x * _scale, _y * _scale, _w * _scale, _h * _scale),
            )
        _sessions.append({"label": _session, "views": _views})
    figs4a = plots.camera_views_panel(LAYOUT["figs4"]["width_mm"], _sessions)
    plots.save_panel(figs4a, FIGURES / "figs4" / "a.svg")
    v_views = {"sessions": values.count(len(_sessions))}
    figs4a
    return (v_views,)


@app.cell
def manuscript_values(
    v_cameras,
    v_clips,
    v_corpus,
    v_errors,
    v_fig1,
    v_fig3,
    v_rate,
    v_rates,
    v_cameras_missing,
    v_selection,
    v_size,
    v_tolerance,
    v_slow,
    v_views,
):
    # Every number the text cites, as printed, for main.typ (#v("section.key")).
    _cv = [
        runs.head_threshold(runs.cv_name(runs.REFERENCE, f))
        for f in runs.RESULTS.cross_validation.folds
    ]
    v_threshold = {
        "folds": values.span(min(_cv), max(_cv)),
        "published": values.num(runs.head_threshold(runs.REFERENCE)),
    }
    manuscript = {
        "benchmark": v_fig3,
        "cameras": v_cameras,
        "clips": v_clips,
        "corpus": v_corpus,
        "errors": v_errors,
        "rate": v_rate,
        "rates": v_rates,
        "selection": v_selection,
        "optional_cameras": v_cameras_missing,
        "size": v_size,
        "task": v_fig1,
        "threshold": v_threshold,
        "tolerance": v_tolerance,
        "slow": v_slow,
        "views": v_views,
    }
    values.write(
        FIGURES.parent / "values.toml", manuscript, "notebooks/manuscript_figures.py"
    )
    mo.ui.table(
        [
            {"value": f"{s}.{k}", "printed": v}
            for s, keys in manuscript.items()
            for k, v in keys.items()
        ],
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
