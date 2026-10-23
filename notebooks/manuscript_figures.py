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

    from utils import architecture, corpus_flow, plots, quality, runs, style
    from zephyr.benchmarks import analysis
    from zephyr.corpus.config import DatasetSpec

    style.use_style()

    FIGURES = Path(
        os.environ.get(
            "ZEPHYR_MANUSCRIPT_FIGURES", runs.REPO / "manuscript" / "figures"
        )
    )
    LAYOUT = tomllib.loads((FIGURES / "layout.toml").read_text(encoding="utf8"))
    # Quality-control reports of the reference checkpoint (Fig. 5).
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
    # The walkthrough's typical test-A clip: Zephyr's correlation on it is test A's
    # median (seed 17, 2026-10-19).
    stem = "behavior_828421_2025-11-21_17-41-34_span2_FaceCamera"
    truth = runs.truth(stem)
    zephyr = runs.prediction("Zephyr", stem)
    origin = float(zephyr["times"][0])
    return origin, stem, truth, zephyr


@app.cell
def fig1bc_md():
    mo.md("""
    ## Fig. 1b, c: sniffing in the task

    One session of a new animal, predicted by the fold that holds the animal out.
    b: both clips' breathing rate and speed around odor onset, apart for sites where
    the animal stopped and was rewarded, stopped and was not (unrewarded), or ran past
    (skipped: the task saw no stop), and breathing rate against speed; c: a minute of
    it, with licks, rewards and odor sites. The behaviour comes from the session's processed
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
def fig1bc_data():
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
    _lags = np.arange(-2.0, 5.0 + 1e-9, 1 / 60)
    task_clips, _rows = [], {}
    for _span in (1, 2):
        _stem = f"{_session}_span{_span}_FaceCamera"
        _p, _t = runs.prediction(_name, _stem), runs.truth(_stem)
        _g = _p["times"]
        _clip = pd.DataFrame(
            {
                "time": _g,
                "speed": np.interp(_g, _speed.index.to_numpy(), _speed.to_numpy()),
                "video": analysis.breathing_rate(_p["head"], _g),
                "thermistor": analysis.breathing_rate(_t["inhale"], _g),
            }
        )
        task_clips.append(_clip)
        # odor onsets with the whole window inside the clip
        _inside = odor_sites[
            odor_sites["odor"].between(_g[0] - _lags[0], _g[-1] - _lags[-1])
        ]
        for _kind, _group in _inside.groupby("kind"):
            for _signal in ("video", "thermistor", "speed"):
                _rows.setdefault((_kind, _signal), []).append(
                    analysis.event_aligned(
                        _g, _clip[_signal].to_numpy(), _group["odor"].to_numpy(), _lags
                    )
                )
    _parts = []
    for (_site, _signal), _blocks in _rows.items():
        _x = np.concatenate(_blocks)
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
    return lick_times, odor_sites, task_aligned, task_clips


@app.cell
def fig1b(task_aligned, task_clips):
    fig1b = plots.task_summary_panel(
        *size("fig1", "b"),
        task_aligned,
        pd.concat(task_clips, ignore_index=True),
        1 / float(np.median(np.diff(task_clips[0]["time"]))),
    )
    plots.save_panel(fig1b, FIGURES / "fig1" / "b.svg")
    _counts = task_aligned.groupby(["site", "signal"])["n"].first()
    mo.vstack([fig1b, mo.ui.table(_counts.reset_index(), selection=None)])
    return


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
    _net = runs.network("Zephyr")[0]
    _hz = runs.clips()[stem][1].output_fs
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
    same test-A clip around its largest change in breathing rate: the network's trace
    over the thermistor and its onset probability, with the thermistor's inhales
    (black) and exhales (grey) and the head's inhales (blue).
    """)
    return


@app.function
def rate_change(truth: dict, origin: float, window: float) -> float:
    """Start (s from the clip's origin) of the *window* with the largest change in
    breathing rate: breaths in its second half minus its first."""
    inhales = truth["inhale"] - origin

    def change(t0: float) -> int:
        first = np.count_nonzero((inhales >= t0) & (inhales < t0 + window / 2))
        second = np.count_nonzero(
            (inhales >= t0 + window / 2) & (inhales < t0 + window)
        )
        return abs(second - first)

    return float(max(np.arange(0.0, inhales[-1] - window, 0.5), key=change))


@app.cell
def fig1e(origin, stem, truth, zephyr):
    _window = 8.0
    _t0 = rate_change(truth, origin, _window)
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
    folds = pd.read_parquet(dataset / "folds.parquet")
    return dataset, folds, spec


@app.cell
def fig2a(dataset, spec):
    # From the dataset version the corpus starts at, the hand-picked ephys sessions
    # joining it there.
    _steps = corpus_flow.shorten(corpus_flow.flow(dataset, spec)[1:], sets=False)
    _steps[0]["name"] = "recorded"
    fig2a = corpus_flow.flow_panel(*size("fig2", "a"), _steps)
    plots.save_panel(fig2a, FIGURES / "fig2" / "a.svg")
    fig2a
    return


@app.cell
def fig2b(folds):
    # Every thermistor breath's rate over the corpus, and per animal.
    fig2b = plots.corpus_rate_panel(
        *size("fig2", "b"), runs.animal_breath_rates(), step=0.5
    )
    plots.save_panel(fig2b, FIGURES / "fig2" / "b.svg")
    fig2b
    return


@app.cell
def fig2c(folds):
    fig2c = plots.corpus_animals_panel(*size("fig2", "c"), folds[folds["set"].notna()])
    plots.save_panel(fig2c, FIGURES / "fig2" / "c.svg")
    fig2c
    return


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
    _folds = pd.read_parquet(runs.root() / "dataset" / "folds.parquet")
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
def fig3a(methods, origin, stem, truth):
    # The same clip as Fig. 1e, 10 s of it starting just before the rate change.
    _scores = (
        runs.scores(methods)
        .query("stem == @stem")
        .groupby("method")["correlation"]
        .mean()
        .to_dict()
    )
    _t0 = float(np.floor(rate_change(truth, origin, 8.0))) + 1.0
    fig3a = plots.trace_stack_panel(
        *size("fig3", "a"),
        {n: runs.prediction(n, stem) for n in methods},
        truth,
        _scores,
        origin,
        _t0,
        6.0,
    )
    plots.save_panel(fig3a, FIGURES / "fig3" / "a.svg")
    fig3a
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
    mo.vstack([fig3c, fig3d, fig3e, mo.ui.table(_rate.round(3), selection=None)])
    return nets, whole


@app.cell
def table1(by_fold, cv_methods, fold_tables, methods, on_cpu, test_a, timing):
    # Table 1 (benchmark): per method on new animals, each score averaged within each
    # unit, then over the units of a fold, then over folds (mean and 95% interval;
    # methods without cross-validation runs, the published split alone). Rate metrics
    # from 10-s windows. Written as CSV for main.typ.
    _a = by_fold[by_fold["set"] == test_a]
    _windows = fold_tables(methods, ("windows",))["windows"].query("set == @test_a")
    _unit = _a.drop_duplicates(["fold", "session"]).set_index(["fold", "session"])[
        "unit"
    ]
    _windows["unit"] = [
        _unit.get((f, st.rsplit("_span", 1)[0]))
        for f, st in zip(_windows["fold"], _windows["stem"], strict=True)
    ]
    _rates = analysis.rate_agreement(
        _windows.dropna(subset=["unit"]), by=["method", "fold", "unit"]
    ).reset_index()
    _rates[["MAE (Hz)", "RMSE (Hz)"]] *= 60  # breaths per minute
    _columns = {
        "MAE (Hz)": ("MAE", 1, _rates),
        "RMSE (Hz)": ("RMSE", 1, _rates),
        "MAPE (%)": ("MAPE", 1, _rates),
        "r": ("rate_r", 2, _rates),
        "correlation": ("trace_r", 2, _a),
        "inhale": ("inhale_f1", 2, _a),
        "exhale_f1": ("exhale_f1", 2, _a),
    }
    _summary = {
        c: analysis.fold_scores(t, c, by=["method"]).set_index("method")
        for c, (_, _, t) in _columns.items()
    }

    def _cell(row, digits):
        mean = 0.0 if abs(row["mean"]) < 0.5 * 10**-digits else row["mean"]
        if pd.isna(row["low"]):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ({row['low']:.{digits}f}–{row['high']:.{digits}f})"

    _rows = []
    for _m in methods:
        _row = {
            "method": plots.DISPLAY.get(_m, _m),
            "splits": str(int(_summary["inhale"].loc[_m, "folds"])),
            "seconds": f"{timing.loc[_m, 's per clip (median)']:.2f}"
            + ("*" if _m in on_cpu else ""),
        }
        for _c, (_name, _digits, _) in _columns.items():
            _row[_name] = _cell(_summary[_c].loc[_m], _digits).replace("-0.", "−0.")
        _rows.append(_row)
    benchmark = pd.DataFrame(_rows)
    (FIGURES.parent / "tables").mkdir(exist_ok=True)
    benchmark.to_csv(FIGURES.parent / "tables" / "benchmark.csv", index=False)
    benchmark_diff = {
        c: _summary[c][["mean", "low", "high", "minus ref", "diff low", "diff high"]]
        for c in ("inhale", "correlation")
    }
    cv_counts = {
        "folds": int(_a.loc[_a["method"] == runs.REFERENCE, "fold"].nunique()),
        "animals": int(_a.loc[_a["method"] == runs.REFERENCE, "subject"].nunique()),
        "units": int(_a.loc[_a["method"] == runs.REFERENCE, "unit"].nunique()),
        "cv methods": cv_methods,
    }
    mo.vstack(
        [
            mo.md(f"{cv_counts}"),
            mo.ui.table(benchmark, selection=None),
            *(mo.ui.table(v.round(3).reset_index()) for v in benchmark_diff.values()),
        ]
    )
    return benchmark_diff, cv_counts


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
    fig4e = plots.tolerance_panel(*size("fig4", "e"), tolerance, _nets)
    plots.save_panel(fig4e, FIGURES / "fig4" / "e.svg")
    fig4e
    return (tolerance,)


@app.cell
def fig4c(test_a, with_unit):
    # Fig. 4c: Zephyr trained on fewer animals (all their sessions), and on fewer
    # sessions of all its animals (open), scored per unit on new animals (test A), the
    # same units for every run.
    _names = [runs.REFERENCE, *runs.available(runs.LEARNING_CURVE)]
    _trained = {n: runs.training(n) for n in _names}
    _every = _trained[runs.REFERENCE]
    _recs = with_unit(runs.scores(_names, "per_recording"))
    _groups = {"New animals": _recs["set"] == test_a}
    _rows = []
    for _group, _keep in _groups.items():
        for _metric in ["inhale", "correlation"]:
            # One run at a time: each has its own known units.
            _ci = pd.concat(
                analysis.bootstrap(
                    _recs[_keep & (_recs["method"] == _n)].assign(set=_group),
                    _metric,
                    reference=_n,
                    unit="unit",
                )
                for _n in _names
            )
            for _, _r in _ci.iterrows():
                _t = _trained[_r["method"]]
                _all = _every["subject"].isin(set(_t["subject"])).sum()
                _rows.append(
                    {
                        "group": _group,
                        "metric": _metric,
                        "method": _r["method"],
                        "animals": _t["subject"].nunique(),
                        "sessions": len(_t),
                        "full": len(_t) == _all,
                        "units": _r["units"],
                        **{k: _r[k] for k in ("mean", "low", "high")},
                    }
                )
    learning = pd.DataFrame(_rows)
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
    fig4c = plots.learning_curve_panel(
        *size("fig4", "c"),
        learning,
        {"New animals": style.color(test_a)},
    )
    plots.save_panel(fig4c, FIGURES / "fig4" / "c.svg")
    mo.vstack(
        [
            fig4c,
            mo.ui.table(learning.round(3), selection=None),
            mo.md("New animals (test A), by whether the rig was in training"),
            mo.ui.table(rig_coverage.round(3), selection=None),
        ]
    )
    return


@app.cell
def fig4df(methods, test_a):
    # Two test-A clips, neither an outlier: among clips whose correlation is at least
    # the median, the one with most near misses (predictions 25-100 ms off; d), around
    # its densest near misses; and, among the lowest tenth of clips by correlation, the
    # 2-s stretch in which every inhalation is found (no error, no excluded span, at
    # least 4 true inhalations) from the least confident onset output: the lowest
    # median onset probability at the head's events (f).
    _clips = (
        runs.scores(methods)
        .query("method == 'Zephyr' and set == @test_a")
        .sort_values("inhale")
        .set_index("stem")
    )
    _lags = runs.tables("Zephyr")["lags"].query("set == @test_a")
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
        p, t = runs.prediction("Zephyr", stem), runs.truth(stem)
        err = runs.errors("Zephyr", stem)["time"].to_numpy()
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
    _cases = {"d": _clips.loc[_near], "f": _clips.loc[_upper]}
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
        _errors = runs.errors("Zephyr", _stem)
        _p = runs.prediction("Zephyr", _stem)
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
    return


@app.cell
def fig5_md():
    mo.md(f"""
    ## Fig. 5: judging predictions without a thermistor

    The quality-control reports of the reference checkpoint on test sets A and B, and
    its reference (onset calibration on the validation animals), from
    `{QC}`; made with `zephyr quality-control reference` and `run`.
    """)
    return


@app.cell
def fig5_data():
    qc_videos, qc_windows = quality.tables(str(QC / "test"))
    mo.stop(qc_videos.empty, mo.md("No quality-control reports yet."))
    _sets = runs.clip_sets()
    qc_videos = qc_videos.assign(set=qc_videos["stem"].map(_sets))
    qc_windows = qc_windows.assign(set=qc_windows["stem"].map(_sets))
    judged = {m: b for m, b in quality.JUDGED.items() if m in qc_videos}
    truth_f1 = "truth_inhale_f1"
    return judged, qc_videos, qc_windows, truth_f1


@app.cell
def fig5a(test_a):
    # Fig. 5a: what our network's errors on new animals are, per fold, then over folds.
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
    fig5a = plots.error_profile_panel(*size("fig5", "a"), error_profile, _baseline)
    plots.save_panel(fig5a, FIGURES / "fig5" / "a.svg")
    mo.vstack(
        [
            fig5a,
            mo.ui.table(error_profile.round(3), selection=None),
            mo.md(f"Baselines: {_baseline}"),
        ]
    )
    return (error_profile,)


@app.cell
def fig5bc(judged, qc_videos, qc_windows, test_a, truth_f1):
    # Fig. 5b-c on new animals (test set A) only: the five measures that rank its
    # videos best, and the windows the three best at ranking windows keep.
    _videos = qc_videos[qc_videos["set"] == test_a]
    _windows = qc_windows[qc_windows["set"] == test_a]
    qc_ranks = {
        "Videos": analysis.quality_agreement(_videos, judged, truth_f1),
        "Windows": analysis.quality_agreement(_windows, judged, truth_f1),
    }
    _top = qc_ranks["Videos"].nlargest(5, "oriented")
    fig5b = plots.qc_agreement_panel(*size("fig5", "b"), {"Videos": _top})
    plots.save_panel(fig5b, FIGURES / "fig5" / "b.svg")
    _best = qc_ranks["Windows"].sort_values("oriented", ascending=False)["measure"]
    _fractions = tuple(np.linspace(0.05, 1, 20))
    _checks = [c for c in _windows if c.startswith("ok_")]
    _w = _windows.dropna(subset=[truth_f1])
    _pass = _w[_w[_checks].fillna(True).astype(bool).all(axis=1)]
    fig5c = plots.qc_retention_panel(
        *size("fig5", "c"),
        {
            m: analysis.retention(_w, m, truth_f1, judged[m], _fractions)
            for m in _best.head(3)
        },
        analysis.retention(_w, truth_f1, truth_f1, "higher", _fractions),
        (len(_pass) / len(_w), float(_pass[truth_f1].mean())),
    )
    plots.save_panel(fig5c, FIGURES / "fig5" / "c.svg")
    qc_kept = {
        "windows": len(_w),
        "pass every check": len(_pass) / len(_w),
        "F1 all": float(_w[truth_f1].mean()),
        "F1 passing": float(_pass[truth_f1].mean()),
        "F1 failing": float(_w.drop(_pass.index)[truth_f1].mean()),
    }
    mo.vstack(
        [
            mo.hstack([fig5b, fig5c]),
            mo.md(f"{qc_kept}"),
            mo.ui.table(qc_ranks["Videos"].round(3), selection=None),
            mo.ui.table(qc_ranks["Windows"].round(3), selection=None),
        ]
    )
    return


@app.cell
def fig5e(test_a, with_unit):
    # Fig. 5e: Zephyr trained differently, minus Zephyr, on new animals (test set A):
    # paired bootstrap over animal x rig units.
    _names = [runs.REFERENCE, *runs.available(runs.VARIANTS)]
    _recs = with_unit(runs.scores(_names, "per_recording"))
    _recs = _recs[_recs["set"] == test_a]
    variants = {
        m: analysis.bootstrap(_recs, m, unit="unit") for m in ("inhale", "correlation")
    }
    fig5e = plots.variants_panel(*size("fig5", "e"), variants)
    plots.save_panel(fig5e, FIGURES / "fig5" / "e.svg")
    mo.vstack(
        [fig5e, *(mo.ui.table(v.round(3), selection=None) for v in variants.values())]
    )
    return


@app.cell
def fig5d():
    _reference = quality.reference(str(QC / "reference.json"))
    mo.stop(_reference is None, mo.md("No reference yet."))
    fig5d = plots.calibration_panel(*size("fig5", "d"), _reference)
    plots.save_panel(fig5d, FIGURES / "fig5" / "d.svg")
    fig5d
    return


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


if __name__ == "__main__":
    app.run()
