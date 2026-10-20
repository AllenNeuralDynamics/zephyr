import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
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
def _():
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
def _():
    # The walkthrough's typical test-A clip: Zephyr's correlation on it is test A's
    # median (seed 17, 2026-10-19).
    stem = "behavior_828421_2025-11-21_17-41-34_span2_FaceCamera"
    truth = runs.truth(stem)
    zephyr = runs.prediction("Zephyr", stem)
    origin = float(zephyr["times"][0])
    return origin, stem, truth, zephyr


@app.cell
def _():
    mo.md("""
    ## Fig. 1c: the ground truth

    The crop at three moments of one breath (inhalation onset, halfway to exhalation,
    exhalation onset) above the thermistor, the moments marked.
    """)
    return


@app.cell
def _(origin, stem, truth):
    # The clip's slowest breath, so its phases are far enough apart to see.
    _inhales = truth["inhale"]
    _longest = int(np.argmax(np.diff(_inhales)))
    _onset, _next = float(_inhales[_longest]), float(_inhales[_longest + 1])
    _exhale = float(truth["exhale"][truth["exhale"] > _onset][0])
    _moments = [_onset, (_onset + _exhale) / 2, _exhale]
    _frame_t, _frames = runs.gray(stem)
    _picked = [
        min(int(np.searchsorted(_frame_t, t)), len(_frames) - 1) for t in _moments
    ]
    _crops = [np.asarray(_frames[i, 0]) for i in _picked]
    # Number each crop by its frame in the clip's own video (the network sees a
    # subset of the camera's frames, on the recipe's selection grid).
    _native = pd.read_parquet(runs.clips()[stem][0].timestamps)["Time"].to_numpy()
    _numbers = [int(np.argmin(np.abs(_native - _frame_t[i]))) for i in _picked]
    fig1c = plots.breath_phases_panel(
        *size("fig1", "c"),
        runs.thumbnail(stem),
        runs.box(stem),
        _crops,
        [f"Frame {n:,}" for n in _numbers],
        truth,
        origin,
        _onset - origin - 0.5,
        max(2.0, _next - _onset + 1.0),
        [t - origin for t in _moments],
    )
    plots.save_panel(fig1c, FIGURES / "fig1" / "c.svg")
    fig1c
    return


@app.cell
def _():
    mo.md("""
    ## Fig. 1d: the network

    Read off the loaded checkpoint, so the layer sizes are the model's own; the input
    is three consecutive crops of the same clip.
    """)
    return


@app.cell
def _(stem, truth):
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
def _():
    mo.md("""
    ## Fig. 1e: an example prediction

    Eight seconds of the same test-A clip around its largest change in breathing rate:
    the network's trace over the thermistor and its onset probability, with the
    thermistor's inhales (black) and the head's (blue).
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
def _(origin, truth, zephyr):
    _window = 8.0
    _t0 = rate_change(truth, origin, _window)
    fig1e = plots.prediction_panel(
        *size("fig1", "e"), truth, zephyr, origin, _t0, _window
    )
    plots.save_panel(fig1e, FIGURES / "fig1" / "e.svg")
    fig1e
    return


@app.cell
def _():
    mo.md("""
    ## Fig. 2: the corpus

    From the local dataset copy (`ZEPHYR_TEST_ROOT`/dataset) and the corpus spec:
    the selection flow, the breathing-rate mix and the split.
    """)
    return


@app.cell
def _():
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
def _(dataset, spec):
    # From the dataset version the corpus starts at, the hand-picked ephys sessions
    # joining it there.
    _steps = corpus_flow.shorten(corpus_flow.flow(dataset, spec)[1:])
    _steps[0]["name"] = "recorded"
    fig2a = corpus_flow.flow_panel(*size("fig2", "a"), _steps)
    plots.save_panel(fig2a, FIGURES / "fig2" / "a.svg")
    fig2a
    return


@app.cell
def _(folds):
    # Every thermistor breath's rate per set; test C (the 359 Hz sessions) is reported
    # in the Supplementary only.
    _rates = {k: v for k, v in runs.breath_rates().items() if k != runs.SETS["test_c"]}
    fig2b = plots.rate_distribution_panel(*size("fig2", "b"), _rates)
    plots.save_panel(fig2b, FIGURES / "fig2" / "b.svg")
    fig2b
    return


@app.cell
def _(folds):
    fig2c = plots.split_panel(*size("fig2", "c"), folds)
    plots.save_panel(fig2c, FIGURES / "fig2" / "c.svg")
    fig2c
    return


@app.cell
def _():
    mo.md("""
    ## Fig. 3: the benchmark

    Every method on test set A (new animals), as each run scored itself, and the
    walkthrough's clip-by-clip tables (onset lags, rate windows) for the networks.
    """)
    return


@app.cell
def _():
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
    recordings = runs.scores(methods, "per_recording")
    test_a = runs.SETS["test_a"]
    return methods, recordings, test_a


@app.cell
def _(methods, origin, stem, truth):
    # The same clip as Fig. 1e, 10 s of it starting just before the rate change.
    _scores = (
        runs.scores(methods)
        .query("stem == @stem")
        .groupby("method")["correlation"]
        .mean()
        .to_dict()
    )
    _t0 = float(np.floor(rate_change(truth, origin, 8.0))) - 1.0
    fig3a = plots.trace_stack_panel(
        *size("fig3", "a"),
        {n: runs.prediction(n, stem) for n in methods},
        truth,
        _scores,
        origin,
        _t0,
        10.0,
    )
    plots.save_panel(fig3a, FIGURES / "fig3" / "a.svg")
    fig3a
    return


@app.cell
def _(methods, recordings, test_a):
    _ci = {m: analysis.bootstrap(recordings, m) for m in ("correlation", "inhale")}
    # Inference time per clip (walkthrough harness): networks on this machine's GPU,
    # the pixel methods and Facemap in numpy on its CPU.
    _timing = runs.wall_time(methods).set_index("method")
    fig3b = plots.benchmark_panel(
        *size("fig3", "b"),
        recordings,
        _ci,
        methods,
        test_a,
        _timing["s per clip (median)"],
        [m for m in methods if not _timing.loc[m, "device"].startswith("NVIDIA")],
    )
    plots.save_panel(fig3b, FIGURES / "fig3" / "b.svg")
    fig3b
    return


@app.cell
def _(test_a):
    _nets = ["Zephyr", "PhysNet", "TS-CAN"]
    _ready = [n for n in _nets if runs.done(n)]
    mo.stop(not _ready, mo.md("Run `runs.tables` for the networks first."))
    _whole = {
        k: pd.concat([runs.tables(n)[k] for n in _ready], ignore_index=True)
        for k in ("lags", "windows")
    }
    _lags = _whole["lags"][_whole["lags"]["set"] == test_a]
    fig3c = plots.lag_panel(*size("fig3", "c"), _lags, _ready)
    plots.save_panel(fig3c, FIGURES / "fig3" / "c.svg")
    _windows = _whole["windows"][_whole["windows"]["set"] == test_a]
    fig3d = plots.bland_altman_panel(*size("fig3", "d"), _windows, "Zephyr")
    plots.save_panel(fig3d, FIGURES / "fig3" / "d.svg")
    mo.hstack([fig3c, fig3d])
    return


@app.cell
def _():
    mo.md("""
    ## Fig. 4: generalization and its limits

    New animals against new sessions, animal by animal, by breathing rate (d, the
    curve against training size, waits for its runs); and one failure
    of the worst test-A clip.
    """)
    return


@app.cell
def _(recordings, test_a):
    _shown = ["Facemap", "TS-CAN", "RhythmFormer", "PhysFormer", "PhysNet", "Zephyr"]
    fig4a = plots.sets_panel(
        *size("fig4", "a"),
        recordings,
        _shown,
        ["correlation", "inhale"],
        [test_a, runs.SETS["test_b"]],
    )
    plots.save_panel(fig4a, FIGURES / "fig4" / "a.svg")
    fig4b = plots.animals_panel(
        *size("fig4", "b"), recordings, ["Zephyr", "PhysNet"], "inhale", test_a
    )
    plots.save_panel(fig4b, FIGURES / "fig4" / "b.svg")
    mo.hstack([fig4a, fig4b])
    return


@app.cell
def _(test_a):
    _nets = ["Zephyr", "PhysNet", "TS-CAN"]
    _tables = {n: runs.tables(n) for n in _nets}
    whole = {
        k: pd.concat([t[k] for t in _tables.values()], ignore_index=True)
        for k in ("breaths", "false_positives", "lags")
    }
    whole = {k: v[v["set"] == test_a] for k, v in whole.items()}
    fig4c = plots.rate_band_panel(
        *size("fig4", "c"),
        analysis.f1_by_rate(
            whole["breaths"], whole["false_positives"], analysis.RATE_BINS_HZ[1:]
        ),
        _nets,
    )
    plots.save_panel(fig4c, FIGURES / "fig4" / "c.svg")
    fig4d = plots.tolerance_panel(
        *size("fig4", "d"),
        whole["lags"],
        whole["breaths"],
        whole["false_positives"],
        _nets,
    )
    plots.save_panel(fig4d, FIGURES / "fig4" / "d.svg")
    mo.hstack([fig4c, fig4d])
    return


@app.cell
def _(methods, test_a):
    # Two test-A clips, neither an outlier: among clips whose correlation is at least
    # the median, the one with most near misses (predictions 25-100 ms off; e), and the
    # clip at the 10th percentile of Zephyr's inhale F1 (g); each around its densest
    # near misses (e) or errors (g).
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
    _others = _clips.drop(_near)
    _tenth = _others.index[int(0.1 * len(_others))]
    _cases = {"e": _clips.loc[_near], "g": _clips.loc[_tenth]}
    _shown = []
    for _panel, _row in _cases.items():
        _stem = _row.name
        _errors = runs.errors("Zephyr", _stem)
        _p = runs.prediction("Zephyr", _stem)
        _origin = float(_p["times"][0])
        _t = (
            _errors.loc[_errors["near_miss"], "time"]
            if _panel == "e"
            else _errors["time"]
        ).to_numpy()
        _at = float(_t[np.argmax([np.sum(np.abs(_t - x) < 1.5) for x in _t])])
        _frame_t, _frames = runs.gray(_stem)
        _picked = dict.fromkeys(
            min(int(np.searchsorted(_frame_t, _at + dt)), len(_frames) - 1)
            for dt in np.linspace(-0.6, 0.6, 7)
        )
        _fig = plots.failure_panel(
            *size("fig4", _panel),
            runs.truth(_stem),
            _p,
            _errors,
            _origin,
            _at - _origin,
            [(_frame_t[i] - _origin, np.asarray(_frames[i, 0])) for i in _picked],
        )
        plots.save_panel(_fig, FIGURES / "fig4" / f"{_panel}.svg")
        _shown += [
            mo.md(f"**{_panel}**: {_stem}, inhale F1 {_row['inhale']:.3f}"),
            _fig,
        ]
    mo.vstack(_shown)
    return


@app.cell
def _():
    mo.md(f"""
    ## Fig. 5: judging predictions without a thermistor

    The quality-control reports of the reference checkpoint on test sets A and B, and
    its reference (onset calibration on the validation animals), from
    `{QC}`; made with `zephyr quality-control reference` and `run`.
    """)
    return


@app.cell
def _(test_a):
    qc_videos, qc_windows = quality.tables(str(QC / "test"))
    mo.stop(qc_videos.empty, mo.md("No quality-control reports yet."))
    _sets = runs.clip_sets()
    qc_videos = qc_videos.assign(set=qc_videos["stem"].map(_sets))
    qc_windows = qc_windows.assign(set=qc_windows["stem"].map(_sets))
    judged = {m: b for m, b in quality.JUDGED.items() if m in qc_videos}
    truth_f1 = "truth_inhale_f1"
    _a = qc_videos[qc_videos["set"] == test_a]
    # A test-A video of middling accuracy, so both passing and failing windows show.
    qc_stem = _a.sort_values(truth_f1)["stem"].iloc[len(_a) // 4]
    return judged, qc_stem, qc_videos, qc_windows, truth_f1


@app.cell
def _(qc_stem, qc_windows):
    _traces = quality.traces(str(QC / "test"), qc_stem)
    _w = qc_windows[qc_windows["stem"] == qc_stem]
    _shown = [
        m
        for m in ("expected_precision", "head_trace_f1", "flat_breath_frac")
        if m in _w
    ]
    fig5a = plots.qc_video_panel(*size("fig5", "a"), _traces, _w, _shown, 30, 40)
    plots.save_panel(fig5a, FIGURES / "fig5" / "a.svg")
    fig5a
    return


@app.cell
def _(judged, qc_videos, qc_windows, truth_f1):
    _ranks = {
        "Videos": analysis.quality_agreement(qc_videos, judged, truth_f1),
        "Windows": analysis.quality_agreement(qc_windows, judged, truth_f1),
    }
    fig5b = plots.qc_agreement_panel(*size("fig5", "b"), _ranks)
    plots.save_panel(fig5b, FIGURES / "fig5" / "b.svg")
    _best = _ranks["Windows"].sort_values("oriented", ascending=False)["measure"]
    _fractions = tuple(np.linspace(0.05, 1, 20))
    _checks = [c for c in qc_windows if c.startswith("ok_")]
    _w = qc_windows.dropna(subset=[truth_f1])
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
    mo.hstack([fig5b, fig5c])
    return


@app.cell
def _():
    _reference = quality.reference(str(QC / "reference.json"))
    mo.stop(_reference is None, mo.md("No reference yet."))
    fig5d = plots.calibration_panel(*size("fig5", "d"), _reference)
    plots.save_panel(fig5d, FIGURES / "fig5" / "d.svg")
    fig5d
    return


if __name__ == "__main__":
    app.run()
