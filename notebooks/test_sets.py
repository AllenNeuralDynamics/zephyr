import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo

    from utils import style, tables, test_sets

    style.use_style()


@app.cell
def _():
    mo.md("""
    # Test sets, clip by clip
    """)
    return


@app.cell
def _():
    root_text = mo.ui.text(
        value=os.environ.get(test_sets.ROOT_ENV, "C:/Data/zephyr-local"),
        label="Local folder",
        full_width=True,
    )
    root_text
    return (root_text,)


@app.cell
def _(root_text):
    root = Path(root_text.value)
    absent = test_sets.missing(root)
    mo.stop(
        bool(absent),
        mo.md("Missing from the folder:\n\n" + "\n".join(f"- `{p}`" for p in absent)),
    )
    found = list(test_sets.networks(root))
    pick = mo.ui.multiselect(found, value=found, label="Networks")
    pick
    return pick, root


@app.cell
def _(pick, root):
    names = list(pick.value)
    mo.stop(not names, mo.md("Pick a network."))
    df = test_sets.scores(root, names)
    info = test_sets.clip_info(root)
    return df, info, names


@app.cell
def _():
    mo.md("""
    ## Scores
    """)
    return


@app.cell
def _(df):
    tables.table(test_sets.summary(df).round(3), lower=["kl_ibi"], group="set", skip=[])
    return


@app.cell
def _():
    mo.md("""
    ## By video
    """)
    return


@app.cell
def _(df, names):
    test_sets.by_video_figure(df, names)
    return


@app.cell
def _(df, info, names):
    worst = (
        df[df["network"] == names[0]]
        .merge(info, on=["stem", "session"])
        .sort_values("correlation")
    )
    mo.ui.table(
        worst[
            ["stem", "set", "subject_id", "rig", "date", "correlation", "inhale_f1"]
            + list(test_sets.COVARIATES)
        ].round(4),
        selection=None,
        page_size=15,
    )
    return


@app.cell
def _():
    mo.md("""
    ## By animal
    """)
    return


@app.cell
def _(df, info, names):
    test_sets.by_animal_figure(df, info, names)
    return


@app.cell
def _():
    mo.md("""
    ## What goes with a bad clip
    """)
    return


@app.cell
def _(names):
    network = mo.ui.dropdown(names, value=names[0], label="Network")
    network
    return (network,)


@app.cell
def _(df, info, network):
    test_sets.covariates_figure(df, info, network.value)
    return


@app.cell
def _():
    mo.md("""
    ## Time offset between prediction and thermistor
    """)
    return


@app.cell
def _(network, root):
    done, total = test_sets.n_predicted(root, network.value)
    run_lags = mo.ui.run_button(
        label=f"Predict the remaining {total - done} of {total} clips ({network.value})"
    )
    run_lags if done < total else None
    return done, run_lags, total


@app.cell
def _(df, done, info, network, root, run_lags, total):
    mo.stop(
        done < total and not run_lags.value,
        mo.md(f"{done} of {total} clips predicted by {network.value}."),
    )
    lag_table = test_sets.lag_table(test_sets.lags(root, network.value), info, df)
    test_sets.lag_diagnosis_figure(lag_table)
    return (lag_table,)


@app.cell
def _():
    mo.md("""
    ## Thermistor quality
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(lag_table, root):
    test_sets.qc_figure(test_sets.qc_table(root, lag_table))
    return


@app.cell
def _(lag_table, root):
    qc = test_sets.qc_table(root, lag_table)
    QC_FEATURES = [
        "snr_db",
        "amplitude",
        "amp_cv",
        "break_frac",
        "outside_frac",
        "rate_cv",
        "mean_rate",
        "session_consistency_z",
        "session_good_frac",
    ]
    sessions = qc.groupby("session").agg(
        {c: "median" for c in [*QC_FEATURES, "r_best"]}
        | {k: "first" for k in ("subject_id", "rig", "set", "date")}
    )
    sessions["date"] = sessions["date"].astype(str)
    axis_x = mo.ui.dropdown(QC_FEATURES, value="snr_db", label="X")
    axis_y = mo.ui.dropdown(QC_FEATURES, value="amplitude", label="Y")
    axis_z = mo.ui.dropdown(QC_FEATURES, value="break_frac", label="Z")
    color_by = mo.ui.dropdown(
        ["r_best", "subject_id", "rig", "set", "date"], value="r_best", label="Colour"
    )
    mo.hstack([axis_x, axis_y, axis_z, color_by])
    return QC_FEATURES, axis_x, axis_y, axis_z, color_by, sessions


@app.cell
def _(QC_FEATURES, sessions):
    import itertools

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    # Rank-regress session r_best on every triple of features; the best triples
    # are the ones whose thresholds separate good from bad sessions most cleanly.
    _ranks = np.column_stack([rankdata(sessions[c]) for c in QC_FEATURES])
    _ranks = (_ranks - _ranks.mean(0)) / _ranks.std(0)
    _y = rankdata(sessions["r_best"])
    _y = (_y - _y.mean()) / _y.std()

    def _r2(idx):
        _x = _ranks[:, idx]
        _b = np.linalg.lstsq(_x, _y, rcond=None)[0]
        return 1 - ((_y - _x @ _b) ** 2).sum() / (_y**2).sum()

    triples = (
        pd.DataFrame(
            [
                (*(QC_FEATURES[i] for i in c), _r2(list(c)))
                for c in itertools.combinations(range(len(QC_FEATURES)), 3)
            ],
            columns=["x", "y", "z", "r2_vs_r_best"],
        )
        .sort_values("r2_vs_r_best", ascending=False)
        .round(3)
        .head(10)
    )
    triples
    return


@app.cell
def _(axis_x, axis_y, axis_z, color_by, sessions):
    import plotly.express as px

    _fig = px.scatter_3d(
        sessions.reset_index(),
        x=axis_x.value,
        y=axis_y.value,
        z=axis_z.value,
        color=color_by.value,
        hover_data=["session", "subject_id", "rig", "date", "r_best"],
        height=700,
    )
    _fig.update_traces(marker=dict(size=5))
    for _axis, _feature in zip(
        ("xaxis", "yaxis", "zaxis"), (axis_x.value, axis_y.value, axis_z.value)
    ):
        if _feature == "amplitude":
            _fig.update_layout(scene={_axis: dict(type="log")})
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## Edge cases
    """)
    return


@app.cell
def _(lag_table, network, root):
    cases = test_sets.edge_cases(lag_table)
    test_sets.edge_case_figure(root, network.value, cases)
    return (cases,)


@app.cell
def _():
    mo.md("""
    ## Raw data
    """)
    return


@app.cell
def _(cases, df, network):
    stems = sorted(df.loc[df["network"] == network.value, "stem"].unique())
    options = {label: stem for label, stem in cases.items()} | {s: s for s in stems}
    clip = mo.ui.dropdown(
        options, value=next(iter(cases)), label="Clip", full_width=True
    )
    clip
    return (clip,)


@app.cell
def _(clip, network, root):
    traces = test_sets.clip_traces(root, network.value, clip.value)
    duration = float(traces["times"][-1] - traces["times"][0])
    length = mo.ui.slider(1, 60, value=6, step=0.5, label="Window (s)", show_value=True)
    length
    return duration, length, traces


@app.cell
def _(duration, length):
    start = mo.ui.slider(
        0,
        max(0.0, duration - length.value),
        value=max(0.0, duration - length.value) / 2,
        step=0.1,
        label="Start (s)",
        show_value=True,
        full_width=True,
    )
    start
    return (start,)


@app.cell
def _(length, start, traces):
    test_sets.raw_figure(traces, start.value, length.value)
    return


@app.cell
def _():
    mo.md("""
    ## One animal, rig by rig
    """)
    return


@app.cell
def _(lag_table):
    animal = mo.ui.dropdown(
        test_sets.animal_options(lag_table), value=None, label="Animal", full_width=True
    )
    excerpt = mo.ui.slider(
        1, 30, value=4, step=0.5, label="Excerpt (s)", show_value=True
    )
    mo.hstack([animal, excerpt])
    return animal, excerpt


@app.cell
def _(animal, excerpt, lag_table, network, root):
    mo.stop(animal.value is None, mo.md("Pick an animal."))
    test_sets.animal_rig_figure(
        root, network.value, lag_table, animal.value, excerpt.value
    )
    return


if __name__ == "__main__":
    app.run()
