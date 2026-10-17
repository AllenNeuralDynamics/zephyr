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
        value=os.environ.get(test_sets.ROOT_ENV, "C:/Data/zephyr-local-v2"),
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
def _(df, info, network):
    per_clip = test_sets.clip_table(df, info, network.value)
    return (per_clip,)


@app.cell
def _():
    mo.md("""
    ## Thermistor quality
    """)
    return


@app.cell
def _(per_clip, root):
    test_sets.qc_figure(test_sets.qc_table(root, per_clip))
    return


@app.cell
def _(per_clip, root):
    qc = test_sets.qc_table(root, per_clip)
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
        {c: "median" for c in [*QC_FEATURES, "r"]}
        | {k: "first" for k in ("subject_id", "rig", "set", "date")}
    )
    sessions["date"] = sessions["date"].astype(str)
    axis_x = mo.ui.dropdown(QC_FEATURES, value="snr_db", label="X")
    axis_y = mo.ui.dropdown(QC_FEATURES, value="amplitude", label="Y")
    axis_z = mo.ui.dropdown(QC_FEATURES, value="break_frac", label="Z")
    color_by = mo.ui.dropdown(
        ["r", "subject_id", "rig", "set", "date"], value="r", label="Colour"
    )
    mo.hstack([axis_x, axis_y, axis_z, color_by])
    return QC_FEATURES, axis_x, axis_y, axis_z, color_by, sessions


@app.cell
def _(QC_FEATURES, sessions):
    import itertools

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    # Rank-regress session r on every triple of features; the best triples
    # are the ones whose thresholds separate good from bad sessions most cleanly.
    _ranks = np.column_stack([rankdata(sessions[c]) for c in QC_FEATURES])
    _ranks = (_ranks - _ranks.mean(0)) / _ranks.std(0)
    _y = rankdata(sessions["r"])
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
            columns=["x", "y", "z", "r2_vs_r"],
        )
        .sort_values("r2_vs_r", ascending=False)
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
        hover_data=["session", "subject_id", "rig", "date", "r"],
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
    ## Raw data
    """)
    return


@app.cell
def _(per_clip):
    _worst = per_clip.sort_values("r")
    clip = mo.ui.dropdown(
        {f"{s} (r {r:+.2f})": s for s, r in zip(_worst["stem"], _worst["r"])},
        value=f"{_worst['stem'].iloc[0]} (r {_worst['r'].iloc[0]:+.2f})",
        label="Clip (worst first)",
        full_width=True,
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
def _(per_clip):
    animal = mo.ui.dropdown(
        test_sets.animal_options(per_clip), value=None, label="Animal", full_width=True
    )
    excerpt = mo.ui.slider(
        1, 30, value=4, step=0.5, label="Excerpt (s)", show_value=True
    )
    mo.hstack([animal, excerpt])
    return animal, excerpt


@app.cell
def _(animal, excerpt, per_clip, network, root):
    mo.stop(animal.value is None, mo.md("Pick an animal."))
    test_sets.animal_rig_figure(
        root, network.value, per_clip, animal.value, excerpt.value
    )
    return


@app.cell
def _():
    mo.md("""
    ## Sessions below a threshold
    """)
    return


@app.cell
def _():
    low_below = mo.ui.number(0, 1, step=0.05, value=0.8, label="Session r below")
    low_which = mo.ui.dropdown(
        {"worse clip": 0, "better clip": 1}, value="worse clip", label="Clip"
    )
    mo.hstack([low_below, low_which])
    return low_below, low_which


@app.cell
def _(per_clip, low_below, network, root):
    low = test_sets.low_sessions(per_clip, low_below.value)
    mo.stop(low.empty, mo.md(f"No session under {low_below.value:g}."))
    _tr = test_sets.clip_traces(root, network.value, low["stems"].iloc[0][0])
    _span = float(_tr["times"][-1] - _tr["times"][0])
    low_window = mo.ui.range_slider(
        0,
        _span,
        value=[_span / 2, _span / 2 + 8],
        step=0.1,
        label=f"Start and stop (s) | {len(low)} sessions",
        show_value=True,
        full_width=True,
    )
    low_window
    return low, low_window


@app.cell
def _(low, low_which, low_window, network, root):
    _start, _stop = low_window.value
    test_sets.low_figure(
        root, network.value, low, _start, max(_stop - _start, 0.1), low_which.value
    )
    return


@app.cell
def _():
    mo.md("""
    ## Thumbnail similarity
    """)
    return


@app.cell
def _():
    view = mo.ui.dropdown(
        test_sets.VIEWS, value="face crop (what the network sees)", label="Image"
    )
    view
    return (view,)


@app.cell
def _(per_clip, low_below, root, view):
    sim_table = test_sets.nearest_train(root, view.value, per_clip)
    test_sets.similarity_figure(sim_table, low_below.value)
    return (sim_table,)


@app.cell
def _():
    mo.md("""
    ## Most alike training thumbnails
    """)
    return


@app.cell
def _(low, root, sim_table, view):
    test_sets.nearest_figure(root, view.value, sim_table, low)
    return


if __name__ == "__main__":
    app.run()
