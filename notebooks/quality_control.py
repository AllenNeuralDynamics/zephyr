import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    from utils import plots, quality, style
    from zephyr.benchmarks import analysis

    style.use_style()


@app.cell
def _():
    mo.md("""
    # Quality control without a thermistor

    `zephyr quality-control` judges a prediction from the network's output alone: is
    the rhythm plausible, and does the onset head agree with the trace. This notebook
    checks those measures where the truth is known: on held-out animals, each
    measure against the clip's real inhale F1, per video and per window.

    Give a folder of reports (local or `s3://`, copied once into
    `notebooks/cache/quality-control/`) and, optionally, the reference it used;
    `ZEPHYR_QC_REPORTS` and `ZEPHYR_QC_REFERENCE` fill them in.
    """)
    return


@app.cell
def _():
    run = mo.ui.text(value=quality.REPORTS, label="Reports folder", full_width=True)
    ref = mo.ui.text(value=quality.REFERENCE, label="Reference file", full_width=True)
    mo.vstack([run, ref])
    return ref, run


@app.cell
def _(run):
    videos, windows = quality.tables(run.value)
    mo.stop(
        videos.empty,
        mo.callout(
            mo.md(
                f"No reports in `{run.value}`."
                if run.value
                else "Give a reports folder."
            ),
            kind="danger",
        ),
    )
    mo.stop(
        "truth_inhale_f1" not in videos,
        mo.callout(
            mo.md(
                "These videos have no thermistor, so there is nothing to check against."
            ),
            kind="warn",
        ),
    )
    judged = {m: b for m, b in quality.JUDGED.items() if m in videos}
    truth = "truth_inhale_f1"
    return judged, truth, videos, windows


@app.cell
def _():
    mo.md(r"""
    ## 1. The videos
    """)
    return


@app.cell
def _(judged, truth, videos):
    _columns = ["stem", "duration_s", truth, "truth_correlation", *judged]
    mo.ui.table(videos[_columns].round(3), selection=None, page_size=10)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Each measure against the true inhale F1, per video
    """)
    return


@app.cell
def _(judged, truth, videos):
    plots.quality_scatter_figure(videos, list(judged), truth, "True inhale F1")
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Which measures rank predictions as they should
    """)
    return


@app.cell
def _(judged, truth, videos, windows):
    ranks = {
        "Videos": analysis.quality_agreement(videos, judged, truth),
        "Windows": analysis.quality_agreement(windows, judged, truth),
    }
    mo.vstack(
        [
            plots.quality_agreement_figure(ranks),
            mo.ui.table(
                ranks["Videos"]
                .merge(ranks["Windows"], on="measure", suffixes=(" videos", " windows"))
                .round(3),
                selection=None,
            ),
        ]
    )
    return (ranks,)


@app.cell
def _():
    mo.md(r"""
    ## 4. Keeping the best videos
    """)
    return


@app.cell
def _(judged, ranks, truth, videos):
    _best = ranks["Videos"].sort_values("oriented", ascending=False)["measure"].head(4)
    plots.retention_figure(
        {m: analysis.retention(videos, m, truth, judged[m]) for m in _best},
        analysis.retention(videos, truth, truth, "higher"),
        "true inhale F1",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. One video
    """)
    return


@app.cell
def _(judged, run, truth, videos):
    _worst = videos.sort_values(truth)["stem"].tolist()
    video = mo.ui.dropdown(_worst, value=_worst[0], label="Video (worst first)")
    shown = mo.ui.multiselect(
        list(judged),
        value=[
            m
            for m in ("expected_precision", "head_trace_f1", "flat_breath_frac")
            if m in judged
        ],
        label="Measures",
    )
    start = mo.ui.slider(
        0, 290, step=5, value=30, label="Excerpt start (s)", debounce=True
    )
    mo.hstack([video, shown, start], justify="start")
    return shown, start, video


@app.cell
def _(run, shown, start, video, windows):
    _traces = quality.traces(run.value, video.value)
    plots.quality_video_figure(
        _traces,
        windows[windows["stem"] == video.value],
        shown.value,
        start.value,
        start.value + 10,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. The reference
    """)
    return


@app.cell
def _(ref):
    _reference = quality.reference(ref.value)
    mo.stop(_reference is None, mo.md("No reference given."))
    plots.calibration_figure(_reference)
    return


if __name__ == "__main__":
    app.run()
