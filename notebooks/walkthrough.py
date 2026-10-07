import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from numpy.typing import NDArray

    from utils import breathing, plots, results, style

    style.use_style()


@app.cell
def _():
    mo.md("""
    # Zephyr: from face video to breathing

    Zephyr recovers a mouse's breathing trace from face video alone. This notebook
    follows one test clip through the pipeline, compares every method on it, and
    reports the benchmark and two ablations. It runs inference only; nothing is trained.
    """)
    return


@app.cell
def _():
    missing: dict[str, str] = results.missing()
    mo.stop(
        bool(missing),
        mo.callout(
            mo.md(
                "Missing inputs. Produce each with:\n\n"
                + "\n".join(f"- **{k}**: `{v}`" for k, v in missing.items())
            ),
            kind="danger",
        ),
    )
    clips: dict[str, results.TestClip] = results.clips()
    return (clips,)


@app.cell
def _(clips: dict[str, results.TestClip]):
    clip: mo.ui.dropdown = mo.ui.dropdown(
        list(clips), value="face_6_part_1", label="Test clip"
    )
    start: mo.ui.slider = mo.ui.slider(
        0, 285, step=5, value=60, label="Excerpt start (s)", debounce=True
    )
    mo.hstack([clip, start], justify="start")
    return clip, start


@app.cell
def _(clip: mo.ui.dropdown, clips: dict[str, results.TestClip]):
    test_clip: results.TestClip = clips[clip.value]
    truth: pd.DataFrame = results.truth(test_clip.entry)
    traces: pd.DataFrame = results.method_traces(test_clip.entry)  # cpu, cached
    scores: pd.DataFrame = results.method_scores(test_clip.entry, traces)
    return scores, test_clip, traces, truth


@app.cell
def _():
    mo.md("""
    ## 1. The data

    A face video, cropped to the box around the nose (orange), and a thermistor
    in the nostril as ground truth. The network sees the 96 x 96 crop as four
    channels: gray level, frame difference, and horizontal and vertical optical flow.
    Drag the time slider to scrub through the excerpt: the video and the four
    channels follow the cursor on the traces.
    """)
    return


@app.cell
def _(start: mo.ui.slider):
    now: mo.ui.slider = mo.ui.slider(
        start.value,
        start.value + 10,
        step=0.05,
        value=start.value + 2,
        label="Time (s)",
        full_width=True,
    )
    now
    return (now,)


@app.cell
def _(
    now: mo.ui.slider,
    start: mo.ui.slider,
    test_clip: results.TestClip,
    traces: pd.DataFrame,
    truth: pd.DataFrame,
):
    t: float = now.value
    fig1 = plots.scrubber(
        results.raw_frame(test_clip, t),
        results.channel_frames(test_clip.entry, t),
        test_clip.box,
        t,
        traces,
        truth,
        start.value,
        10,
    )
    fig1
    return


@app.cell
def _():
    mo.md("""
    ## 2. The network

    The network maps the channel stack to a breathing trace, and to the probability
    that an inhalation starts at each time (the onset head). Inhale events are read
    from the head, never from the trace, and scored against the thermistor's within
    17 ms; the triangles on both panels are the head's events.
    """)
    return


@app.cell
def _(start: mo.ui.slider, traces: pd.DataFrame, truth: pd.DataFrame):
    t0: float = start.value
    t1: float = start.value + 10
    truth_on, _ = results.events(truth["Time"].to_numpy(), truth["Signal"].to_numpy())
    head_on: NDArray = results.head_events(traces)

    def inside(x: NDArray) -> NDArray:
        return x[(x >= t0) & (x <= t1)]

    shown: pd.DataFrame = traces[(traces.Time >= t0) & (traces.Time <= t1)]
    shown_truth: pd.DataFrame = truth[(truth.Time >= t0) & (truth.Time <= t1)]

    fig2, (top, bottom) = style.figure(
        "double", 0.36, nrows=2, sharex=True, height_ratios=[2, 1]
    )
    z: pd.Series = (
        shown_truth.Signal - shown_truth.Signal.mean()
    ) / shown_truth.Signal.std()
    top.plot(shown_truth.Time, z, color=style.TRUTH, label="Thermistor")
    top.plot(shown.Time, shown.Zephyr, color=style.color("Zephyr"), label="Zephyr")
    plots.mark_events(top, inside(truth_on), 3.3, style.TRUTH)
    plots.mark_events(top, inside(head_on), 2.8, style.color("Zephyr"), "^")
    top.set_ylabel("Breathing (z-score)")
    top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    plots.onset_head(bottom, shown.Time, shown.onset)
    plots.mark_events(bottom, inside(truth_on), 1.1, style.TRUTH)
    plots.mark_events(bottom, inside(head_on), 1.02, style.OBJECTIVES[plots.HEAD], "^")
    bottom.set_xlim(t0, t1)
    bottom.set_xlabel("Time (s)")
    plots.panel_letters([top, bottom])
    fig2
    return


@app.cell
def _():
    mo.md("""
    ## 3. Every method on the same clip
    """)
    return


@app.cell
def _(
    scores: pd.DataFrame,
    start: mo.ui.slider,
    traces: pd.DataFrame,
    truth: pd.DataFrame,
):
    fig3, ax3 = style.figure("double", 0.42)
    plots.trace_stack(ax3, traces, truth, scores, start.value, 10)
    fig3
    return


@app.cell
def _(scores: pd.DataFrame):
    table: pd.DataFrame = scores.round(3).reset_index(names="method")
    mo.ui.table(table, selection=None)
    return


@app.cell
def _():
    mo.md("""
    ## 4. Benchmark

    Every method on the 24 held-out clips. Each dot is one trained network (or
    one deterministic run); bars are means. Zephyr's inhale events come from its
    onset head; the methods without a head are scored with DSP on their trace, and
    exhale F1 is DSP for all (tick labels say which).
    """)
    return


@app.cell
def _():
    benchmark: pd.DataFrame = results.benchmark()
    plots.benchmark_figure(benchmark)
    return (benchmark,)


@app.cell
def _(benchmark: pd.DataFrame):
    mo.ui.table(plots.benchmark_means(benchmark).round(3), selection=None)
    return


@app.cell
def _():
    mo.md("""
    ## 5. Ablation: inputs and objective

    Each input set is trained with and without the onset-event objective, five
    networks each. Inhalations can be detected in two ways: by signal processing on
    the predicted trace (middle), or from the onset head's probability (right). The
    head's F1 is far higher, and flow improves both. The middle column is the one
    place inhale events come from DSP on a predicted trace: it is the comparison
    this ablation is for, and the signal-only networks have no trained head.
    """)
    return


@app.cell
def _():
    ablation: pd.DataFrame = results.ablation()
    plots.ablation_figure(ablation)
    return


@app.cell
def _():
    mo.md("""
    ## 6. Ablation: what the network uses

    For the full network, one 16 x 16 patch of one channel at a time is replaced by
    the training mean and the drop in event F1 is measured. A dark region is not
    needed, not necessarily uninformative: channels can compensate for each other.
    """)
    return


@app.cell
def _():
    mo.stop(
        not results.OCCLUSION.exists(),
        mo.callout(
            mo.md(f"Occlusion maps missing. Produce them with `{results.OCCLUSION_COMMAND}`"),
            kind="warn",
        ),
    )
    occlusion: dict[str, NDArray] = results.occlusion()
    frames: dict[str, NDArray] = results.mean_frames()
    plots.channel_importance_figure(occlusion, frames)
    return frames, occlusion


@app.cell
def _(frames: dict[str, NDArray], occlusion: dict[str, NDArray]):
    plots.importance_by_clip_figure(occlusion, frames)
    return


@app.cell
def _():
    mo.md("""
    ## 7. Ablation: temporal stretch augmentation

    Does temporal aug even help?
    """)
    return


@app.cell
def _():
    stretch: pd.DataFrame = results.stretch_summary()
    stretch["no stretch minus stretch"] = (
        stretch[results.NO_STRETCH] - stretch[results.STRETCH]
    )
    mo.ui.table(stretch.round(3).reset_index(names="metric"), selection=None)
    return


@app.cell
def _(
    start: mo.ui.slider,
    test_clip: results.TestClip,
    truth: pd.DataFrame,
):
    _stretch: pd.DataFrame = results.zephyr_outputs(  # the stretch network, as "Zephyr"
        test_clip.entry, results.REFERENCE_RUN / "seed-42"
    )
    variant: pd.DataFrame = results.no_stretch_traces(test_clip.entry)  # cpu, cached
    inhales: dict[str, NDArray] = {
        "Truth": results.events(truth["Time"].to_numpy(), truth["Signal"].to_numpy())[
            0
        ],
        "Zephyr": results.head_events(_stretch),
        results.NO_STRETCH: results.head_events(variant),
    }
    plots.variant_traces_figure(_stretch, variant, truth, inhales, start.value, 10)
    return


@app.cell
def _():
    mo.md("""
    ## 8. Is breathing rate sampled evenly? (training set)

    Each breath's rate is 1 / (time to the next inhalation). (a) Share of each
    recording's breaths per rate bin; the pooled share of *time* (dotted) is what
    random training windows see, since fast breaths are short. (b) The same per
    video, part 2 dashed: one colour per recording, so the spread between a pair is
    the variation within a recording. (c) Power spectrum of each video's training
    target. `rate_balance` in a fold's `[train_params]` draws windows evenly over
    their true rate instead (bins 2-15 Hz by default). (d) below is what the
    plotted network was trained on: windows drawn with its own `rate_balance`.
    """)
    return


@app.cell
def _():
    _breaths: pd.DataFrame = breathing.train_breaths()
    plots.training_rates_figure(
        breathing.share_by_bin(_breaths, ["recording"]),
        breathing.share_by_bin(_breaths, ["video", "recording"]),
        breathing.train_spectra(),
        breathing.time_share_by_bin(_breaths),
        breathing.PLOT_BINS_HZ,
    )
    return


@app.cell
def _():
    plots.drawn_rates_figure(
        breathing.drawn_share_by_bin(0), breathing.drawn_share_by_bin()
    )
    return


@app.cell
def _():
    mo.md("""
    ## 9. The test set by breathing rate

    Every thermistor breath is binned by its own rate; rows are the two test groups
    and, for comparison, the training clips (in-sample: the network was fitted to
    them, so that row is a reference, not a score). One line per recording
    (colours restart in each row), pooled dashed.
    **Breaths detected**: share matched by an onset-head event within one frame
    (17 ms); dotted is chance, the same head events circularly shifted.
    **Local F1**: misses binned by the breath's rate, false events by the
    thermistor's rate at that time; from the head, and next to it from DSP on the
    trace (the only DSP column, kept as a comparison). **Waveform correlation**: Pearson r of the
    trace against the filtered thermistor in 3 s windows (dots), binned means as
    lines. **Rate distribution**: where each recording's breaths fall.
    """)
    return


@app.cell
def _():
    rate_tables: dict[str, pd.DataFrame] = breathing.all_tables()  # cpu, cached
    _b: pd.DataFrame = rate_tables["breaths"]
    plots.test_rates_figure(
        breathing.recall_by_bin(_b),
        breathing.recall_by_bin(_b, "chance"),
        breathing.f1_by_bin(_b, rate_tables["false_positives"], "head"),
        breathing.f1_by_bin(_b, rate_tables["false_positives"], "DSP"),
        rate_tables["windows"],
        breathing.share_by_bin(_b, ["recording"], within="stratum"),
        breathing.bin_centres(),
        breathing.PLOT_BINS_HZ,
    )
    return


if __name__ == "__main__":
    app.run()
