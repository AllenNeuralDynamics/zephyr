import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from numpy.typing import NDArray

    from utils import plots, results, style
    from zephyr.evaluate import head_event_indices

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
def _(start):
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
def _(now, start, test_clip, traces, truth):
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
    that an inhalation starts at each time (the onset head). Events are read from
    the trace, or from the head, and scored against the thermistor within 17 ms.
    """)
    return


@app.cell
def _(start: mo.ui.slider, traces: pd.DataFrame, truth: pd.DataFrame):
    t0: float = start.value
    t1: float = start.value + 10
    times: NDArray = traces["Time"].to_numpy()
    truth_on, _ = results.events(truth["Time"].to_numpy(), truth["Signal"].to_numpy())
    trace_on, _ = results.events(times, traces["Zephyr"].to_numpy())
    head_on: NDArray = times[head_event_indices(traces["onset"].to_numpy(), times)]

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
    plots.mark_events(top, inside(trace_on), 2.8, style.color("Zephyr"), "^")
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
    one deterministic run); bars are means. Zephyr is scored twice on the event
    metrics: with events detected on its trace, and with events from its onset head (green).
    """)
    return


@app.cell
def _():
    benchmark: pd.DataFrame = results.benchmark()
    plots.benchmark_figure(benchmark)
    return


@app.cell
def _():
    mo.md("""
    ## 5. Ablation: inputs and objective

    Each input set is trained with and without the onset-event objective, five
    networks each. Inhalations can be detected in two ways: by signal processing on
    the predicted trace (middle), or from the onset head's probability (right). The
    head's F1 is far higher, and flow improves both.
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
    occlusion: dict[str, NDArray] = results.occlusion()
    frames: dict[str, NDArray] = results.mean_frames()
    plots.channel_importance_figure(occlusion, frames)
    return frames, occlusion


@app.cell
def _(frames, occlusion):
    plots.importance_by_clip_figure(occlusion, frames)
    return


if __name__ == "__main__":
    app.run()
