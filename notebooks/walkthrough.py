import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from utils import breathing, plots, results, style, tables, thermistor

    style.use_style()


@app.cell
def _():
    mo.md("""
    # Zephyr: from face video to breathing

    Zephyr recovers a mouse's breathing trace from face video alone. This notebook
    follows one test clip through the pipeline, compares every method on it, and
    reports the benchmark and two ablations. It runs inference only; nothing is trained.

    The task and the dataset come from the Breathing CodaBench challenge: see the
    [challenge repository](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge)
    and its [documentation site](https://allenneuraldynamics.github.io/breathing-codabench-challenge/)
    for the dataset, the evaluation protocol and how to submit.
    """)
    return


@app.cell
def _():
    videos = {p.stem: p for p in sorted((mo.notebook_dir() / "figures").glob("*.mp4"))}
    video = mo.ui.dropdown(videos, value=next(iter(videos), None), label="Video")
    video
    return (video,)


@app.cell
def _(video):
    mo.video(src=str(video.value), controls=True) if video.value else mo.md(
        "No videos in `notebooks/figures/`."
    )
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
        list(clips), value="face_3_part_2", label="Test clip"
    )
    start: mo.ui.slider = mo.ui.slider(
        0, 285, step=5, value=30, label="Excerpt start (s)", debounce=True
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
    mo.md(r"""
    ## 1. The data

    A face video, cropped to the box around the nose (orange), and a thermistor
    in the nostril as ground truth. The network sees the $96 \times 96$ crop as four
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
    mo.md(r"""
    ## 3. Every method on the same clip

    Five methods turn the same face video into a breathing trace. Two are
    non-learned (Pixel, Facemap-style ridge), two are published video networks
    refitted to mouse breathing (TS-CAN, PhysNet), and one is Zephyr. All see the
    same $96 \times 96$ nose crop at 60 Hz and are scored against the thermistor.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Zephyr

    1. **Input**: four channels per frame (gray, frame difference, optical flow $x$ and $y$),
       so the network never has to compute motion itself.
    2. **Frame encoder** (spatial only): four stride-2 convolutions, then a $2 \times 2$
       pooled grid (not a global mean, so opposite motion in different parts of the
       nose cannot cancel) and a linear layer give one vector in $\mathbb{R}^{128}$ per frame.
    3. **Time**: the vectors are interpolated onto a fixed 60 Hz grid using the real
       frame timestamps.
    4. **TCN** (temporal only): six residual blocks with dilations $1, 2, 4, \dots, 32$ see about
       2 s of context each side, with symmetric padding (offline, non-causal).
    5. **Pooled branch**: the TCN's output averaged over pairs of frames (30 Hz),
       a convolution, and a learned upsampling back to 60 Hz. Averaging removes
       the frame-to-frame noise of an encoder that sees each frame alone, so the
       trace comes out smooth (`signal_pool = 2`).
    6. **Two heads**: the breathing trace, read from the pooled branch, and the
       probability that an inhalation starts, read from the full-rate and the
       pooled features together (`onset_input = "both"`), so it keeps frame-level
       timing. Trained with a multi-scale correlation loss on the trace plus a
       weighted cross-entropy on the onset head. Inhale events come from that head.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Crop<br/>T x 4 x 96 x 96<br/>gray, diff, flow x, flow y"] --> B
        subgraph B["Frame encoder (per frame)"]
            direction TB
            B1["4 x Conv2d, stride 2<br/>32, 64, 96, 128 ch<br/>BatchNorm + GELU"] --> B2["Avg-pool to 2 x 2<br/>Linear, 128-d"]
        end
        B --> C["Resample to 60 Hz grid<br/>(frame timestamps)"]
        C --> D
        subgraph D["Temporal net (TCN)"]
            direction TB
            D1["1x1 Conv1d, 128 ch"] --> D2["6 x residual block<br/>2 x dilated Conv1d, k = 3<br/>dilation 1, 2, 4, 8, 16, 32"]
        end
        D --> P
        subgraph P["Pooled branch"]
            direction TB
            P1["Avg-pool x 2 (30 Hz)"] --> P2["Conv1d, k = 3"] --> P3["ConvTranspose1d<br/>back to 60 Hz"]
        end
        P --> E["Signal head<br/>1x1 Conv1d"]
        P --> F["Onset head<br/>1x1 Conv1d"]
        D -- "full rate" --> F
        E --> G(["Breathing trace"])
        F --> H(["P(inhale onset)"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Are the frame embeddings smooth?

    Step 3 interpolates the frame embeddings ($\mathbb{R}^{128}$) onto the 60 Hz grid, which
    assumes they change smoothly from frame to frame. Below, the embeddings over
    the first 60 s of the clip chosen at the top. (a, b) Three seconds of every
    dimension under the thermistor. (c) Their average spectrum against the
    filtered thermistor's (the training target), both scaled to unit area.
    (d) Per dimension, the $R^2$ of rebuilding each frame by linear interpolation
    between its two neighbours: $R^2 = 1$ is smooth, $R^2 = 0$ is no better than
    the mean, and the same frames in shuffled order give about $-0.5$.
    """)
    return


@app.cell
def _(test_clip: results.TestClip):
    _times, _emb = results.frame_embeddings(test_clip.entry)  # cpu, cached
    _truth: pd.DataFrame = results.filtered_truth(test_clip.entry)
    _reference: NDArray = np.interp(
        _times, _truth["Time"].to_numpy(), _truth["Signal"].to_numpy()
    )
    _shuffled: NDArray = _emb[np.random.default_rng(0).permutation(len(_emb))]
    plots.embedding_smoothness_figure(
        _times,
        _emb,
        _reference,
        {
            "Embeddings": results.mean_spectrum(_times, _emb),
            "Thermistor": results.mean_spectrum(_times, _reference),
        },
        {
            "Embeddings": results.interpolation_r2(_times, _emb),
            "Thermistor": results.interpolation_r2(_times, _reference),
            "Shuffled": results.interpolation_r2(_times, _shuffled),
        },
        _times[0] + 10,
        3,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### TS-CAN (Liu et al., 2020)

    A published two-stream network for pulse from video, refitted here to mouse
    breathing on the **gray channel only**.

    1. **Input**: the crop is shrunk to $36 \times 36$. A *motion* stream gets the
       normalised frame difference $\dfrac{I(t+1) - I(t)}{I(t+1) + I(t)}$; an
       *appearance* stream gets the gray frame.
    2. **Attention**: the appearance stream produces a soft spatial mask that
       re-weights the motion stream twice, so motion is read mostly where the
       nose is.
    3. **Temporal shift**: no temporal convolution. In every motion convolution a
       third of the channels are shifted one frame forward and a third one back,
       over segments of 10 frames.
    4. **Output**: it predicts the *derivative* of the breathing trace. Inference
       integrates it, removes slow drift (smoothness-priors detrend) and band-passes
       to 1-15 Hz. It has no onset head, so inhale events are found by DSP.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray crop, 36 x 36<br/>T frames"] --> M["Normalised frame difference"]
        A --> P["Appearance frame"]
        subgraph S["Two streams, shared 10-frame segments"]
            direction TB
            M --> M1["Conv + temporal shift<br/>32 ch, tanh"]
            P --> P1["Conv, 32 ch, tanh"]
            P1 -- "attention mask 1" --> M2
            M1 --> M2["x mask, avg-pool"]
            M2 --> M3["Conv + temporal shift<br/>64 ch, tanh"]
            P1 --> P2["avg-pool, Conv, 64 ch, tanh"]
            P2 -- "attention mask 2" --> M4
            M3 --> M4["x mask, avg-pool"]
        end
        M4 --> D["Dense 128, tanh<br/>Dense 1"]
        D --> E["Predicted derivative"]
        E --> F["Integrate, detrend,<br/>band-pass 1-15 Hz"]
        F --> G(["Breathing trace"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### PhysNet (Yu et al., 2019)

    A published 3-D convolutional encoder-decoder for pulse from video, also on the
    **gray channel only**.

    1. **Input**: the whole gray clip as a video volume ($T \times 96 \times 96$).
    2. **Encoder**: 3-D convolutions mix space and time together; max-pooling halves
       the space, and twice also the time (4x shorter).
    3. **Decoder**: two transposed convolutions along time restore the original
       frame count.
    4. **Output**: average over space, then a $1 \times 1$ convolution gives the breathing
       trace directly. No onset head, so inhale events are found by DSP.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray clip<br/>1 x T x 96 x 96"] --> B["Conv3d 1x5x5, 16 ch<br/>pool space"]
        B --> C["2 x Conv3d 3x3x3<br/>32, 64 ch<br/>pool space + time"]
        C --> D["2 x Conv3d 3x3x3, 64 ch<br/>pool space + time"]
        D --> E["2 x Conv3d 3x3x3, 64 ch<br/>pool space"]
        E --> F["2 x Conv3d 3x3x3, 64 ch"]
        F --> G["2 x ConvTranspose3d<br/>upsample time x 4"]
        G --> H["Mean over space<br/>1x1 Conv"]
        H --> I(["Breathing trace"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pixel (non-learned floor)

    No training and no thermistor. Mean optical flow over the crop gives one
    $(x, y)$ velocity per frame; both are band-passed to 1-15 Hz, projected on their
    dominant direction (first principal axis), integrated to a displacement and
    band-passed again. The sign is unknown, so it is fixed afterwards using only
    the traces themselves. Anything a network does better than this is not just
    "the nose moves".
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        P1["Optical flow<br/>over crop"] --> P2["Mean (x, y)<br/>per frame"]
        P2 --> P3["Band-pass,<br/>dominant axis"]
        P3 --> P4["Integrate,<br/>band-pass"]
        P4 --> P5(["Breathing trace"])
    """)
    return


@app.cell
def _():
    mo.md("""
    ### Facemap-style ridge (field-standard baseline)

    The way breathing is usually read out of mouse face video. Two 100-component
    SVD bases are fitted on training frames: the *motion* (absolute frame
    difference) and the *movie* (raw pixels). Each frame is projected on both,
    z-scored, copied at several time lags, and a ridge regression maps that to the
    thermistor. The ridge strength is chosen leaving one recording out. It is
    linear and has no memory beyond the lags; no onset head, so DSP finds events.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        F1["Frames"] --> F2["Motion + movie<br/>PCA, 100 comp. each"]
        F2 --> F3["z-score,<br/>time lags"]
        F3 --> F4["Ridge regression<br/>to thermistor"]
        F4 --> F5(["Breathing trace"])
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
def _():
    mean_table: pd.DataFrame = results.mean_scores()  # cached traces, scored here
    tables.table(mean_table.round(3).reset_index(names="method"), lower=["kl_ibi"])
    return


@app.cell
def _():
    band_r: pd.DataFrame = breathing.band_correlation_summary()
    band_f1: pd.DataFrame = breathing.event_f1_band_summary()
    fig3b = plots.band_comparison_figure(band_r, band_f1)
    fig3b
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
    tables.table(
        plots.benchmark_means(benchmark).round(3),
        lower=["KL-IBI (lower is better)"],
        skip=["runs"],
    )
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
    mo.md(r"""
    ## 6. Ablation: what the network uses

    For the full network, one $16 \times 16$ patch of one channel at a time is replaced by
    the training mean and the drop in event F1 is measured. A dark region is not
    needed, not necessarily uninformative: channels can compensate for each other.
    """)
    return


@app.cell
def _():
    mo.stop(
        not results.OCCLUSION.exists(),
        mo.callout(
            mo.md(
                f"Occlusion maps missing. Produce them with `{results.OCCLUSION_COMMAND}`"
            ),
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
    tables.table(
        stretch.round(3).reset_index(names="metric"),
        across_columns=[results.STRETCH, results.NO_STRETCH],
    )
    return


@app.cell
def _(start: mo.ui.slider, test_clip: results.TestClip, truth: pd.DataFrame):
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
    mo.md(r"""
    ## 8. Is breathing rate sampled evenly? (training set)

    Each breath's rate is $1 / \Delta t$, with $\Delta t$ the time to the next
    inhalation. (a) Share of each
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
    mo.md(r"""
    ## 9. The test set by breathing rate

    Every thermistor breath is binned by its own rate; rows are the two test groups
    and, for comparison, the training clips (in-sample: the network was fitted to
    them, so that row is a reference, not a score). One line per recording
    (colours restart in each row), pooled dashed.
    **Breaths detected**: share matched by an onset-head event within one frame
    (17 ms); dotted is chance, the same head events circularly shifted.
    **Local F1**: misses binned by the breath's rate, false events by the
    thermistor's rate at that time; from the head, and next to it from DSP on the
    trace (the only DSP column, kept as a comparison). **Waveform correlation**: Pearson $r$ of the
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


@app.cell
def _():
    mo.md(r"""
    ## 10. Out of distribution: the side camera

    A second rig films the face from the side: three sessions (13, 14, 15), two
    clips each, with the same nostril thermistor. Zephyr never trained on it.
    Below, two face test clips and two side clips: the full frame with the crop
    (orange), the $96 \times 96$ crop the network sees, and the filtered thermistor over
    the excerpt chosen at the top (same amplitude scale in every panel). The side
    view shows the nose in profile, and those mice breathe slower (about 2 Hz,
    against about 5 Hz on the face rig).
    """)
    return


@app.cell
def _(start: mo.ui.slider):
    _columns: dict[str, str] = {
        "face_6_part_1": "Face, held-out animal",
        "face_9_part_1": "Face, held-out session",
        "ood_side_right_13_part_1": "Side, session 13",
        "ood_side_right_15_part_1": "Side, session 15",
    }
    _all: dict[str, results.TestClip] = {
        **results.clips(),
        **results.clips(results.OOD_CLIPS),
    }
    _panels: list[plots.DomainPanel] = [
        plots.DomainPanel(
            title=f"{title}\n{name}",
            frame=results.raw_frame(_all[name], start.value),
            box=_all[name].box,
            thermistor=results.filtered_truth(_all[name].entry),
            rate_hz=results.breathing_rate(_all[name].entry),
        )
        for name, title in _columns.items()
    ]
    plots.domain_figure(_panels, start.value, 8)
    return


@app.cell
def _():
    mo.md(r"""
    ### The thermistor itself

    Six features of every thermistor recording, for the 32 training clips, the 24
    test clips and the 6 side clips. Each clip is placed by its distance from the
    training clips in robust $z$ (median and MAD of the training set; $\log_{10}$ for
    amplitude and resolution, which span decades); the grey band is the training
    clips' usual range ($\pm 2$ robust $z$). The side clips breathe about twice as
    slowly, with much more regular intervals, and a rate outside anything the
    network trained on. Signal to noise is the 0.5-15 Hz against the 25-100 Hz power
    of the filtered trace; breath size is each breath's peak-to-trough range.
    """)
    return


@app.cell
def _():
    thermistor_table: pd.DataFrame = thermistor.features()
    plots.thermistor_figure(
        thermistor.shift_from_train(thermistor_table),
        {k: label for k, (label, *_) in thermistor.FEATURES.items()},
    )
    return (thermistor_table,)


@app.cell
def _(thermistor_table: pd.DataFrame):
    mo.ui.table(thermistor.summary_table(thermistor_table), selection=None)
    return


@app.cell
def _():
    mo.md(r"""
    Four kinds of network, scored on what each could see (one dot per clip):

    1. **Face only**: the face benchmark network. Near-perfect on held-out face
       clips, at chance on every side clip.
    2. **Face + side 13, 14**: that network fine-tuned on the face data plus two
       side sessions. Still fine on the face, still at chance on held-out
       session 15: more data from the new rig does not transfer to another
       session of it.
    3. **Side 14, 15 only** and **Side 13, 14 only**: trained from scratch on two
       side sessions each, one session held out (13, then 15). Both fit the
       sessions they trained on (in-sample, a reference, not a score) yet are at
       chance on the held-out one.
    4. **PhysNet (DSP)**: PhysNet trained the same way, on the same two pairs of
       side sessions with the same training budget, on the gray channel as in the
       benchmark. It has no onset head, so its inhale F1 is DSP on its trace. It
       does not learn this rig at all: correlation $r \approx 0.05$ even on the
       sessions it trained on, against $r \approx 0.78$ for Zephyr. Not for lack of
       training: on the face rig the same PhysNet reaches $r = 0.67$ after one epoch
       and $0.95$ by epoch 50. Most likely the profile view's nostril motion is
       too small to find in raw gray pixels, where Zephyr is given frame
       difference and optical flow; this is untested.

    Face scores are the runs' own evaluations. Every side clip is scored here with
    the same per-clip scorer (`evaluate` refuses clips a network trained on, so
    trained-on clips cannot come from it), which compares prediction and
    thermistor at the same instants: two side clips start almost 1 s before their
    thermistor.
    """)
    return


@app.cell
def _():
    plots.ood_figure(results.ood_scores())  # cpu, cached
    return


@app.cell
def _():
    mo.md("""
    ### The side-only network by breathing rate

    The figure of §9 for the full Zephyr network trained on side sessions 14 and
    15 only (**Side 14, 15 only** above): the top row is the held-out session 13,
    the bottom row the sessions it trained on (in-sample, a reference, not a
    score). The rate bins start at 1 Hz rather than 2 Hz: these mice breathe at
    about 2 Hz, and a third to two thirds of each clip's breaths are slower than 2 Hz.
    """)
    return


@app.cell
def _():
    _bins = breathing.SIDE_BINS_HZ
    _side = breathing.side_tables(results.OOD_RUNS["Side 14, 15 only"])  # cpu, cached
    _b = _side["breaths"]
    plots.test_rates_figure(
        breathing.recall_by_bin(_b, bins=_bins),
        breathing.recall_by_bin(_b, "chance", bins=_bins),
        breathing.f1_by_bin(_b, _side["false_positives"], "head", bins=_bins),
        breathing.f1_by_bin(_b, _side["false_positives"], "DSP", bins=_bins),
        _side["windows"],
        breathing.share_by_bin(_b, ["recording"], within="stratum", bins=_bins),
        breathing.bin_centres(_bins),
        _bins,
    )
    return


@app.cell
def _():
    mo.md("""
    ## 11. Half the frame rate

    Can the network work from 30 Hz video? The same full network, trained from
    scratch with the CNN seeing every other frame (30 Hz instead of 60). The TCN, the
    heads and the scored output stay on the 60 Hz grid: the frame embeddings are
    interpolated up to it, so the two networks are scored on the same instants.
    """)
    return


@app.cell
def _():
    tables.table(
        results.frame_rate_scores().round(3).reset_index(names="metric"),
        across_columns=["Zephyr, 60 Hz frames", "Zephyr, 30 Hz frames"],
    )
    return


@app.cell
def _():
    rate_r, rate_f1 = breathing.frame_rate_summaries()  # cpu, cached
    plots.band_comparison_figure(
        rate_r, rate_f1, methods=list(breathing.FRAME_RATE_SERIES)
    )
    return


if __name__ == "__main__":
    app.run()
