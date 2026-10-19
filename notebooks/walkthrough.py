import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd

    from utils import plots, runs, style, tables
    from zephyr.benchmarks import analysis

    style.use_style()


@app.cell
def _():
    mo.md(f"""
    # Zephyr: from face video to breathing

    Zephyr recovers a mouse's breathing trace from face video alone. This notebook
    follows one test clip through the pipeline, compares every method on it, then
    scores them all on the corpus's three test sets: **A**, animals never trained on;
    **B**, new sessions of training animals; **C**, sessions filmed at 359 Hz. It runs
    inference only; nothing is trained.

    Every network and baseline was trained and scored on Code Ocean
    (`zephyr-compute-sandbox`) and uploaded to
    [`{runs.S3_RUNS}`](https://us-west-2.console.aws.amazon.com/s3/buckets/aind-scratch-data?prefix=vr-foraging/zephyr-runs/).
    What a cell needs of a run is downloaded once into `notebooks/cache/runs/`
    (credentials from `AWS_PROFILE`), and every prediction is cached in
    `notebooks/cache/walkthrough/`, so only the first run computes anything. The
    clips come from a local copy of the dataset and its feature cache
    (`ZEPHYR_TEST_ROOT`, default `{runs.DEFAULT_ROOT}`).
    """)
    return


@app.cell
def _():
    _root = runs.root()
    _wanted = [
        _root / "dataset" / "folds.parquet",
        _root / "dataset" / "clips.toml",
        _root / "cache" / runs.test_sets.RECIPE,
    ]
    _missing = [str(p) for p in _wanted if not p.exists()]
    if not list((_root / "dataset").glob("full_test_*.toml")):
        _missing.append(str(_root / "dataset" / "full_test_{a,b,c}.toml"))
    mo.stop(
        bool(_missing),
        mo.callout(
            mo.md(
                "Missing from the local dataset copy:\n\n"
                + "\n".join(f"- `{m}`" for m in _missing)
                + "\n\nCopy them from `s3://aind-scratch-data/vr-foraging/zephyr-dataset/` "
                "and `zephyr-cache-v2/`, or point `ZEPHYR_TEST_ROOT` at a copy."
            ),
            kind="danger",
        ),
    )
    _error = None
    try:
        run_table = runs.runs_table()
    except Exception as _e:  # noqa: BLE001 -- shown to the user (credentials, network)
        _error = _e
    mo.stop(
        _error is not None,
        mo.callout(
            mo.md(f"Cannot list the runs on S3 ({_error}). Set `AWS_PROFILE`."),
            kind="danger",
        ),
    )
    names = runs.available(runs.METHODS)
    clip_scores = runs.scores(runs.ALL)
    recording_scores = runs.scores(runs.ALL, "per_recording")
    sets = list(runs.SETS.values())
    mo.vstack(
        [
            mo.md(
                "**Runs.** A method whose run has not finished is left out of every "
                "cell below until it has."
            ),
            mo.ui.table(run_table, selection=None),
        ]
    )
    return clip_scores, names, recording_scores, sets


@app.cell
def _(sets):
    test_set = mo.ui.dropdown(sets, value=sets[0], label="Test set")
    return (test_set,)


@app.cell
def _(test_set):
    # Test A's median clip (78th of 156) by its correlation averaged over the ten
    # methods finished on 2026-10-18: typical by construction, and favouring none.
    _median_a = "behavior_806527_2025-10-30_17-20-31_span2_FaceCamera"
    _stems = [s for s, k in runs.clip_sets().items() if k == test_set.value]
    clip = mo.ui.dropdown(
        _stems, value=_median_a if _median_a in _stems else _stems[0], label="Clip"
    )
    return (clip,)


@app.cell
def _(clip, test_set):
    stem = clip.value
    truth = runs.truth(stem)
    zephyr = runs.prediction("Zephyr", stem)
    origin = float(zephyr["times"][0])
    _length = float(zephyr["times"][-1] - origin)
    start = mo.ui.slider(
        0,
        max(5, int(_length) - 10),
        step=5,
        value=min(30, max(0, int(_length) - 10)),
        label="Excerpt start (s)",
        debounce=True,
    )
    mo.hstack([test_set, clip, start], justify="start")
    return origin, start, stem, truth, zephyr


@app.cell
def _():
    mo.md(r"""
    ## 1. The data

    A face video, cropped to the box around the nose (orange), and a thermistor in
    the nostril as ground truth. Every method sees the same $96 \times 96$ gray crop at
    60 Hz. Drag the time slider to scrub through the excerpt.
    """)
    return


@app.cell
def _(start):
    now = mo.ui.slider(
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
def _(now, origin, start, stem, truth, zephyr):
    _frame_t, _frames = runs.gray(stem)
    _i = min(int(np.searchsorted(_frame_t, origin + now.value)), len(_frames) - 1)
    plots.clip_figure(
        runs.thumbnail(stem),
        runs.box(stem),
        np.asarray(_frames[_i, 0]),
        now.value,
        truth,
        zephyr,
        origin,
        start.value,
        10,
    )
    return


@app.cell
def _():
    mo.md("""
    ## 2. The network

    The network maps the crops to a breathing trace, and to the probability that an
    inhalation starts at each time (the onset head). Inhale events are read from the
    head, never from the trace, and scored against the thermistor's within 1.5 frames
    (25 ms); the triangles on both panels are the head's events.
    """)
    return


@app.cell
def _(origin, start, truth, zephyr):
    plots.network_figure(truth, zephyr, origin, start.value, 10)
    return


@app.cell
def _():
    mo.md(r"""
    ### Are the frame embeddings smooth?

    The encoder gives one vector in $\mathbb{R}^{128}$ per frame, and those are
    interpolated onto the 60 Hz output grid with the real frame times, which assumes
    they change smoothly from frame to frame. Below, the embeddings over the first
    60 s of the clip. (a, b) Three seconds of every dimension under the target.
    (c) Their average spectrum against the target's, both scaled to unit area. (d)
    Per dimension, the $R^2$ of rebuilding each frame by linear interpolation between
    its two neighbours: 1 is smooth, 0 no better than the mean, and the same frames
    in shuffled order give about $-0.5$.
    """)
    return


@app.cell
def _(origin, stem, truth):
    _times, _emb = runs.frame_embeddings(stem)  # gpu, cached
    _times = _times - origin
    _reference = np.interp(_times, truth["times"] - origin, truth["signal"])
    _shuffled = _emb[np.random.default_rng(0).permutation(len(_emb))]
    plots.embedding_smoothness_figure(
        _times,
        _emb,
        _reference,
        {
            "Embeddings": analysis.mean_spectrum(_times, _emb),
            "Target": analysis.mean_spectrum(_times, _reference),
        },
        {
            "Embeddings": analysis.interpolation_r2(_times, _emb),
            "Target": analysis.interpolation_r2(_times, _reference),
            "Shuffled": analysis.interpolation_r2(_times, _shuffled),
        },
        _times[0] + 10,
        3,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Every method on the same clip

    Twelve ways of turning the same crops into a breathing trace. Five published video
    networks (TS-CAN, PhysNet, PhysFormer, RhythmFormer; PhysNet, PhysFormer and
    RhythmFormer also with an onset head) are refitted to mouse breathing, each on its
    own paper's loss; Facemap's ridge regression is the field's standard readout; the
    pixel methods are a non-learned floor. All train on the same sessions, stop early
    on the same validation animals, and are scored the same way. The clip shown by
    default is test A's median one by correlation averaged over the methods, so it is
    neither a showcase nor a failure for any of them.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Zephyr

    1. **Input**: the gray crop, one frame at a time, with its timestamp.
    2. **Frame encoder**: a 3-D first layer (3 frames $\times$ 5 $\times$ 5, stride 2) sees
       the change between neighbouring frames itself; three stride-2 convolutions follow
       per frame (64, 96, 128 channels), then a $2 \times 2$ pooled grid (not a global
       mean, so opposite motion in two parts of the nose cannot cancel) and a linear
       layer give one vector in $\mathbb{R}^{128}$ per frame.
    3. **Time**: the vectors are interpolated onto a fixed 60 Hz grid using the real
       frame timestamps, so frame drops and jitter do not shift the output.
    4. **TCN**: six residual blocks with dilations $1, 2, 4, \dots, 32$ (253 samples,
       4.2 s of context, centred: offline, non-causal).
    5. **Two heads**: the breathing trace, and the probability that an inhalation
       starts. Trained with a multi-scale correlation loss on the trace plus a weighted
       cross-entropy on the onset head; 0.86 M parameters.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray crop<br/>T x 1 x 96 x 96"] --> B
        subgraph B["Frame encoder"]
            direction TB
            B0["Conv3d 3 x 5 x 5, stride 2<br/>32 ch, 3 frames"] --> B1["3 x Conv2d, stride 2<br/>64, 96, 128 ch<br/>BatchNorm + GELU"] --> B2["Avg-pool to 2 x 2<br/>Linear, 128-d"]
        end
        B --> C["Resample to 60 Hz grid<br/>(frame timestamps)"]
        C --> D
        subgraph D["Temporal net (TCN)"]
            direction TB
            D1["1x1 Conv1d, 128 ch"] --> D2["6 x residual block<br/>2 x dilated Conv1d, k = 3<br/>dilation 1, 2, 4, 8, 16, 32"]
        end
        D --> E["Signal head<br/>1x1 Conv1d"]
        D --> F["Onset head<br/>1x1 Conv1d"]
        E --> G(["Breathing trace"])
        F --> H(["P(inhale onset)"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### TS-CAN (Liu et al., 2020)

    A two-stream network for pulse from video. The crop is shrunk to $36 \times 36$; a
    *motion* stream reads the normalised frame difference
    $\dfrac{I(t+1) - I(t)}{I(t+1) + I(t)}$ and an *appearance* stream the frame, whose
    soft spatial masks re-weight the motion stream twice. No temporal convolution: in
    every motion convolution a third of the channels shift one frame forward and a
    third one back. It predicts the *derivative* of the trace (MSE, the paper's loss);
    inference integrates it, removes slow drift and band-passes to 1-15 Hz. No onset
    head: DSP finds its events.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray crop, 36 x 36"] --> M["Normalised frame difference"]
        A --> P["Appearance frame"]
        subgraph S["Two streams, 10-frame segments"]
            direction TB
            M --> M1["Conv + temporal shift"]
            P --> P1["Conv"]
            P1 -- "attention masks" --> M2["x mask, pool, conv + shift, x mask"]
            M1 --> M2
        end
        M2 --> D["Dense 128, Dense 1"]
        D --> E["Predicted derivative"]
        E --> F["Integrate, detrend,<br/>band-pass 1-15 Hz"]
        F --> G(["Breathing trace"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### PhysNet (Yu et al., 2019)

    A 3-D convolutional encoder-decoder for pulse from video, on the whole
    $T \times 96 \times 96$ gray volume: 3-D convolutions mix space and time, pooling
    halves space (and twice time), two transposed convolutions restore the frame count,
    and a mean over space gives the trace. Negative Pearson loss (the paper's).
    **PhysNet + head** adds a second $1 \times 1$ output trained with Zephyr's onset loss,
    so the comparison with Zephyr's head is like for like.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray clip<br/>1 x T x 96 x 96"] --> B["Conv3d 1x5x5, pool space"]
        B --> C["Conv3d 3x3x3 blocks<br/>pool space + time (x2)"]
        C --> F["Conv3d 3x3x3 blocks"]
        F --> G["2 x ConvTranspose3d<br/>time x 4"]
        G --> H["Mean over space"]
        H --> I(["Breathing trace"])
        H -.-> J(["P(inhale onset)<br/>+ head only"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### PhysFormer (Yu et al., 2022) and RhythmFormer (Zou, Guo et al., 2024)

    The video transformers of the field, reimplemented from the authors' code, on the
    gray crop upsampled to the 128 px they were built for and windows of 320 frames
    (5.3 s), their own published context.

    - **PhysFormer** (7.4 M parameters): a convolutional stem, tube tokens of
      $4 \times 4 \times 4$, and 12 blocks of *temporal-difference* attention (the queries
      and keys come from a central-difference 3-D convolution, $\theta = 0.7$), then
      upsampling in time and a mean over space. Loss
      $0.1 \cdot \text{Pearson} + b \cdot (\text{CE} + \text{KL})$ over the window's
      spectrum, $b$ rising from 1 to 5 over training.
    - **RhythmFormer** (3.2 M): a stem fusing each frame with its four neighbouring
      differences, then three *temporal periodic* stages that halve time 1, 2 and 3
      times and attend with bi-level routing (each region to its 40 most related ones).
      Loss $0.2 \cdot \text{Pearson} + \text{CE}$. Trained at batch 2, where the others
      use 4: at batch 4 its stem does not fit in the 16 GB GPU.

    Each also comes with an onset head (**+ head**), trained like PhysNet's.
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        A["Gray crop<br/>upsampled to 128 x 128<br/>320 frames"] --> S1["PhysFormer stem<br/>tube tokens 4x4x4"]
        S1 --> T1["12 x temporal-difference<br/>attention blocks"]
        A --> S2["RhythmFormer fusion stem<br/>frame + 4 differences"]
        S2 --> T2["3 periodic stages<br/>time / 2, 4, 8<br/>bi-level routing attention"]
        T1 --> U["Upsample time,<br/>mean over space"]
        T2 --> U
        U --> O(["Breathing trace"])
        U -.-> J(["P(inhale onset)<br/>+ head only"])
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Facemap-style ridge (the field's standard readout)

    Two 100-component SVD bases are fitted on training frames: the *motion* (absolute
    frame difference) and the *movie* (raw pixels). Each frame is projected on both,
    z-scored and copied at lags of up to $\pm 15$ frames, and a ridge regression maps
    that to the thermistor, its strength chosen leaving one recording out. Linear, no
    memory beyond the lags, no onset head.

    ### Pixel methods (non-learned floor)

    No training: **motion** is the mean absolute frame difference over the crop,
    **PCA** the most rhythmic of the top 10 principal components of the pixels,
    **SNR** a band-power-weighted average of the pixels with their signs aligned; each
    band-passed to 1-15 Hz. The sign and lag of each are then fitted on the training
    clips (*calibrated*). Anything a method does better than these is not just "the
    face changes in time with breathing".
    """)
    return


@app.cell
def _():
    mo.mermaid("""
    flowchart LR
        F1["Gray frames"] --> F2["Motion + movie<br/>SVD, 100 comp. each"]
        F2 --> F3["z-score,<br/>time lags"]
        F3 --> F4["Ridge regression<br/>to thermistor"]
        F4 --> F5(["Breathing trace"])
        F1 --> P1["Motion energy, PCA<br/>or SNR-weighted pixels"]
        P1 --> P2["Band-pass 1-15 Hz,<br/>sign and lag from training"]
        P2 --> P5(["Breathing trace"])
    """)
    return


@app.cell
def _(clip_scores, names, origin, start, stem, truth):
    _shown = clip_scores[
        (clip_scores["stem"] == stem) & clip_scores["method"].isin(names)
    ]
    _r = _shown.groupby("method")["correlation"].mean().to_dict()
    plots.trace_stack(
        {n: runs.prediction(n, stem) for n in names},  # gpu, cached
        truth,
        _r,
        origin,
        start.value,
        10,
    )
    return


@app.cell
def _(sets):
    mo.md(f"""
    ## 4. Benchmark

    Every method on every test clip, as each run scored itself on Code Ocean. A
    recording's score is the mean of its clips, an animal's the mean of its
    recordings (and seeds), and the dot the mean over animals with its 95% interval
    from resampling animals: animals are what a new lab would bring, so they are the
    unit that varies. Inhale F1 comes from the onset head where there is one, from DSP
    on the trace otherwise; exhale F1 is DSP for all. Sets: {", ".join(sets)}.
    """)
    return


@app.cell
def _(names, recording_scores):
    benchmark = {
        m: analysis.bootstrap(
            recording_scores[recording_scores["method"].isin(names)], m
        )
        for m in plots.METRIC_LABELS
    }
    return (benchmark,)


@app.cell
def _(benchmark, names, sets):
    plots.benchmark_figure(benchmark, names, sets)
    return


@app.cell
def _(benchmark):
    _wide = pd.concat(
        [
            b.set_index(["set", "method"])["mean"].rename(plots.METRIC_LABELS[m])
            for m, b in benchmark.items()
        ],
        axis=1,
    ).reset_index()
    tables.table(_wide.round(3), lower=["KL-IBI (lower is better)"], group="set")
    return


@app.cell
def _(benchmark):
    _cols = ["set", "method", "Zephyr minus", "diff low", "diff high"]
    _diff = pd.concat(
        [
            b.dropna(subset=["Zephyr minus"])[_cols].assign(
                metric=plots.METRIC_LABELS[m]
            )
            for m, b in benchmark.items()
            if m in ("correlation", "inhale")
        ]
    )
    mo.vstack(
        [
            mo.md(
                "**Zephyr against each method**, paired on the same resampled animals: "
                "an interval that excludes 0 is a difference the animals support."
            ),
            mo.ui.table(_diff.round(3).reset_index(drop=True), selection=None),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ### By animal

    Each animal's mean correlation, worst first by Zephyr, for test sets A (animals
    never trained on) and B (new sessions of training animals). A row of one colour
    is an animal every method finds hard; a row where Zephyr alone is red is a
    Zephyr failure.
    """)
    return


@app.cell
def _(names, recording_scores, sets):
    plots.per_animal_figure(
        recording_scores[recording_scores["method"].isin(names)], names, sets[:2]
    )
    return


@app.cell
def _(names):
    _counts = ", ".join(
        f"{n} {sum(runs.cached(n, s) for s in runs.clips())}" for n in names
    )
    mo.md(f"""
    ## 5. Whole test sets, clip by clip

    The rest of the notebook needs every method's trace on every test clip
    ({len(runs.clips())} clips, about five minutes each), predicted here on the GPU and
    cached. The first time that takes hours (the transformers dominate); after that,
    seconds. Clips already predicted: {_counts}.
    """)
    return


@app.cell
def _(names):
    included = mo.ui.multiselect(names, value=names, label="Methods")
    go = mo.ui.run_button(label="Predict and tabulate the missing ones")
    mo.hstack([included, go], justify="start")
    return go, included


@app.cell
def _(go, included):
    _todo = [n for n in included.value if not runs.done(n)]
    mo.stop(
        bool(_todo) and not go.value,
        mo.callout(
            mo.md(f"Not computed yet: {', '.join(_todo)}. Press the button."),
            kind="info",
        ),
    )
    _parts = [
        runs.tables(
            n, lambda it, n=n: mo.status.progress_bar(it, title=n, remove_on_exit=True)
        )
        for n in included.value
    ]
    whole = {
        k: pd.concat([p[k] for p in _parts], ignore_index=True) for k in analysis.TABLES
    }
    shown = list(included.value)
    return shown, whole


@app.cell
def _():
    mo.md("""
    ### Correlation by frequency band

    The trace and the target are both band-passed to each band and correlated, clip by
    clip. Slow breathing (1-3 Hz) and sniffing (above 6 Hz) are different motions of the
    nose; a method can follow one and not the other.
    """)
    return


@app.cell
def _(sets, shown, whole):
    plots.bands_figure(whole["bands"], shown, sets)
    return


@app.cell
def _():
    mo.md("""
    ## 6. Inference time

    Seconds from the preprocessed crops on disk to the finished 60 Hz trace, per clip
    (`zephyr.benchmarks.timing`: the same 8 clips for every method, an untimed warm-up
    clip, files read into the OS cache first). Networks run on this machine's GPU at
    their own inference window (bf16 where supported); the pixel methods and Facemap's
    projection and readout run on the CPU in numpy. Decoding and cropping, common to
    all, are not timed. Cached; delete `cache/walkthrough/timing.json` to time again.
    """)
    return


@app.cell
def _(names):
    time_go = mo.ui.run_button(label="Time the methods not timed yet")
    time_go
    return (time_go,)


@app.cell
def _(names, recording_scores, time_go):
    _todo = [n for n in names if n not in runs.timed()]
    mo.stop(
        bool(_todo) and not time_go.value,
        mo.callout(mo.md(f"Not timed yet: {', '.join(_todo)}."), kind="info"),
    )
    timing_table = runs.wall_time(names)
    _a = recording_scores[recording_scores["set"] == runs.SETS["test_a"]]
    mo.vstack(
        [
            plots.timing_figure(
                timing_table, _a.groupby("method")["correlation"].mean()
            ),
            mo.ui.table(timing_table.round(3), selection=None),
        ]
    )
    return


@app.cell
def _():
    mo.md(f"""
    ## 7. Breathing rate

    The rate rPPG papers report for heart rate: inhales counted in each
    {analysis.RATE_WINDOW_S:g} s window (outside excluded spans), predicted against
    true. MAE, RMSE and Pearson $r$ per method and set, and the Bland-Altman bias and
    95% limits of agreement (predicted minus true). A method can follow the waveform
    poorly and still count breaths, or the reverse.
    """)
    return


@app.cell
def _(shown, whole):
    _agree = analysis.rate_agreement(whole["windows"]).reset_index()
    _show = [
        n
        for n in ("Zephyr", "PhysNet + head", "PhysFormer + head", "TS-CAN", "Facemap")
        if n in shown
    ]
    mo.vstack(
        [
            tables.table(
                _agree.round(3),
                lower=["MAE (Hz)", "RMSE (Hz)", "limits (Hz)"],
                group="set",
                skip=["bias (Hz)", "windows"],
            ),
            plots.rate_agreement_figure(whole["windows"], _show),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## 8. Inhale timing

    For every true inhale with a predicted one within 100 ms, how early or late the
    prediction is. The scorer counts a match within 25 ms (dotted), so a method whose
    events sit just outside is penalised as one miss and one false event: the width
    of this distribution is the onset head's precision.
    """)
    return


@app.cell
def _(shown, whole):
    plots.lag_figure(
        whole["lags"],
        [n for n in shown if runs.ALL[n].head] + [n for n in ("TS-CAN",) if n in shown],
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 9. By breathing rate

    Every true breath is binned by its own rate ($1 / \Delta t$ to the next inhale);
    misses count in their breath's bin, false events in the bin of the true rate at
    their time, and F1 is computed per bin (top). Below, where the breaths of the
    training set and of each test set fall: a method fails first where training had
    few breaths.
    """)
    return


@app.cell
def _(sets, shown, whole):
    _f1 = analysis.f1_by_rate(whole["breaths"], whole["false_positives"])
    _shares = runs.rate_distribution()  # events only, cached
    mo.vstack(
        [
            plots.rate_f1_figure(_f1, _shares, shown, sets),
            plots.splits_rate_figure(_shares),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## 10. Augmentation

    Zephyr trained with stronger geometric augmentation and blur (shift $\\pm 8$ px
    instead of 4, zoom $\\pm 15\\%$ instead of 10, rotation $\\pm 10°$, Gaussian blur up to
    $\\sigma = 1$ px), everything else the same. Per recording, the difference to Zephyr
    (dots) and its mean, and below the paired interval over animals.
    """)
    return


@app.cell
def _(recording_scores, sets):
    _variant = next(iter(runs.VARIANTS))
    _both = recording_scores[recording_scores["method"].isin(["Zephyr", _variant])]
    mo.stop(
        _both["method"].nunique() < 2,
        mo.callout(mo.md(f"{_variant} has not finished."), kind="info"),
    )
    _wide = _both.pivot_table(
        index=["set", "recording"], columns="method", values=["correlation", "inhale"]
    )
    _diff = pd.DataFrame(
        {
            m: _wide[(m, _variant)] - _wide[(m, "Zephyr")]
            for m in ("correlation", "inhale")
        }
    ).reset_index()
    _ci = pd.concat(
        [
            analysis.bootstrap(_both, m).assign(metric=m)
            for m in ("correlation", "inhale")
        ]
    )
    mo.vstack(
        [
            plots.paired_figure(_diff, "more augmentation minus Zephyr", sets),
            mo.ui.table(
                _ci.dropna(subset=["Zephyr minus"]).round(3).reset_index(drop=True),
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## 11. Examples

    The best, median and worst clip of each test set by Zephyr's correlation, six
    seconds from 30 s in: target (black) and Zephyr (blue).
    """)
    return


@app.cell
def _(clip_scores, sets):
    _z = clip_scores[clip_scores["method"] == "Zephyr"]
    _rows = []
    for _set in sets:
        _part = _z[_z["set"] == _set].sort_values("correlation", ignore_index=True)
        for _label, _k in (
            ("best", len(_part) - 1),
            ("median", len(_part) // 2),
            ("worst", 0),
        ):
            _r = _part.iloc[_k]
            _p = runs.prediction("Zephyr", _r["stem"])
            _rows.append(
                {
                    "truth": runs.truth(_r["stem"]),
                    "prediction": _p,
                    "origin": float(_p["times"][0]),
                    "start": 30.0,
                    "title": f"{_set}, {_label}: r = {_r['correlation']:.2f}\n{_r['stem'][:32]}",
                }
            )
    plots.examples_figure(_rows)
    return


@app.cell
def _():
    mo.md("""
    ## 12. Failure modes

    Zephyr's worst sessions over all three test sets, and what goes wrong in them.
    Every missed true inhale (FN) and every predicted inhale with no true one (FP) is
    checked against what is happening there: is it a **near miss** (an error of the
    other kind within 100 ms, so the breath was found but placed more than 25 ms off),
    **fast breathing** (sniffing above 8 Hz), **face motion** (the crop changes more
    than in 90% of the clip's breaths) or a **shallow breath** (smaller than 75% of the
    clip's)? The raw thermistor is drawn under the target it was filtered into.
    """)
    return


@app.cell
def _():
    n_worst = mo.ui.slider(3, 20, value=8, label="Worst sessions")
    rank_by = mo.ui.dropdown(
        {"Correlation": "correlation", "Inhale F1 (head)": "inhale"},
        value="Correlation",
        label="Ranked by",
    )
    mo.hstack([n_worst, rank_by], justify="start")
    return n_worst, rank_by


@app.cell
def _(clip_scores, n_worst, rank_by, recording_scores):
    _z = recording_scores[recording_scores["method"] == "Zephyr"]
    worst = (
        _z.groupby(["set", "session", "subject"])[["correlation", "inhale"]]
        .mean()
        .reset_index()
        .sort_values(rank_by.value)
        .head(n_worst.value)
        .reset_index(drop=True)
    )
    worst_clips = sorted(
        clip_scores.loc[
            (clip_scores["method"] == "Zephyr")
            & clip_scores["session"].isin(worst["session"]),
            "stem",
        ].unique()
    )
    mo.ui.table(worst.round(3), selection=None)
    return worst, worst_clips


@app.cell
def _(worst_clips):
    all_errors = pd.concat(
        [
            runs.errors("Zephyr", s)
            for s in mo.status.progress_bar(
                worst_clips, title="errors", remove_on_exit=True
            )
        ],
        ignore_index=True,
    )
    plots.causes_figure(analysis.cause_counts(all_errors))
    return (all_errors,)


@app.cell
def _(all_errors):
    _per_clip = (
        all_errors.groupby(["stem", "kind"]).size().unstack(fill_value=0).reset_index()
    )
    failure_clip = mo.ui.dropdown(
        list(_per_clip["stem"]), value=_per_clip["stem"].iloc[0], label="Clip"
    )
    error_kind = mo.ui.dropdown(
        ["all", "missed (FN)", "false (FP)"], value="all", label="Errors"
    )
    mo.vstack(
        [
            mo.ui.table(_per_clip, selection=None),
            mo.hstack([failure_clip, error_kind], justify="start"),
        ]
    )
    return error_kind, failure_clip


@app.cell
def _(all_errors, error_kind, failure_clip):
    clip_errors = all_errors[all_errors["stem"] == failure_clip.value].reset_index(
        drop=True
    )
    if error_kind.value != "all":
        clip_errors = clip_errors[clip_errors["kind"] == error_kind.value].reset_index(
            drop=True
        )
    error_index = mo.ui.slider(
        0, max(0, len(clip_errors) - 1), value=0, label="Error", full_width=True
    )
    error_index
    return clip_errors, error_index


@app.cell
def _(clip_errors, error_index, failure_clip):
    mo.stop(clip_errors.empty, mo.md("No errors of that kind in this clip."))
    _stem = failure_clip.value
    _p = runs.prediction("Zephyr", _stem)
    _origin = float(_p["times"][0])
    _err = clip_errors.iloc[error_index.value]
    _frame_t, _frames = runs.gray(_stem)
    _picked = dict.fromkeys(  # distinct frames: near a clip's edge several clip to one
        min(int(np.searchsorted(_frame_t, _err["time"] + _dt)), len(_frames) - 1)
        for _dt in np.linspace(-0.3, 0.3, 7)
    )
    _shown = [(_frame_t[_i] - _origin, np.asarray(_frames[_i, 0])) for _i in _picked]
    _at = _err["time"] - _origin
    mo.vstack(
        [
            mo.md(
                f"**{_err['kind']}** at {_at:.2f} s: true rate {_err['rate_hz']:.1f} Hz, "
                f"face motion at the {100 * _err['motion']:.0f}th and breath size at the "
                f"{100 * _err['amplitude']:.0f}th percentile of the clip; "
                f"{_err['causes']}."
            ),
            plots.failure_figure(
                runs.truth(_stem), _p, clip_errors, _origin, _at, _shown
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
