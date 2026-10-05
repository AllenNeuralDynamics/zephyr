# Benchmark baselines: working report

Status as of 2026-10-03. This is a living document: update the results, caveats
and open items as the work moves on.

Paths are relative to `packages/zephyr-benchmarks/`. Commands quoted in the
history below are as they were run; today `zephyr baseline <step>` is
`zephyr-benchmarks <step>`, and the networks train with `zephyr-benchmarks run`
(see the package README).

## Purpose

The manuscript needs zephyr compared against other ways of recovering a mouse's
breathing trace from face video. The target comparison has four tiers:

1. **Non-learned pixel methods**, a floor with no training at all.
2. **A Facemap-style SVD with a linear readout**, the field-standard approach.
3. **TS-CAN** and 4. **PhysNet**, published video-physiology networks trained on
   the same data.

Everything here is pixel-based. No keypoint tracking is used.

## Protocol

Every method sees the same inputs and is scored the same way as zephyr.

- **Inputs:** the preprocessed 96 x 96 face crops at 60 Hz that `zephyr
  preprocess` writes (`data/features/`), including the stored gray and optical
  flow channels.
- **Training data:** the `train` split only (16 sessions, 32 clips). Anything
  that is fitted (polarity and lag, the Facemap basis and readout, the ridge
  strength) is fitted here, never on test data.
- **Test data:** the `test` split (12 sessions, 24 clips), taken from
  `data/split.json`. It is divided into two strata:
  - `new_animals`: animals not seen in training.
  - `known_animals_new_date`: known animals recorded on a new date.
- **Dev sessions:** sessions 9, 10 and 12 of the `train` split (6 clips),
  reserved in `artifacts/holdout_sessions.json`. Models used for tuning never
  train on them. They are used for learning-rate tuning and the TS-CAN
  input-size experiments, and for nothing else. **They are not the test data**: every number in the Results section comes from
  `data/test`.
- **Scoring:** `zephyr.evaluation.score_clip`, the same function `zephyr
  evaluate` uses. Per-clip scores are averaged within session, and sessions are
  the sampling unit.
  - **Correlation:** zero-lag correlation of the whole trace.
  - **Inhale F1 / exhale F1:** events are detected on the trace as peaks
    (inhale) and troughs (exhale), and matched to the thermistor's within
    **17 ms**, about one frame at 60 Hz.
  - **KL-IBI:** divergence between inter-breath-interval histograms (lower is
    better).
- **Spread:** the pixel and Facemap-style baselines are deterministic, so they
  are single runs without error bars. TS-CAN and PhysNet are means over 2 seeds
  (17 and 42) with the spread shown as +/-. Zephyr's numbers are means over 5
  seeds.

## Methods

### Non-learned pixel methods (`src/zephyr/baselines/pixel.py`)

None of these sees the thermistor. Each is reported in two versions:

- **blind:** polarity set by a fixed rule (flip the trace if the skew of its
  first difference is positive, because inhalation is faster than exhalation so
  the true trace falls faster than it rises), and zero lag.
- **calibrated:** the blind trace plus one global sign and lag, fitted on the
  training split (lag searched within +/- 0.5 s).

| Method | What it does |
|---|---|
| `flow` | Crop-averaged optical flow, projected on its dominant direction, integrated to displacement, band-passed |
| `pca` | Per-pixel band-passed intensity, per-clip PCA, keeps the component with the most concentrated spectrum in the breathing band |
| `snr` | Pixels weighted by the share of their variance inside the breathing band, signed to agree with the best pixel, averaged |

All filters use one breathing band: **1-15 Hz**.

### Facemap-style SVD + ridge (`src/zephyr/baselines/facemap_ridge.py`)

This is **our reimplementation**. It does not use the `facemap` package.

- **Features:** the top 100 principal components of the *motion* (absolute frame
  difference) and of the *movie* (raw frames), on gray crops downsampled to
  48 x 48. Three variants: `motion`, `movie`, `both`.
- **Basis:** fitted once on 2000 frames sampled from each training clip and
  shared by all clips. Facemap itself fits per video.
- **Readout:** ridge regression on lagged copies of the z-scored components
  (lags -15 to +15 frames in steps of 3, 11 lags), so 1100 features per feature
  set (2200 for `both`). There is no temporal network.
- **Regularisation:** chosen by leave-one-session-out on the training sessions.

### TS-CAN and PhysNet (`src/zephyr/baselines/nets.py`)

Both are **our PyTorch reimplementations**, written from the papers and the
authors' architecture descriptions, not copied from any codebase. No code from
rPPG-Toolbox is used (its licence would impose use restrictions on zephyr's MIT
code). Citations: Liu et al., NeurIPS 2020 (TS-CAN); Yu et al., BMVC 2019
(PhysNet).

Shared setup:

- **Same interface as zephyr's network,** so `train.py`, `infer.py` and
  `evaluate.py` run them unchanged (`--arch tscan|physnet`). Neither has an event
  head, so event F1 is trace-based.
- **Input:** the gray channel only (the papers use RGB), from the same 96 x 96
  crops. TS-CAN downsamples to 36 x 36 as in the paper; PhysNet uses the full
  96 x 96.
- **Training recipe:** zephyr's own (AdamW, cosine schedule, EMA, bf16, 200
  epochs x 200 steps, batch 4), not the papers' optimisers. Temporal
  augmentation is off (no time stretch or frame jitter) because TS-CAN's motion
  stream differences consecutive frames. Photometric and spatial augmentation
  stay on.
- **Learning rate** was tuned per network on the train split only (below).

| | TS-CAN | PhysNet |
|---|---|---|
| Parameters | 0.53 M | 0.77 M |
| Window (frames) | 512 | 128 |
| Target | forward difference of the trace, `y[k+1] - y[k]` | the trace |
| Loss | MSE on the standardised forward difference | negative Pearson (scale 1) |
| Trace at inference | running sum, detrend (lambda 100), first-order Butterworth band-pass | direct |

The TS-CAN band-pass uses the same 1-15 Hz band as the other baselines, where the
original used a human breathing band.

**Learning-rate tuning.** Each network was trained for 40 epochs at three rates
(seed 0), validating on the last 25% of each training clip with sessions 9, 10 and
12 held out; the test split was not used. Score is the best full-clip correlation
on that validation tail (EMA weights once settled). The validation tail is
within-session, so these numbers run higher than test-split scores and are only
for choosing between rates.

| Learning rate | PhysNet | TS-CAN |
|---|---|---|
| 1e-4 | 0.918 | |
| 3e-4 | **0.929** | 0.787 |
| 1e-3 | 0.922 | 0.820 |
| 3e-3 | | **0.829** |

The TS-CAN column is for the forward-difference version. Its best rate is again
the top of the grid, so it may be slightly under-tuned. Final runs used 3e-4
(PhysNet) and 3e-3 (TS-CAN).

**TS-CAN target.** The paper trains on the forward difference of the label
(`np.diff`), and a running sum then recovers the label exactly. An earlier
version of this benchmark used a *central* difference instead, a deviation from
the paper. The running sum of a central difference lands half a sample early, so
that version's inhale F1 was 0.32 against exhale F1 0.76. TS-CAN was retrained
with the paper's forward-difference target (each output is placed at the earlier
frame of its frame pair and the reconstruction is an exact running sum). On a
synthetic task with a known answer the reconstructed trace correlates 0.999 with
the truth at zero lag. The earlier central-difference results are not reported
below; the checkpoints are kept in `runs_all/archive-central-diff/` and
`benchmarks/baselines-v1/tscan_central_diff_archive/`.

**Attempts to improve the (central-difference) TS-CAN.** Same protocol (40
epochs, dev sessions plus the validation tail), held-out correlation:

| Input | Learning rate | Held-out correlation |
|---|---|---|
| 36 px (the paper's size) | 1e-3 | 0.785 |
| 36 px | 3e-3 | **0.792** |
| 72 px | 1e-3 | 0.765 |
| 72 px | 3e-3 | 0.751 |

None helped beyond noise (the 0.007 gain from 3e-3 is about the seed-to-seed
spread), so the retrained TS-CAN keeps the paper's 36 px input. 96 px
was not run: about 9 h per full run on this GPU, and a 72 px attempt with
512-frame windows filled the 8 GB of GPU memory, so the larger sizes used
128-frame windows.

## Results

Test split, all 12 sessions, 17 ms event tolerance.

### Baselines

| Method | Variant | Correlation | Inhale F1 | Exhale F1 | KL-IBI |
|---|---|---|---|---|---|
| pixel flow | blind | 0.019 | 0.155 | 0.137 | 0.549 |
| pixel flow | calibrated | 0.016 | 0.151 | 0.164 | 0.549 |
| pixel pca | blind | -0.003 | 0.147 | 0.156 | 0.305 |
| pixel pca | calibrated | 0.004 | 0.162 | 0.149 | 0.314 |
| pixel snr | blind | -0.019 | 0.158 | 0.136 | 0.290 |
| pixel snr | calibrated | 0.012 | 0.150 | 0.157 | 0.290 |
| Facemap-style | motion | 0.235 | 0.252 | 0.242 | 2.326 |
| Facemap-style | movie | 0.399 | 0.293 | 0.317 | 0.728 |
| Facemap-style | both | **0.434** | **0.303** | **0.322** | 0.928 |

Facemap-style by stratum (correlation / inhale F1):

| Variant | `new_animals` | `known_animals_new_date` |
|---|---|---|
| motion | 0.226 / 0.275 | 0.244 / 0.230 |
| movie | 0.425 / 0.301 | 0.373 / 0.284 |
| both | 0.458 / 0.311 | 0.409 / 0.295 |

### TS-CAN and PhysNet (2 seeds)

| Method | Stratum | Correlation | Inhale F1 | Exhale F1 | KL-IBI |
|---|---|---|---|---|---|
| TS-CAN | all | 0.773 +/- 0.001 | 0.696 +/- 0.015 | 0.837 +/- 0.008 | 0.077 |
| TS-CAN | `new_animals` | 0.770 | 0.662 | 0.811 | 0.073 |
| TS-CAN | `known_animals_new_date` | 0.777 | 0.730 | 0.864 | 0.080 |
| PhysNet | all | **0.900** +/- 0.003 | **0.822** +/- 0.004 | **0.870** +/- 0.006 | 0.038 |
| PhysNet | `new_animals` | 0.893 | 0.794 | 0.834 | 0.036 |
| PhysNet | `known_animals_new_date` | 0.908 | 0.850 | 0.907 | 0.040 |

Per seed (all sessions, correlation / inhale F1 / exhale F1 / KL-IBI):

| Seed | TS-CAN | PhysNet |
|---|---|---|
| 17 | 0.774 / 0.707 / 0.843 / 0.070 | 0.898 / 0.819 / 0.875 / 0.047 |
| 42 | 0.772 / 0.686 / 0.831 / 0.083 | 0.903 / 0.825 / 0.866 / 0.030 |

### Zephyr, for reference (5 seeds, from the existing sweep)

| Input | Objective | Correlation | Inhale F1 (trace) | Exhale F1 (trace) | KL-IBI | Inhale F1 (onset head) |
|---|---|---|---|---|---|---|
| gray | multitask | 0.872 | 0.784 | 0.803 | 0.057 | 0.918 |
| gray + diff | multitask | 0.905 | 0.841 | 0.857 | 0.054 | 0.939 |
| gray + flow | multitask | 0.906 | 0.865 | 0.880 | 0.052 | 0.948 |
| gray + diff + flow | multitask | 0.913 | 0.872 | 0.887 | 0.038 | **0.954** |

Trace-based F1 detects events on the reconstructed breathing trace, exactly as
for the baselines. The onset-head column scores the network's own onset head
(`head_evaluation.json`, written by `zephyr benchmark-report head`). The head
only produces inhale events, so it has no correlation, exhale F1 or KL-IBI.

### Best per method (test split)

| Method | Correlation | Inhale F1 | Exhale F1 |
|---|---|---|---|
| Pixel flow / pca / snr | at most 0.02 | at most 0.16 | at most 0.16 |
| Facemap-style (both) | 0.434 | 0.303 | 0.322 |
| TS-CAN | 0.773 | 0.696 | 0.837 |
| PhysNet | 0.900 | 0.822 | 0.870 |
| Zephyr, from the trace | 0.913 | 0.872 | 0.891 |
| Zephyr, from the onset head | n/a | 0.954 | n/a |

The TS-CAN and PhysNet rows have no choice made on the test split (learning
rates came from the dev sessions). The pixel, Facemap-style and zephyr rows are
the best variant by test score.

### Inference time per clip

Wall-clock seconds per 300 s clip, from the preprocessed crops on disk to the
finished 60 Hz trace, mean +/- sd over 8 test clips (`test_1_part_1`,
`test_2_part_2`, `test_4_part_2`, `test_6_part_1`, `test_7_part_2`,
`test_9_part_1`, `test_11_part_1`, `test_12_part_2`). "x real time" is 300 s
divided by the time per clip.

| Method | Device | Params (M) | Seconds per clip | x real time |
|---|---|---|---|---|
| pixel flow | CPU | n/a | 3.48 +/- 0.03 | 86x |
| pixel pca | CPU | n/a | 2.80 +/- 0.16 | 107x |
| pixel snr | CPU | n/a | 1.94 +/- 0.17 | 155x |
| Facemap-style motion | CPU | n/a | 0.95 +/- 0.01 | 316x |
| Facemap-style movie | CPU | n/a | 0.86 +/- 0.01 | 350x |
| Facemap-style both | CPU | n/a | 1.19 +/- 0.02 | 251x |
| PhysNet | GPU | 0.77 | 1.42 +/- 0.01 | 211x |
| TS-CAN | GPU | 0.53 | 0.83 +/- 0.00 | 363x |
| zephyr, gray | GPU | 0.86 | 0.65 +/- 0.01 | 459x |
| zephyr, gray + diff + flow | GPU | 0.86 | 1.70 +/- 0.02 | 176x |
| PhysNet | CPU | 0.77 | 18.99 +/- 1.52 | 16x |
| TS-CAN | CPU | 0.53 | 4.70 +/- 0.02 | 64x |
| zephyr, gray | CPU | 0.86 | 3.42 +/- 0.06 | 88x |
| zephyr, gray + diff + flow | CPU | 0.86 | 4.11 +/- 0.02 | 73x |

Hardware: NVIDIA GeForce RTX 4060 Ti (8 GB), an Intel CPU (PyTorch using 24
threads), PyTorch 2.7.1 (CUDA 12.6). Networks run on the GPU in bf16 and on the CPU in fp32; the
pixel and Facemap-style methods are numpy and run on the CPU only.

What the numbers include and exclude:

- **Included:** reading the crop array, the method itself, and trace
  reconstruction (TS-CAN's integrate-and-filter, the pixel polarity rule). The
  Facemap-style time includes projection and the lagged readout.
- **Not included:** video decoding and cropping (shared by every method), scoring,
  and the preprocessing channels a method reads. **Optical flow is computed once
  in `zephyr preprocess` and costs about 52 s per clip on the CPU (2.9 ms per
  frame).** Add it for any method that reads the flow channel (pixel flow, zephyr
  with flow); it is larger than the inference time of every method above. Gray
  and diff channels are cheap.
- **File cache:** every timed clip is read into the operating system's file cache
  first, so all methods are timed on warm data. Cold disk reads add real cost to
  the cheaper methods (pixel flow measured 6.4 s cold against 3.5 s warm here).
- **Warm-up:** one extra clip, not among the 8, is run and discarded before each
  method.
- **Facemap-style weights** are not saved by the benchmark run, so timing uses
  random weights of the right shape (cost depends on shape, not values).
- **Single machine, single run per clip.** Treat differences under about 10% as
  noise (a repeat of the whole timing run differed by at most 7.7%).

`zephyr baseline collect` regenerates the full tables, including this one, in
`benchmarks/baselines-v1/results.md`.

## Findings

- **The non-learned pixel methods are at chance.** Correlation is about zero
  for every method and both variants.
- **The blind polarity rule fails.** It gives a positive correlation on only
  41-56% of training clips (flow 0.53, pca 0.56, snr 0.41), which is a coin flip.
- **Calibration does not rescue them.** The fitted global sign and lag are tiny
  (flow: sign -1, lag +5; pca: sign -1, lag +2; snr: sign +1, lag -3), with best
  training correlation of 0.017-0.024.
- **The breathing rhythm is in the video.** On two training clips the mean
  brightness of the crop peaks at the same frequency as the thermistor (1.52 Hz).
  The pixel methods just pick out something else, probably head movement or
  sniffing, so a floor near zero could be plausible I guess...
- **Facemap-style with a linear readout recovers part of the signal.**
  - Raw frames matter more than motion alone (correlation 0.40 vs 0.24).
  - Motion alone loses the direction of movement, since the absolute difference
    is rectified, so inhale and exhale can look alike.
  - Combining both is best.
  - Results are similar across the two strata, within about 0.05 correlation.
- **Event F1 around 0.15 on the pixel methods is probably chance.** This has not
  been verified; see [Open items](#open-items).
- **PhysNet is a strong baseline.** It reaches correlation 0.900 and trace-based
  inhale F1 0.822, against zephyr's 0.913 and 0.872 (gray + diff + flow). The two
  PhysNet seeds agree to about 0.005.
  - **Gray-only comparison:** zephyr's gray-only model scores 0.872 / 0.784 /
    0.803, below PhysNet. So zephyr's advantage over PhysNet comes from the extra
    motion channels (diff and flow), not from the architecture alone. These are
    separate claims, and a reviewer is likely to notice.
  - PhysNet generalises about equally to new animals (0.893) and to new dates
    (0.908).
- **A sub-frame timing offset explained the first TS-CAN's weak inhale F1, and
  the paper's forward-difference target fixes most of it.** The first version
  scored correlation 0.764 but inhale F1 only 0.324 (exhale F1 0.761). On the dev
  sessions, loosening the match tolerance from 17 ms to 34 ms lifted inhale F1
  from 0.30 to 0.75, so the peaks were about a frame off. The cause was the
  central-difference target (see Methods). The retrained TS-CAN scores
  correlation 0.773, inhale F1 0.696 and exhale F1 0.837, so inhale F1 more than
  doubled with no timing correction. It still sits below PhysNet (0.900 /
  0.822 / 0.870) and below zephyr gray (0.872 / 0.784 / 0.803). Other reconstruction settings (band,
  filter order, detrend strength) changed scores by only about 0.05.
- **Neither a larger input nor a higher learning rate improved TS-CAN** (dev
  sessions, central-difference version).
- **Inference is cheap for every method on a GPU.** All networks process a
  5-minute clip in under 2 s (zephyr gray 0.65 s, TS-CAN 0.83 s, PhysNet 1.42 s,
  zephyr gray + diff + flow 1.70 s). On the CPU, PhysNet is the slowest by a wide
  margin (19 s per clip); zephyr gray (3.4 s) is faster than TS-CAN (4.7 s).
  Preprocessing optical flow (52 s per clip) costs far more than any of these
  inference times.

## Caveats

- **Not the real Facemap.** This is a Facemap-style reimplementation. It differs
  in the pooled basis, the 48 x 48 downsampled gray crops, and the plain
  eigendecomposition on sampled frames. It has not been compared against the
  `facemap` package. Until it is, call it "a Facemap-style SVD plus linear
  readout" in the manuscript, not "Facemap".
- **The best-per-method table is not a fair ranking.** Variants are picked by
  their own test score. Facemap has 3 variants, the pixel methods 2 each, zephyr
  8 plus the head. For a manuscript, state that variants were chosen on the test
  split, or choose them on validation data.
- **The onset head only gives inhale F1,** and none of the baselines (including
  TS-CAN and PhysNet) has an event head. Their F1 stays trace-based, so the
  zephyr onset-head row is not a like-for-like comparison.
- **TS-CAN and PhysNet are reimplementations,** with a gray-only input, zephyr's
  training recipe instead of the papers' optimisers, and (TS-CAN) a different
  breathing band. They have not been checked against the authors' code.
- **Fewer seeds than zephyr.** TS-CAN and PhysNet have 2 seeds, zephyr 5.
- **TS-CAN's learning rate** was best at the top of the grid (3e-3), so a higher
  rate might help slightly.
- **The dev sessions are only 6 clips,** from animals seen in training, so they
  are a coarse guide for choosing learning rates.
- **The TS-CAN investigation started from a test-split result.** The weak inhale F1
  was first noticed in a test score (seed 17). The cause was diagnosed on the dev
  sessions only, and the fix (the paper's forward-difference target) involves no
  parameter chosen on the test split.
- **Possible sub-frame offsets elsewhere.** PhysNet and zephyr were not checked
  for the same kind of timing offset (see [Open items](#open-items)).
- **Inference timing** is for 8 clips on one machine, on warm file cache, with
  random Facemap-style weights, and excludes preprocessing (including the 52 s per
  clip of optical flow). Networks are timed on both GPU (bf16) and CPU (fp32),
  the numpy methods on CPU only, so the two groups are not like for like.
- **Tight event tolerance.** 17 ms is about one frame. A trace with a little
  timing blur loses most matches even when the rhythm is right, which probably
  limits Facemap's F1 (its lag step is 3 frames, or 50 ms). Not tested.
- **KL-IBI** has not been checked for the baselines. Treat it as unvalidated and
  keep it out of headline comparisons for now.
- **No error bars on the pixel and Facemap-style baselines.** They are single
  deterministic runs.

## Open items

1. **Random-trace control.** Score noise traces through the same event detector
   to find the real chance level for F1 (the pixel methods' 0.15).
2. **Check Facemap-style against the `facemap` package** on one or two clips
   (add it as a dev-only dependency) and compare the leading components.
3. **Timing precision of Facemap's F1.** Rescore with a wider tolerance (for
   example 50 ms), or rerun with single-frame lag steps.
4. **Check PhysNet and zephyr for a sub-frame timing offset.** Scoring PhysNet on
   the dev sessions showed its F1 is sensitive to a fraction of a frame, and the
   cause has not been found. A fix would belong in the model or its training, not
   in a post-hoc parameter.
5. **Decide how to pick variants for the paper** (test split vs validation).
6. **Rerun the baselines with 5 seeds** if the paper needs the same count as
   zephyr (about 2 h per TS-CAN seed and 1 h per PhysNet seed on the RTX 4060 Ti).

## Reproducing

Use `uv run --extra cpu ...` (or `--extra gpu` for TS-CAN and PhysNet); plain
`uv run` removes torch because it is an optional extra. `baseline net` defaults
to each network's tuned learning rate (TS-CAN 3e-3, PhysNet 3e-4); `--lr`
overrides it.

```bash
uv run --extra cpu zephyr baseline pixel     # flow, pca, snr: roughly 30 min on CPU
uv run --extra cpu zephyr baseline facemap   # up to about an hour on CPU
uv run --extra gpu zephyr baseline net --arch physnet --seeds 17 42
uv run --extra gpu zephyr baseline net --arch tscan --seeds 17 42
uv run --extra gpu zephyr baseline timing --n-clips 8   # inference time per clip
uv run --extra cpu zephyr baseline collect   # tables, merges zephyr's sweep
```


Learning-rate tuning for each network (repeat for 1e-4, 3e-4, 1e-3, 3e-3; PhysNet also
needs `--window 128 --scales 1`):

```bash
uv run --extra gpu zephyr train --arch tscan --channels gray --w-onset 0 \
  --time-stretch 1 --select-jitter 0 --lr 0.001 --seed 0 --epochs 40 \
  --run-dir runs_all/tune-tscan-0.001
```

Outputs go to `benchmarks/baselines-v1/` (git-ignored):

```text
pixel/{flow,pca,snr}/{blind,calibrated}/evaluation.json
pixel/{flow,pca,snr}/traces/{train,test}/*.npy   cached output-grid traces
facemap/{motion,movie,both}/evaluation.json
facemap/bases.npz
{tscan,physnet}/runs/gray__signal__seed-*/{best.pt,evaluation.json}
timing.json                                      per-clip inference times
results.json, results.md
```

The pixel and Facemap results above were produced with the code at commit
`b10d803` (branch `baselines`). The onset-head rows and the "best per method"
table came later from `zephyr baseline collect`, which read the sweep's existing
`head_evaluation.json` files, so no re-run was needed.

The TS-CAN and PhysNet results were produced with commit `7154f08` plus a fix
that was not yet committed at the time: `python -m zephyr.train` and
`python -m zephyr.evaluate` did nothing, because commit `00f0419` had removed
their `__main__` blocks while `zephyr benchmark` (and so `zephyr baseline net`)
still launches them that way. The jobs exited successfully without running. The
fix restores the two-line entry points and adds
`tests/test_module_entrypoints.py`. The PhysNet final runs were redone after
the fix; the learning-rate tuning was unaffected because it uses the `zephyr
train` command.

## Code map

| Path | Role |
|---|---|
| `src/zephyr/benchmarks/common.py` | Breathing band, frame loading, polarity rule, lag fit |
| `src/zephyr/benchmarks/pixel.py` | Flow, PCA and SNR methods |
| `src/zephyr/benchmarks/facemap_ridge.py` | SVD basis, lagged design, ridge with session-wise validation |
| `src/zephyr/benchmarks/baselines.py` | Runs and scores the pixel and Facemap baselines |
| `src/zephyr/benchmarks/nets.py` | TS-CAN and PhysNet, the derivative loss and trace reconstruction, `build_model` |
| `src/zephyr/benchmarks/config.py`, `src/zephyr/benchmarks/runner.py` | `train_params.arch` / `tscan_img_size`, and the architecture and checkpoint loader handed to zephyr's own `run` and `evaluate` |
| `src/zephyr/benchmarks/timing.py` | Per-clip inference timing for every method |
| `src/zephyr/benchmarks/collect.py` | `pixel`, `facemap`, `timing` and `collect` subcommands and the results table |
| zephyr's `evaluate.py`, `train.py`, `infer.py` | `build_result`, shared with `zephyr evaluate`; the `Architecture` and loader seams; the model's optional `postprocess` hook (TS-CAN reconstruction) |
| `tests/` of this package | Tests |
