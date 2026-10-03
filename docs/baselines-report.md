# Benchmark baselines: working report

Status as of 2026-10-03. This is a living document: update the results, caveats
and open items as the work moves on.

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
| Target | derivative of the trace | the trace |
| Loss | MSE on the standardised derivative | negative Pearson (scale 1) |
| Trace at inference | cumulative sum, detrend (lambda 100), first-order Butterworth band-pass | direct |

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
| 1e-4 | 0.918 | 0.706 |
| 3e-4 | **0.929** | 0.759 |
| 1e-3 | 0.922 | **0.785** |

TS-CAN's best rate is the top of the grid, so it may be slightly under-tuned.
Final runs used 3e-4 (PhysNet) and 1e-3 (TS-CAN).

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
| TS-CAN | all | 0.764 +/- 0.002 | 0.324 +/- 0.001 | 0.761 +/- 0.003 | 0.067 |
| TS-CAN | `new_animals` | 0.748 | 0.312 | 0.713 | 0.058 |
| TS-CAN | `known_animals_new_date` | 0.779 | 0.335 | 0.809 | 0.077 |
| PhysNet | all | **0.900** +/- 0.003 | **0.822** +/- 0.004 | **0.870** +/- 0.006 | 0.038 |
| PhysNet | `new_animals` | 0.893 | 0.794 | 0.834 | 0.036 |
| PhysNet | `known_animals_new_date` | 0.908 | 0.850 | 0.907 | 0.040 |

Per seed (all sessions, correlation / inhale F1 / exhale F1 / KL-IBI):

| Seed | TS-CAN | PhysNet |
|---|---|---|
| 17 | 0.762 / 0.323 / 0.759 / 0.066 | 0.898 / 0.819 / 0.875 / 0.047 |
| 42 | 0.766 / 0.324 / 0.763 / 0.069 | 0.903 / 0.825 / 0.866 / 0.030 |

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

### Best per method

| Method | Correlation | Inhale F1 | Exhale F1 |
|---|---|---|---|
| Pixel flow / pca / snr | at most 0.02 | at most 0.16 | at most 0.16 |
| Facemap-style (both) | 0.434 | 0.303 | 0.322 |
| TS-CAN | 0.764 | 0.324 | 0.761 |
| PhysNet | 0.900 | 0.822 | 0.870 |
| Zephyr, from the trace | 0.913 | 0.872 | 0.891 |
| Zephyr, from the onset head | n/a | 0.954 | n/a |

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
- **TS-CAN reconstructs the waveform but not the timing of inhales.** Correlation
  is 0.764, but inhale F1 is 0.324 against exhale F1 0.761 on both seeds. Zephyr's
  and PhysNet's two F1s are within about 0.05 of each other. A plausible cause is
  that the integrated, band-passed reconstruction keeps the broad troughs in
  place while blurring the sharp inhale peaks. Not yet established; see
  [Open items](#open-items).

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
- **TS-CAN's learning rate** was best at the top of the grid (1e-3).
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
4. **Improve TS-CAN's inhale timing (in progress).** Test other reconstruction
   settings (band, detrend strength) and a higher learning rate (3e-3), choosing
   them on the held-out validation tail of the training clips and not on the test
   split.
5. **Decide how to pick variants for the paper** (test split vs validation).
6. **Rerun the baselines with 5 seeds** if the paper needs the same count as
   zephyr (about 2 h per TS-CAN seed and 1 h per PhysNet seed on the RTX 4060 Ti).

## Reproducing

Use `uv run --extra cpu ...` (or `--extra gpu` for TS-CAN and PhysNet); plain
`uv run` removes torch because it is an optional extra.

```bash
uv run --extra cpu zephyr baseline pixel     # flow, pca, snr: roughly 30 min on CPU
uv run --extra cpu zephyr baseline facemap   # up to about an hour on CPU
uv run --extra gpu zephyr baseline net --arch physnet --lr 0.0003 --seeds 17 42
uv run --extra gpu zephyr baseline net --arch tscan --lr 0.001 --seeds 17 42
uv run --extra cpu zephyr baseline collect   # tables, merges zephyr's sweep
```

Learning-rate tuning for each network (repeat for 1e-4, 3e-4, 1e-3; PhysNet also
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
| `src/zephyr/baselines/common.py` | Breathing band, frame loading, polarity rule, lag fit |
| `src/zephyr/baselines/pixel.py` | Flow, PCA and SNR methods |
| `src/zephyr/baselines/facemap_ridge.py` | SVD basis, lagged design, ridge with session-wise validation |
| `src/zephyr/baselines/run.py` | Runs and scores the pixel and Facemap baselines |
| `src/zephyr/baselines/nets.py` | TS-CAN and PhysNet, the derivative loss and trace reconstruction, `build_model` |
| `src/zephyr/baseline.py` | `zephyr baseline` CLI and the results table |
| `src/zephyr/evaluate.py` | `build_result`, the result builder shared with `zephyr evaluate`; `--arch`-aware `load_checkpoint` |
| `src/zephyr/train.py`, `src/zephyr/infer.py` | `--arch` flag; trace postprocessing hook for TS-CAN |
| `tests/test_baselines_*.py`, `tests/test_baseline_cli.py`, `tests/test_evaluate_result.py`, `tests/test_module_entrypoints.py` | Tests |
