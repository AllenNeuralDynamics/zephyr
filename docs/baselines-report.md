# Benchmark baselines: working report

Status as of 2026-10-02. This is a living document: update the results, caveats
and open items as the work moves on.

## Purpose

The manuscript needs zephyr compared against other ways of recovering a mouse's
breathing trace from face video. The target comparison has four tiers:

1. **Non-learned pixel methods**, a floor with no training at all.
2. **A Facemap-style SVD with a linear readout**, the field-standard approach.
3. **TS-CAN** and 4. **PhysNet**, published video-physiology networks trained on
   the same data. *Not started yet; see [Open items](#open-items).*

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
- **Spread:** the baselines are deterministic, so they are single runs without
  error bars. Zephyr's numbers are means over 5 seeds.

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
  the planned TS-CAN and PhysNet) has an event head. Their F1 stays trace-based.
- **Tight event tolerance.** 17 ms is about one frame. A trace with a little
  timing blur loses most matches even when the rhythm is right, which probably
  limits Facemap's F1 (its lag step is 3 frames, or 50 ms). Not tested.
- **KL-IBI** has not been checked for the baselines. Treat it as unvalidated and
  keep it out of headline comparisons for now.
- **No error bars on the baselines.** They are single deterministic runs.

## Open items

1. **Random-trace control.** Score noise traces through the same event detector
   to find the real chance level for F1 (the pixel methods' 0.15).
2. **Check Facemap-style against the `facemap` package** on one or two clips
   (add it as a dev-only dependency) and compare the leading components.
3. **Timing precision of Facemap's F1.** Rescore with a wider tolerance (for
   example 50 ms), or rerun with single-frame lag steps.
4. **TS-CAN and PhysNet: code done, training not yet run.** Both are
   reimplemented in PyTorch (`src/zephyr/baselines/nets.py`) behind the same
   interface as zephyr's network, with a new `--arch tscan|physnet` flag in
   `train.py` / `evaluate.py` / `infer.py`. Checked so far: unit tests, a
   3-step CPU smoke train and evaluate for each, and a synthetic task where both
   reach correlation about 0.99 within 40 steps. Not yet done: learning-rate
   tuning on the train split and the 5-seed runs, which need a GPU
   (`zephyr baseline net --arch tscan|physnet`).
   - TS-CAN is trained on the derivative of the trace, as in the original
     paper, and the trace is rebuilt by cumulative sum, detrend and band-pass.
   - Neither can use code from rPPG-Toolbox (a licence that would impose use
     restrictions on zephyr's MIT code). Both are written from the papers (Liu et
     al., NeurIPS 2020 for TS-CAN; Yu et al., BMVC 2019 for PhysNet).
   - Learning rate tuned on the train split only, then 5 seeds each.
5. **Decide how to pick variants for the paper** (test split vs validation).

## Reproducing

Use `uv run --extra cpu ...` (or `--extra gpu`); plain `uv run` removes torch
because it is an optional extra.

```bash
uv run --extra cpu zephyr baseline pixel     # flow, pca, snr: roughly 30 min on CPU
uv run --extra cpu zephyr baseline facemap   # up to about an hour on CPU
uv run --extra cpu zephyr baseline collect   # tables, merges zephyr's sweep
```

Outputs go to `benchmarks/baselines-v1/` (git-ignored):

```text
pixel/{flow,pca,snr}/{blind,calibrated}/evaluation.json
pixel/{flow,pca,snr}/traces/{train,test}/*.npy   cached output-grid traces
facemap/{motion,movie,both}/evaluation.json
facemap/bases.npz
results.json, results.md
```

The pixel and Facemap results above were produced with the code at commit
`b10d803` (branch `baselines`). The onset-head rows and the "best per method"
table came later from `zephyr baseline collect`, which read the sweep's existing
`head_evaluation.json` files, so no re-run was needed.

## Code map

| Path | Role |
|---|---|
| `src/zephyr/baselines/common.py` | Breathing band, frame loading, polarity rule, lag fit |
| `src/zephyr/baselines/pixel.py` | Flow, PCA and SNR methods |
| `src/zephyr/baselines/facemap_ridge.py` | SVD basis, lagged design, ridge with session-wise validation |
| `src/zephyr/baselines/run.py` | Runs and scores the pixel and Facemap baselines |
| `src/zephyr/baseline.py` | `zephyr baseline` CLI and the results table |
| `src/zephyr/evaluate.py` | `build_result`, the result builder shared with `zephyr evaluate` |
| `tests/test_baselines_*.py`, `tests/test_baseline_cli.py`, `tests/test_evaluate_result.py` | Tests |
