# zephyr-benchmarks

Benchmarks of [zephyr](../zephyr/README.md) on the AIND breathing challenge
data, and the methods it is compared against: TS-CAN, PhysNet, pixel methods
and a Facemap-style ridge regression. Imported as `zephyr.benchmarks`; run as
`zephyr-benchmarks <subcommand>`. Not released: it lives in this workspace.

- `configs/` holds every clip list and experiment (with its folds inline) of the benchmark.
- `zephyr-benchmarks run` and `evaluate` are zephyr's own `run` and `evaluate`,
  taking the same files plus `train_params.arch` (`zephyr`, `tscan`, `physnet`,
  `physformer` or `rhythmformer`), so every network trains and scores under one
  protocol, each on its own paper's loss. zephyr folds run with plain `zephyr run` too.
  Only zephyr has an onset head; the others need `w_onset = 0`.

Commands below run from the repository root, after
`uv sync --locked --all-packages --extra gpu` (or `--extra cpu`).

## Dataset

16 labelled training sessions and 12 test sessions of a face camera. Download into `data/`
at the repository root (ignored by Git; no AWS credentials needed):

```bash
# face training clips: videos, frame timestamps, thermistors
aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train/ data/train/

# face test clips: videos and timestamps are public, thermistors are in private/
aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/test/ data/test/
aws s3 sync --no-sign-request --exclude "*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/private/test/ data/test/

```

Session numbers restart in every folder, which is why a recording is the pair
(folder, group). In `configs/clips/` the test sessions 1-6 (animals not in
training) are `face_test_new_animals.toml` and 7-12 (known animals, new dates)
are `face_test_known_animals_new_date.toml`, following the dataset's
`split.json`.

## How to train the benchmark network

The benchmark network is zephyr's CNN + TCN on the gray frames with two heads: the
breathing trace (correlation loss, weight 1.0) and inhalation onsets (weight 0.5). It
trains on all 16 labelled face sessions (32 clips) for a fixed budget of 200 epochs x
200 steps, with no validation split and no early stopping. It is then scored on the 12
test sessions, reported as two groups of 6: animals not seen in training, and known
animals recorded on new dates. The recipe is the `gray-multitask` fold of
`configs/experiments/gray-temporal-stem.toml`, run with seeds 17, 42, 101, 202 and 314.

1. **Download the face data** into `data/train/` and `data/test/`: the first
   commands under [Dataset](#dataset). Training needs a CUDA GPU.

2. **Preprocess** the training clips and both test groups into a feature cache
   (`data/features-v2`). The clip lists already carry the crop boxes, so no
   annotation is needed:

   ```bash
   C=packages/zephyr-benchmarks/configs
   uv run zephyr preprocess $C/clips/face_train.toml $C/clips/face_test_new_animals.toml $C/clips/face_test_known_animals_new_date.toml --cache data/features-v2
   ```

3. **Check the plumbing** with a 2-step run (it writes to `runs/smoke/`):

   ```bash
   uv run zephyr run $C/experiments/gray-temporal-stem.toml --folds gray-multitask --cache data/features-v2 --smoke --seeds 42 --output-dir runs/smoke
   ```

4. **Train and score.** A single seed, then all five one after another (a seed whose
   `best.pt` already exists is not retrained, only re-scored):

   ```bash
   uv run zephyr run $C/experiments/gray-temporal-stem.toml --folds gray-multitask --cache data/features-v2 --seeds 42
   uv run zephyr run $C/experiments/gray-temporal-stem.toml --folds gray-multitask --cache data/features-v2
   ```

   Each seed writes `runs/gray-temporal-stem/gray-multitask/seed-<n>/` with `best.pt`,
   the resolved config, the training video list and `evaluation.json`.

5. **Summarise across seeds** by running this in `uv run python`:

   ```python
   import glob, json, statistics

   runs = sorted(
       glob.glob("runs/gray-temporal-stem/gray-multitask/seed-*/evaluation.json")
   )
   groups = [json.load(open(path))["groups"] for path in runs]
   metrics = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi", "head_inhale_f1")
   print(f"{len(runs)} seeds")
   for group in ("new_animals", "known_animals_new_date", "all"):
       means = [statistics.mean(g[group]["summary"][m] for g in groups) for m in metrics]
       print(f"{group:24s}", "  ".join(f"{m} {v:.3f}" for m, v in zip(metrics, means)))
   ```

`configs/experiments/baselines-nets.toml` reproduces the inputs of the reference
baseline runs (`benchmarks/baselines-v1/`): the same settings and the same 32 training
clips in the same order (`tests/test_reproduction.py` checks this against local
reference runs). GPU training is not bit-deterministic, so a retrained network lands
within about the seed-to-seed spread rather than on the exact numbers.

## Comparison methods

- **TS-CAN and PhysNet** (the `baseline-tscan` and `baseline-physnet` folds of
  `configs/experiments/baselines-nets.toml`) train through zephyr's own loop on the gray
  frames:

  ```bash
  uv run zephyr-benchmarks run $C/experiments/baselines-nets.toml
  ```

- **PhysFormer and RhythmFormer** (`arch = "physformer"`, `"rhythmformer"`;
  `transformer_nets.py`)
  are the transformers of the field, reimplemented from the authors' code with their
  published layers and settings: PhysFormer (Yu et al., CVPR 2022, 7.4 M parameters) and
  RhythmFormer (Zou, Guo et al., 2024, 3.2 M). They train like TS-CAN and PhysNet, on the
  gray frames upsampled to the 128 px they were built for. Their attention spans a whole
  window, so set `infer_window` to the training `window` (e.g. both 320, about 5 s at
  60 Hz): the stitching then keeps the middle half of each window. Their papers trained
  them at lr 1e-4 (PhysFormer) and 9e-3 with a one-cycle schedule (RhythmFormer), batch 4.
  Each trains on its own paper's loss (`transformer_losses.py`), as TS-CAN does: negative
  Pearson plus a cross-entropy over the window's spectrum (one bin per breath per minute
  across the 1-15 Hz band) against the thermistor's rate in that window. PhysFormer's is
  `0.1 Pearson + b (CE + KL)` with `b` rising from 1 to 5 over the run's `epochs`,
  RhythmFormer's `0.2 Pearson + CE`; `w_corr` and `scales` are not used. The optimiser
  and schedule stay zephyr's, as for every network.

- **Pixel and Facemap-style baselines** take an experiment (and `--fold <name>`
  when it has several) so they see the same
  train/test clips; `timing` times every method per clip and `collect` tables
  them all with the runs:

  ```bash
  uv run zephyr-benchmarks pixel $C/experiments/baselines-nets.toml --fold baseline-tscan
  uv run zephyr-benchmarks facemap $C/experiments/baselines-nets.toml --fold baseline-tscan
  uv run zephyr-benchmarks collect --runs runs/baselines-nets
  ```

  The pixel methods are `pca`, `snr` and `motion` (all three by default). `motion` is
  motion energy, the mean absolute frame difference over the crop, band-passed. It is
  rectified, so it follows how fast the face moves rather than the breath
  itself; its `calibrated` variant fits the sign and lag on the training clips.

## Reports beyond `evaluation.json`

`configs/results/*.toml` (`ResultsManifest`) names the finished runs a report compares
and the label each is shown under: a `root` (an `s3://` prefix or a folder), then one
`[[method]]` per method (`run` folder, `kind` `network`/`facemap`/`pixel`, the baseline's
`variant`, and `head = true` for a network with an onset head) and `[[variant]]`s of the
first method. `corpus-v3.toml` is the corpus benchmark that `notebooks/walkthrough.py`
reports. To report other runs, copy it under a new name.

`analysis.py` measures, clip by clip, what a run's own scores do not: every true inhale
and whether it was found, every false event, onset lags, breathing rate counted in 10 s
windows, and correlation by frequency band (`clip_tables`), plus each error's likely
cause (`errors`); then, over clips, rate agreement (MAE, RMSE, Bland-Altman),
F1 by breathing rate and a bootstrap over animals with paired differences (`bootstrap`).
Events are matched exactly as the scorer matches them; inhales of a network with an
onset head come from the head.

`zephyr-benchmarks occlusion --checkpoint <best.pt> --clips <a>.toml ... --cache <dir> --out <file.npz>`
is inference-only analysis, not an experiment: one patch-occlusion map per clip of event
F1 on the first `--window-s` seconds of each clip (`--patch`, `--stride`).
