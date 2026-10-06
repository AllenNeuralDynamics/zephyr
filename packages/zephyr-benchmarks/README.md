# zephyr-benchmarks

Benchmarks of [zephyr](../zephyr/README.md) on the AIND breathing challenge
data, and the methods it is compared against: TS-CAN, PhysNet, pixel methods
and a Facemap-style ridge regression. Imported as `zephyr.benchmarks`; run as
`zephyr-benchmarks <subcommand>`. Not released: it lives in this workspace.

- `configs/` holds every clip list, fold and experiment of the benchmark.
- `zephyr-benchmarks run` and `evaluate` are zephyr's own `run` and `evaluate`,
  taking the same files plus `train_params.arch` (`zephyr`, `tscan` or
  `physnet`), so every network trains and scores under one protocol. zephyr
  folds run with plain `zephyr run` too.

Commands below run from the repository root, after
`uv sync --locked --all-packages --extra gpu` (or `--extra cpu`).

## Dataset

16 labelled training sessions and 12 test sessions of a face camera, plus 3
labelled sessions from a different rig (`side_right`). Download into `data/`
at the repository root (ignored by Git; no AWS credentials needed):

```bash
# face training clips: videos, frame timestamps, thermistors
aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train/ data/train/

# face test clips: videos and timestamps are public, thermistors are in private/
aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/test/ data/test/
aws s3 sync --no-sign-request --exclude "*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/private/test/ data/test/

# other rig (side_right, sessions 13-15), ~2.6 GiB
aws s3 sync --no-sign-request --exclude "*" --include "video_side_right_13_*" --include "video_side_right_14_*" --include "video_side_right_15_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/test/ data/ood/
aws s3 sync --no-sign-request --exclude "*" --include "thermistor_13_*" --include "thermistor_14_*" --include "thermistor_15_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/private/test/ data/ood/
```

Session numbers restart in every folder, which is why a recording is the pair
(folder, group). In `configs/clips/` the test sessions 1-6 (animals not in
training) are `face_test_new_animals.toml` and 7-12 (known animals, new dates)
are `face_test_known_animals_new_date.toml`, following the dataset's
`split.json`.

## How to reproduce the benchmark network

The reference network is the CNN + TCN on `gray + diff + flow` input with two
heads: the breathing trace (correlation loss, weight 1.0) and inhalation onsets
(weight 0.5). It trains on all 16 labelled face sessions (32 clips) for a fixed
budget of 200 epochs x 200 steps, with no validation split and no early
stopping. It is then scored on the 12 test sessions, reported as two groups of
6: animals not seen in training, and known animals recorded on new dates. The
recipe is `configs/folds/benchmark-gray-diff-flow-multitask.toml`, run with
seeds 17, 42, 101, 202 and 314 by
`configs/experiments/benchmark-gray-diff-flow-multitask.toml`.

1. **Download the face data** into `data/train/` and `data/test/`: the first
   three commands under [Dataset](#dataset). The `side_right` data is not
   needed. Training needs a CUDA GPU.

2. **Preprocess** the training clips and both test groups into the feature
   cache the experiment reads (`data/features-v2`). The clip lists already
   carry the crop boxes, so no annotation is needed:

   ```bash
   C=packages/zephyr-benchmarks/configs
   uv run zephyr preprocess $C/clips/face_train.toml $C/clips/face_test_new_animals.toml $C/clips/face_test_known_animals_new_date.toml --cache data/features-v2
   ```

3. **Check the plumbing** with a 2-step run (it writes to `runs/smoke/`):

   ```bash
   uv run zephyr run $C/experiments/benchmark-gray-diff-flow-multitask.toml --smoke --seeds 42 --output-dir runs/smoke
   ```

4. **Train and score.** Each seed took about 5 hours on the GPU the reference
   runs used. A single seed, then all five one after another (a seed whose
   `best.pt` already exists is not retrained, only re-scored):

   ```bash
   uv run zephyr run $C/experiments/benchmark-gray-diff-flow-multitask.toml --seeds 42
   uv run zephyr run $C/experiments/benchmark-gray-diff-flow-multitask.toml
   ```

   Each seed writes
   `runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask/seed-<n>/`
   with `best.pt`, the resolved config, the training video list and
   `evaluation.json`.

5. **Summarise across seeds** by running this in `uv run python`:

   ```python
   import glob, json, statistics

   runs = sorted(
       glob.glob("runs/benchmark-gray-diff-flow-multitask/*/seed-*/evaluation.json")
   )
   groups = [json.load(open(path))["groups"] for path in runs]
   metrics = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi", "head_inhale_f1")
   print(f"{len(runs)} seeds")
   for group in ("new_animals", "known_animals_new_date", "all"):
       means = [statistics.mean(g[group]["summary"][m] for g in groups) for m in metrics]
       print(f"{group:24s}", "  ".join(f"{m} {v:.3f}" for m, v in zip(metrics, means)))
   ```

The configs reproduce the reference runs' inputs exactly: the same settings,
the same 32 training clips in the same order, the same sequence of training
windows per seed, and byte-identical preprocessed features
(`tests/test_reproduction.py` checks this against local reference runs). GPU
training is not bit-deterministic, so a retrained network lands within about
the seed-to-seed spread rather than on the exact numbers. Scoring the
original checkpoints with `zephyr evaluate --device cuda --amp bf16`
reproduces their stored results exactly.

## Comparison methods

- **TS-CAN and PhysNet** (`configs/folds/baseline-{tscan,physnet}.toml`, run by
  `configs/experiments/baselines-nets.toml`) train through zephyr's own loop on
  `channels = "gray"`; see `docs/baselines-report.md`:

  ```bash
  uv run zephyr-benchmarks run $C/experiments/baselines-nets.toml
  ```

- **Pixel and Facemap-style baselines** take a fold file so they see the same
  train/test clips; `timing` times every method per clip and `collect` tables
  them all with the runs:

  ```bash
  uv run zephyr-benchmarks pixel $C/folds/benchmark-gray-diff-flow-multitask.toml --cache data/features-v2
  uv run zephyr-benchmarks facemap $C/folds/benchmark-gray-diff-flow-multitask.toml --cache data/features-v2
  uv run zephyr-benchmarks collect --runs runs/baselines-nets
  ```

- `zephyr-benchmarks report --runs <dir> --out <dir>` builds the figures and
  table for the full 4 inputs x 2 objectives x 5 seeds grid from run output.

## Other camera view

`configs/clips/ood_side_right.toml` lists the three `side_right` sessions;
`configs/folds/ood-hold{13,14,15}.toml` each select two of them with
`groups = [...]` to fine-tune the benchmark network (`init_from`) on the face
data (weight 0.5) plus those sessions (0.5), and score the held-out session and
both face groups. `configs/experiments/ood-finetune.toml` runs all three.
Preprocess `configs/clips/ood_side_right.toml` first.
