# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video, covering
everything from raw video to a trained, evaluated checkpoint.

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), the AWS CLI, and
`ffmpeg`/`ffprobe` on `PATH`. A display is only needed for the optional
annotation step. Everything runs through the single `zephyr <subcommand>`
command; `zephyr <subcommand> --help` prints a one-line summary, so see each
module's docstring (e.g. `src/zephyr/run.py`) for its flags.

## How inputs are described

Zephyr knows nothing about where data came from. Every input is a **TOML file
of local paths**, validated by a pydantic model (`src/zephyr/config.py`); paths
are relative to the file they are written in.

- A **clip list** names videos (everything else -- timestamps, thermistor,
  recording group -- is derived from the file name, or given explicitly), the
  `[preprocess]` recipe, and a hand-placed crop `box` per clip.
- A **fold** names the clip lists to train on (with relative weights), the
  clip lists to score (each a named group in the report), the training
  hyperparameters, and optionally a checkpoint to start from. Leakage is
  refused at load: no test clip may share a video, or a *recording* (video
  folder + group), with a training clip.
- An **experiment** names folds, the seeds each runs with, and where results go.

`zephyr schema <dir>` writes JSON schemas of all three for editor validation.
`configs/` holds the examples used below.

## Example dataset

The examples use the AIND breathing challenge data: 16 labelled training
sessions and 12 test sessions of a face camera, plus 3 labelled sessions from a
different rig (`side_right`). Download into `data/` (ignored by Git; no AWS
credentials needed):

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

## Clone to an evaluated network

1. **Install.**

   ```bash
   git clone https://github.com/AllenNeuralDynamics/zephyr
   cd zephyr
   uv sync --locked
   ```

2. **Scan** a folder into a clip list (videos only; the scan validates that
   every derived file exists, so unlabelled clips need `--unlabelled`):

   ```bash
   uv run zephyr clips scan data/train --glob 'video_face_*.mp4' -o configs/clips/my_train.toml
   ```

3. **Annotate**: place one crop box per recording (one box covers a
   recording's parts; `o` gives a single clip its own). Boxes are written into
   the clip list as `box = [x, y, w, h]`. The shipped lists already carry
   boxes, so skip this unless you have new data or want a different crop:

   ```bash
   uv run zephyr annotate configs/clips/my_train.toml --prepare
   uv run zephyr annotate configs/clips/my_train.toml --width 360 --height 270 --box-size 96
   ```

4. **Preprocess** into a feature cache (~1-2 min per clip on CPU; the 56 face
   clips take ~39 GiB). Entries are keyed by video, recipe and box, so a cache
   directory is safe to share and re-running only does what changed:

   ```bash
   uv run zephyr preprocess configs/clips/face_train.toml configs/clips/face_test_new_animals.toml configs/clips/face_test_known_animals_new_date.toml --cache data/features-v2
   ```

5. **Run.** `zephyr run` trains each (fold, seed) into
   `<output_dir>/<fold>/seed-<n>/` -- with the resolved config, the list of
   training videos, checkpoints and `evaluation.json` -- then scores the
   fold's test groups. Check the plumbing first with `--smoke` (1 epoch of 2
   steps; add `--device cpu --amp off --num-workers 0` without a GPU):

   ```bash
   uv run zephyr run configs/experiments/benchmark-gray-diff-flow-multitask.toml --smoke --seeds 42 --output-dir runs/smoke
   ```

   The full benchmark network (200 epochs x 200 steps, five seeds) needs a
   GPU:

   ```bash
   uv run zephyr run configs/experiments/benchmark-gray-diff-flow-multitask.toml
   ```

   `evaluation.json` reports each clip, each recording, and each named test
   group (plus the onset head's inhale F1 for multitask networks).

6. **Evaluate** any checkpoint on any clip lists; each list is reported as a
   group. A checkpoint records the videos it trained on and `evaluate` refuses
   clips that share a video or recording with them (older checkpoints without
   the record are scored with a warning):

   ```bash
   uv run zephyr evaluate --checkpoint runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask/seed-17/best.pt --clips configs/clips/face_test_new_animals.toml configs/clips/face_test_known_animals_new_date.toml --cache data/features-v2 --out evaluation.json
   ```

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

1. **Install with GPU support.** Training needs a CUDA GPU.

   ```bash
   git clone https://github.com/AllenNeuralDynamics/zephyr
   cd zephyr
   uv sync --locked --extra gpu
   ```

2. **Download the face data** into `data/train/` and `data/test/`: the first
   three commands under [Example dataset](#example-dataset). The `side_right`
   data is not needed.

3. **Preprocess** the training clips and both test groups into the feature
   cache the experiment reads (`data/features-v2`). The shipped clip lists
   already carry the crop boxes, so no annotation is needed:

   ```bash
   uv run zephyr preprocess configs/clips/face_train.toml configs/clips/face_test_new_animals.toml configs/clips/face_test_known_animals_new_date.toml --cache data/features-v2
   ```

4. **Check the plumbing** with a 2-step run (it writes to `runs/smoke/`):

   ```bash
   uv run zephyr run configs/experiments/benchmark-gray-diff-flow-multitask.toml --smoke --seeds 42 --output-dir runs/smoke
   ```

5. **Train and score.** Each seed took about 5 hours on the GPU the reference
   runs used. A single seed:

   ```bash
   uv run zephyr run configs/experiments/benchmark-gray-diff-flow-multitask.toml --seeds 42
   ```

   All five seeds (17, 42, 101, 202, 314), one after another; a seed whose
   `best.pt` already exists is not retrained, only re-scored:

   ```bash
   uv run zephyr run configs/experiments/benchmark-gray-diff-flow-multitask.toml
   ```

   Each seed writes
   `runs/benchmark-gray-diff-flow-multitask/benchmark-gray-diff-flow-multitask/seed-<n>/`
   with `best.pt`, the resolved config, the training video list and
   `evaluation.json`.

6. **Summarise across seeds** by running this in `uv run python`:

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

## Beyond the quickstart

- **Other camera view.** `configs/clips/ood_side_right.toml` lists the three
  `side_right` sessions; `configs/folds/ood-hold{13,14,15}.toml` each select
  two of them with `groups = [...]` to fine-tune the benchmark network
  (`init_from`) on the face data (weight 0.5) plus those sessions (0.5), and
  score the held-out session and both face groups.
  `configs/experiments/ood-finetune.toml` runs all three. Preprocess
  `configs/clips/ood_side_right.toml` first.
- **Baselines** (`zephyr baseline pixel|facemap|timing|collect`) take a fold
  file so they see the same train/test clips; TS-CAN and PhysNet are trained
  with `zephyr run` on a fold whose `train_params.arch` is `tscan` or `physnet`
  (with `channels = "gray"`).
- `zephyr benchmark-report --runs <dir> --out <dir>` builds the figures and
  table for the full 4 inputs x 2 objectives x 5 seeds grid from run output.
- `zephyr.infer.predict_clip` runs inference on an arbitrary preprocessed clip
  and checkpoint; see its docstring.
