# Designing experiments

An **experiment** is a TOML file listing one or more **folds**, the **seeds** each fold
runs with, and where results go. A fold says which clips train, which stop training
early, and which are scored. The app writes one-fold experiments. Write the file by hand
for anything more.

```toml title="experiments/mouse-5-held-out.toml"
seeds = [17]
output_dir = "../runs/my-rig"          # paths are relative to this file
features_dir = "../cache"

[[fold]]
name = "mouse-5-held-out"              # its run folder: <output_dir>/<name>/seed-<n>/

[[fold.train]]
clips = "../clips.toml"
groups = ["m1-day1", "m1-day2", "m2-day1", "m3-day1"]

[fold.val]                             # stop early on held-out animals
clips = "../clips.toml"
groups = ["m4-day1"]

[fold.test]                            # report group name -> clips to score
mouse_5 = { clips = "../clips.toml", groups = ["m5-day1", "m5-day2"] }

[fold.train_params]
epochs = 200
```

## Split by animal

A network scored on new sessions of an animal it trained on looks better than it will
on a new animal. For results you report:

- **Hold out whole animals** for testing, and for validation.
- **Rotate** which animals are held out, one fold per rotation (cross-validation), and
  pool the test scores.
- Keep **one clip list** per dataset, so folds select sessions with `groups`.

A video on two sides of a fold is refused when the experiment loads. If a check refuses
your experiment, fix the experiment, never the check.

## Validation and early stopping

Validation keeps the best weights as `best.pt` and stops a run that has stopped
improving. It scores every `score_every` epochs (default 4). It stops after `patience`
scorings without improvement (default 8), but never before `min_epochs` (default 24).

There are two ways to validate:

| | Validates on | Set with |
| --- | --- | --- |
| **Held-out clips** (recommended) | whole clips of other animals, which never train | a `[fold.val]` selection |
| **Time tail** | the last `val_fraction` (25 %) of every training clip | the default, when there is no `[fold.val]` |

Validation clips may not be training or test clips. `val_fraction = 0` without
`[fold.val]` trains for exactly `epochs` epochs.

## Several folds

Each `[[fold]]` is its own block with a distinct `name`. Leave-one-animal-out, for
example, is one fold per animal:

```toml
seeds = [1, 2, 3]
output_dir = "../runs/leave-one-out"
features_dir = "../cache"

[[fold]]
name = "hold-m1"
[[fold.train]]
clips = "../clips.toml"
groups = ["m2-day1", "m3-day1", "m4-day1"]
[fold.test]
m1 = { clips = "../clips.toml", groups = ["m1-day1"] }

[[fold]]
name = "hold-m2"
[[fold.train]]
clips = "../clips.toml"
groups = ["m1-day1", "m3-day1", "m4-day1"]
[fold.test]
m2 = { clips = "../clips.toml", groups = ["m2-day1"] }
```

`zephyr run <experiment> --folds hold-m1` runs only some folds, and `--seeds 2` only
some seeds. Seeds and machine settings belong to the experiment, never to a fold.

## Mixing datasets

A fold can train on several clip lists. `weight` sets each source's share of the
sampled training windows (relative: 1 and 1 is the same as 0.5 and 0.5):

```toml
[[fold.train]]
clips = "../rig-a.toml"
weight = 2

[[fold.train]]
clips = "../rig-b.toml"
groups = ["s1", "s2"]
weight = 1
```

All lists of a fold must share one `[preprocess]` recipe.

## Starting from a trained network

`init_from` starts a fold from a checkpoint's weights, with a fresh optimiser and
schedule. It needs less data than training from scratch, provided your clip lists'
`[preprocess]` (frame size, box size, `select_fs`, `output_fs`) matches the checkpoint's.

```toml
[[fold]]
name = "fine-tune"
init_from = "../runs/reference/best.pt"
```

## Training settings

Every setting under `[fold.train_params]` is optional. The defaults are the reference
network's. See [Experiments reference](../reference/experiments.md) for all of them.
The ones you are most likely to change:

| Setting | Default | Meaning |
| --- | --- | --- |
| `epochs` | 200 | maximum epochs |
| `steps_per_epoch` | 200 | optimiser steps per epoch |
| `batch_size` | 4 | windows per step |
| `window` | 512 | window length, in output samples (8.5 s at 60 Hz) |
| `lr` | 3e-4 | peak learning rate |
| `rate_balance` | 0 | 0 to 1: draw windows evenly across breathing rates |
| `w_onset` | 0.5 | weight of the onset head's loss (0 trains no onset head) |

## Running

```bash
uv run zephyr run experiments/mouse-5-held-out.toml --smoke    # always first
uv run zephyr run experiments/mouse-5-held-out.toml
```

- A seed whose `best.pt` exists is not trained again: `zephyr run` only scores it.
- `--resume` continues stopped runs from `last.pt`.
- `--restart` deletes each run folder and trains from scratch.
- `--evaluate-only` never trains: it scores `best.pt` as it is.
- Without a GPU: `--device cpu --amp off --num-workers 0`.

Use a new `output_dir` for a new experiment rather than deleting old results.
