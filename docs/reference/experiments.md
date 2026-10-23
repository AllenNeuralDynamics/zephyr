# Experiments

An experiment is a TOML file of folds, seeds and where results go. Paths are relative to
the file, and a typo in a key is an error. See [Designing
experiments](../guides/experiments.md) for how to use them.

## Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `seeds` | required | each fold runs once per seed; distinct |
| `output_dir` | required | runs go to `<output_dir>/<fold name>/seed-<n>/` |
| `features_dir` | required | the feature cache the folds read |
| `device` | CUDA if present | machine setting; the command line overrides it |
| `amp` | `bf16` | `bf16`, `fp16` or `off` |
| `num_workers` | 4 | data-loading workers |
| `save_traces` | `false` | also write each test clip's predicted trace into `traces/` |
| `[[fold]]` | required | one block per fold |

## `[[fold]]`

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | the run folder: letters, digits, `.`, `_`, `-`; distinct |
| `init_from` | none | a checkpoint whose weights to start from (fresh optimiser and schedule) |
| `[[fold.train]]` | required | one or more training sources |
| `[fold.val]` | none | clips scored whole to keep `best.pt` and stop early, instead of `val_fraction` |
| `[fold.test]` | required | report group name → clips to score; `all` is reserved |
| `[fold.train_params]` | defaults | training settings (below) |

A **selection** (each test group, `[fold.val]`, and each training source) names a clip
list and optionally some of its groups:

```toml
held_out = { clips = "clips.toml", groups = ["3", "4"] }   # omit groups for the whole list
```

A training source also takes `weight` (default 1), its relative share of the sampled
windows.

## `[fold.train_params]`

### Schedule

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | 200 | maximum epochs |
| `steps_per_epoch` | 200 | optimiser steps per epoch |
| `batch_size` | 4 | windows per step |
| `window` | 512 | window length, in output samples |
| `lr` | 3e-4 | peak learning rate |
| `warmup_steps` | 200 | linear warm-up |
| `weight_decay` | 1e-4 | |
| `grad_clip` | 1.0 | gradient norm clip |
| `dropout` | 0.1 | |
| `ema_decay` | 0.999 | moving average of the weights, selected on once it has settled; 0 disables it |
| `save_every` | 0 | also keep `epoch-<n>.pt` every this many epochs |

### Validation

| Key | Default | Meaning |
| --- | --- | --- |
| `val_fraction` | 0.25 | the tail of every training clip held out for validation, without `[fold.val]`; 0 trains a fixed `epochs` |
| `score_every` | 4 | epochs between validation scorings |
| `patience` | 8 | scorings without improvement before stopping |
| `min_epochs` | 24 | never stop before this |
| `val_windows` | 2048 | windows scored per validation |

### Losses and architecture

| Key | Default | Meaning |
| --- | --- | --- |
| `w_corr` | 1.0 | weight of the trace's correlation loss |
| `w_onset` | 0.5 | weight of the onset head's loss |
| `scales` | `[1, 4, 16]` | time scales of the correlation loss, in samples |
| `temporal_kernel` | 1 | frames the encoder's first layer spans (odd); 3 or 5 make it a 3D convolution |
| `signal_pool` | 1 | the signal head reads the TCN output average-pooled by this factor |
| `onset_input` | `full` | what the onset head reads: `full`, `pooled` or `both` (the last two need `signal_pool > 1`) |

### Sampling

| Key | Default | Meaning |
| --- | --- | --- |
| `rate_balance` | 0 | 0 to 1: weight training windows by `share(rate bin) ** -rate_balance`; 1 gives every rate bin equal mass |
| `rate_bins_hz` | 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15 | bin edges for `rate_balance` |

### Inference

| Key | Default | Meaning |
| --- | --- | --- |
| `infer_window` | 1024 | window length for whole-clip inference |
| `frame_chunk` | 256 | frames encoded per batch |

## `[fold.train_params.augmentation]`

On by default.

| Key | Default | Meaning |
| --- | --- | --- |
| `select_jitter` | 0.25 | jitter of which camera frames are selected |
| `shift_px` | 4 | random shift of the crop, pixels |
| `scale` | 0 | random zoom (fraction) |
| `rotation` | 0 | random rotation about the crop's centre, degrees |
| `blur` | 0 | maximum Gaussian blur sigma, pixels |
| `brightness` | 0.15 | random brightness change |
| `contrast` | 0.2 | random contrast change |
| `noise` | 2.0 | pixel noise, gray levels |
| `flip` | 0 | probability of a horizontal flip |

## Checking an experiment

```python
from pathlib import Path

from zephyr.config import Experiment, load

experiment = load(Experiment, Path("experiments/first-network.toml"))
for fold in experiment.folds():
    print(fold.name, fold.train_params.epochs)
```

Loading refuses a video on both sides of a fold, unlabelled training clips, missing
boxes and mixed `[preprocess]` recipes, and says which clip failed.
