# Zephyr

Zephyr is a CNN plus temporal convolutional network for predicting a mouse's
breathing trace from video. It includes the model, channel encoding, crop
annotation, feature preprocessing, training targets, windowed datasets,
training, and whole-clip inference. Its model code and relevant Git history
were extracted from the
[Breathing from Video Challenge](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge).

Zephyr has no dependency on the challenge scorer. It includes local training
diagnostics, held-out evaluation, and diagnostic plots. The challenge
repository retains submission packaging and the authoritative competition
scorer.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- AWS CLI for downloading the public dataset. No AWS credentials are needed.
- `ffmpeg` and `ffprobe` on your `PATH` for video decoding.
- A display and Tk support for interactive crop annotation. Preprocessing
  and training can run headless.

Run these commands from the root of a Zephyr clone:

```bash
uv sync --locked
```

The tracked `.python-version` prefers Python 3.13, then 3.12 or 3.11;
`uv.lock` fixes package versions for a repeatable install.

The package also installs `zephyr-annotate`, `zephyr-preprocess`,
`zephyr-train`, and `zephyr-evaluate` commands. The examples below
use the equivalent `python -m zephyr.<module>` form.

The default training device is CUDA when available, otherwise CPU. The package
declares PyTorch as a dependency; install a build suitable for your hardware.

## Download the data

The [Breathing from Video Challenge](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge)
publishes a labelled training dataset in a public S3 bucket. Each clip needs
three matching files: the face-camera video, its frame timestamps, and the
thermistor trace. From the Zephyr repository root, use the AWS CLI to download
only those files into `data/train/`:

```bash
aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train/ data/train/
```

The filter selects `video_face_{session}_part_{part}.mp4`, its matching
`video_face_{session}_part_{part}.parquet`, and
`thermistor_{session}_part_{part}.parquet`. It excludes the side-camera files.
The command works in Bash and PowerShell; no AWS credentials are needed. A
dry run with `--dryrun` listed 96 files (32 clips times three files) from the
public train prefix. `data/` is ignored by Git. The private test split is not
needed to train or validate Zephyr.

## Choose the crop

`artifacts/session_boxes_face.json` contains the hand-placed face-camera crop
boxes for all 16 public training sessions from the original baseline. Reuse it
as-is to reproduce that geometry; no annotation step is needed for these clips.
For new data or a different crop, cache a frame per clip and open the
annotation UI:

```bash
uv run python -m zephyr.annotate --prepare --splits train
uv run python -m zephyr.annotate --splits train --width 360 --height 270 --box-size 96
```

Click to place the box. Arrow keys move it; `[` and `]` resize it; `n` and `p`
change sessions. Edits save automatically to
`artifacts/session_boxes_face.json`. The annotation UI requires a desktop
display. If you switch cameras, use `--camera side` and create a corresponding
`artifacts/session_boxes_side.json`.

## Preprocess

Decode and crop each clip into memory-mappable channel arrays. Preprocessing
also filters and resamples the thermistor onto the model's 60 Hz output grid
and writes the training targets. The public training split is the default:

```bash
uv run python -m zephyr.preprocess --boxes-json artifacts/session_boxes_face.json
```

This takes about 45 minutes and writes roughly 20 GiB under `data/features/`
for the full public dataset. It requires crop boxes for every selected clip.
Use `--limit 1 --workers 1` for a small check, or `--sessions 7` to redo one
session after changing its box. Run `uv run python -m zephyr.preprocess --help`
for the other options.

## Train

```bash
uv run python -m zephyr.train
```

Training uses every non-reserved public session. The checked-in
`artifacts/holdout_sessions.json` reserves sessions 9, 10, and 12; they are
excluded from training, normalization statistics, validation, and checkpoint
selection. Keep this file unchanged if you want comparable runs. The default
validation set is the last 25% in time of each remaining clip, leaving its
first 75% for training. Thus validation measures performance later in the
same sessions; it does not estimate generalization to unseen sessions.

Every epoch writes windowed `val_loss` and `val_corr` to `history.json`.
Every four epochs (and at the final epoch), Zephyr predicts the full clips and
measures only their validation tails. `xcorr`, `inhale_f1`, and `exhale_f1`
are local diagnostics; `xcorr` selects `best.pt` and drives early stopping.
Runs go under `runs_all/zephyr/<timestamp>-<channels>/` and contain `best.pt`,
`last.pt`, `history.json`, and `best_per_clip.json`. The latter lists each
validation clip's correlation and event F1. The reserved sessions remain
untouched until final held-out evaluation.

The default inputs are gray, signed frame difference, and two optical-flow
channels. Preprocessing always stores all four; `--channels` selects the
subset used for a training run without regenerating features:

```bash
uv run python -m zephyr.train --channels gray
uv run python -m zephyr.train --channels gray+diff
uv run python -m zephyr.train --channels diff+flow
```

Use `--device cpu --amp off` on a machine without a suitable CUDA device.
`--resume` continues from the latest `last.pt` in the selected run directory.
Run `uv run python -m zephyr.train --help` for augmentation, validation, and
optimization settings. Local diagnostics are not official challenge scores.

## Evaluate reserved sessions

After training, evaluate a checkpoint on the three reserved public sessions.
Keep the raw thermistor parquets in `data/train/`; the evaluator reads them
directly and needs the preprocessed features for those sessions.

```bash
uv run python -m zephyr.evaluate --checkpoint runs_all/zephyr/<run>/best.pt --plot
```

The command checks that the checkpoint reserved every session it scores. It
prints per-clip correlation, inhale and exhale event F1, and inter-breath
interval divergence. With `--plot`, it writes rate breakdown and reserved
session diagnostic plots under the run's `diagnosis/` directory. Use `--out`
to save metrics as JSON. Several checkpoints can be ensembled when `--plot`
is omitted.

`zephyr.evaluation` contains local copies of the metric calculations used
by this command, so evaluation does not import the challenge scorer. The
[challenge scorer](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge/tree/main/scoring)
remains authoritative for Codabench results; changes there may make these
local metrics differ. Because inspecting reserved results can influence model
choices, use them for final assessment after choosing a checkpoint with the
training validation tail.

## Run inference

`zephyr.infer.predict_clip` stitches overlapping windows into a full trace.
For a preprocessed public clip and a training checkpoint:

```python
from pathlib import Path

import numpy as np
import torch

from zephyr.channels import ChannelSet
from zephyr.dataset import load_manifest
from zephyr.infer import predict_clip
from zephyr.model import BreathingNet

state = torch.load("runs_all/zephyr/<run>/best.pt", map_location="cpu", weights_only=False)
channels = ChannelSet.parse(state.get("channels") or state["feature_config"]["channel_names"])
model = BreathingNet(channels=channels)
model.load_state_dict(state["model"])
_, clips = load_manifest(Path("data/features"), "train", "face")
clip = clips[0]
signal, onset_probability = predict_clip(
    model, clip, state["mean"], state["std"], device="cpu", amp_dtype=None
)
times = np.load(clip.times)
```

`signal` and `onset_probability` align with `times`. To package predictions
for Codabench or compute official challenge scores, use the separate
[challenge repository](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge).
