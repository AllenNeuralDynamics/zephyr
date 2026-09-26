# Zephyr

Zephyr is a CNN plus temporal convolutional network for predicting a mouse's
breathing trace from video. It includes the model, channel encoding, crop
annotation, feature preprocessing, training targets, windowed datasets,
training, and whole-clip inference. Its model code and relevant Git history
were extracted from the
[Breathing from Video Challenge](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge).

Zephyr has no dependency on the challenge scorer. Local training diagnostics
are for choosing checkpoints. The challenge repository retains submission
packaging, official scoring, held-out evaluation, and competition plots.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- AWS CLI for downloading the public dataset. No AWS credentials are needed.
- `ffmpeg` and `ffprobe` on your `PATH` for video decoding.
- A display and Tk support for interactive crop annotation. Preprocessing
  and training can run headless.

Run these commands from the root of a Zephyr clone:

```bash
uv sync
```

The default training device is CUDA when available, otherwise CPU. The package
declares PyTorch as a dependency; install a build suitable for your hardware.

## Download the data

The public training data contains videos, frame timestamp parquets, and
thermistor parquets. Download it to `data/train/`:

```bash
aws s3 sync --no-sign-request s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train/ data/train/
```

Use the AWS CLI for dataset downloads; this command works in Bash and PowerShell. `data/` is ignored by Git. A clip uses matching
`video_face_{session}_part_{part}.mp4`,
`video_face_{session}_part_{part}.parquet`, and
`thermistor_{session}_part_{part}.parquet` files. The model does not need the
private test split for training.

## Choose the crop

`artifacts/session_boxes_face.json` contains the hand-placed face-camera crop
boxes from the original baseline. Reuse it as-is to reproduce that geometry.
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
`artifacts/holdout_sessions.json` reserves sessions that are excluded from
training and checkpoint selection. Validation uses a time tail of the
remaining clips; local full-clip correlation selects `best.pt`. Runs go under
`runs_all/zephyr/<timestamp>-<channels>/` and contain `best.pt`, `last.pt`,
`history.json`, and per-clip validation results.

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
