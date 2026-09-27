# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video, covering
everything from raw video to a trained, evaluated checkpoint.

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), the AWS CLI, and
`ffmpeg`/`ffprobe` on `PATH`. A display is only needed for the optional
annotation step. Everything runs through the single `zephyr <subcommand>`
command; `zephyr <subcommand> --help` only prints a one-line summary, so see
each module's own docstring (e.g. `src/zephyr/train.py`) for its full set of
flags.

## Clone to an evaluated network

1. **Install.**

   ```bash
   git clone https://github.com/AllenNeuralDynamics/zephyr
   cd zephyr
   uv sync --locked
   ```

2. **Download the labelled public training data** into `data/train/` (ignored
   by Git; no AWS credentials needed):

   ```bash
   aws s3 sync --no-sign-request --exclude "*" --include "video_face_*" --include "thermistor_*" s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train/ data/train/
   ```

3. **Crop boxes** -- already checked in at `artifacts/session_boxes_face.json`
   for all 16 public sessions, so skip to step 4 unless you have new data or
   want a different crop:

   ```bash
   uv run zephyr annotate --prepare --splits train
   uv run zephyr annotate --splits train --width 360 --height 270 --box-size 96
   ```

4. **Preprocess** into channel arrays and training targets (~45 min, ~20 GiB
   under `data/features/`):

   ```bash
   uv run zephyr preprocess --boxes-json artifacts/session_boxes_face.json
   ```

5. **Train.** Reserves sessions 9, 10, and 12 (`artifacts/holdout_sessions.json`)
   from training/validation/checkpoint selection. Add `--device cpu --amp off`
   without a CUDA GPU; `--channels gray` (or `gray+diff`, `diff+flow`, ...)
   selects a channel subset without reprocessing.

   ```bash
   uv run zephyr train
   ```

6. **Evaluate** the trained checkpoint on the reserved sessions:

   ```bash
   uv run zephyr evaluate --checkpoint runs_all/zephyr/<run>/best.pt --plot
   ```

   Prints per-clip correlation and event F1; `--plot` also writes diagnostic
   plots under the run's `diagnosis/` directory.

## Beyond the quickstart

- `zephyr benchmark` / `benchmark-data` / `benchmark-report` run a fixed
  train/test factorial sweep against the organizer's official split, as an
  alternative to the single-run workflow above.
- `zephyr.infer.predict_clip` runs inference on an arbitrary preprocessed clip
  and checkpoint; see its docstring.
