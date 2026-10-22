# Zephyr on your own data

This guide takes you from your own recordings to a network trained and scored on them,
on one machine. The [README](../README.md) is the reference for every command and
setting; this page says what your data must look like and in what order to run things.

## What zephyr needs

For each recording, three files:

| File | Format | Notes |
| --- | --- | --- |
| **Video** | anything `ffmpeg` decodes (`.mp4`, `.avi`, ...) | A fixed camera that sees the animal's snout. Gray or colour (it is converted to gray). |
| **Frame timestamps** | `.parquet`, a float column `Time` in seconds | One row per frame, in the order the frames decode, increasing. |
| **Breathing reference** | `.parquet`, float columns `Time` (seconds) and `Signal` | A thermistor or any other breathing sensor. Leave it out for videos you only want predictions on. |

Rules the files must follow. Each is checked, and each error names the clip it failed on:

- **One clock.** The two `Time` columns must be on the same clock. Their origin does
  not matter, and video and sensor need not start together: scoring compares them over
  the span both cover. Zephyr cannot synchronise them for you.
- **Every frame has a timestamp.** `preprocess` refuses a video whose decoded frame
  count differs from the number of timestamp rows. If you have no hardware timestamps,
  write the video's own (see [Timestamps from the video](#timestamps-from-the-video)).
- **Inhale is up.** An inhale is a **peak** of `Signal` and an exhale a trough, as for
  a thermistor in the nose, which cools when the animal breathes in. Negate a sensor
  whose inhale is a trough (a pressure sensor, for example) before you write the file.
  Units and offset do not matter, because every clip's trace is z-scored.
- **The sensor is fast enough.** `Signal` is low-pass filtered at 40 Hz at its own rate,
  so it must be sampled above 80 Hz. The corpus thermistors run at about 1 kHz.
- **The camera is fast enough.** The network sees frames at `select_fs` (60 Hz by
  default), picked from the camera's own frames, so the camera must run at least that
  fast. For a slower camera, set `select_fs` to its rate or lower (for example 30 for a
  30 Hz camera). The prediction is still made and scored on a 60 Hz grid.

## Laying out the files

A clip list names only the videos. Zephyr derives each clip's timestamps, breathing
file and recording (`group`) from the video's file name, so a list stays short.

The default naming is that of the example dataset:

```
data/my-rig/
  video_face_12_part_1.mp4        # video_<camera>_<session>_part_<part>.mp4
  video_face_12_part_1.parquet    # its frame timestamps: <video stem>.parquet
  thermistor_12_part_1.parquet    # thermistor_<session>_part_<part>.parquet
```

The session number (`12`) is the clip's `group`. If you name your files this way,
nothing else needs configuring.

For other names, add a `[derive]` table to the clip list. `pattern` is a regular
expression searched in the video's file stem: it must capture `group`, and every
group it captures can be used by the two templates. The templates are paths relative
to the video's folder:

```toml
# Files named mouse3_day2.mp4, mouse3_day2_times.parquet, mouse3_day2_breath.parquet
[derive]
pattern = '(?P<group>.+)'
timestamps = "{stem}_times.parquet"
thermistor = "{stem}_breath.parquet"
```

You can also give any clip's files explicitly, which overrides `[derive]`:

```toml
[[clip]]
video = "sessions/2026-03-01/face.mp4"
timestamps = "sessions/2026-03-01/face_times.parquet"
thermistor = "sessions/2026-03-01/sniff.parquet"
group = "2026-03-01"
```

Write `group` explicitly when the session lives in a folder name rather than in the
file name. A fold picks clips by group, so give every session its own group.

### Timestamps from the video

With no hardware timestamps, a constant-rate video's own frame times will do. They are
only correct if the breathing sensor's `Time` starts when the video starts. This writes
`<video stem>.parquet` next to each video:

```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

for video in map(Path, sys.argv[1:]):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    times = [float(line.strip(",")) for line in out.split() if line.strip(",")]
    pd.DataFrame({"Time": times}).to_parquet(video.with_suffix(".parquet"), index=False)
```

## Setting up a machine

- **Software.** Python 3.11+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` and
  `ffprobe` on `PATH`. From a clone of the repository:

  ```bash
  uv sync --locked --all-packages --extra gpu   # CUDA; --extra cpu without a GPU
  uv sync --locked --all-packages --extra gpu --extra ui   # also the event-detector window
  ```

  Placing crop boxes (`zephyr annotate`) opens a Tk window, so it needs a display. Do it
  on a desktop, then copy the clip list to a headless machine for training.
- **GPU.** Training wants a CUDA GPU. It runs on CPU at about 1.7 s per step, and a
  full run of 200 epochs × 200 steps is about 19 hours. Without a GPU, pass
  `--device cpu --amp off --num-workers 0`.
- **Disk.** The feature cache stores each clip's cropped gray frames as uint8, at
  `select_fs`. A 96 × 96 box at 60 Hz is about 2 GB per hour of video. Decode time
  scales with the camera's native resolution and rate, not with the box. The videos
  are not needed once a clip is cached.

## From files to a trained network

Run the commands from the repository root. The paths below are examples.

1. **Make a clip list.** With the default names, scan the folder. The scan checks that
   every derived file exists:

   ```bash
   uv run zephyr clips scan data/my-rig --glob 'video_*.mp4' -o configs/my-rig.toml
   ```

   `--unlabelled` marks every clip `thermistor = false`. The scan knows only the default
   names, so for any other layout write the list yourself. It needs a `[preprocess]`
   table, your `[derive]`, and one `[[clip]]` per video, with paths relative to the
   list:

   ```toml
   [preprocess]
   target_size = [360, 270]

   [derive]
   pattern = '(?P<group>.+)'
   timestamps = "{stem}_times.parquet"
   thermistor = "{stem}_breath.parquet"

   [[clip]]
   video = "../data/my-rig/mouse3_day2.mp4"
   ```

   Every later command loads the list and names any file it cannot find.

2. **Place one crop box per recording**, around the snout and nostrils, the same size
   for every clip:

   ```bash
   uv run zephyr annotate configs/my-rig.toml --prepare
   uv run zephyr annotate configs/my-rig.toml --width 360 --height 270 --box-size 96
   ```

   `--width/--height` set `target_size`, the frame size boxes are placed on. The video
   is scaled to it before cropping. The corpus uses 360 × 270 with 96-pixel boxes;
   keep both if you want to start from a checkpoint trained on it.

3. **Check the breathing events** (optional; needs `--extra ui`). Training and scoring
   use the inhale peaks a detector finds on `Signal`: `peak_detection` unless you choose
   another. `zephyr annotate events configs/my-rig.toml` shows each clip's trace and
   events and lets you pick a detector, mark bad stretches as `excluded`, and leave out
   bad clips. A sensor that is noisier or slower than a nasal thermistor is the usual
   reason to change the detector.

4. **Preprocess** the list into a feature cache. A re-run does only what changed:

   ```bash
   uv run zephyr preprocess configs/my-rig.toml --cache data/features-my-rig
   ```

5. **Write an experiment.** Start from `examples/experiment.toml`. Each `[[fold]]` names
   the groups it trains on and the groups it scores. Hold out **whole animals** for
   testing: a network scored on new sessions of an animal it trained on looks better
   than it is on a new animal. Give each fold a `[fold.val]` of other held-out animals
   to stop on. Without one, the last quarter of every training clip is used:

   ```toml
   seeds = [17]
   output_dir = "../runs/my-rig"
   features_dir = "../data/features-my-rig"

   [[fold]]
   name = "mouse-5-held-out"
   [[fold.train]]
   clips = "my-rig.toml"
   groups = ["m1-day1", "m1-day2", "m2-day1", "m3-day1"]
   [fold.val]
   clips = "my-rig.toml"
   groups = ["m4-day1"]
   [fold.test]
   mouse_5 = { clips = "my-rig.toml", groups = ["m5-day1", "m5-day2"] }
   ```

   For results you report, rotate which animals are held out (one fold per rotation)
   and pool the test scores.

6. **Smoke-run, then run.** The smoke run takes two optimiser steps and catches a bad
   config in minutes:

   ```bash
   uv run zephyr run configs/my-rig-experiment.toml --smoke
   uv run zephyr run configs/my-rig-experiment.toml
   ```

   Each fold and seed trains into `<output_dir>/<fold>/seed-<n>/`. That folder holds
   `best.pt` and `evaluation.json` with every test clip's correlation, inhale and
   exhale F1, and rate error.

7. **Score a checkpoint on more clips** with `zephyr evaluate`. It refuses clips the
   checkpoint trained on:

   ```bash
   uv run zephyr evaluate --checkpoint runs/my-rig/mouse-5-held-out/seed-17/best.pt \
       --clips configs/new-animals.toml --cache data/features-my-rig --out new.json
   ```

### How much data

The corpus network trained on 5-minute clips from the sessions of about 30 animals
(50 animals on 15 rigs in all, the rest held out; see the
[dataset README](../../zephyr-dataset/README.md)). Our learning curves
(zephyr-benchmarks) show how scores grow with the number of training animals. With
only a few animals, expect lower scores on new ones. Starting from a trained
checkpoint (`init_from = "<best.pt>"` in the fold) needs less data, provided your
`[preprocess]` recipe matches the one it was trained with.

## Predicting on videos without a sensor

Put those videos in a clip list with `thermistor = false`, boxed and preprocessed like
any other, and run quality control. You get the trace and the network's judgement of
it:

```bash
uv run zephyr quality-control run --checkpoint best.pt --clips configs/unlabelled.toml \
    --cache data/features-my-rig --out predictions/
```

Each video gets a `<video stem>.qc.parquet` (columns `time`, `signal`, `onset`, `inhale`:
the z-scored trace on a 60 Hz grid, the onset head's inhale probability, and the
inhales it found), plus a `<video stem>.qc.json` that rates the prediction over the
whole video and per 30 s window. The [README](../README.md#quality-control-without-a-thermistor)
explains the measures. To predict from Python, use `zephyr.infer.predict_clip`.

## Errors you may meet

| Message | Cause and fix |
| --- | --- |
| `cannot derive timestamps: pattern ... does not match the video name` | Your names differ from the default: add `[derive]` or give the paths explicitly. |
| `N file(s) do not exist (mark an unlabelled clip thermistor = false)` | A derived file is missing, or the clip has no breathing reference. |
| `decoded N frames but the timestamp parquet has M rows` | The timestamps are not one row per frame: dropped frames, or timestamps from another file. |
| `select_fs 60.0 Hz exceeds the clip's native rate` | The camera is slower than `select_fs`: lower `select_fs` in `[preprocess]`. |
| `Digital filter critical frequencies must be 0 < Wn < fs/2` | The breathing sensor is sampled at 80 Hz or less. |
| `boxes of mixed sizes` | Every clip must crop to one box size: re-place them with one `--box-size`. |
| `test group ...: ... is also a training clip` | A video is on both sides of a fold: fix the fold's groups. |
| `N clip(s) have no box` | Run `zephyr annotate` on the list. |
