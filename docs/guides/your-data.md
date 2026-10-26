# Preparing your own data

This guide covers what your recordings must look like before zephyr can use them, and
how to lay them out so a clip list stays short. Once your files follow it, the
[tutorial](../tutorial/index.md) works on them unchanged.

## What zephyr needs

For each recording, three files:

| File | Format | Notes |
| --- | --- | --- |
| **Video** | anything `ffmpeg` decodes (`.mp4`, `.avi`, ...) | A fixed camera that sees the animal's snout. Gray or colour; it is converted to gray. |
| **Frame timestamps** | `.parquet`, a float column `Time` in seconds | One row per frame, in decode order, increasing. |
| **Breathing reference** | `.parquet`, float columns `Time` (seconds) and `Signal` | A thermistor or any other breathing sensor. Leave it out for videos you only want predictions on. |

## Rules the files must follow

Each rule is checked, and each error names the clip it failed on. `zephyr clips check`
(**Check files** in the app) checks them all at once.

**One clock.**
:   Both `Time` columns must be on the same clock. Their origin does not matter, and
    video and sensor need not start together: scoring compares them over the span both
    cover. Zephyr cannot synchronise them for you.

**Every frame has a timestamp.**
:   A video whose decoded frame count differs from its number of timestamp rows is
    refused. With no hardware timestamps, write the video's own (see
    [below](#timestamps-from-the-video)).

**Inhale is up.**
:   An inhale is a **peak** of `Signal` and an exhale a trough, as for a thermistor in
    the nose, which cools when the animal breathes in. Negate a sensor whose inhale is a
    trough (a pressure sensor, for example) before you write the file. Units and offset
    do not matter, because every clip's trace is z-scored.

**The sensor is fast enough.**
:   `Signal` is low-pass filtered at 40 Hz at its own rate, so it must be sampled above
    80 Hz. The reference thermistors run at about 1 kHz.

**The camera is fast enough.**
:   The network sees frames at `select_fs` (60 Hz by default), picked from the camera's
    own frames, so the camera must run at least that fast. For a slower camera, set
    `select_fs` to its rate or lower (for example 30 for a 30 Hz camera). The prediction
    is still made and scored on a 60 Hz grid.

## Laying out the files

A clip list names only the videos. Zephyr derives each clip's timestamps, breathing
file and session (`group`) from the video's file name. The default naming is:

```text
data/my-rig/
  video_face_12_part_1.mp4        # video_<camera>_<session>_part_<part>.mp4
  video_face_12_part_1.parquet    # its timestamps: <video stem>.parquet
  thermistor_12_part_1.parquet    # thermistor_<session>_part_<part>.parquet
```

The session number (`12`) becomes the clip's `group`. Name your files this way and
nothing else needs configuring.

### Other names

For other names, give the list a `[derive]` table (**Other file names** on the app's
Clips page, or `--pattern`, `--timestamps` and `--thermistor` for `clips scan`).

- `pattern` is a regular expression searched in the video's file name, without the
  extension. It must capture `group`, and every group it captures can be used by the
  two templates.
- `timestamps` and `thermistor` are templates for paths relative to the video's
  folder, over `{stem}` and the pattern's groups.

```toml
# Files named mouse3_day2.mp4, mouse3_day2_times.parquet, mouse3_day2_breath.parquet
[derive]
pattern = '(?P<group>.+)'
timestamps = "{stem}_times.parquet"
thermistor = "{stem}_breath.parquet"
```

### Files in session folders

You can also give any clip's files explicitly, which overrides `[derive]`. Do this when
the session is in a folder name rather than in the file name:

```toml
[[clip]]
video = "sessions/2026-03-01/face.mp4"
timestamps = "sessions/2026-03-01/face_times.parquet"
thermistor = "sessions/2026-03-01/sniff.parquet"
group = "2026-03-01"
```

!!! tip "One list for a whole dataset"
    Put every session of a dataset in **one clip list**, each session its own group.
    Experiments then choose which sessions train and which are scored by group, so you
    never copy clips between lists.

### Videos without a sensor

Mark a clip `thermistor = false` (or scan with `--unlabelled`). It can be boxed,
preprocessed and predicted, but never trained on or scored.

## Timestamps from the video

With no hardware timestamps, give a video its frame rate instead: `frame_rate = 240` on
its clip (or `clips scan --frame-rate 240`, `predict --frame-rate 240`) puts frame *i*
at *i* / 240 s. That is only right if the camera dropped no frames, and the times start
at 0, not on any other clock, so it suits predicting rather than training against a
sensor.

To train against a sensor, the video's own frame times will do if the sensor's `Time`
starts when the video starts. This script writes `<video stem>.parquet` next to each
video given on the command line:

```python title="write_timestamps.py"
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

```bash
uv run python write_timestamps.py data/my-rig/*.mp4
```

## How much data

The reference network trained on 5-minute clips from the sessions of about 30 animals.
With only a few animals, expect lower scores on new ones. Starting from a trained
checkpoint (`init_from` in the fold, see [Designing experiments](experiments.md)) needs
less data, provided your `[preprocess]` recipe (frame size, box size, rates) matches the
one it was trained with.

## Cameras and crops

- Place boxes on frames shrunk so the box covers the snout, mouth and whisker pad. The
  reference setting is 360 × 270 frames with 96-pixel boxes.
- One box covers a whole recording. If the camera moved within a recording, give the
  affected clip its own box (++o++ on the Boxes page).
- Every clip of a list must crop to the same box size.
