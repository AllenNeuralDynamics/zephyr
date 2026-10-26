# Clip lists

A clip list is a TOML file naming videos, the recipe they are preprocessed with, and
each clip's crop box and breathing labels. Paths are relative to the file. A typo in a
key is an error, never silently ignored. `zephyr schema <dir>` writes a JSON schema for
editor validation.

```toml
[preprocess]
target_size = [360, 270]          # frame size boxes are placed on
select_fs = 60                    # rate of the frames the CNN sees, Hz

[derive]                          # optional: how file names map to files
pattern = '(?P<group>.+)_span(?P<span>\d+)_FaceCamera$'
timestamps = "{stem}.parquet"
thermistor = "{group}_span{span}_thermistor.parquet"

[events]                          # the list's breathing-event detector
method = "adaptive_prominence"
params = {highpass_hz = 1.0, distance_s = 0.05, prominence_frac = 0.2, window_s = 1.5}

[[clip]]
video = "../data/vr-foraging/behavior_789903_2025-10-29_20-06-45_span1_FaceCamera.mp4"
box = [230, 98, 96, 96]           # x, y, w, h in target_size pixels
selection = "include"
reviewed = true
excluded = [[112.4, 118.0]]       # seconds on the sensor's clock
```

## `[preprocess]`

Part of every feature cache key: change it and the clips are preprocessed again. All
lists in one fold must share it.

| Key | Default | Meaning |
| --- | --- | --- |
| `target_size` | required | `[width, height]` the video is shrunk to before cropping; boxes are in these pixels |
| `select_fs` | 60 | rate of the frames the CNN sees, Hz; at most the camera's rate |
| `output_fs` | 60 | rate of the grid the network predicts on, Hz. The TCN's context in seconds scales with `1 / output_fs`. Scoring is always at 60 Hz |
| `onset_sigma_s` | 0.02 | width of the onset head's Gaussian target, s |

## `[derive]`

How a clip's omitted `timestamps`, `thermistor` and `group` are derived from its video
name.

| Key | Default | Meaning |
| --- | --- | --- |
| `pattern` | `video_.+_(?P<group>\d+)_part_(?P<part>\d+)$` | regex searched in the video's file stem; must capture `group` |
| `timestamps` | `{stem}.parquet` | template, relative to the video's folder |
| `thermistor` | `thermistor_{group}_part_{part}.parquet` | template, relative to the video's folder |

## `[events]`

The breathing-event detector of every clip without its own `events`. Without
`[events]`, a list uses `peak_detection` with its defaults.

| Key | Meaning |
| --- | --- |
| `method` | `peak_detection`, `adaptive_prominence` or `hysteresis_cycles` |
| `params` | the detector's parameters (below); omitted ones take their defaults |

| Detector | Parameters (defaults) |
| --- | --- |
| `peak_detection` | `highpass_hz` (0.2), `distance_s` (0.05), `prominence_frac` (0.1, of the whole clip's range) |
| `adaptive_prominence` | `highpass_hz` (0.2), `distance_s` (0.05), `prominence_frac` (0.25, of the local range), `window_s` (2.0) |
| `hysteresis_cycles` | `highpass_hz` (0.2), `baseline_s` (1.0), `hysteresis_frac` (0.15), `window_s` (2.0) |

Changing a detector rebuilds only the clips' targets at the next `preprocess`, without
decoding video.

## `[[clip]]`

| Key | Default | Meaning |
| --- | --- | --- |
| `video` | | the clip's video. Omit it for a sensor-only entry (a whole session), which only the Breathing page reads |
| `timestamps` | derived | frame-timestamp parquet, column `Time` |
| `frame_rate` | none | frames per second of a video with no timestamps, instead of `timestamps`: frame *i* is at *i* / `frame_rate` s, which assumes the camera dropped no frames (`clips scan --frame-rate`, `predict --frame-rate`) |
| `thermistor` | derived | breathing-sensor parquet, columns `Time` and `Signal`; `false` for an unlabelled clip |
| `group` | derived | the recording (session). A recording is the pair (video folder, group) |
| `frame` | the list's `target_size` | `[width, height]` this video is shrunk to instead, e.g. another camera's (`clips frame --input-scale`, or the Boxes page's **Input scale**) |
| `box` | none | `[x, y, w, h]` in the clip's frame pixels (its `frame`, else `target_size`). Every clip needs one before preprocessing, and all must share one size |
| `events` | the list's | this clip's own detector, `{method = "...", params = {...}}` |
| `selection` | none | `"exclude"`: left out of everything downstream. `"include"`: a note only |
| `reviewed` | `false` | a person confirmed the mark and spans; suggestions never change it |
| `excluded` | `[]` | `[[start, end], ...]` seconds on the sensor's clock: never scored or trained on |
| `included` | `[]` | spans marked good: a note only |

## Who writes what

| Writer | Writes |
| --- | --- |
| `clips scan` / Clips page | `[preprocess]`, `[derive]`, `[events]`, the `video` of each clip |
| Boxes page, `annotate --auto` | `box`, and `frame` (Boxes page only) |
| `clips frame` | `frame`, `box` |
| Breathing page, `annotate events --auto` | `[events]`, `events`, `selection`, `reviewed`, `excluded`, `included` |
| you, in a text editor | anything; comments and layout are kept by every writer |

Rewrites go through `tomlkit`, so your comments, ordering and layout survive.
