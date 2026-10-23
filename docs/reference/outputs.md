# Output files

## A project folder

```text
mouse-breathing/
  clips.toml                    # clip lists
  experiments/<name>.toml       # experiments, written once
  frames/frame_<video>-<hash>.npy   # one cached frame per clip, for boxes
  cache/                        # the feature cache (see below)
  runs/<experiment>/<fold>/seed-<n>/   # a run
  runs/<experiment>/smoke/<fold>/seed-<n>/   # its smoke run
  predictions/<name>/           # predictions and quality-control reports
```

## A run folder

`zephyr run` trains each fold and seed into `<output_dir>/<fold name>/seed-<n>/`:

| File | Holds |
| --- | --- |
| `config.json` | the fully resolved fold, seed and machine settings the run used |
| `train_videos.json` | the videos it trained on; `evaluate` refuses them |
| `history.json` | one row per epoch: learning rate, losses, validation scores, seconds |
| `best.pt` | the checkpoint with the best validation score; the one to use |
| `last.pt` | the latest weights, for `--resume` |
| `epoch-<nnn>.pt` | weights every `save_every` epochs (off by default) |
| `best_per_clip.json` | each validation clip's scores at `best.pt` |
| `evaluation.json` | the test scores (below) |
| `traces/feat-<prefix>.parquet` | each test clip's predicted trace, with `save_traces` or `--save-traces` |

A checkpoint carries its architecture, preprocessing recipe and training record, so it
is all `evaluate` and `predict` need.

## evaluation.json

```json
{
  "clips": [
    {"clip_id": "vr-foraging/behavior_806527_2025-10-24_18-04-57_span1_FaceCamera",
     "recording": "C:/data/vr-foraging#behavior_806527_2025-10-24_18-04-57",
     "group": "test", "correlation": 0.81, "inhale_f1": 0.88, "exhale_f1": 0.85,
     "kl_ibi": 0.04, "head_inhale_f1": 0.93, "n_head_events": 1402}
  ],
  "per_recording": [ ... ],
  "groups": {
    "test": {"recordings": [ ... ], "summary": {"correlation": 0.81, "correlation_sd": 0.0, "correlation_n": 1, ...}},
    "all":  { ... }
  },
  "summary": { ... },
  "summary_by_recording": { ... },
  "composite": 0.83
}
```

A metric that cannot be computed (no inhales, for example) is `null`. See
[Scoring](scoring.md) for what each metric means.

## Traces

`traces/feat-<prefix>.parquet` (from a run) holds one row per output sample of a test
clip:

| Column | Holds |
| --- | --- |
| `time` | seconds, on the video's clock |
| `signal` | the z-scored predicted trace that was scored |
| `onset` | the onset head's inhale probability (networks with an onset head) |

## Predictions

`zephyr predict` writes `<video stem>.prediction.parquet` per video:

| Column | Holds |
| --- | --- |
| `time` | seconds, on the video's clock, on a 60 Hz grid |
| `signal` | the z-scored predicted breathing trace (inhale up) |
| `onset` | the onset head's inhale probability, 0 to 1 |
| `inhale` | true where the onset head found an inhale |

```python
import pandas as pd

p = pd.read_parquet(
    "predictions/new-animals/"
    "behavior_808728_2025-10-24_20-45-03_span1_FaceCamera.prediction.parquet"
)
inhales = p.loc[p["inhale"], "time"]
rate_hz = 1 / inhales.diff().median()
```

## Quality-control reports

`zephyr quality-control run` writes `<video stem>.qc.json` (the measures and checks,
whole clip and per window) and `<video stem>.qc.parquet` (time, trace, onset
probability, inhale). See [Quality control](../guides/quality-control.md).

## The feature cache

One entry per clip, keyed by video, preprocessing recipe and box:

| File | Holds |
| --- | --- |
| `<prefix>.json` | the entry's sidecar: which clip, recipe and box it was made from |
| `gray-<prefix>.npy` | the clip's cropped gray frames at `select_fs`, uint8: the network's only input |
| `ftime-<prefix>.npy`, `time-<prefix>.npy` | the selected frames' times and the 60 Hz output grid |
| `target-<prefix>.parquet` | the sensor's training target on the output grid |
| `events-<prefix>.npz` | the clip's event set: inhales, exhales and excluded spans |

Entries are safe to share between lists and experiments. A changed box or recipe makes
a new entry. A changed detector rebuilds only the targets, without decoding video. Read
entries through `zephyr.features`; never rename them (see [Working across
machines](../guides/machines.md)).
