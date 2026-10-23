# Commands

Every step is a subcommand of `zephyr`. Run them with `uv run zephyr <subcommand>` from
a clone. `zephyr <subcommand> --help` prints only a one-line summary; this page lists
every flag.

| Command | Does |
| --- | --- |
| [`gui`](#gui) | the desktop app |
| [`clips scan`](#clips-scan) | make a clip list from a folder of videos |
| [`clips check`](#clips-check) | read every file of a list and report what would be refused |
| [`annotate`](#annotate) | cache one frame per clip, or match boxes from those placed |
| [`annotate events`](#annotate-events) | give every clip the default breathing-event detector |
| [`preprocess`](#preprocess) | decode, crop and gray clips into the feature cache |
| [`run`](#run) | train an experiment's folds and score their test groups |
| [`evaluate`](#evaluate) | score checkpoints on clip lists |
| [`predict`](#predict) | predict breathing on videos, no sensor needed |
| [`quality-control run`](#quality-control-run) | judge predictions without a sensor |
| [`quality-control reference`](#quality-control-reference) | calibrate a checkpoint's onset head, once |
| [`schema`](#schema) | write JSON schemas of the config files |

## gui

```text
zephyr gui [path] [--page {project,clips,boxes,breathing,train,results,predict}]
```

| Flag | Meaning |
| --- | --- |
| `path` | a project folder, or a clip list in one. A new folder becomes a project |
| `--page` | open on this page (default: Clips, or Project without a path) |

Needs the `ui` extra. See [The app](app.md).

## clips scan

```text
zephyr clips scan <directory> --glob <pattern> -o <list.toml> [options]
```

| Flag | Meaning |
| --- | --- |
| `--glob` | video file pattern, e.g. `'*_FaceCamera.mp4'`; repeat to combine |
| `-o`, `--out` | the clip list to write |
| `--target-size W H` | the frame size boxes are placed on (default: the first video's native size) |
| `--unlabelled` | mark every clip `thermistor = false` (prediction only) |
| `--pattern` | `[derive]` pattern: a regex over the video's file stem, capturing `group` |
| `--timestamps` | `[derive]` timestamps template |
| `--thermistor` | `[derive]` breathing-file template |
| `--checkpoint` | a list to predict with this network: no sensor, its `[preprocess]`, each video its own group unless `--pattern` |

The scan checks that every derived file exists. The list's `[events]` starts as
`adaptive_prominence`.

## clips check

```text
zephyr clips check <list.toml> [--json report.json]
```

Reads every file the list names and reports, per clip, what preprocessing or training
would refuse. It exits non-zero if any clip has a problem. `--json` also writes the
report.

## annotate

```text
zephyr annotate <list.toml> (--prepare | --auto) [--frame-dir DIR] [--min-score S] [--checkpoint best.pt]
```

| Flag | Meaning |
| --- | --- |
| `--prepare` | cache one frame per clip (at 150 s, or halfway) into `--frame-dir`, then exit |
| `--auto` | box the unplaced recordings where the placed boxes match their frame, and save |
| `--frame-dir` | where frames are cached (default `artifacts/frames`; the app uses the project's `frames/`) |
| `--min-score` | the lowest match score `--auto` accepts (default 0.5) |
| `--checkpoint` | with `--auto`: boxes are this network's crop size (a list from `clips scan --checkpoint`) |

Placing boxes by hand is the app's: `zephyr gui <list> --page boxes`.

## annotate events

```text
zephyr annotate events <list.toml> --auto
```

Makes the default detector the list's `[events]` and removes every clip's own `events`.
Marks, spans and `reviewed` are untouched. Choosing detectors by hand is the app's:
`zephyr gui <list> --page breathing`.

## preprocess

```text
zephyr preprocess <list.toml>... --cache <dir> [--workers N] [--limit N]
```

| Flag | Meaning |
| --- | --- |
| `--cache` | the feature cache folder (made if missing) |
| `--workers` | clips processed at once (each decode runs in its own `ffmpeg`); 1 disables concurrency |
| `--limit` | process only the first N clips (for tests) |

## run

```text
zephyr run <experiment.toml> [options]
```

| Flag | Meaning |
| --- | --- |
| `--smoke` | cut every fold to 1 epoch of 2 steps, under `<output_dir>/smoke/` |
| `--folds NAME...` | run only these folds |
| `--seeds N...` | run only these seeds |
| `--resume` | continue runs that have a `last.pt` |
| `--restart` | delete each run folder and train from scratch |
| `--evaluate-only` | never train: score each run's `best.pt` on its test groups |
| `--save-traces` | also write each test clip's trace into `traces/` |
| `--cache` | feature cache (default: the experiment's `features_dir`) |
| `--output-dir` | run root (default: the experiment's `output_dir`) |
| `--device` | `cuda`, `cpu`, ... (default: the experiment's, else CUDA if present) |
| `--amp {bf16,fp16,off}` | mixed precision (default bf16) |
| `--num-workers` | data-loading workers (default 4) |

A run that has a `best.pt` is not trained again: `zephyr run` only scores it.

## evaluate

```text
zephyr evaluate --checkpoint <best.pt>... --clips <list.toml>... --cache <dir> [options]
```

| Flag | Meaning |
| --- | --- |
| `--checkpoint` | one or more checkpoints; several are z-scored and averaged |
| `--clips` | clip lists to score; each is reported as its own group |
| `--name` | group name for a single `--clips` list (default: its file stem) |
| `--cache` | the feature cache |
| `--out` | write the full result as JSON |
| `--plot`, `--plot-dir` | also write diagnostic plots (one checkpoint), by default into `diagnosis/` beside it |
| `--device`, `--amp` | machine settings |
| `--infer-window`, `--frame-chunk` | inference window and frame batch sizes |

Clips a checkpoint trained on are refused.

## predict

```text
zephyr predict --checkpoint <best.pt> --clips <list.toml>... --out <dir> [options]
```

| Flag | Meaning |
| --- | --- |
| `--out` | the folder for `<video stem>.prediction.parquet` files; one checkpoint per folder |
| `--cache` | keep the preprocessed frames here and reuse them (default: a temporary folder, removed) |
| `--groups G...` | only these groups |
| `--restart` | remove the predictions in `--out` first and predict every clip |
| `--device`, `--amp`, `--workers` | machine settings |
| `--infer-window`, `--frame-chunk` | inference window and frame batch sizes |

The lists' `[preprocess]` must be the checkpoint's, and every box the size it trained
on.

## quality-control run

```text
zephyr quality-control run --checkpoint <best.pt> --clips <list.toml>... --cache <dir> --out <dir> [options]
```

| Flag | Meaning |
| --- | --- |
| `--reference` | the checkpoint's reference, for expected precision |
| `--groups G...` | only these groups of the lists |
| `--window-s` | window width in seconds (default 30) |
| `--threshold MEASURE=BOUND` | override a pass bound (repeat; `nan`: not checked) |
| `--no-traces` | skip the `.qc.parquet` traces |
| `--device`, `--amp`, `--infer-window`, `--frame-chunk` | machine settings |

## quality-control reference

```text
zephyr quality-control reference --checkpoint <best.pt> --clips <list.toml>... --cache <dir> --out <ref.json> [options]
```

| Flag | Meaning |
| --- | --- |
| `--groups G...` | the labelled groups to calibrate on (never ones the checkpoint trained on) |
| `--max-clips N` | predict at most N clips |
| `--device`, `--amp`, `--infer-window`, `--frame-chunk` | machine settings |

## schema

```text
zephyr schema <directory>
```

Writes JSON schemas of clip lists and experiments, for editor completion and validation.
