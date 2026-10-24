# raw/

The files the corpus was built from, and the record of each build step. Training and
evaluation never read them: everything they need is in `../clips/`, `../clips.toml` and
`../sessions.parquet`. They are kept so that a session can be checked or re-cut without
going back to the source bucket (`aind-open-data`), and so that every session's way in or
out of the corpus can be traced.

The build is `zephyr-dataset` (package `zephyr-dataset` in
https://github.com/AllenNeuralDynamics/zephyr), run from the spec
`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`. Every threshold and every hand
decision (`[[review]]`) is in that spec. The steps run in this order, and each one
writes the files listed under it.

Times are Harp seconds throughout. A session id is the packaged session's name; the
subject always comes from the session's metadata, never from the name.

## How a session gets in

{flow}

The {no_video} sessions without a face or side video carry no `dropped` note: their
`cameras` lacks the missing camera.

## 1. inventory

| file | what |
| --- | --- |
| `docdb.json.gz` | the DocDB query result the inventory was built from, saved as returned |
| `inventory.parquet` | one row per packaged session found: data asset, source location, subject, rig, date, software version |

## 2. fetch

| file | what |
| --- | --- |
| `thermistor/<session_id>.parquet` | the whole session's thermistor (`RawVoltage`): `Time`, `Signal`, one row per sample, as recorded. {scored} files |

## 3. lists and qc

| file | what |
| --- | --- |
| `sessions.toml` | a zephyr clip list with one thermistor-only entry per session: the detector settings, the inhale marks, the usable spans and whether it was reviewed by hand. `group` is the subject. Open it with `zephyr gui sessions.toml --page breathing` (after copying `thermistor/` beside it) |
| `qc.parquet` | one row per session: the signal-quality measures (`snr_db`, `amplitude`, `mean_rate`, `break_frac`, `outside_frac`, `rate_cv`, `amp_cv`, `rail_frac`, `consistency_z`, `good_frac`, `artifacts`, `accepted_s`, `dead_sensor`), the automatic `suggested` decision with its `reason`, and the spans a review `included` or `excluded` |

## 4. select and videos

| file | what |
| --- | --- |
| `camera_metadata/<session_id>_<camera>.parquet` | the camera's `metadata.csv`, unmodified (frame number, camera time, Harp time per frame). For the face and side cameras of every session that has both videos; for the front camera only of the sessions the folds assign that have one |
| `videos/<session_id>_<camera>.parquet` | the repaired frame times: `ReferenceTime` (Harp), `CameraFrameNumber`, `CameraFrameTime`, one row per video frame |
| `cameras.parquet` | one row per session and camera: the timestamp checks before (`before_*`) and after (`after_*`) the repair (dropped frames, steps back, jitter, frame count against the video), the rows repaired, the source `folder`, and `clean`. A required camera (face, side) that is not clean drops its session |
| `selection.json` | the selection: the spec it was made with, the counts above, and per session its metadata, QC metrics, marks, spans, the two clips' start and end, the camera folders, and for a dropped session `dropped` (why) |

## 5. sync

| file | what |
| --- | --- |
| `sync.parquet` | one row per selected session: when the thermistor's device (`device_s`) and the reference Harp board (`reference_s`) logged the same command, and their difference (`offset_s`), or the `error` that stopped the check. A session off the reference clock is left out of the split |

## 6. folds and cross-validation

| file | what |
| --- | --- |
| `folds.parquet` | one row per selected session ({selected}): subject, rig, date, camera rate, breathing-rate bands, `group_c` (the 359 Hz sessions), and the `set` it was first drawn into with its animal's `role` (`small` is left from a subset no longer kept). The sessions with a `set` are the corpus; `cross_validation` drew the folds of `../sessions.parquet` from them, and group C keeps its `set` in every fold. Results are never reported on this first draw |

## 7. clips

| file | what |
| --- | --- |
| `clips.parquet` | one row per span and camera: its files, the span's start and end, the clip's first frame, frame count, start, end and rate, or the `error` that stopped it |

The clips themselves (`../clips/`) are whole GOPs stream-copied from the source videos,
never re-encoded, at each camera's own rate.

## Known gaps

- The front camera is optional: it is cut only where it was clean and pointed at the
  animal (`../README.md` has the sessions with clips of each camera).
- {dropped} sessions were dropped because a required camera's timestamps could not be
  repaired; they are in `selection.json` with their reason.
- More in the package README's "Known data issues".
