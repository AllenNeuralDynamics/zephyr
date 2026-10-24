# zephyr-dataset

Videos of head-fixed mice foraging in virtual reality, each with the breathing recorded
by a nasal thermistor: {assigned} sessions of {animals} animals, two 300 s clips per
session, from each of three cameras. It is the corpus zephyr's networks are trained and
scored on (https://github.com/AllenNeuralDynamics/zephyr).

Written by `zephyr-dataset` (package `zephyr-dataset` of that repository) from the spec
`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`; the package README says why
each session is in or out. `raw/` holds what the clips were cut from and the record of
every build step (`raw/README.md`); nothing here needs it.

## Files

| path | what |
| --- | --- |
| `clips.toml` | the clips as a zephyr clip list: one entry per span and camera, with its video, frame times, thermistor, `group` (the session) and `box` (the crop) |
| `sessions.parquet` | one row per session: its animal, rig, date, camera rate, breathing rates and its set in each cross-validation fold |
| `clips/<session>_<span>_<camera>.mp4` | the span cut from the camera's video: stream-copied (never re-encoded) at the camera's own rate, fragmented MP4 |
| `clips/<session>_<span>_<camera>.parquet` | that clip's frame times: one `Time` column (Harp seconds), one row per frame |
| `clips/<session>_<span>_thermistor.parquet` | the thermistor over the span and a margin either side: `Time` (Harp seconds), `Signal` (raw ADC units) |
| `thumbnails/<session>_<camera>.png` | the first frame of the session's first clip, at full resolution |

Times of a session's videos and thermistor are on one clock (Harp seconds), so a frame's
time indexes the thermistor directly.

{cameras}

The front camera is optional: it is cut only where its timestamps were clean and it
pointed at the animal. In `clips.toml` the side and front clips are
`selection = "exclude"`, so zephyr uses the face camera unless a fold selects them.

## The folds

`sessions.parquet` holds {n_folds} cross-validation folds (`fold_0` to
`fold_{last_fold}`). In each, a session is one of:

| set | holds out | to measure |
| --- | --- | --- |
| `train` | | |
| `val` | whole animals, drawn from those the fold trains on | early stopping and tuning |
| `test_a` | whole animals: every animal outside group C is test A in exactly one fold | a new animal (the goal) |
| `test_b` | a block of consecutive dates of a training animal | a new session of a known animal |
| `test_c` | some 359 Hz sessions of the two animals recorded at that rate (group C) | a rare camera rate |

Report a method over all folds: pool its test A scores, average within each animal first,
and bootstrap over animals. `group` in `clips.toml` is the session, so a fold selects its
sets by session; no session appears on both sides of a fold.

## sessions.parquet

| column | what |
| --- | --- |
| `session_id` | the session's name, the prefix of its clips |
| `subject_id` | the animal, from the session's metadata |
| `rig_name`, `rig` | the rig as recorded, and with names of one physical rig merged |
| `date`, `month` | when it was recorded |
| `camera_hz` | the face camera's frame rate |
| `samples` | breathing-rate samples over both clips (60 per second where a rate is defined) |
| `t_1-4`, `t_4-8`, `t_8-15` | of those, how many at 1-4, 4-8 and 8-15 Hz |
| `q50` | the median breathing rate (Hz) |
| `group_c` | recorded at 359 Hz: the session keeps one set (`train`, `val` or `test_c`) in every fold |
| `cv_group` | the fold in which the animal is test A (-1: group C) |
| `cv_b_fold` | the fold in which the session is test B (-1: none) |
| `fold_<k>` | the session's set in fold k |
