# zephyr-dataset

Allen-internal, never released. Builds zephyr's corpus of VR-foraging sessions with a
sniff detector: the inventory of packaged sessions, their raw thermistor traces, a clip
list with QC marks and spans, and the final selection of clips.

```bash
uv run zephyr-dataset build packages/zephyr-dataset/configs/vr-foraging-sniff.toml
```

or one step at a time: `inventory`, `fetch`, `lists`, `qc`, `select`, `videos`, `status`.
Review the marks with `uv run zephyr annotate events <root>/sessions.toml`, then run
`select` again. After `videos`, run `select` once more: it drops sessions whose camera
timestamps are not clean (see [Known data issues](#known-data-issues)).

The dataset folder (`root` in the spec) holds only what later stages need:

| path | what |
| --- | --- |
| `docdb.json.gz` | the saved DocDB return the inventory is built from (reused unless `--refresh`) |
| `inventory.parquet` | every packaged session found (asset, source, subject, rig, date, version) |
| `thermistor/<session_id>.parquet` | the whole session's raw `RawVoltage` (`Time`, `Signal`) |
| `sessions.toml` | the clip list, one thermistor-only entry per session: detector, marks, spans, reviewed flags (source of truth); `group` is the subject from the metadata |
| `qc.parquet` | each session's QC metrics and suggestion |
| `camera_metadata/<session_id>_<camera>.parquet` | each selected camera's `metadata.csv`, unmodified (a cache; the video's frame count in its metadata) |
| `videos/<session_id>_<camera>.parquet` | the repaired camera timestamps (`ReferenceTime`, `CameraFrameNumber`, `CameraFrameTime`) |
| `cameras.parquet` | each camera's checks before and after the repair, and whether it is clean |
| `results.json` | the selection: per session metadata, metrics, marks, spans, clips, camera folders, and `dropped` (why) for sessions left out for their cameras |

## Known data issues

Camera timestamps (`behavior-videos/<camera>/metadata.csv`) are checked by `videos`
(`zephyr/corpus/camera.py`). Every problem below is in the **raw data**, not made by the
repair. Status for the corpus built on 2026-10-11 (2,215 selected sessions, FaceCamera and
SideCamera):

| problem | sessions | what happens now |
| --- | --- | --- |
| one-row `ReferenceTime` rollback (~1 s early, same row on both cameras) | 104 | **repaired**: the row is interpolated from its neighbours; checked against the camera clock to within 5-9 us |
| one-row `ReferenceTime` dip of ~3 ms (no backward step, same row on both cameras) | 6 | **dropped** |
| lasting `ReferenceTime` step (0.62 s back, never recovers; the thermistor shows no step) | 1 | **dropped** |
| the video has fewer frames than its timestamps (4-253; metadata complete) | 8 | **dropped** |

The 15 dropped sessions (0.7%) are listed with their reason in `results.json`
(`dropped`). None is rare: every one's animal has 18-91 other clean selected sessions,
and every rig keeps 20 or more. 7 of the 8 short videos were recorded on 2026-06-01/02,
mostly on FRG.4 rigs, which suggests an acquisition problem on those days.

They could be recovered, in theory, which is why the raw `metadata.csv` is kept in
`camera_metadata/` (a re-repair needs no S3):

- **3 ms dip**: generalise the repair to use the camera clock as the reference: blank any
  row whose `ReferenceTime` step disagrees with the `CameraFrameTime` step by more than
  ~1 ms and is followed by a compensating step, then interpolate. This also covers the
  1 s rollback.
- **lasting step**: the timestamps after the step agree with the thermistor (end offset as
  in clean sessions), so the frames before it are ~0.63 s late. Shift them back by the
  offset the camera clock measures across the step.
- **short video**: the timestamps cannot say where the frames are missing. Finding them
  needs decoding the video (e.g. frame-to-frame differences against the camera's frame
  intervals); without that, frames cannot be matched to timestamps.

Dropping is done by `select` from `cameras.parquet` (a camera's `clean`), so a repaired
camera comes back in automatically on the next `videos` + `select`.
