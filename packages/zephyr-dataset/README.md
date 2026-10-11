# zephyr-dataset

Allen-internal, never released. Builds zephyr's corpus of VR-foraging sessions with a
sniff detector: the inventory of packaged sessions, their raw thermistor traces, a clip
list with QC marks and spans, the final selection, and the selected spans cut from the
videos.

```bash
uv run zephyr-dataset <step> packages/zephyr-dataset/configs/vr-foraging-sniff.toml --profile IAmAnywhere
```

Steps, in order: `inventory`, `fetch`, `lists`, `qc`, `select`, `videos`, `select` (again:
it drops sessions whose camera timestamps are not clean, see
[Known data issues](#known-data-issues)), `clips`; `status` reports progress and `build`
runs them all.

## Where the dataset lives

`root` in the spec is a local folder or a URL; the corpus is at
`s3://aind-scratch-data/vr-foraging/zephyr-dataset` (about 40 GB of metadata, timestamps
and thermistors, plus about 3.9 TB of clips). Every step reads and writes it directly,
with the AWS profile given by `--profile` (or `AWS_PROFILE`); the source bucket,
`aind-open-data`, is read anonymously. Writes never rename or delete remote objects (the
`IAmAnywhere` role may not delete), and a file appears only once it is complete.

Two things need local files, because zephyr itself reads only local paths:

- `qc` (zephyr's session selection) refuses a remote root. Copy what it reads, run it on
  the copy and upload its two outputs:

  ```bash
  aws s3 sync s3://aind-scratch-data/vr-foraging/zephyr-dataset <copy> --exclude "clips/*" --exclude "videos/*" --exclude "camera_metadata/*"
  # a spec with root = "<copy>", then: zephyr-dataset qc <that spec>
  aws s3 cp <copy>/sessions.toml s3://aind-scratch-data/vr-foraging/zephyr-dataset/
  aws s3 cp <copy>/qc.parquet s3://aind-scratch-data/vr-foraging/zephyr-dataset/
  ```

- Reviewing (`zephyr annotate events <copy>/sessions.toml`) and preprocessing
  (`clips.toml`) work on such a copy. Every path in the lists is relative, so a copy
  anywhere works; sync `clips/` only for the clips you need. Decisions made by hand go
  in the spec's `[[review]]`, so `qc` writes them back on every run.

## The dataset folder

It holds only what later stages need:

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
| `clips/<session_id>_<span>_<camera>.mp4` | the span cut from the camera's video: whole GOPs, stream-copied (never re-encoded) as fragmented MP4, at the camera's native rate |
| `clips/<session_id>_<span>_<camera>.parquet` | that clip's frame times, one `Time` column (repaired `ReferenceTime`, Harp seconds), one row per frame |
| `clips/<session_id>_<span>_thermistor.parquet` | the thermistor over both cameras' clips ± `[clips] thermistor_pad_s` (Harp seconds) |
| `thumbnails/<session_id>_<camera>.png` | the camera at full resolution, losslessly: the first frame of the session's first clip |
| `clips.parquet` | every clip: frames, first frame, span, start and end, rate, bytes, or the error that stopped it |
| `clips.toml` | the clips as a zephyr clip list (`group` = subject, no boxes yet) |

### How `clips` cuts

A span is drawn in thermistor seconds, which share the Harp clock with the cameras'
repaired `ReferenceTime`, so it maps to frame indices directly (row *i* of the timestamps
is frame *i*: `videos` guarantees no dropped frames and as many rows as frames). The cut
runs from the keyframe at or before the span's first frame to the keyframe at or after its
last (or the end of the video), so a clip is a few seconds longer than its span and needs
no re-encoding. Each video's frame rate, time base and keyframes are read from the
stream, never assumed (most cameras are 240.38 Hz, the 15 sessions of rig 4B in
2026-02/03 are 359.2 Hz). ffmpeg writes the cut to a pipe; it is uploaded only after a
check that it holds one packet per expected frame, evenly spaced.

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
`camera_metadata/` (a re-repair never reads the source bucket again):

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
