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
[Known data issues](#known-data-issues)), `sync`, `folds`, `clips`; `status` reports
progress and `build` runs them all. The required cameras (`[source] cameras`, face and
side) decide the selection; an optional one (`[source] optional_cameras`, the front
camera) is checked by `videos` and cut by `clips` where a session the folds assign has it
clean, and never selects or drops a session.

## Why these sessions

The goal is a network that works **on animals it has never seen**.

**The corpus.** 7,898 packaged sessions → 4,120 kept by the query (dataset version 0.6.0+,
plus 15 hand-reviewed ephys sessions) → 1,591 pass thermistor QC (see
[Thermistor QC](#thermistor-qc)) → 1,518 fit two 300 s clips in good stretches → 1,515
have both cameras → **1,506** with clean camera timestamps (see
[Known data issues](#known-data-issues)). They come from 50 animals on 15 rigs (names of
one physical rig merged, `FRG.4-B` = `4B`), 2025-10 to 2026-10.
`notebooks/corpus_selection.py` draws where every session goes and why.

**The split** (`folds`, written to `folds.parquet`, the record of the split):

| set | question | held out as | animals | sessions |
| --- | --- | --- | --- | --- |
| test A | new animal? (the goal) | 10 whole animals | 10 | 78 |
| test B | new session of a known animal and rig? | sessions of training animals | 26 | 115 |
| test C | the rare 359 Hz camera? | sessions of its two animals (rig 4B) | 2 | 5 |
| validation | early stopping, tuning | 8 whole animals (+ 2 group C sessions) | 9 | 67 |
| train | | the rest | 32 | 295 |

- **Whole animals** for test A and validation: an animal's sessions share its face and
  breathing, and validation should reward the same thing test A measures. Comparing A
  with B tells generalisation from memorising animals.
- **Balanced draws:** test A and validation are drawn (seeded) to cover rigs and months
  and to match the share of time at 1-4, 4-8 and 8-15 Hz: train, validation, test A and
  test B are all within about 2 points of 44 / 46 / 10%. This favours animals that used
  several rigs, which is accepted.
- **Caps per animal and rig**, spread over its dates, because such sessions repeat each
  other: train 10, validation 3, test A 5, test B 5 (at most a third of the pair, leaving at
  least 2 to train). 560 of 1,506 sessions are used; only their clips are cut. Sessions
  with a dark video, two whose traces were rejected on review (`[folds] exclude`) and those
  whose SniffDetector is off the Behavior board's clock (`sync`) are taken out after the
  draw (see [Known data issues](#known-data-issues)).
- **Group C:** 841301 and 828415 were recorded at 359.2 Hz. Rather than hold the animals
  out (training would never see that rate), their 359 Hz sessions are split 6 / 2 / 4 and
  1 / 0 / 1 (train / val / test C). Both breathe slowly, so test C confounds frame rate with
  breathing rate.
- **Reporting:** average within each animal first, then report test A as the mean over its
  10 animals with a bootstrap interval over animals; more precision means more animals,
  not more sessions.
- **Set aside:** a temporal holdout (new animals is what matters), leaving a rig out (rigs
  are similar and animals move between them). Cross-validation over animals is drawn
  beside this split, never in place of it: see [Cross-validation](#cross-validation).

Test A animals: 806527, 808728, 815103, 828421, 841306, 841312, 849495, 863704, 864846,
879295. Validation animals: 789917, 828417, 828423, 841299, 846441, 864847, 866061, 866063.

### Thermistor QC

The thermistor is the target, so a session or stretch is used only where the target can
be trusted. `qc` scores every session in 30 s windows (`[qc]`), and a clip comes only from
a run of good windows at least 300 s long. A good window has:

- **amplitude at least 100** (`min_amplitude`, the mean 1 s 5th-95th range, in raw
  thermistor units), so every clip has a strong signal;
- **no clipping** (`rails`, `max_rail_frac = 0`): no second touching the thermistor's
  floor (about 18-20, where it saturates) or the ADC's top (4095);
- breaths inside the rate band and events that alternate (`max_outside`, `max_breaks`).

A session is used only if its events alternate over the whole session (at most 0.3% of
events repeat the previous kind, `max_session_breaks`) and are consistent with the rest of
the corpus (`outlier_z`). Of the 4,120 queried sessions, 2,038 fail the amplitude, 257 the
alternation, 229 have no good 300 s run, 1 fails the consistency check and 4 were
excluded on review ([[review]]).

Why these thresholds: scored by gray networks, sessions with an amplitude under 100 had a
median correlation of 0.3 and inhale F1 near 0.1 (0.85 above), alternation breaks
separated the bad sessions best (AUC 0.87), and on validation clips, never used to choose
the rules, the clips the rules leave out score lower than those they keep with every gray
network (full network: median r 0.86 against 0.89). Clipping does not lower the scores
measurably; it is left out because a clipped breath is a distorted target. `qc` scores
whole sessions, whatever marks the list holds, so re-running it never changes a metric.

## Cross-validation

One split scores 10 test animals, so a few hard animals move test A. `zephyr-dataset
cross_validation <spec>` (`[cross_validation]`) draws folds over the same 560 sessions so
that every animal is a test animal once; it writes `cross_validation.parquet` and only
reads `folds.parquet`, which stays the record of the published split. `build` does not
run it.

- **Test A:** the 48 animals outside group C in 5 groups (the best balanced of seeded
  draws, judged by its worst group); fold k tests group k whole, so each animal is scored
  once by a network that never saw it.
- **Validation:** 8 whole animals per fold, drawn from those it trains on.
- **Test B:** each animal with at least 4 sessions is cut by date into one block for each
  fold it trains in, and that fold tests the block. Blocks rather than random sessions,
  because sessions days apart look alike: a held-out session should not have its
  neighbours in training. Each such session is test B once.
- **Group C** keeps its sets from `folds.parquet` in every fold.
- **Sizes** per fold: 75-128 test A, 70-119 test B, 87-144 validation and 213-268
  training sessions; every set of every fold is within about 2 points of the corpus' rate
  mix.
- **Columns:** `session_id`, `subject_id`, `cv_group` (the animal's test A fold, -1 for
  group C), `cv_b_fold` (the session's test B fold, -1 for none) and `fold_<k>` (its set in
  fold k). Every fold passes `folds.check`.

Report a method by pooling its test A scores of all folds, averaged within each animal,
with a bootstrap over animals; one seed per fold is enough when seeds vary less than folds.

## Where the dataset lives

`root` in the spec is a local folder or a URL; the corpus is at
`s3://aind-scratch-data/vr-foraging/zephyr-dataset` (about 40 GB of metadata, timestamps
and thermistors, plus about 1 TB of clips for the 560 sessions the folds use). Every
step reads and writes it directly, with the AWS profile given by `--profile` (or
`AWS_PROFILE`); the source bucket, `aind-open-data`, is read anonymously. Writes never
rename remote objects, and a file appears only once it is complete. The one deletion is
`clips --prune`, which removes the clips of sessions `folds.parquet` does not assign; it
needs a profile allowed to delete (your SSO login; `IAmAnywhere` may not delete).

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
| `camera_metadata/<session_id>_<camera>.parquet` | each selected camera's `metadata.csv`, unmodified (a cache; the video's frame count in its metadata); an optional camera's only for the sessions the folds assign |
| `videos/<session_id>_<camera>.parquet` | the repaired camera timestamps (`ReferenceTime`, `CameraFrameNumber`, `CameraFrameTime`) |
| `cameras.parquet` | each camera's checks before and after the repair, whether it is clean, and its source `folder` |
| `sync.parquet` | each selected session's read-dump times (the WRITE to `OperationControl`) of the SniffDetector and of `[sync] reference`, and their difference (`offset_s`), or the error that stopped the check |
| `results.json` | the selection: per session metadata, metrics, marks, spans, clips, camera folders, and `dropped` (why) for sessions left out for their cameras |
| `folds.parquet` | the split: per selected session its subject, rig, date, camera rate, rate bands, the animal's `role` and the session's `set` (`train`, `val`, `test_a`, `test_b`, `test_c`, or none); see [Why these sessions](#why-these-sessions) |
| `clips/<session_id>_<span>_<camera>.mp4` | the span cut from the camera's video: whole GOPs, stream-copied (never re-encoded) as fragmented MP4, at the camera's native rate |
| `clips/<session_id>_<span>_<camera>.parquet` | that clip's frame times, one `Time` column (repaired `ReferenceTime`, Harp seconds), one row per frame |
| `clips/<session_id>_<span>_thermistor.parquet` | the thermistor over the required cameras' clips ± `[clips] thermistor_pad_s` (Harp seconds); an optional camera's clip lies inside it and never changes it |
| `thumbnails/<session_id>_<camera>.png` | the camera at full resolution, losslessly: the first frame of the session's first clip |
| `clips.parquet` | every clip: frames, first frame, span, start and end, rate, bytes, or the error that stopped it |
| `cross_validation.parquet` | per assigned session its animal's test A fold, its test B fold and its set in each fold (see [Cross-validation](#cross-validation)) |
| `clips.toml` | the clips as a zephyr clip list (`group` = session, so folds can pick sessions and test B and C can hold out sessions of training animals; face clips carry their `box`; `[clips] exclude_cameras` lists the side and front cameras as `selection = "exclude"`, so zephyr uses the face camera only and their files stay in the folder) |

`clips` cuts only the sessions `folds.parquet` assigns, once it exists.

### Crop boxes for the face clips

A script, not a build step and not in the spec; the work folder is local. `clips.toml`
names clips that are not on your machine, so the face clips go in a local `face.toml` with
one frame per clip, the session's thumbnail:

```bash
uv run python -m zephyr.corpus.boxes prepare <spec> --work boxes --profile IAmAnywhere
uv run zephyr annotate boxes/face.toml --remote --frame-dir boxes/frames --box-size 96
uv run python -m zephyr.corpus.boxes publish <spec> --work boxes --profile IAmAnywhere
```

`prepare` template-matches the labelled snouts (`rois.EXAMPLES`, cut from our thumbnails)
against each session's thumbnail and puts a 96x96 box (in the 360x270 frame, 192x192 native)
on the best match if it scores at least `--min-score` (0.58) and sits where a snout can;
both spans of a session share the box. The rest have no box: place them in the UI, which
is also where you correct the others (`auto.csv` lists the scores). `publish` writes the
boxes into the dataset's `clips.toml` (face clips only), and `clips` keeps them when it
rewrites the list. The match centres on the nose-tip pad; the nostrils are not resolvable
at 720x540, so a global offset is applied later if wanted. `python -m zephyr.corpus.rois`
is the helper used to find this method (OWLv2 or template matching, with a preview image);
it needs `--extra rois` for OWLv2.

## Known data issues

Camera timestamps (`behavior-videos/<camera>/metadata.csv`) are checked by `videos`
(`zephyr/corpus/camera.py`). Every problem below is in the **raw data**, not made by the
repair. Of the 1,515 sessions that pass QC, fit two clips and have both cameras
(FaceCamera and SideCamera):

| problem | sessions | what happens |
| --- | --- | --- |
| one-row `ReferenceTime` rollback (~1 s early, same row on both cameras) | 73 | **repaired**: the row is interpolated from its neighbours; checked against the camera clock to within 5-9 us |
| one-row `ReferenceTime` dip of ~3 ms (no backward step, same row on both cameras) | 3 | **dropped** |
| lasting `ReferenceTime` step (0.62 s back, never recovers; the thermistor shows no step) | 1 | **dropped** |
| the video has fewer frames than its timestamps (6-154; metadata complete) | 5 | **dropped** |

The 9 dropped sessions are listed with their reason in `results.json` (`dropped`). All five
short videos were recorded on 2026-06-01/02, which suggests an acquisition problem on
those days.

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

**Dark videos** (not a timestamp problem): six sessions of animals 815103, 828423 and
828425, recorded 2026-01-23 and 02-09 to 02-16 on rigs 12C, 13A and 13B, are dark for the
whole video (thumbnail mean luminance 4 to 38 of 255; every other session is above 53). Only
the cut thumbnails were checked, so sessions the folds do not use are not screened. They are
`[folds] exclude` and their clips and thumbnails are removed by `clips --prune`.

Dropping is done by `select` from `cameras.parquet` (a required camera's `clean`), so a
repaired camera comes back in automatically on the next `videos` + `select`.

**Front camera** (optional): 546 of the 560 sessions the folds assign have a
`FrontCamera` folder (same 720x540 at 240 Hz as the others). The 14 without one are all of
animals 841301 (12) and 828415 (2), 2026-02 and 03: six of test C's sessions and eight of
their training and validation sessions. A front camera that is not clean is not cut, and
the session keeps its face and side clips.

**Harp clocks** (`sync`): every Harp device answers the read dump that starts a session
(its WRITE to `OperationControl`, register 10), and devices on one clock do so within a few
ms of each other. `sync` reads only that register of the SniffDetector and of the Behavior
board (`[sync] reference`) from the raw session. Of the 1,506 selected sessions, 1,502 are
1-10 ms apart (median +3.5 ms); four are not, so every thermistor time of theirs is off by
as much, and `folds` takes them out after the draw (`[sync] max_offset_s` = 50 ms):

| session | rig | SniffDetector - Behavior | set it had |
| --- | --- | --- | --- |
| `behavior_828418_2025-11-14_17-22-06` | 4B | +884 ms | train |
| `behavior_807093_2026-01-21_16-45-31` | 4A | -321 ms | none |
| `828415_2026-02-21_00-26-46` | 4B | -202 ms | train |
| `behavior_815104_2026-01-23_21-03-24` | 4D | -144 ms | test B |

The thermistor of `behavior_815104_2026-01-23_21-03-24` lags the gray network's
prediction by about the same 150 ms. Their clips and thumbnails are removed by
`clips --prune`.
