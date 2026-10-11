# zephyr-dataset

Allen-internal, never released. Builds zephyr's corpus of VR-foraging sessions with a
sniff detector: the inventory of packaged sessions, their raw thermistor traces, a clip
list with QC marks and spans, and the final selection of clips.

```bash
uv run zephyr-dataset build packages/zephyr-dataset/configs/vr-foraging-sniff.toml
```

or one step at a time: `inventory`, `fetch`, `lists`, `qc`, `select`, `status`. Review the
marks with `uv run zephyr annotate events <root>/sessions.toml`, then run `select` again.

The dataset folder (`root` in the spec) holds only what later stages need:

| path | what |
| --- | --- |
| `docdb.json.gz` | the saved DocDB return the inventory is built from (reused unless `--refresh`) |
| `inventory.parquet` | every packaged session found (asset, source, subject, rig, date, version) |
| `thermistor/<session_id>.parquet` | the whole session's raw `RawVoltage` (`Time`, `Signal`) |
| `sessions.toml` | the clip list, one thermistor-only entry per session: detector, marks, spans, reviewed flags (source of truth); `group` is the subject from the metadata |
| `qc.parquet` | each session's QC metrics and suggestion |
| `results.json` | the selection: per session metadata, metrics, marks, spans, clips, FaceCamera |
