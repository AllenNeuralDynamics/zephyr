# Quality control

!!! warning "Experimental"
    Quality control is under active development. Its measures and the report format
    may change. It reports scores, never a verdict: what to make of them is your call.

On a video without a breathing sensor nothing can be scored, yet you still want to know
how far to trust a prediction. Quality control measures each prediction from the
network's own output, its trace and onset probability, with label-free
**measures**, over the whole video and per window. With the checkpoint's reference,
each measure is also placed among the checkpoint's own validation windows, where the
truth was known.

It costs nothing beyond the prediction itself. No measure needs extra
network passes, other checkpoints or the video itself.

## Run it

Quality control runs with the prediction, in the same pass. In the app, tick **judge
them (quality control)** on the Predict page (Clip list or
Videos tab). From the command line, add
`--quality-control` to `zephyr predict`:

```bash
uv run zephyr predict --checkpoint best.pt --clips new.toml --out predictions/ \
    --quality-control [--qc-reference qc-reference.json]
```

Beside each `<video stem>.prediction.parquet` it writes `<video stem>.qc.json`: every
measure over the whole clip (`summary`) and per window (`windows`, 30 s), and with a
reference each one's `percentiles`. The reference is `qc-reference.json` beside the
checkpoint, which training writes, unless `--qc-reference` names another. The report is
made from the prediction alone, so it is the same whether or not the video has a
thermistor.

## In the app

The Predict page's **Quality** tab, beside the list of a predictions folder's videos,
shows the folder's scores, one measure at a time:

- an overview with one row per video: the spread of the measure over its 30-s windows
  (box: middle half; whiskers: 5–95%; line: median);
- when the checkpoint has a reference, two rows above them: its validation windows that
  were predicted at least as well as **Reference split at true F1 ≥** (0.9 by default)
  and the rest, so you see what good and poor predictions scored where the truth was
  known; without a reference, the videos are shown against each other;
- a table of the videos with each measure's median (and, with a reference, its median
  percentile), sortable to bring the least typical first.

Clicking a video in the overview or the table opens it: its trace, onset probability
and crop, with the measure over its windows in a strip under the trace (the reference's
good windows shaded). Clicking a window in the strip zooms the trace to it. The tab only
reads the folder; nothing is passed or failed.

## Read the scores

Every summary and window keeps each measure's value, `null` when it has none (for
example expected precision without a reference). With a reference, `percentiles` says
where each value lies among the checkpoint's validation windows: the percentage of
them with a lower value. A percentile is a position, not a pass or a fail.

```python
from zephyr.quality_control import load_reports

summaries, windows = load_reports("qc/")
windows[["start", "end", "head_trace_f1", "pct_head_trace_f1"]]
```

`load_reports` reads a folder back as two tables, with percentiles as `pct_<measure>`
columns. The reference itself holds every validation window's measures beside its true
scores (`Reference.validation`), so you can see what values went with good and bad
predictions there.

## The measures

| Family | Measure | A better prediction tends to |
| --- | --- | --- |
| rhythm | `too_fast_frac` | have fewer breaths faster than `breathing_ceiling_hz` (15 Hz), which a mouse cannot do |
| rhythm | `alternation_break_frac` | alternate inhales and exhales |
| rhythm | `flat_breath_frac` | have few stretches where the trace swings less than `flat_ratio` of the clip's median breath |
| head | `head_trace_f1` | have the onset head's inhales agree with peaks on its own trace |
| head | `peak_prob` | have high inhale peaks (a decided head) |
| head | `ambiguous_peak_frac` | have few peaks between `ambiguous_floor` and the checkpoint's onset threshold |
| head | `onset_background` | have a low probability between inhales |
| head | `expected_precision` | have a high calibrated chance that the asserted inhales are real (needs a reference) |

The breathing rate is reported as a description: usual rates change with running,
odours and task.

!!! warning "Measures are proxies"
    The measures do not see the truth, and only animals with a sensor show how well
    they track it. A video unlike the validation windows is different, not necessarily
    wrong; one like them can still be wrong.

## The reference

A **reference** holds a calibration of the checkpoint's onset head, for the
expected-precision measure, and every measure on the windows of labelled clips the
checkpoint never trained on, its validation animals, beside their true scores. Training
writes it beside `best.pt` as `qc-reference.json` when the run chooses its onset
threshold on validation clips (`head_thresholds`). For a checkpoint without one, make
it yourself:

```bash
uv run zephyr quality-control reference --checkpoint best.pt --clips clips.toml \
    --groups m4-day1 m6-day1 --cache cache/ --out qc-reference.json
```

The reference is tied to its checkpoint by SHA-256, and a run with another checkpoint
refuses it. It finds the head's inhales at the checkpoint's own onset threshold, as
every measure does. Ship it beside `best.pt`. `--max-clips` predicts fewer clips.
