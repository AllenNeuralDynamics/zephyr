# Quality control

!!! warning "Experimental"
    Quality control is under active development. Its measures, their pass bounds and
    the report format may change, and they are not yet validated on enough held-out
    animals to be relied on. Use its checks as a hint about where to look, not as a
    verdict on a prediction.

On a video without a breathing sensor nothing can be scored, yet you still want to know
whether to trust a prediction. `zephyr quality-control` judges each prediction from the
network's own output, its trace and onset probability, with label-free **measures**.
Each measure is checked against a pass bound, over the whole video and per window.

It costs one ordinary inference per video, nothing more. No measure needs extra
network passes, other checkpoints or the video itself.

## Run it

In the app, tick **judge them (quality control)** on the Predict page's Clip list tab.
From the command line:

```bash
uv run zephyr quality-control run --checkpoint best.pt --clips new.toml \
    --cache cache/ --out qc/ [--reference qc-reference.json]
```

The clips must be preprocessed (boxed, in `--cache`). `thermistor = false` is fine.
Clips the checkpoint trained on are refused.

For each video it writes:

`<video stem>.qc.json`
:   every measure over the whole clip (`summary`) and per window (`windows`, about
    `--window-s` wide, 30 s by default), each with its pass/fail in `checks`.

`<video stem>.qc.parquet`
:   one row per output sample: time, trace, onset probability and inhale. `--no-traces`
    skips it.

A finished report is kept on a re-run. A folder holding reports made with other
settings is refused.

## Read the checks

Every summary and window keeps each measure's value and adds `checks`: `true` passes,
`false` fails, and `null` means the measure has no value (for example expected
precision without a reference). **No video is failed as a whole.** Which checks matter
is your call.

```python
from zephyr.quality_control import load_reports

summaries, windows = load_reports("qc/")
flat = windows[windows["ok_flat_breath_frac"].eq(False)]  # failed, not unchecked
```

`load_reports` reads a folder back as two tables, with the checks as `ok_<measure>`
columns.

## The measures

| Family | Measure | Fails when |
| --- | --- | --- |
| rhythm | `too_fast_frac` | breaths faster than `breathing_ceiling_hz` (15 Hz), which a mouse cannot do |
| rhythm | `alternation_break_frac` | inhales and exhales do not alternate |
| rhythm | `flat_breath_frac` | the trace swings less than `flat_ratio` of the clip's median breath between two inhales |
| head | `head_trace_f1` | the onset head's inhales disagree with peaks on its own trace |
| head | `peak_prob` | the head's inhale peaks are low (unsure) |
| head | `ambiguous_peak_frac` | many peaks sit between `ambiguous_floor` and 0.5 |
| head | `onset_background` | the probability between inhales is high |
| head | `expected_precision` | the calibrated chance that the asserted inhales are real is low (needs a reference) |
| head | `inhale_phase_spread` | the head puts inhales at a wandering point of the breathing cycle |

The breathing rate is reported as a description, never judged: usual rates change with
running, odours and task. Only rates faster than a mouse can breathe fail.

### Pass bounds

The default bounds each pass 99 % of the good windows (true inhale F1 ≥ 0.9) of the
reference dataset's validation animals. Override one with
`--threshold <measure>=<bound>`. A bound of `nan` leaves that measure unchecked. Every
report keeps the bound it used beside each value.

!!! warning "Measures are proxies"
    The measures do not see the truth, and only held-out animals with a sensor show how
    well they track the real F1. If you change a bound, choose it on validation animals,
    never on the test set you then report.

## The reference (optional)

The expected-precision measure needs a **reference**: a calibration of the checkpoint's
onset head, fitted once on labelled clips the checkpoint never trained on, such as its
validation animals.

```bash
uv run zephyr quality-control reference --checkpoint best.pt --clips clips.toml \
    --groups m4-day1 m6-day1 --cache cache/ --out qc-reference.json
```

The reference is tied to its checkpoint by SHA-256, and a run with another checkpoint
refuses it. Ship it beside `best.pt`. `--max-clips` predicts fewer clips.

## Labelled clips

A labelled clip's report also holds a `truth` block, its real scores, computed as
`zephyr evaluate` would. No measure reads it. It is there to check the measures.
