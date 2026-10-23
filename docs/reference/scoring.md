# Scoring

How `zephyr run` and `zephyr evaluate` score a prediction against the breathing sensor.

## The metrics

| Metric | Range | Better | What it measures |
| --- | --- | --- | --- |
| `correlation` | −1 to 1 | higher | Pearson correlation of the predicted and true traces at zero lag (no lag search) |
| `inhale_f1` | 0 to 1 | higher | F1 of the inhales found as peaks on the predicted trace (DSP) |
| `exhale_f1` | 0 to 1 | higher | F1 of the exhales found as troughs on the predicted trace (DSP) |
| `head_inhale_f1` | 0 to 1 | higher | F1 of the inhales the **onset head** asserts. This is the inhale score of a network with an onset head |
| `kl_ibi` | 0 to ∞ | lower | KL divergence between the true and predicted distributions of inter-inhale intervals: whether the rhythm matches, regardless of timing |

`evaluation.json` also holds a `composite` (0.5 × inhale F1 + 0.2 × exhale F1 +
0.2 × correlation + 0.1 × e^−KL), for orientation only.

## Rules

**By time, never by sample index.**
:   Video and sensor need not start together. A clip is scored at the prediction's
    timestamps, over the span both cover. Outside its span the sensor holds its edge
    value.

**The truth events are the clip's own.**
:   True inhales and exhales come from the event set the clip's detector produced, the
    same set its training target was built from, as cached. They are never re-detected
    at scoring. A cache entry whose events are stale is refused as "not in the feature
    cache": re-run `zephyr preprocess` on its list, which rebuilds only the targets.

**Event tolerance: 1.5 frames.**
:   A predicted event matches a true one on the nearest frame, or the frame before or
    after, of the prediction's own output grid: 25 ms at 60 Hz, 50 ms at 30 Hz. Each
    true event is matched at most once (an optimal one-to-one assignment).

**Excluded spans are never scored.**
:   A clip's `excluded` spans, and clips marked `selection = "exclude"`, are left out.

**Scoring is on a 60 Hz grid.**
:   Whatever `output_fs` a network predicts at, scoring resamples to 60 Hz.

**Training clips are refused.**
:   A checkpoint records the videos it trained on, and `evaluate` refuses to score
    them.

!!! warning "Compare like with like"
    Scores made before and after a scoring rule changed (time alignment, event
    tolerance) are not comparable. Re-score old checkpoints before you compare them.

## Groups and summaries

`evaluation.json` reports each clip, each recording (a session's clips from one folder),
and each named test group, plus `all`. Group summaries average over recordings, with
the standard deviation (`_sd`) and the count (`_n`) of each metric. See [Output
files](outputs.md#evaluationjson).

## The onset head's target

The onset head learns a Gaussian (width `onset_sigma_s`, 20 ms) centred on each true
inhale's exact time, between frames where the inhale falls there. It is built per
training window from the cached event times and never stored.
