---
title: "Inferring breathing from orofacial video in mice, validated at scale against intranasal thermistry"
author: "Bruno F. Cruz^1^, [\[authors\]]{custom-style="Author note"}"
subtitle: "^1^Allen Institute for Neural Dynamics, Seattle, WA, USA"
---

# Abstract

Olfaction is one of the primary senses in rodents. Contrary to vision and audition,
it relies heavily on active sampling through sniffing. Ground-truth methods for measuring sniffing
in mice typically rely on sensors implanted in the nasal cavity which require invasive surgery and
often fail to yield a usable signal.
Here we show that the breathing waveform and the
timing of each inhalation can be inferred from ordinary video of the face. We assembled a
corpus of [560]{custom-style="Provisional"} highly curated sessions from [50]{custom-style="Provisional"} head-fixed mice that pairs face video with a
synchronized intranasal thermistor. On [ten]{custom-style="Provisional"} held-out animals never seen in training, our
network detected inhalation onsets with an F1 score of [0.95]{custom-style="Provisional"} (within 25 ms) and
recovered the waveform with a correlation of [0.82]{custom-style="Provisional"}. Video can thus provide
breath-by-breath timing in head-fixed mice without a nasal implant.

# Introduction

Breathing is more than a vital function in rodents. Sniffing is a form of active sensing: a movement the animal makes to sample its
surroundings. Each inhalation draws odour into the nose, so sniffing sets when and how
olfactory information is sampled [@kepecs2006; @verhagen2007; @wachowiak2011], and rodents
adjust it constantly. Mice breathe at about 2–4 Hz at rest and sniff at up to about 12 Hz
when they explore or anticipate an odour, switching between the two within a fraction of
a second [@welker1964; @wesson2008]. Human breathing, by contrast, stays near 0.2–0.3 Hz, so
a rodent's breathing rate varies over a range several times wider and changes far faster.
Responses of olfactory bulb neurons are locked to precise phases of the sniff cycle
[@shusterman2011]. The respiratory rhythm, which
originates in the brainstem [@smith1991], also coordinates whisking and other orofacial
movements [@moore2013; @kleinfeld2014], and it entrains oscillations far beyond the olfactory
system, in somatosensory and prefrontal cortex and in the hippocampus
[@ito2014; @heck2017; @tort2018; @karalis2022; @folschweiller2023]. The inhalation onset is
therefore a natural reference event for olfactory coding, and breathing is a variable
worth recording in many experiments on awake mice, not only in studies of olfaction.

In practice, measuring breathing with this precision in a behaving mouse means implanting
a sensor. The standard methods are a thermistor or a pressure cannula placed in the nasal
cavity [@grimaud2018; @wesson2008; @reisert2014; @mcafee2016]; plethysmography, piezoelectric
sensors and electromyography each trade timing precision, freedom of movement or
invasiveness against one another [@grimaud2018]. Nasal implants require surgery, can obstruct
airflow and damage the nasal epithelium, and cannulae clog and must be cleared, which
stresses the animal further [@mcafee2016; @grimaud2018]. Their signals also degrade: as one
recent account puts it, chronic methods "are invasive, and none are perfectly robust"
[@rafilson2025], and recordings are routinely discarded after inspection [@weinreb2024].
Measuring breathing from video would remove the need for this surgery, a refinement in
the sense of the 3Rs [@tannenbaum2015].

Video is a natural candidate. Machine vision has transformed the measurement of behaviour
[@mathis2018; @pereira2022; @weinreb2024], and face video is now routinely used to explain
neural activity [@stringer2019; @musall2019; @syeda2024]. Because breathing and orofacial
movements are coordinated [@moore2013; @kleinfeld2014], the face should carry the breath.
But facial motion is not breathing. The muscles of the nose, whisker pad and jaw have many
degrees of freedom and are also used to whisk, lick, chew and groom, so the face moves for
many reasons besides breathing, and even the nose has its own rhythm that can run between
breaths [@kurnikova2017]. Current tools label a "sniffing" component of facial motion
without checking it against a breathing sensor [@syeda2024; @hakim2025]. Studies that recorded a
breathing sensor together with video have measured how the two align rather than predicted
one from the other [@kurnikova2017; @findley2021; @weinreb2024]. Thermal imaging of the nose
detects inhalation onsets in head-restrained mice [@esquivelzeta2017; @mutlu2018], but it
requires a thermal camera and has been validated in few animals, and conventional cameras
have so far yielded breathing rate only, in restrained, anaesthetized or sleeping animals
[@kunczik2019; @breuer2023; @geuther2022]. In humans, decades of work on camera-based
physiology, from colour changes in the skin [@verkruysse2008] and motion magnification
[@wu2012] to networks trained end to end [@chen2018; @liu2020; @yu2019; @yu2022; @zou2025], have made
video monitoring of pulse practical, including in neonatal intensive care
[@villarroel2019]. Breathing from face video remains much harder: in the field's reference
benchmark, a multitask network's breathing rate correlated with the reference at only
[0.14]{custom-style="Provisional"} [@liu2023toolbox; @narayanswamy2024], and networks lose much of their accuracy on
data from a new dataset [@liu2023toolbox]. To our knowledge, no method infers the breathing
waveform and inhalation onsets from conventional video and has been validated against an
implanted sensor in animals it was not trained on. Nor is there a public dataset to develop
one.

Here we attempt to close this gap. We assembled a corpus of face video and synchronized intranasal
thermistry from [1,506]{custom-style="Provisional"} sessions in [50]{custom-style="Provisional"} head-fixed mice on [15]{custom-style="Provisional"} rigs across N months, applied
the same quality rules to every session, and drew a split in which test and validation
animals are held out whole. On this benchmark we compared the signals that current tools
extract (motion energy and facial motion components with a linear readout) with networks
developed for human camera physiology and with our own network, all trained and scored
under one protocol. The simple signals failed, and every trained network recovered
breathing on new animals, which shows that the information is in the video and that a
learned model is needed to read it out. Our network detected [95%]{custom-style="Provisional"} of inhalation onsets
in animals it had never seen, and the gap between new animals and new sessions of known
animals locates the remaining difficulty in the individual animal. Because a method meant
to replace a sensor will be used where there is no sensor, we also developed measures,
computed from the network's output alone, that flag unreliable predictions, and checked
them against held-out animals. We release
the corpus, trained models and an evaluation that reproduces our scores.

# Results

## A corpus of face video paired with intranasal thermistry

Mice were head-fixed on a treadmill in a virtual-reality foraging task while two cameras,
one facing the snout and one from the side, filmed them, and a thermistor implanted in the
nasal cavity recorded the temperature of the air at each breath (Fig. 1a).
The thermistor was read through a dedicated signal-conditioning board [\[cite
sniff-detector repo / RRID\]]{custom-style="Author note"}. We used only the face camera
here; side-camera video of every session is released with the corpus for future work. An inhalation cools the thermistor, so its trace falls at each inhalation and
rises at each exhalation, and we marked inhalation onsets and exhalations on it with a
peak detector (Fig. 1b; Methods).

We built the corpus from every recorded session of this task with a thermistor
([7,898]{custom-style="Provisional"} sessions) by applying one written specification to all of them
(Fig. 2a; Methods). Because the thermistor is the training target, we kept
only sessions in which it gave a high-quality signal, and only stretches long enough for two
300-s clips. Sessions that fall short of this bar are not necessarily unusable, but they
would be weaker targets, and clips the rules left out were indeed predicted worse
(Methods). We also confirmed that the thermistor and the cameras were on one clock in
every session kept (Fig. 2b), and removed sessions with timing faults or
dark video (Supplementary Table [1]{custom-style="Provisional"}).

The resulting corpus holds [1,506]{custom-style="Provisional"} sessions from [50]{custom-style="Provisional"} mice on [15]{custom-style="Provisional"} rigs over
[12]{custom-style="Provisional"} months. Breathing in it covers the full range of the mouse: [44%]{custom-style="Provisional"} of the time
at 1–4 Hz, [46%]{custom-style="Provisional"} at 4–8 Hz and [10%]{custom-style="Provisional"} at 8–15 Hz (Fig. 2c). From it
we drew the benchmark split (Fig. 2d). Test set A holds out [10]{custom-style="Provisional"} whole
animals ([78]{custom-style="Provisional"} sessions) and asks whether a model works on an animal it has never seen.
Test set B holds out [115]{custom-style="Provisional"} other sessions of training animals and asks whether it works
on a new day of a known animal. Test set C holds the [5]{custom-style="Provisional"} sessions recorded with a rare
[359]{custom-style="Provisional"}-Hz camera. Validation holds out [8]{custom-style="Provisional"} further animals and is the only set used
for early stopping and model selection. Every set has the corpus's mix of breathing rates (Methods). In all, [560]{custom-style="Provisional"} sessions are assigned, with two 300-s clips each
([≈93]{custom-style="Provisional"} h of face video). Because sessions of one animal resemble each other, we treat
the animal as the unit of replication: scores are averaged within each animal and reported
as the mean over animals with a bootstrap interval over animals.



> **[Figure 1 placeholder]**
>
> **a** Rig: head-fixed mouse on treadmill, face and side cameras, intranasal thermistor,
    signal-conditioning board and task controller on one Harp clock.

> **b** Face-camera frames at three phases of one breath, and the thermistor trace with
    detected inhalation onsets and exhalations.

> **c** The pipeline: 96 × 96 crop of the snout → per-frame CNN encoder → resampling of
    frame embeddings to a 60-Hz grid by frame timestamps → temporal convolutional network →
    breathing trace and inhalation-onset probability.

> **d** Predicted and measured breathing on [20]{custom-style="Provisional"} s of a test-A animal, with predicted and
    true inhalation onsets.
>
> **Source: `notebooks/walkthrough.py` §1–3; rig schematic drawn by hand**

**Fig. 1 \|** **Measuring breathing from face video.** **a**, Acquisition. **b**, Ground truth. **c**, Network. **d**, Example prediction on an animal held out from training.





> **[Figure 2 placeholder]**
>
> **a** Session funnel from every packaged session to the assigned sessions, each loss
    labelled by the first rule it fails.

> **b** Thermistor board minus task controller start times, per session; the four
    excluded sessions marked.

> **c** Breathing-rate distribution of the corpus, and per set (train, validation, A, B, C).

> **d** The split: animals × sessions, coloured by set; rigs and months on the margins.

> **e** Thumbnail gallery of face crops across rigs and months.
>
> **Source: `notebooks/corpus_selection.py`; `folds.parquet`; `sync.parquet`**

**Fig. 2 \|** **The corpus.** **a**, Selection. **b**, Synchronization. **c**, Breathing rates. **d**, Benchmark split with whole held-out animals. **e**, Appearance across rigs.



## Simple video signals do not recover breathing; trained networks do

We first asked whether the signals that current tools extract from face video carry
breathing. Motion energy, the mean absolute difference between frames, is the most common
summary of facial movement [@stringer2019; @musall2019]. Fitted with a global sign and lag on
the training clips, it was uncorrelated with the thermistor on test set A
(**r** = [0.02]{custom-style="Provisional"}), as were two other pixel methods that select the components of the crop
with most power in the breathing band (**r** = [0.01]{custom-style="Provisional"} each;
Fig. 3a,b). Their inhalation F1 scores ([0.17–0.22]{custom-style="Provisional"}) are near what
a trace with the right rate and random timing would reach [\[random-trace control\]]{custom-style="Author note"}.
A linear readout of the leading facial motion components, as in Facemap [@syeda2024], did
better, but recovered only part of the signal (**r** = [0.49]{custom-style="Provisional"}, inhalation F1
[0.52]{custom-style="Provisional"}).

We then trained networks to predict the thermistor trace from video. Ours encodes each
frame of the snout crop with a convolutional network, whose first layer spans three frames
so that it sees change between frames, places the frame embeddings on a 60-Hz grid by
their timestamps, and passes them to a dilated temporal convolutional network
(Fig. 1c; Methods). It has two outputs: the breathing trace, trained with
a correlation loss, and the probability of an inhalation onset at each time step, trained
against a smoothed onset train. It reads only the gray image, with no precomputed motion
channels. For comparison we trained four networks developed for physiology from human face
video, TS-CAN [@liu2020], PhysNet [@yu2019], PhysFormer [@yu2022] and RhythmFormer [@zou2025], on the
same clips and crops, with the same training schedule and model selection on validation
animals, each with the loss of its original paper (Methods).

Every network recovered breathing on new animals (Fig. 3b). Ours had
the highest scores on test set A, with inhalation onsets detected at F1 [0.95]{custom-style="Provisional"} within
25 ms ([1.5]{custom-style="Provisional"} frames at 60 Hz) and a waveform correlation of [0.82]{custom-style="Provisional"}; PhysNet was
close behind (F1 [0.91]{custom-style="Provisional"}, **r** = [0.81]{custom-style="Provisional"}), and TS-CAN, RhythmFormer and PhysFormer
reached correlations of [0.68–0.77]{custom-style="Provisional"}. The margin between our network and PhysNet is
small [\[report paired bootstrap difference over animals\]]{custom-style="Author note"}, and we do not consider the
architecture the main result. What matters is the gap between learned and simple methods:
the breathing signal is present in ordinary face video, and reading it out requires a model
trained on paired data. The onset output contributed modestly: detecting inhalations as
peaks of the predicted trace gave F1 [0.93]{custom-style="Provisional"}, against [0.95]{custom-style="Provisional"} from the onset
probability. Exhalations, detected on the trace, were found less reliably (F1 [0.81]{custom-style="Provisional"}).

Inhalation timing was precise. Matched onsets lagged the thermistor by a median of
[X]{custom-style="Provisional"} ms (interquartile range [X–X]{custom-style="Provisional"} ms; Fig. 3c), well within one
breath at the fastest sniffing rates. Breathing rate, counted in 10-s windows, agreed with
the thermistor with a mean absolute error of [X]{custom-style="Provisional"} breaths per minute and a bias of
[X]{custom-style="Provisional"} (95% limits of agreement [X]{custom-style="Provisional"} to [X]{custom-style="Provisional"}; Fig. 3d), the measures
used for camera respiration in humans [@villarroel2019].



> **[Figure 3 placeholder]**
>
> **a** One test-A clip, 20 s: thermistor and the trace of every method (motion energy,
    pixel PCA, pixel SNR, Facemap-style readout, TS-CAN, PhysNet, PhysFormer,
    RhythmFormer, ours), with true and predicted onsets.

> **b** Correlation and inhalation F1 on test set A, one dot per animal, mean and 95%
    bootstrap interval over animals; dashed line, random-trace F1.

> **c** Onset lag distribution (predicted minus thermistor) for the networks.

> **d** Bland–Altman plot of breathing rate in 10-s windows, ours against thermistor.
>
> **Source: `notebooks/walkthrough.py` §3–7; manifest `configs/results/corpus-v3.toml`**

**Fig. 3 \|** **Breathing from face video on animals held out from training.** **a**, Example traces. **b**, Benchmark. **c**, Onset timing. **d**, Breathing rate.



## New animals are harder than new sessions

The split separates two kinds of generalization. For every network, scores on new sessions
of known animals (test set B) were higher than on new animals (test set A): our network's
correlation was [0.94]{custom-style="Provisional"} on B against [0.82]{custom-style="Provisional"} on A, and PhysNet's [0.93]{custom-style="Provisional"} against
[0.81]{custom-style="Provisional"} (Fig. 4a). A new recording day of a known animal costs
little; most of the error comes from animals the network has never seen. Scores varied between animals in test set A
(Fig. 4b) [\[range of per-animal F1; which animals are worst and why\]]{custom-style="Author note"}. Because ten animals are a small sample, we repeated
training in a five-fold cross-validation in which each of [48]{custom-style="Provisional"} animals is held out once
(Methods); [\[CV result: pooled per-animal F1 and r with bootstrap interval\]]{custom-style="Author note"}.

[\[Scaling: test-A score against number of training animals (4, 8, 16, 32). One
sentence on whether it saturates.\]]{custom-style="Author note"}

Accuracy depended on breathing rate (Fig. 4c) [\[F1 and r per rate band
1–4, 4–8, 8–15 Hz\]]{custom-style="Author note"}. The 359-Hz camera of test set C was predicted as well as the common
cameras (F1 [0.99]{custom-style="Provisional"}, **r** = [0.95]{custom-style="Provisional"}), but its two animals breathe slowly, so this set
cannot separate the effect of frame rate from that of breathing rate. To see where the
network fails, we examined every missed and every false inhalation in the worst sessions
(Fig. 4d) [\[causes and their shares: grooming, licking, fast motion,
dark or blurred video\]]{custom-style="Author note"}.

Finally, we tested a camera view absent from training: the side camera of a different rig
[\[confirm which view and rig\]]{custom-style="Author note"}. Without adaptation the network [\[zero-shot result\]]{custom-style="Author note"};
fine-tuned on [two]{custom-style="Provisional"} labelled sessions of that view, it reached [\[result on the
held-out session\]]{custom-style="Author note"}. The network therefore does not transfer to a new view on its own, but a
small amount of paired data from that view is enough to adapt it [\[only if the
fine-tuning result supports this\]]{custom-style="Author note"}.



> **[Figure 4 placeholder]**
>
> **a** Test A against test B per method (correlation and F1), lines joining each
    method's two scores.

> **b** Per-animal scores of test A and cross-validation, ordered.

> **c** F1 and correlation by breathing-rate band; scaling with the number of training
    animals (inset).

> **d** Failure examples: video frames and traces around a missed inhalation and a false
    one, labelled by cause.

> **e** New view: zero-shot and after fine-tuning on [k]{custom-style="Provisional"} sessions.
>
> *Source: `notebooks/walkthrough.py` §4, §8, §11; `test_sets.py`; new runs (CV, scaling,
    `ood-finetune`)*

**Fig. 4 \|** **Generalization and its limits.** **a**, New animals against new sessions. **b**, Variation between animals. **c**, Breathing rate. **d**, Failures. **e**, New camera view.



## Flagging unreliable predictions without a thermistor

A method that replaces a sensor will be used where there is no sensor to check it. We
therefore developed measures computed only from the network's output, which judge a
prediction window by window (Fig. 5a; Methods). Some test the rhythm: breaths
faster than a mouse can breathe, inhalations and exhalations that fail to alternate, and
breaths too shallow to be real. Others test the onset output: how sharply it peaks, how
much of its probability is ambiguous, and how consistently onsets fall at the same phase
of the predicted breath. Each measure has a pass bound chosen on the validation animals
only, as the value that passes [99%]{custom-style="Provisional"} of their windows in which the true inhalation F1
was at least [0.9]{custom-style="Provisional"}. No measure reads the thermistor, and none judges breathing rate
from below, because slow breathing is normal.

On test-set animals, which played no part in setting the bounds, the measures tracked the
true accuracy [\[result: rank correlation of each measure with true F1 per video; AUC
for windows with F1 < 0.8\]]{custom-style="Author note"} (Fig. 5b). Keeping only windows that passed every
check raised the mean inhalation F1 from [X]{custom-style="Provisional"} to [X]{custom-style="Provisional"} while keeping [X]{custom-style="Provisional"}% of the
recording (Fig. 5c). A calibration of the onset output on validation animals
turns each predicted inhalation's peak into the probability that it is real
(Fig. 5d).

[\[Application, if one is added: run on sessions or animals with no usable thermistor
and show one analysis that needs inhalation times.\]]{custom-style="Author note"}



> **[Figure 5 placeholder]**
>
> **a** One test-A video: trace, onset probability and each measure over time, failing
    windows shaded.

> **b** Each measure against true inhalation F1, per video.

> **c** Mean F1 against fraction of windows kept, keeping the best-judged first; oracle
    and random orderings for reference.

> **d** Calibration of the onset output: predicted against observed precision.
>
> **Source: `notebooks/quality_control.py`**

**Fig. 5 \|** **Judging predictions without a thermistor.** **a**, Measures over time. **b**, Agreement with true accuracy. **c**, Effect of keeping windows that pass. **d**, Onset calibration.



## Running the method

The method needs a face video and a GPU or a CPU. Predicting a 300-s clip took
[0.65]{custom-style="Provisional"} s on a consumer GPU ([≈460]{custom-style="Provisional"} times faster than real time) and [3.4]{custom-style="Provisional"} s on a
CPU [\[re-time on corpus-v3 model\]]{custom-style="Author note"}, and it needs no optical flow, which would cost more
than inference itself. One crop box is placed per session; boxes for our corpus were placed
by template matching and corrected by hand. We release the code, trained weights, the
corpus (face and side video, thermistor and timestamps) with its fixed split, and an evaluation that reproduces the scores reported here
(Data and Code availability).

# Methods

## Corpus construction

All settings below are in one specification file, so a rebuild applies the same rules
[\[cite repository path / version\]]{custom-style="Author note"}. Of the [7,898]{custom-style="Provisional"} packaged sessions, [4,120]{custom-style="Provisional"}
were recorded with dataset version [0.6.0]{custom-style="Provisional"} or later, or were among [15]{custom-style="Provisional"}
hand-reviewed electrophysiology sessions.

_Thermistor quality._ Each session's thermistor was scored in 30-s windows. A window was
good if the mean 1-s range between the 5th and 95th percentiles was at least [100]{custom-style="Provisional"} raw
units, no sample touched the sensor's floor (about [18–20]{custom-style="Provisional"} units) or the converter's
ceiling ([4,095]{custom-style="Provisional"}), breaths fell inside the rate band, and inhalations and exhalations
alternated. A clip could come only from a run of good windows at least 300 s long. A
session was used only if, over the whole session, at most [0.3%]{custom-style="Provisional"} of events repeated
the previous kind, and if its metrics were consistent with the rest of the corpus
([outlier z-score]{custom-style="Provisional"}). Of the [4,120]{custom-style="Provisional"} sessions, [2,038]{custom-style="Provisional"} failed the amplitude
criterion, [257]{custom-style="Provisional"} the alternation criterion, [229]{custom-style="Provisional"} had no good 300-s run, [1]{custom-style="Provisional"}
failed the consistency check and [4]{custom-style="Provisional"} were excluded on review, leaving [1,591]{custom-style="Provisional"}.
The thresholds were checked on validation clips, which were not used to choose them: in
sessions with amplitude below [100]{custom-style="Provisional"}, networks reached a median correlation of
[0.3]{custom-style="Provisional"} and inhalation F1 near [0.1]{custom-style="Provisional"} (against [0.85]{custom-style="Provisional"} above), alternation breaks
separated poor sessions best (AUC [0.87]{custom-style="Provisional"}), and clips the rules left out scored lower
than those they kept with every network (median **r** [0.86]{custom-style="Provisional"} against [0.89]{custom-style="Provisional"}).
Clipping did not lower scores measurably; it was excluded because a clipped breath is a
distorted target.

_Clips and cameras._ [1,518]{custom-style="Provisional"} sessions fitted two 300-s clips and [1,515]{custom-style="Provisional"} had both
cameras. Camera timestamps were checked against the camera's own clock: [73]{custom-style="Provisional"} sessions
with a one-frame rollback were repaired by interpolation, and [9]{custom-style="Provisional"} with other faults
were dropped ([1,506]{custom-style="Provisional"} remaining). Clips were cut from the original video by stream copy,
without re-encoding, at the camera's native rate.

_Synchronization._ Every Harp device answers the read command that starts a session. We
compared the time of this answer on the thermistor board and on the task controller. In
[1,502]{custom-style="Provisional"} of [1,506]{custom-style="Provisional"} sessions they were [1–10]{custom-style="Provisional"} ms apart (median [3.5]{custom-style="Provisional"} ms).
The four others were [0.14–0.88]{custom-style="Provisional"} s apart and were removed (threshold [50]{custom-style="Provisional"} ms),
together with [six]{custom-style="Provisional"} sessions whose video was dark throughout.

## Benchmark split

Test set A and validation hold out whole animals. They were chosen as the best balanced of
[2,000]{custom-style="Provisional"} seeded draws, covering rigs and months and matching the corpus's share of time
at 1–4, 4–8 and 8–15 Hz to within about [2]{custom-style="Provisional"} percentage points. Because sessions of one
animal on one rig resemble each other, the number of sessions per animal and rig was capped
at [10]{custom-style="Provisional"} for training, [3]{custom-style="Provisional"} for validation and [5]{custom-style="Provisional"} for each test set, spread over
dates. The two animals recorded with the [359]{custom-style="Provisional"}-Hz camera were split by session
([6/2/4]{custom-style="Provisional"} and [1/0/1]{custom-style="Provisional"} for train, validation and test C) so that training includes
that rate. Cross-validation folds were drawn alongside this split without changing it
(Supplementary Note [1]{custom-style="Provisional"}).

# References

::: {#refs}
:::
