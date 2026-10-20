// Manuscript draft. Build: typst compile manuscript/main.typ
// Numbers in yellow are provisional (corpus-v3, seed 17, pooled over sessions);
// they become per-animal means with bootstrap intervals over 5 seeds.
// Red notes are for the authors and are removed before submission.

#import "figures/fig1.typ": body as fig1
#import "figures/fig2.typ": body as fig2
#import "figures/fig3.typ": body as fig3
#import "figures/fig4.typ": body as fig4
#import "figures/fig5.typ": body as fig5

// A figure assembled at its printed width (183 mm), scaled to the text width.
#let fit(body) = layout(size => scale(reflow: true, size.width / 189mm * 100%, body))

#let tbd(body) = highlight(fill: rgb("#fff3b0"), body)
#let note(body) = text(fill: rgb("#b42318"), size: 8.5pt)[\[#body\]]

#set document(title: "Inferring mouse breathing from orofacial video")
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 11pt, lang: "en")
#set par(justify: true, leading: 0.75em, spacing: 1.2em)
#set heading(numbering: none)
#show heading.where(level: 2): set text(size: 11pt, style: "italic")
#set figure(supplement: [Fig.])
#set figure.caption(separator: [ | ])
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(left)
#show figure.where(kind: table): set figure.caption(position: top)

// Figure reference with a panel letter: #fig(<fig:corpus>, "b") gives "Fig. 2b".
#let fig(label, panel) = [#ref(label)#panel]
// Placeholder for a figure that is not drawn yet; `source` says what will draw it.
#let mock(panels, source) = rect(
  width: 100%, inset: 10pt, fill: luma(246), stroke: (paint: luma(160), dash: "dashed"),
)[
  #set text(size: 8.5pt, fill: luma(70))
  #set align(left)
  #panels
  #v(2pt)
  _Source: #source _
]
#show heading.where(level: 1): set text(size: 13pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Inferring sniffing from orofacial video in mice]
  #v(0.6em)
  VrForaging Platform#super[1], #note[authors] \
  #text(size: 9.5pt)[#super[1]Allen Institute for Neural Dynamics, Seattle, WA, USA]
]

#v(1em)

= Abstract

Olfaction is one of the primary senses in rodents. Contrary to vision and audition,
olfaction relies heavily on active sampling through sniffing. Ground-truth methods for measuring sniffing
in mice typically rely on sensors implanted in the nasal cavity which require invasive surgery and
often fail to yield a usable signal.
Here we show that the breathing waveform and the
timing of each inhalation can be inferred from ordinary video of the face. We assembled a
corpus of #tbd[560] curated sessions from #tbd[50] head-fixed mice that pairs face video with a
synchronized intranasal thermistor. On #tbd[ten] held-out animals never seen in training, our
network detected inhalation onsets with an F1 score of #tbd[0.95] (within 25 ms) and
recovered the waveform with an average correlation of #tbd[0.82]. Video can thus provide
breath-by-breath timing in head-fixed mice without a nasal implant.

= Introduction

Breathing is more than a vital function in rodents. Sniffing is a form of active sensing: a movement the animal makes to sample its
surroundings. Each inhalation draws odor into the nose, so sniffing sets when and how
olfactory information is sampled @kepecs2006 @verhagen2007 @wachowiak2011, and rodents
adjust it constantly. Mice breathe at about 2–4 Hz at rest and sniff at up to about 12 Hz
when they explore or anticipate an odor, switching between the two within a fraction of
a second @welker1964 @wesson2008. Human breathing, by contrast, stays near 0.2–0.3 Hz, so
a rodent's breathing rate varies over a range several times wider and changes far faster.
Responses of olfactory bulb neurons are locked to precise phases of the sniff cycle
@shusterman2011. The respiratory rhythm
originates in the brainstem @smith1991, and it also reports the animal's state: breathing
changes with arousal and emotion @homma2008 @delnegro2018, brainstem neurons of the
breathing rhythm generator themselves promote arousal @yackle2017, and fear-conditioned
mice breathe at a steady 4 Hz while they freeze @bagur2021. The inhalation onset is
therefore a natural reference event for olfactory coding, and breathing is a variable
worth recording in many experiments on awake mice, not only in studies of olfaction.

In practice, measuring breathing with this precision in a behaving mouse means implanting
a sensor. The standard methods are a thermistor or a pressure cannula placed in the nasal
cavity @grimaud2018 @wesson2008 @reisert2014 @mcafee2016; plethysmography, piezoelectric
sensors and electromyography each trade timing precision, freedom of movement or
invasiveness against one another @grimaud2018. Nasal implants require surgery, can obstruct
airflow and damage the nasal epithelium, and cannulae clog and must be cleared, which
stresses the animal further @mcafee2016 @grimaud2018. Their signals also degrade: as one
recent account puts it, chronic methods "are invasive, and none are perfectly robust"
@rafilson2025, and recordings are routinely discarded after inspection @weinreb2024.
Measuring breathing from video would remove the need for this surgery, a refinement in
the sense of the 3Rs @tannenbaum2015.

Videography is a natural candidate. Machine vision has transformed the measurement of behaviour
@mathis2018 @pereira2022 @weinreb2024, and face video is now routinely used to explain
neural activity @stringer2019 @musall2019 @syeda2024. Because breathing and orofacial
movements are coordinated @moore2013 @kleinfeld2014, the face should carry the sniffing information.
But facial motion is not breathing. The muscles of the nose, whisker pad and jaw have many
degrees of freedom and are also used to whisk, lick, chew and groom, so the face moves for
many reasons besides breathing, and even the nose has its own rhythm that can run between
breaths @kurnikova2017. Current tools label a "sniffing" component of facial motion
without checking it against a breathing sensor @syeda2024 @hakim2025. Studies that recorded a
breathing sensor together with video have measured how the two align rather than predicted
one from the other @kurnikova2017 @findley2021 @weinreb2024. Thermal imaging of the nose
detects inhalation onsets in head-restrained mice @esquivelzeta2017 @mutlu2018, but it
requires a thermal camera and has been validated in few animals, and conventional cameras
have so far yielded breathing rate only, in restrained, anaesthetized or sleeping animals
@kunczik2019 @breuer2023 @geuther2022. In humans, decades of work on camera-based
physiology, from colour changes in the skin @verkruysse2008 and motion magnification
@wu2012 to networks trained end to end @chen2018 @liu2020 @yu2019 @yu2022 @zou2025, have made
video monitoring of pulse practical, including in neonatal intensive care
@villarroel2019. Breathing from face video remains much harder: in the field's reference
benchmark, a multitask network's breathing rate correlated with the reference at only
#tbd[0.14] (TODO CHECK) @liu2023toolbox @narayanswamy2024, and networks lose much of their accuracy on held-out data @liu2023toolbox. To our knowledge, no method infers the breathing
waveform and inhalation onsets from conventional video and has been validated against an
implanted sensor in animals it was not trained on. Nor is there a public dataset to develop
one.

Here we attempt to close both of these gaps. We assembled a corpus of face video and synchronized intranasal
thermistor recordings from #tbd[560] sessions in #tbd[50] head-fixed mice on #tbd[15] rigs, applied
the same quality rules to every session, and drew a split in which test and validation
animals are held out whole. On this benchmark we compared the signals that current tools
extract (motion energy and facial motion components with a linear readout) with networks
developed for human camera physiology and with our own network, all trained and scored
under one protocol. The simple signals failed, and every trained network recovered
breathing on new animals. Our network detected inhalation onsets with an F1 score of
#tbd[0.95] in animals it had never seen, and the gap between new animals and new sessions of known
animals locates the remaining difficulty in the individual animal. Because a method meant
to replace a sensor will be used where there is no sensor, we also developed measures,
computed from the network's output alone, that flag unreliable predictions, and checked
them against held-out animals. We release the corpus, trained models and an evaluation that reproduces our scores.

= Results

== A corpus of face video paired with ground-truth sniffing recordings

Mice were head-fixed on a circular treadmill, while engaging in an olfactory virtual-reality foraging task.
Briefly, mice ran along a virtual corridor of odor-cued reward sites and could stop at a site to collect a probabilistic reward. Running, stopping and odor sampling alternate throughout a session, so breathing ranges from rest to fast sniffing (#fig(<fig:overview>, "b") and #fig(<fig:corpus>, "b")). While the mice performed the task, high-speed video was recorded from two angles (#fig(<fig:overview>, "a")). A snout-facing camera captured detailed movements of the nose and whiskers, while a side camera recorded the overall posture and locomotion. For all analysis in this study, we only used the face camera view, but we are including side-camera video for every session with the corpus for future work. All mice were additionally implanted with a small thermistor in the nasal cavity to record temperature changes as a proxy for breathing. The signal was conditioned and digitized using a custom, open-source, board #note[cite
sniff-detector repo / RRID] at 250 Hz.
As inhalation draws cool air, temperature in the thermistor drops, and it rises again during exhalation as the subject expells warm air. This periodic signal reflects the breathing cycle that can be readily extracted with simple digital-signal processing (DSP) peak detection algorithms (#fig(<fig:overview>, "c")\; Methods).

We built the corpus from every recorded session of this task with a thermistor
(#tbd[4,120] sessions) by applying a uniform quality-control pipeline
(#fig(<fig:corpus>, "a")\; Methods). Because the thermistor is the training target, we kept
only sessions in which it gave a high-quality signal for a sufficient duration. We note that sessions that
fall short of this bar are not necessarily unusable, but they would be weaker targets
#note[check whether clips the rules left out are predicted worse, and add the analysis to Methods].

Of the #tbd[1,506] sessions that passed every check, we kept #tbd[560], capped per animal
and rig so that no animal or rig dominates (Methods). The resulting corpus holds these
#tbd[560] sessions from #tbd[50] mice on #tbd[15] rigs over #tbd[13] months (October 2025
to October 2026). Breathing in it covers the full range of the mouse: in the training set,
#tbd[25%] of breaths were at 2–4 Hz, #tbd[54%] at 4–8 Hz and #tbd[20%] at 8–15 Hz
(#fig(<fig:corpus>, "b")). We split them into benchmark sets (#fig(<fig:corpus>, "c")). Test set A holds out #tbd[10] whole
animals and asks whether a model works on an animal it has never seen.
Test set B holds out sessions of training animals and asks whether it works
on a new day of a known animal. In all protocols, the test set was only used for final evaluation and scoring.  Every set has the corpus's mix of breathing rates (Methods). In all, #tbd[560] sessions are assigned, with two 300-s clips each
(#tbd[≈93] h of face video). Because sessions of one animal resemble each other, we treated
the subject as the unit of replication: scores are averaged within each mouse and reported
as the mean over animals with a bootstrap interval over animals.

#figure(
  fit(fig1),
  caption: [*Measuring breathing from face video.* *a*, Acquisition. *b*, Task and
    behaviour aligned to reward-site entry. *c*, Ground truth: the face frame with the crop
    the network sees (orange), the crop at three moments of one breath, and the thermistor
    with its inhalations and exhalations. *d*, Network. *e*, Eight seconds of a test-A
    animal as it starts sniffing: the predicted breathing over the thermistor, and the
    onset probability, with the thermistor's inhalations (black) and the network's (blue).],
) <fig:overview>

#figure(
  fit(fig2),
  caption: [*The corpus.* *a*, Selection: the recorded sessions, step by step, to the
    sets of the split; each loss is labelled with the first rule the session failed
    (rules and thresholds in @tab:qc). *b*, Distribution of breathing rate (thermistor breaths, 2–15 Hz) in each set. *c*, The split: one row
    per animal (grouped by role), one dot per session at its date, coloured by set.],
) <fig:corpus>

== Simple video signals do not recover breathing; trained networks do

We first asked whether the signals that current tools extract from face video carry
breathing. Motion energy, the mean absolute difference between frames, is the most common
summary of facial movement @stringer2019 @musall2019. Fitted with a global sign and lag on
the training clips, it was uncorrelated with the thermistor on test set A
($r = #tbd[0.02]$), as were two other pixel methods that select the components of the crop
with most power in the breathing band ($r <= #tbd[0.01]$ each;
#fig(<fig:benchmark>, "a,b")). Their inhalation F1 scores (#tbd[0.17–0.22]) are near what
a trace with the right rate and random timing would reach #note[random-trace control].
A linear readout of the leading facial motion components, as in Facemap @syeda2024, did
better, but recovered only part of the signal ($r = #tbd[0.50]$, inhalation F1
#tbd[0.56]).

We then trained networks to predict the thermistor trace from video. Ours encodes each
frame of the snout crop with a convolutional network, whose first layer spans three frames
so that it sees change between frames, places the frame embeddings on a 60-Hz grid by
their timestamps, and passes them to a dilated temporal convolutional network
(#fig(<fig:overview>, "d")\; Methods). It has two outputs: the breathing trace, trained with
a correlation loss, and the probability of an inhalation onset at each time step, trained
against a smoothed onset train. It reads only a small 96x96 crop of the grayscale image over the snout of the animal. For comparison we trained four networks developed for physiology from human face
video, TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025, on the
same clips and crops, with the same training schedule and model selection on validation
animals, each with the loss of its original paper (Methods).

Every network recovered breathing on new animals (#fig(<fig:benchmark>, "b")\; an example
in #fig(<fig:overview>, "e")). Ours had
the highest scores on test set A, with inhalation onsets detected at F1 #tbd[0.95] within
25 ms (#tbd[1.5] frames at 60 Hz) and a waveform correlation of #tbd[0.82]\; PhysNet was
close behind (F1 #tbd[0.92], $r = #tbd[0.80]$), and TS-CAN, RhythmFormer and PhysFormer
reached correlations of #tbd[0.66–0.75]. We note that while the margin between our network and PhysNet is
small #note[report paired bootstrap difference over animals], it infers faster:
a 5-min clip took #tbd[0.93] s on a consumer GPU (RTX 4060 Ti), against #tbd[1.56] s for
PhysNet and #tbd[5.6–6.2] s for the two transformers (#fig(<fig:benchmark>, "b")), and
#tbd[3.4] s on a CPU, against #tbd[19] s #note[re-time on CPU with the corpus-v3 models]
(Supplementary Table #tbd[2]). The onset output contributed modestly: detecting inhalations as
peaks of the predicted trace gave F1 #tbd[0.94], against #tbd[0.95] from the onset
probability. Exhalations, detected on the trace, were found less reliably (F1 #tbd[0.80]).

Inhalation timing was precise. Matched onsets lagged the thermistor by a median of
#tbd[0.9] ms (interquartile range #tbd[−5.2 to 7.3] ms; #fig(<fig:benchmark>, "c")), well within one
breath at the fastest sniffing rates. Breathing rate, counted in 10-s windows, agreed with
the thermistor with a mean absolute error of #tbd[4.4] breaths per minute and a bias of
#tbd[+2.4] (95% limits of agreement #tbd[−13.0] to #tbd[+17.8]\; #fig(<fig:benchmark>, "d")), the measures
used for camera respiration in humans @villarroel2019.

#figure(
  fit(fig3),
  caption: [*Breathing from face video on animals held out from training.* *a*, Ten
    seconds of one test-A clip (the clip of Fig. 1e): the thermistor (top) and the
    trace of each method over it (grey), with each method's correlation on the whole
    clip; vertical lines, thermistor inhalations. *b*, Correlation and inhalation F1 on
    test set A: one dot per animal (mean of its recordings), and the mean over animals
    with its 95% bootstrap interval; right, inference time per 5-min clip from the
    preprocessed crops (median of 8 clips; networks on a consumer GPU, hatched methods on
    a CPU). *c*, Onset lag of matched inhalations (predicted
    minus thermistor; every thermistor inhalation with a prediction within 100 ms) for the
    networks on test set A; dotted lines, the 25-ms tolerance; grey, lags scored as a miss
    and a false inhalation.
    *d*, Breathing rate in 10-s windows on test set A: our network minus the thermistor
    against their mean (colour, number of windows), with the bias (solid) and 95% limits
    of agreement (dotted).],
) <fig:benchmark>

== The method generalizes to new sessions and subjects

The split separates two kinds of generalization. For every network, scores on new sessions
of known animals (test set B) were higher than on new animals (test set A): our network's
correlation was #tbd[0.92] on B against #tbd[0.82] on A, and PhysNet's #tbd[0.92] against
#tbd[0.80] (#fig(<fig:general>, "a")). A new recording day of a known animal costs
little; most of the error comes from animals the network has never seen. Scores varied between animals in test set A
(#fig(<fig:general>, "b")) #note[range of per-animal F1; which animals are worst and why]. Because ten animals are a small sample, we repeated
training in a five-fold cross-validation in which each of #tbd[48] animals is held out once
(Methods); #note[CV result: pooled per-animal F1 and r with bootstrap interval].

To ask how much data a new lab would need, we trained on subsets of #tbd[4, 8, 16] and
#tbd[32] animals and scored each on test set A (#fig(<fig:general>, "f")) #note[scaling runs;
one sentence on whether it saturates].

Accuracy depended on breathing rate (#fig(<fig:general>, "c")) #note[F1 and r per rate band
2–4, 4–8, 8–15 Hz]. To see where the
network fails, we examined every missed (false negative, FN) and every false inhalation (false positive, FP) on test set A
(#fig(<fig:general>, "e,g")) #note[causes and their shares: grooming, licking, fast motion,
dark or blurred video]. Most of the failures, about three quarters, were small differences between the predicted and true inhalation times rather than gross misdetections (#fig(<fig:general>, "e")). Relaxing the tolerance from 25 to 100 ms raised inhalation F1 on
test set A from #tbd[0.95] to #tbd[0.99] and cut the errors left from #tbd[9.3%] to
#tbd[2.4%] of breaths (#fig(<fig:general>, "d")).

#figure(
  fit(fig4),
  caption: [*Generalization and its limits.* *a*, Mean over animals on new animals
    (test set A) and new sessions of known animals (test set B), per method. *b*,
    Inhalation F1 of each test-A animal (mean of its recordings), ranked by our network.
    *c*, Inhalation F1 on test set A by breathing rate (top) and the share of breaths at
    each rate (bottom). *d*, Inhalation F1 on test set A as the matching tolerance is relaxed (pairs within
    100 ms); for our network, F1 and errors left (misses and false inhalations, as a share
    of breaths) at 25 ms (the tolerance used throughout), 50 ms (dotted lines) and 100 ms. *e*, Near misses: among test-A clips with a correlation at least the median, the one
    with the most predictions 25–100 ms from a thermistor inhalation (r = #tbd[0.92], F1
    #tbd[0.91]), around its densest near misses. *f*, Correlation and inhalation F1 on test set A against the
    number of training animals. *g*, Errors in the test-A clip at the 10th percentile of
    inhalation F1 (r = #tbd[0.52], F1 #tbd[0.91]), around its densest errors. In *e* and *g*: crops around the error
    (top), the thermistor and our trace with the thermistor's (black) and the network's
    (blue) inhalations, misses and false inhalations marked, and the onset probability.],
) <fig:general>

== Known failure modes and limitations

A method that replaces a sensor will be used where there is no sensor to check it and thus cannot be manually validated. We
therefore developed measures computed only from the network's output, which judge a
prediction window by window (#fig(<fig:qc>, "a")\; Methods). We show that using these metrics, and training them on validation animals, allows us to fine tune the bounds for reliable prediction without access to the thermistor. Indeed, on test-set animals, which played no part in setting the bounds, the measures tracked the
true accuracy #note[result: rank correlation of each measure with true F1 per video; AUC
for windows with F1 < 0.8] (#fig(<fig:qc>, "b")). Keeping only windows that passed every
check raised the mean inhalation F1 from #tbd[X] to #tbd[X] while keeping #tbd[X]% of the
recording (#fig(<fig:qc>, "c")). A calibration of the onset output on validation animals
turns each predicted inhalation's peak into the probability that it is real
(#fig(<fig:qc>, "d")).

#note[Application, if one is added: run on sessions or animals with no usable thermistor
and show one analysis that needs inhalation times.]

#figure(
  fit(fig5),
  caption: [*Judging predictions without a thermistor.* *a*, One test-A video: ten
    seconds of the predicted trace with its inhalations and the onset probability
    (left), and over the whole video three of the measures (each scaled to its range)
    and the true inhalation F1, per window; shaded, windows that fail a check; grey,
    the excerpt. *b*, Spearman correlation of each measure with the true inhalation F1
    on test sets A and B, per video and per window, signed so that a positive value
    means the measure ranks predictions correctly. *c*, Mean true inhalation F1 of the
    windows kept, best-judged first, for the three best measures and for the truth
    itself (oracle); star, the windows that pass every check; dotted, all windows.
    *d*, Calibration of the onset output on the validation animals: the predicted
    probability that an inhalation is real against the share matched to a thermistor
    inhalation.],
) <fig:qc>

= Methods

== Data collection

_Surgery_

_Data acquisition_

All data shown in this paper was acquired from using the VrForaging platform (github). Video data was acquired using 240Hz or 360Hz Flir Cameras and compressed online using a custom ffmpeg pipeline. A second pass compression was applied offline to further reduce file size(TODO). The remaining of the devices implemented the Harp protocol thus providing synchronization at acquisition time. Specifically, we use a Behavior board (todo) to trigger camera frames at the start of each session, a Olfactometer (todo) to deliver odors with temporal precision, and a Sniff-detector board (see below) to monitor sniffing.

_Video encoding_

During acquisition, frames were encoded in real time on the GPU with H.264 (NVENC; variable
bit rate in constant-quality mode, CQ 18, preset p3, high-quality tuning) with 4:2:0 chroma
subsampling, full range and BT.709 colour metadata, and written to Matroska files. In the
offline second pass, each video was re-encoded on the CPU with H.264 (libx264; CRF 18, preset
slow). Frames were first converted to BT.709 at 10-bit precision without dithering, then
reduced to 8-bit 4:2:0 at limited (TV) range with ordered dithering, and written to MP4
files with colour metadata and the index at the start of the file. Clips for the corpus were
cut from these files by stream copy, without a further decoding/encoding.

== Sniff-detector board
TODO SONYA

== Corpus construction

Sessions were selected with the criteria in @tab:qc, applied in the order listed. Thermistor
quality was assessed in 30-s windows, and clips were taken only from stretches of at least
300 s in which every window passed. Large artifacts, and the #tbd[2] s around them, were
removed first. Inhalation onsets and exhalations were detected on the thermistor with an
adaptive peak detector (parameters in Supplementary Table #tbd[3]).

#figure(
  kind: table,
  supplement: [Table],
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, left, right),
    stroke: none,
    inset: (x: 5pt, y: 3.5pt),
    table.hline(),
    table.header([*Stage*], [*Rule*], [*Threshold*], [*Removed*]),
    table.hline(stroke: 0.5pt),
    [Window (30 s)], [Amplitude: mean over 1-s blocks of the 5th–95th percentile range], [≥ #tbd[100] raw units], [#tbd[2,038]],
    [], [Clipping: samples at or beyond the sensor floor or the ADC ceiling], [#tbd[0]% at ≤ #tbd[20] or ≥ #tbd[4,090]], [—],
    [], [Breaths outside the rate band], [≤ #tbd[10]% outside #tbd[2–15] Hz], [—],
    [], [Alternation breaks: events repeating the previous kind], [≤ #tbd[5]%], [—],
    [Session], [Alternation breaks over the whole session], [≤ #tbd[0.3]%], [#tbd[257]],
    [], [A run of good windows long enough to cut clips], [≥ #tbd[300] s], [#tbd[229]],
    [], [Event consistency with the rest of the corpus], [robust $z <= #tbd[3]$], [#tbd[1]],
    [], [Dead sensor: SNR below the corpus's low-SNR mode], [modes ≥ #tbd[3] dB apart], [—],
    [Clips], [Two 300-s clips fit in good stretches], [#tbd[2] × #tbd[300] s], [#tbd[73]],
    [Cameras], [Both cameras present], [], [#tbd[3]],
    [], [Camera timestamps clean (#tbd[73] one-frame rollbacks repaired)], [], [#tbd[9]],
    [], [Low video exposure], [], [#tbd[6]],
    [Excluded on review], [Thermistor trace rejected on inspection], [], [#tbd[4]],
    [], [Synchronization issues due to unplugged cables], [], [#tbd[4]],
    table.hline(),
  ),
  caption: [*Corpus selection.* From #tbd[4,120] recorded sessions to #tbd[1,591] after
    thermistor rules and #tbd[1,506] after clip and camera rules. Synchronization and exposure issues were
    excluded from the #tbd[560] sessions of the split. "—": these rules are scored per window and remove no
    session on their own #note[check the dead-sensor and window-rule counts].],
) <tab:qc>

The peak detector and the thresholds were set by manual inspection: they were adjusted
until the detected events fitted the raw thermistor trace, judged by eye and by keeping low
the number of consecutive inhalations (I→I) or exhalations (E→E) with no event of the other
kind between them. Clips were cut at the camera's native rate (see Video encoding).

== Data splits

Every split used in this work followed the same rules.

- *Whole animals for new-animal tests and validation.* Test animals and validation animals
  contributed no session to training, so these sets measure performance on animals the
  network has never seen.
- *Separate sessions for known-animal tests.* Test sessions of training animals were held
  out as blocks of consecutive dates, so that a test session's neighbours in time were not in
  training.
- *Balance.* Held-out animals were chosen as the best of many seeded draws, so that every set
  covered the rigs and months of the corpus and matched its share of time at 1–4, 4–8 and
  8–15 Hz.
- *Caps per animal and rig.* Sessions of one animal on one rig resemble each other, so their
  number was capped in every set and spread over dates, and heavily recorded animals do not
  dominate.
- *High frame-rate cameras.* Animals recorded with the #tbd[359]-Hz camera were split by session, so that
  training includes that rate. Their remaining #tbd[5] sessions form a third test set (test
  C), a check on a camera rate rare in training; its scores are in Supplementary Table
  #tbd[X] and are not discussed in the main text.
- *Validation only for selection.* Validation animals were used for early stopping and model
  selection, and never for testing.

The split reported in the main text and the cross-validation folds (Supplementary Note
#tbd[1]) were both drawn under these rules (@tab:split) #note[add any further splits here].

#figure(
  kind: table,
  supplement: [Table],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    stroke: none,
    inset: (x: 6pt, y: 3.5pt),
    table.hline(),
    table.header([*Set*], [*Held out as*], [*Animals*], [*Sessions*]),
    table.hline(stroke: 0.5pt),
    [Train], [—], [#tbd[31–33]], [#tbd[213–295]],
    [Validation], [whole animals], [#tbd[9]], [#tbd[67–144]],
    [Test A], [whole animals], [#tbd[9–10]], [#tbd[75–128]],
    [Test B], [sessions of training animals], [#tbd[26–30]], [#tbd[70–119]],
    [Test C], [#tbd[359]-Hz sessions], [#tbd[2]], [#tbd[5]],
    table.hline(stroke: 0.5pt),
    [Total], [], [#tbd[50]], [#tbd[560]],
    table.hline(),
  ),
  caption: [*Split sizes.* Minimum and maximum over the #tbd[6] splits: the published
    split and the #tbd[5] cross-validation folds, in which each of the #tbd[48] animals
    outside the #tbd[359]-Hz group is a test-A animal once. Validation includes #tbd[2]
    sessions of the #tbd[359]-Hz animals; test B animals are training animals, so animals
    do not add up across sets.],
) <tab:split>

== Preprocessing

Each frame was downsampled to 360 × 270 pixels by area averaging and cropped to a
96 × 96-pixel box over the snout (192 × 192 pixels of the original frame), placed once per
session. Frames were converted to gray and selected on a 60-Hz grid by their camera
timestamps; each selected frame keeps its own timestamp, so the network knows when every
frame was taken. Each channel was standardized with the mean and standard deviation of
the training clips. The training target is the thermistor trace filtered at its own rate
(second-order Butterworth, high-pass #tbd[0.2] Hz and low-pass #tbd[40] Hz, applied
forwards and backwards), resampled onto the 60-Hz output grid and z-scored per clip.

== Network architecture

The network has a frame encoder, a resampling step and a temporal network
(#fig(<fig:overview>, "d")\; @tab:arch).

_Frame encoder._ A convolutional network maps each crop to a 128-dimensional embedding.
Its first layer is a 3D convolution over three consecutive frames (kernel 3 × 5 × 5,
spatial stride 2, 32 channels), so it sees change between frames without precomputed
motion channels. Three 2D convolutions follow (kernel 3 × 3, stride 2; 64, 96 and 128
channels), each with batch normalization and a GELU activation. The 6 × 6 output is
average-pooled to 2 × 2 rather than to a single value, so that opposite motion in two
parts of the snout does not cancel, and a linear layer gives the embedding.

_Resampling._ Embeddings are computed at the selected frames' timestamps and linearly
interpolated onto a uniform 60-Hz grid. Irregular frame intervals therefore do not shift
the prediction in time, and the output grid is the same for every camera.

_Temporal network._ A non-causal temporal convolutional network reads the embedding
sequence: a 1 × 1 projection to 128 channels, then six residual blocks with dilations 1,
2, 4, 8, 16 and 32, each two dilated convolutions (kernel 3) with batch normalization,
GELU and dropout. Its receptive field is #tbd[253] output samples (#tbd[4.2] s). Two
1 × 1 convolutions read its output: the breathing trace, and the logit of an inhalation
onset at each time step. The model has #tbd[0.86] million parameters (#tbd[0.25] million
in the encoder).

#figure(
  kind: table,
  supplement: [Table],
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    inset: (x: 6pt, y: 3pt),
    table.hline(),
    table.header([*Stage*], [*Layer*], [*Output*]),
    table.hline(stroke: 0.5pt),
    [Input], [gray crop, three consecutive frames], [3 × 96 × 96],
    [Encoder], [Conv3D 3×5×5, stride (1, 2, 2), BN, GELU], [32 × 48 × 48],
    [], [Conv 3×3, stride 2, BN, GELU], [64 × 24 × 24],
    [], [Conv 3×3, stride 2, BN, GELU], [96 × 12 × 12],
    [], [Conv 3×3, stride 2, BN, GELU], [128 × 6 × 6],
    [], [average pool to 2×2, flatten, linear, GELU], [128 per frame],
    [Resampling], [linear interpolation by frame timestamps], [128 at 60 Hz],
    [Temporal], [Conv 1×1], [128],
    [], [6 residual blocks of 2 × (Conv 3, dilation $d$, BN, GELU) and dropout; $d = 1, 2, …, 32$], [128],
    [Heads], [Conv 1×1 (trace), Conv 1×1 (onset logit)], [1 + 1],
    table.hline(),
  ),
  caption: [*Network architecture.* BN, batch normalization.],
) <tab:arch>

== Training

The loss was the sum of a correlation term on the trace and a classification term on
onsets. The correlation term is $1 - r$ between the predicted and target traces,
averaged over three time scales (the trace, and copies average-pooled by 4 and 16
samples), so that errors in the slow envelope are not hidden by the fast breathing
component; it ignores amplitude, which is arbitrary in the thermistor. The onset term is
a binary cross-entropy against a target of Gaussians (s.d. #tbd[20] ms) centred on each
thermistor inhalation, weighted by #tbd[0.5], with positive samples up-weighted by the
inverse of their share in each batch.

Training used windows of #tbd[512] output samples (#tbd[8.5] s) drawn at random from the
training clips, in batches of #tbd[16]. Weights were optimized with AdamW (learning rate
#tbd[6 × 10#super[−4]], weight decay #tbd[10#super[−4]]), with #tbd[200] steps of linear
warm-up followed by cosine decay to 1% of the peak rate, gradient clipping at norm
#tbd[1] and bf16 mixed precision, and an exponential moving average of the weights
(decay #tbd[0.999]) was kept. Each frame's position on the selection grid was jittered
by up to a quarter of a frame interval, its timestamp moving with it, and each window was
augmented by translation (up to #tbd[4] pixels), zoom (#tbd[±10%]), brightness and
contrast changes (#tbd[±15%] and #tbd[±20%]) and additive Gaussian noise (s.d. #tbd[2]
grey levels). Every #tbd[4] epochs of #tbd[100] steps, the model predicted the whole
clips of the validation animals, and the checkpoint with the highest mean correlation was
kept; training stopped after #tbd[12] scorings without improvement, after at least
#tbd[40] and at most #tbd[200] epochs. Once the averaged weights had settled, selection
used them. Each model was trained with #tbd[5] seeds #note[seeds, GPU, time per run]
(@tab:train). The comparison networks were trained with the same procedure, each on the
loss of its original paper, with the learning rate chosen on the validation animals
(Supplementary Note #tbd[2]).

#figure(
  kind: table,
  supplement: [Table],
  table(
    columns: (auto, auto),
    align: (left, left),
    stroke: none,
    inset: (x: 6pt, y: 3pt),
    table.hline(),
    table.header([*Setting*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Input], [gray, 96 × 96 crop, 60-Hz selection grid],
    [Window, batch], [#tbd[512] samples (#tbd[8.5] s), #tbd[16]],
    [Optimizer], [AdamW, learning rate #tbd[6 × 10#super[−4]], weight decay #tbd[10#super[−4]]],
    [Schedule], [#tbd[200]-step linear warm-up, cosine decay to 1%],
    [Regularization], [dropout #tbd[0.1] (temporal network), gradient clip #tbd[1], weight EMA #tbd[0.999]],
    [Loss], [multi-scale $1 - r$ (scales 1, 4, 16) + #tbd[0.5] × onset BCE],
    [Onset target], [Gaussians, s.d. #tbd[20] ms, per-batch positive weight],
    [Augmentation], [frame jitter ±0.25 frame, shift ±#tbd[4] px, zoom ±#tbd[10]%, brightness ±#tbd[15]%, contrast ±#tbd[20]%, noise #tbd[2] grey levels],
    [Selection], [mean correlation on validation animals, every #tbd[4] epochs],
    [Stopping], [patience #tbd[12] scorings, #tbd[40]–#tbd[200] epochs of #tbd[100] steps],
    [Precision], [bf16 mixed precision],
    table.hline(),
  ),
  caption: [*Training settings* of the network reported in the main text.],
) <tab:train>

== Inference and event detection

A clip was predicted in overlapping windows of #tbd[1,024] output samples, discarding
#tbd[126] samples (half the receptive field) at each window edge, so the stitched trace
equals a single pass over the whole clip. Inhalation onsets were the peaks of the onset
probability above #tbd[0.5], at least #tbd[50] ms apart, and exhalations the troughs of
the predicted trace. For the comparison networks, which have no onset output,
inhalations were the peaks of the predicted trace.

== Comparison methods

Motion energy and two pixel methods (the principal component of the crop, and a pixel
average weighted by power in the breathing band) were computed from the same crops
without training, with one global sign and lag fitted on the training clips. The
Facemap-style readout was a ridge regression from lagged principal components of the
crop's movie and motion energy @syeda2024, fitted on the training clips. TS-CAN
@liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025 were
reimplemented from their papers and trained as above #note[input sizes, windows and
learning rates in Supplementary Note 2].

== Scoring

Predictions and thermistor were compared by time, over the span both cover. The trace
was scored by its Pearson correlation with the target. A predicted inhalation or
exhalation matched a thermistor event within #tbd[1.5] frames of the 60-Hz output grid
(25 ms), each event matching at most once, and F1 was computed per clip. Breathing rate
was counted in 10-s windows. Scores were averaged within each animal and reported as the
mean over animals, with 95% intervals from a bootstrap over animals #note[number of
draws; paired differences].

== Applying the method to new video

The method needs a face video, its frame timestamps and a GPU or a CPU. One crop box is
placed per session over the snout; boxes for our corpus were placed by template matching
against labelled examples and corrected by hand in the annotation tool. Preprocessing
crops and downsamples each frame and stores the gray channel; inference then predicts the
breathing trace and onset probability on a 60-Hz grid, and the quality measures can be
computed from that output alone #note[commands: `zephyr preprocess`, `zephyr evaluate`,
`zephyr quality-control run`].

== Data and code availability

We release the code, trained weights, the corpus (face and side video, thermistor and
timestamps) with its fixed split, and an evaluation that reproduces the scores reported
here #note[repository URL, data DOI, licence; confirm the corpus can be released].

#bibliography("refs.bib", style: "nature")
