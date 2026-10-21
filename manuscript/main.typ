// Manuscript draft. Build: typst compile manuscript/main.typ
// Numbers in yellow are provisional (corpus-v3, seed 17, pooled over sessions);
// they become means over animal x rig units with bootstrap intervals, over 5 seeds.
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
network detected inhalation onsets with an F1 score of #tbd[0.94] (within 25 ms) and
recovered the waveform with an average correlation of #tbd[0.79]. Video can thus provide
breath-by-breath timing in head-fixed mice without a nasal implant.

= Introduction

Breathing is more than a vital function in rodents. Sniffing is a form of active sensing: a movement the animal makes to sample its
surroundings. Each inhalation draws odor into the nose, so sniffing sets when and how
olfactory information is sampled @kepecs2006 @verhagen2007 @wachowiak2011, and rodents
adjust it constantly. Mice breathe at about 2–4 Hz at rest and sniff at up to about 12 Hz
when they explore or anticipate an odor, switching between the two within a fraction of
a second @wesson2008. Human breathing, by contrast, stays near 0.2–0.3 Hz, so
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
#tbd[0.94] in animals it had never seen, and the gap between new animals and new sessions of known
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
#tbd[560] sessions, each with two 300-s clips (#tbd[≈93] h of face video), from #tbd[50]
mice on #tbd[15] rigs over #tbd[13] months (October 2025 to October 2026). Animals
contributed a median of #tbd[8.5] sessions over a median of #tbd[34]
days, and #tbd[22] of them were recorded on more than one rig (#fig(<fig:corpus>, "c")).
Breathing in the corpus covers the full range of the mouse: #tbd[26%] of breaths were at
2–4 Hz, #tbd[54%] at 4–8 Hz and #tbd[20%] at 8–15 Hz (#fig(<fig:corpus>, "b")).

We evaluated models under two protocols. In the first, whole animals are held out (test
set A): it asks whether a model works on an animal it has never seen. In the second,
sessions of training animals are held out (test set B): it asks whether a model works on a
new day of a known animal. Validation animals, held out whole, were used only to select
models, and test sessions only for final scoring. The results below use one draw of these
sets; the same rules gave the cross-validation folds (Methods, @tab:split). Because
sessions of one animal on one rig resemble each other, the unit of comparison is an animal
on one rig (an animal × rig unit): scores are averaged within each unit and reported as
the mean over units, with a bootstrap interval over units (Methods, Statistics). Test set
A holds #tbd[21] such units from #tbd[10] animals, and test set B #tbd[44] from
#tbd[26].

#figure(
  fit(fig1),
  caption: [*Measuring breathing from face video.* *a*, The rig: a head-fixed mouse on a wheel facing the virtual-reality screens, the
    cameras with example frames, and the odor delivery line. *b*, Task and
    behaviour aligned to reward-site entry. *c*, Ground truth: the face frame with the crop
    the network sees (orange), the crop at three moments of one breath, and the thermistor
    with its inhalations and exhalations (one breath of one clip). *d*, Network. *e*, Eight
    seconds of one test-A clip as the animal starts sniffing: the predicted breathing over the thermistor, and the
    onset probability, with the thermistor's inhalations (black) and the network's (blue).],
) <fig:overview>

#figure(
  fit(fig2),
  caption: [*The corpus.* *a*, Selection: the recorded sessions, step by step, to the
    corpus; each loss is labelled with the first rule the session failed (rules and
    thresholds in @tab:qc). *b*, Distribution of breathing rate (thermistor breaths,
    2–15 Hz) over the corpus (black; n = #tbd[1,630,355] breaths) and for each animal (grey;
    n = #tbd[50]). *c*, Sessions per animal, by the number of rigs the animal was recorded
    on (n = #tbd[50] animals, #tbd[560] sessions).],
) <fig:corpus>

== Simple video signals do not recover breathing; trained networks do

We first asked whether the signals that current tools extract from face video carry
breathing. Motion energy, the mean absolute difference between frames, is the most common
summary of facial movement @stringer2019 @musall2019. Fitted with a global sign and lag on
the training clips, it was uncorrelated with the thermistor on test set A
($r = #tbd[0.02]$), as were two other pixel methods that select the components of the crop
with most power in the breathing band ($r <= #tbd[0.01]$ each;
#fig(<fig:benchmark>, "a,b")\; @tab:benchmark). Their inhalation F1 scores (#tbd[0.16–0.20]) are near what
a trace with the right rate and random timing would reach #note[random-trace control].
A linear readout of the leading facial motion components, as in Facemap @syeda2024, did
better, but recovered only part of the signal ($r = #tbd[0.46]$, inhalation F1
#tbd[0.49]).

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

Every network recovered breathing on new animals (#fig(<fig:benchmark>, "b")\;
@tab:benchmark\; an example in #fig(<fig:overview>, "e")). Ours
detected inhalation onsets best on test set A, at F1 #tbd[0.94] within 25 ms (#tbd[1.5]
frames at 60 Hz), against #tbd[0.91] for PhysNet (paired difference #tbd[0.03], 95%
interval #tbd[0.01–0.05]). Its waveform correlation, #tbd[0.79], matched PhysNet's
(#tbd[0.80]\; difference #tbd[−0.01], #tbd[−0.05 to 0.03]), and TS-CAN, RhythmFormer and
PhysFormer reached correlations of #tbd[0.67–0.77]. It also infers faster than PhysNet:
a 5-min clip took #tbd[0.87] s on a consumer GPU (RTX 4060 Ti), against #tbd[1.58] s for
PhysNet and #tbd[5.8–6.0] s for the two transformers (#fig(<fig:benchmark>, "b")), and
#tbd[3.4] s on a CPU, against #tbd[19] s #note[re-time on CPU with the corpus-v3 models]
(Supplementary Table #tbd[2]). The onset output contributed modestly: detecting inhalations as
peaks of the predicted trace gave F1 #tbd[0.92], against #tbd[0.94] from the onset
probability. Exhalations, detected on the trace, were found less reliably (F1 #tbd[0.76]).

Accuracy depended on breathing rate (#fig(<fig:benchmark>, "c")). Our network was
least accurate on slow breathing (F1 #tbd[0.87] at 2–3 Hz) and near perfect from 6 Hz up
(F1 ≥ #tbd[0.99]), whereas every other network also lost accuracy above 10 Hz; PhysFormer
reached only #tbd[0.50] at 2–3 Hz and RhythmFormer #tbd[0.65] above 12 Hz.

Inhalation timing was precise. Matched onsets lagged the thermistor by a median of
#tbd[0.9] ms (interquartile range #tbd[−5.2 to 7.3] ms), well within one breath at the
fastest sniffing rates, and the lag hardly changed with breathing rate, from a median of
#tbd[+3.4] ms at 2–3 Hz to #tbd[−3.5] ms at 11–12 Hz (#fig(<fig:benchmark>, "d")). Breathing rate, counted in 10-s windows, agreed with
the thermistor with a mean absolute error of #tbd[4.4] breaths per minute and a bias of
#tbd[+2.4] (95% limits of agreement #tbd[−13.0] to #tbd[+17.8]\; #fig(<fig:benchmark>, "e")), the measures
used for camera respiration in humans @villarroel2019. @tab:benchmark gives these rate
measures for every method, as reported for camera physiology in humans @liu2023toolbox.

#figure(
  fit(fig3),
  caption: [*Breathing from face video on animals held out from training.* *a*, Six
    seconds of one test-A clip (the clip of Fig. 1e): the thermistor (top) and the
    trace of each method over it (grey), with each method's correlation on the whole
    clip; vertical lines, thermistor inhalations. *b*, Inhalation F1 on test set A against
    inference time per 5-min clip from the preprocessed crops (median of 8 clips): one dot
    per animal × rig unit (mean of its recordings; n = #tbd[21] units from #tbd[10]
    animals), and the mean over units with its 95% bootstrap interval; filled, networks on
    a consumer GPU; open, methods on a CPU. *c*, Inhalation F1 on test set A by breathing
    rate for each network (top; colours as in *b*) and the share of breaths at each rate
    (bottom); n = #tbd[221,626] thermistor breaths at 2–15 Hz. *d*, Our network's onset lag (predicted minus thermistor; every
    thermistor inhalation with a prediction within 50 ms) against the breath's rate on test
    set A: the share of each rate bin's lags (colour), with the median (solid) and
    interquartile range (dashed); dotted lines, the 25-ms tolerance; n = #tbd[218,608]
    inhalations. *e*, Breathing rate in 10-s windows on test set A: our network minus the
    thermistor against their mean (colour, number of windows), with the bias (solid) and
    95% limits of agreement (dotted); n = #tbd[4,679] windows. *a*–*e*, test set A,
    #tbd[156] clips from #tbd[78] sessions.],
) <fig:benchmark>

#figure(
  kind: table,
  supplement: [Table],
  {
    // Written by notebooks/manuscript_figures.py (test set A, seed 17).
    let rows = csv("tables/benchmark.csv").slice(1)
    set text(size: 7.5pt)
    table(
      columns: 9,
      align: (left,) + (right,) * 8,
      stroke: none,
      inset: (x: 3pt, y: 2.5pt),
      table.hline(),
      table.header(
        [], [], table.cell(colspan: 4, align: center)[*Breathing rate (10-s windows)*],
        table.cell(colspan: 3, align: center)[*Waveform and events*],
        [*Method*], [*Inference (s)*], [*MAE ↓*], [*RMSE ↓*], [*MAPE (%) ↓*],
        [*r ↑*], [*Trace r ↑*], [*Inhale F1 ↑*], [*Exhale F1 ↑*],
      ),
      table.hline(stroke: 0.5pt),
      ..rows.flatten(),
      table.hline(),
    )
  },
  caption: [*Benchmark on new animals (test set A).* Mean ± s.e.m. over the #tbd[21]
    animal × rig units of the #tbd[10] test-A animals, each the mean of its recordings. Breathing rate is counted in 10-s windows;
    MAE and RMSE in breaths per minute, MAPE relative to the thermistor's rate, and r the
    Pearson correlation of the windowed rates. Trace r, correlation of the predicted
    breathing trace with the thermistor; F1 within 25 ms. Inference, seconds per 5-min
    clip from the preprocessed crops on a consumer GPU (\*, on a CPU). #tbd[Seed 17.]],
) <tab:benchmark>

== The method generalizes to new sessions and subjects

The two protocols separate two kinds of generalization. For every network, scores on new sessions
of known animals (test set B) were higher than on new animals (test set A): our network's
correlation was #tbd[0.93] on B against #tbd[0.79] on A, and PhysNet's #tbd[0.92] against
#tbd[0.80] (#fig(<fig:general>, "a")). A new recording day of a known animal costs
little; most of the error comes from animals the network has never seen. Scores varied between the animal × rig units of test set A
(#fig(<fig:general>, "b")) #note[range of per-unit F1; which units are worst and why]. Because ten animals are a small sample, we repeated
training in a five-fold cross-validation in which each of #tbd[48] animals is held out once
(Methods); #note[CV result: F1 and r over units with bootstrap interval].

To ask how much data is necessary to generalized to new held-out animals, we retrained our network on nested subsets of
#tbd[4], #tbd[8] and #tbd[16] of the #tbd[32] training animals, with all their sessions
(#tbd[35], #tbd[74] and #tbd[148] of #tbd[295]), and scored each on the same test sets
(#fig(<fig:general>, "c")). On new animals (test set A), scores barely depended on the
number of training animals: inhalation F1 was #tbd[0.93] with #tbd[4] animals and
#tbd[0.94] with #tbd[32] (paired difference #tbd[0.02], 95% interval #tbd[−0.01 to
0.04]), and the waveform correlation #tbd[0.77] against #tbd[0.79]. Keeping all #tbd[32]
animals but a quarter of their sessions (#tbd[74]) gave much the same scores (F1
#tbd[0.93], $r$ #tbd[0.77]). A network
trained on a few animals thus generalizes to new ones about as well as one trained on
all 32 animals #note[one seed per size; repeat with the
other seeds, and with the cross-validation folds (full-gray-data-size-cv4-v3)].

To see where the
network fails, we examined every missed (false negative, FN) and every false inhalation (false positive, FP) on test set A
(#fig(<fig:general>, "d")\; their causes are in #fig(<fig:qc>, "a")). Most of the failures, about three quarters, were small differences between the predicted and true inhalation times rather than gross misdetections (#fig(<fig:general>, "d")). Relaxing the tolerance from 25 to 100 ms raised inhalation F1 on
test set A from #tbd[0.95] to #tbd[0.99] and cut the errors left from #tbd[9.3%] to
#tbd[2.4%] of breaths (#fig(<fig:general>, "e")).

A low waveform correlation did not mean that the breaths were lost. Across test-A clips,
correlation and inhalation F1 were only loosely related (Spearman $rho = #tbd[0.41]$):
in the #tbd[16] clips with the lowest tenth of correlations ($r < #tbd[0.61]$), median F1
was still #tbd[0.76], and a quarter reached #tbd[0.9] or more. In the clip shown,
the waveform correlation was #tbd[0.48] yet every inhalation was found (F1
#tbd[1.00]); over the 2 s shown, through fast sniffing and a pause, the predicted trace
departs from the thermistor in shape and level ($r = #tbd[0.36]$, RMSE #tbd[1.7]
s.d.), while each predicted onset falls on a thermistor inhalation
(#fig(<fig:general>, "f")). Onset times can thus be used even where the waveform is
poor.

#figure(
  fit(fig4),
  caption: [*Generalization and its limits.* *a*, Mean over animal × rig units on new
    animals (test set A; n = #tbd[21] units) and new sessions of known animals (test set B;
    n = #tbd[44]), per method. *b*, Inhalation F1 of each test-A animal × rig unit (mean of
    its recordings; n = #tbd[21]), ranked by our network. *c*, Inhalation F1 (top) and
    correlation (bottom) of our network retrained on nested subsets of the training
    animals, with all their sessions (filled, joined) or with all #tbd[32] animals and a
    quarter of their sessions (#tbd[74]; open, drawn beside #tbd[32]), on new animals
    (test set A): mean over the same #tbd[21] units with 95% bootstrap interval; one seed
    per run. *d*, Near misses: among test-A clips with a correlation at least the median,
    the one with the most predictions 25–100 ms from a thermistor inhalation (r =
    #tbd[0.92], F1 #tbd[0.91]), around its densest near misses. *e*, Inhalation F1 on test
    set A as the matching tolerance is relaxed (pairs within 100 ms); for our network, F1
    and errors left (misses and false inhalations, as a share of breaths) at 25 ms (the
    tolerance used throughout), 50 ms (dotted lines) and 100 ms; n = #tbd[223,588]
    thermistor breaths per network. *f*, A clip with low correlation whose inhalations are
    all found: among the #tbd[16] test-A clips in the lowest tenth of correlation, the
    one with the highest inhalation F1 (r = #tbd[0.48], F1 #tbd[1.00]), around the 2-s
    stretch with the largest difference between trace and thermistor (RMSE, both z-scored
    over the clip) among stretches with no error or excluded span and at least 4
    inhalations (r = #tbd[0.36], RMSE #tbd[1.7]). In *d* and *f* (one clip each):
    crops (top), the thermistor and our trace with the thermistor's (black) and the
    network's (blue) inhalations, misses (FN) and false inhalations (FP) marked, and the
    onset probability.],
) <fig:general>

== Known failure modes and limitations

We first asked what our network's remaining errors are. Of the #tbd[9,432] missed and
#tbd[11,304] false inhalations on test set A, #tbd[82]% and #tbd[69]% were near misses: a
prediction and a thermistor inhalation within 100 ms of each other that the 25-ms
tolerance did not pair (#fig(<fig:qc>, "a")). The other, gross errors did not gather
where one might expect. Fast breathing, above 8 Hz, accounted for #tbd[5]% of gross misses
and no gross false inhalations, against #tbd[19]% of all breaths; face motion was no more
common around errors than around breaths in general (#tbd[11–12]% against #tbd[10]%).
Only shallow breaths were over-represented, at #tbd[32–37]% of gross errors against
#tbd[25]% of breaths (Methods).

A method that replaces a sensor will be used where there is no sensor to check it. We
therefore computed measures from the network's output alone, which judge each 30-s
window of a prediction without the thermistor; their bounds were set on validation
animals (Methods). On new animals (test set A), which played no part in setting them,
the best measures ranked the #tbd[156] videos by their true inhalation F1 with Spearman
correlations of #tbd[0.63–0.80] (#fig(<fig:qc>, "b")). The #tbd[88]% of windows that
passed every check had a mean inhalation F1 of #tbd[0.96], against #tbd[0.81] for the
windows that failed and #tbd[0.95] for all (#fig(<fig:qc>, "c")). The onset output is
also calibrated: on validation animals, the probability the network gives a predicted
inhalation matched the share that were real (expected calibration error #tbd[0.002];
#fig(<fig:qc>, "d")), so each inhalation can carry its own confidence.

Finally, the network's accuracy did not hinge on its training recipe
(#fig(<fig:qc>, "e")). Training with more augmentation, or with every breathing-rate bin
weighted equally, changed inhalation F1 on test set A by less than #tbd[0.01] (paired
intervals including zero), and doubling the crop to 192 × 192 pixels lowered it slightly
(#tbd[−0.013], 95% interval #tbd[−0.021 to −0.006]); no variant changed the waveform
correlation #note[one seed per variant].

#note[Application, if one is added: run on sessions or animals with no usable thermistor
and show one analysis that needs inhalation times.]

#figure(
  fit(fig5),
  caption: [*When to trust a prediction.* *a*, Our network's errors on test set A
    (#tbd[9,432] misses, FN; #tbd[11,304] false inhalations, FP): the share that are near
    misses (a prediction and a thermistor inhalation within 100 ms), and, among the other
    errors (#tbd[1,748] FN, #tbd[3,508] FP), the share in fast breathing (> 8 Hz), in face
    motion (top 10% of the clip) and on shallow breaths (bottom 25% of the clip); grey
    ticks, the share of all #tbd[223,588] thermistor breaths with each property. *b*,
    Spearman correlation with the true inhalation F1 of the five measures that rank the
    #tbd[156] test-A videos best, signed so that a positive value means the measure ranks
    predictions correctly; colours, families of measures. *c*, Mean true inhalation F1 of
    the test-A windows kept (#tbd[1,560] 30-s windows), best-judged first, for the three
    measures that rank windows best and for the truth itself (oracle); star, the windows
    that pass every check; dotted, all windows. *d*, Calibration of the onset output on the
    validation animals: the predicted probability that an inhalation is real against the
    share matched to a thermistor inhalation. *e*, Our network trained differently, minus
    our network, on test set A: inhalation F1 and correlation, mean over the #tbd[21]
    units with a paired 95% bootstrap interval; one seed each.],
) <fig:qc>

= Methods

== Data collection

_Surgery._ All procedures were approved by the Allen Institute Institutional Animal Care
and Use Committee #note[protocol number; mice: strain, sex, age at surgery, housing and
light cycle; analgesia and post-operative care]. Thermistors were implanted over the
nasal epithelium as described @mcafee2016. Briefly, under isoflurane anaesthesia (1–5%)
in a stereotaxic frame, the skin over the skull and the snout was removed and the
periosteum cleared. A 0.5-mm burr hole was thinned through the nasal bone #tbd[3.1] mm
anterior to the frontonasal suture and #tbd[0.5] mm right of the midline, exposing the
epithelium, and the thermistor tip was lowered until it touched the epithelium without
pressing on it. The hole was sealed with silicone (Kwik-Cast) and the probe fixed with
dental cement (Metabond), which was also used to attach a head bar, with the
thermistor's connector fixed on top of it. Mice recovered on a 37 °C heat plate.

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
    [Window (30 s)], [Signal amplitude (5th–95th percentile range)], [≥ #tbd[100] raw units], [#tbd[2,038]],
    [], [Clipped samples], [#tbd[0]% at ≤ #tbd[20] or ≥ #tbd[4,090]], [—],
    [], [Breaths outside #tbd[2–15] Hz], [≤ #tbd[10]%], [—],
    [], [Repeated events (I→I or E→E)], [≤ #tbd[5]%], [—],
    [Session], [Repeated events over the session], [≤ #tbd[0.3]%], [#tbd[257]],
    [], [Longest run of good windows], [≥ #tbd[300] s], [#tbd[229]],
    [], [Event amplitude within expected range], [robust $z <= #tbd[3]$], [#tbd[1]],
    [], [Dead sensor (low SNR)], [modes ≥ #tbd[3] dB apart], [—],
    [Clips], [Room for two clips in good stretches], [#tbd[2] × #tbd[300] s], [#tbd[73]],
    [Cameras], [A camera missing], [], [#tbd[3]],
    [], [Timestamp alignment issues], [], [#tbd[13]],
    [], [Low exposure], [], [#tbd[6]],
    [Manual review], [], [], [#tbd[4]],
    table.hline(),
  ),
  caption: [*Corpus selection.* Rules in the order applied, with the number of sessions
    each removed. Of #tbd[4,120] recorded sessions, #tbd[1,591] passed the thermistor rules
    and #tbd[1,506] the clip and camera rules. Window rules mark 30-s windows as good or
    bad and remove no session on their own ("—"), except amplitude, which is also applied
    to the session as a whole (the #tbd[2,038] sessions removed); a session without a long
    enough run of good windows is removed. Repeated events are an inhalation followed by another
    inhalation, or an exhalation by another exhalation. Timestamps with a single one-frame
    rollback were repaired (#tbd[73]) rather than removed. Low exposure and unplugged sync
    cables were found after the sets were drawn and removed from them, leaving the
    #tbd[560] sessions of the corpus
    #note[check the dead-sensor and window-rule counts].],
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

The draw used for the results in the main text and the cross-validation folds (Supplementary Note
#tbd[1]) were both drawn under these rules (@tab:split) #note[add any further splits here].

_Training-set size._ To measure how scores depend on the amount of training data
(#fig(<fig:general>, "c")), we retrained our network, with the settings of @tab:train, on
parts of the published split's training set, keeping its validation and test sets: on
nested subsets of #tbd[4], #tbd[8] and #tbd[16] of the #tbd[32] training animals with all
of their sessions (#tbd[35], #tbd[74] and #tbd[148] sessions), and on #tbd[74] sessions
drawn from all #tbd[32] animals (a median of #tbd[2] per animal, against #tbd[8.5] in the
full set). Every run was scored on the same test-A units.

_Rig coverage._ The #tbd[4]- and #tbd[8]-animal runs trained on only some of the test-A
rigs, so their test-A units fall into those on a rig in the training set and those on a
rig it lacked (#tbd[10] and #tbd[11] units with #tbd[4] animals, #tbd[15] and #tbd[6] with
#tbd[8]; the larger runs trained on every test-A rig). Scores were averaged per unit and
then over the units of each group: F1 #tbd[0.94] and #tbd[0.91] ($r$ #tbd[0.79] and
#tbd[0.76]) with #tbd[4] training animals, and #tbd[0.96] and #tbd[0.91] ($r$ #tbd[0.82]
and #tbd[0.75]) with #tbd[8]. The comparison is not paired, since different units fall
in each group #note[one seed; repeat with the other seeds and the cross-validation
folds].

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

_Timing._ Inference time was the wall-clock time to turn the preprocessed crops of a
5-min clip, already in memory, into its 60-Hz trace, so decoding and cropping the video
are not included. Each method was run once on another clip to warm up, then on
#tbd[8] clips, and we report the median. Networks ran one clip at a time (batch size 1)
in bf16 on one consumer GPU (NVIDIA RTX 4060 Ti), with frames read in chunks of
#tbd[256]\; PhysFormer and RhythmFormer were run in windows of #tbd[320] samples, the
length they were built for, and the other networks in windows of #tbd[1,024]. Motion
energy, the pixel methods and the Facemap-style readout ran on the CPU.

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
was counted in 10-s windows, and its error (MAE, RMSE, MAPE) and correlation computed over
the windows of each unit (see Statistics).

_Error analysis._ Every missed thermistor inhalation and every unmatched predicted
inhalation of test set A was examined (#fig(<fig:qc>, "a")). An error was a near miss when an
error of the other kind lay within 100 ms of it. For the other errors we recorded the true
breathing rate there (the rate of the thermistor breath it falls in), the face motion
around it (mean absolute difference between successive crops within 0.25 s) and the size
of the breath (peak-to-peak thermistor signal within 0.25 s), the last two as quantiles
over the true breaths of the same clip. Face motion in the top 10% and breath size in the
bottom 25% of a clip are thus, by construction, the shares of all breaths with those
properties.

_Training variants._ To test how much the results depend on the training recipe
(#fig(<fig:qc>, "e")), we retrained our network with the settings of @tab:train except for one
change each: more augmentation (shifts of up to 8 instead of 4 pixels, zoom 0.15 instead
of 0.1, rotation up to 10°, and blur), every breathing-rate bin weighted equally in each
batch, or 192 × 192-pixel crops from frames downsampled to 720 × 540 (batch 4, 400 steps
per epoch, learning rate 3 × 10#super[−4]).

== Statistics

Unless otherwise stated, our sampling is over a subject recorded on one rig (i.e., an
animal × rig). Sessions of one animal on one rig share the implant, the camera's
position and the lighting, so they are not independent samples; an animal recorded on a
second rig is seen through a different camera and enclosure. Every score was first
averaged over the clips and sessions of a unit, and units were then
treated as the samples for plots and statistics. Test set A holds #tbd[21] units from #tbd[10] animals, test set B
#tbd[44] units from #tbd[26] animals, and validation #tbd[25] units from #tbd[9] animals.

_Scores per method._ A method's score on a test set is the mean of its unit scores. Its
95% interval comes from a percentile bootstrap: we drew the test set's units with
replacement #tbd[2,000] times, took the mean of each draw, and report the 2.5th and
97.5th percentiles of those means. @tab:benchmark gives the mean ± s.e.m. over units
instead.

_Pooled analyses._ Analyses of single breaths (F1 by breathing rate, onset lag,
matching tolerance) and of 10-s windows (rate agreement) do not average per unit: they
pool every breath or window of the test set, so units with more recordings contribute
more, and they carry no interval. Their sample size is the number of breaths or windows
pooled. Each figure legend gives the sample size of its panels.
#note[units of one animal are not independent of each other; check intervals with a
bootstrap that resamples animals and then units within them]

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
