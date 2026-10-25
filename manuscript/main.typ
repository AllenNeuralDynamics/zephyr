// Manuscript draft, in the Allen Institute preprint template (the submodule template/:
// `git submodule update --init`). Build from the repository root:
//   typst compile --font-path manuscript/template/fonts manuscript/main.typ
// Results numbers come from values.toml, which notebooks/manuscript_figures.py writes
// (#v("section.key")): re-run the notebook, never edit them here. Numbers in yellow
// (#tbd) are typed by hand and still to be confirmed; red notes are for the authors
// and are removed before submission.
#import "template/preprint.typ": *
#import "figures/fig1.typ": body as fig1
#import "figures/fig2.typ": body as fig2
#import "figures/fig3.typ": body as fig3
#import "figures/fig4.typ": body as fig4
#import "figures/fig5.typ": body as fig5
#import "figures/fig6.typ": body as fig6
#import "figures/fig7.typ": body as fig7
#import "figures/figs1.typ": body as figs1
#import "figures/figs2.typ": body as figs2
#import "figures/figs3.typ": body as figs3
#import "figures/figs4.typ": body as figs4

// A figure assembled at its printed width (183 mm), scaled to the text width.
#let fit(body) = layout(size => scale(reflow: true, size.width / 189mm * 100%, body))

// The numbers the figure notebook computes (values.toml, written by
// notebooks/manuscript_figures.py), as printed: #v("section.key").
#let generated = toml("values.toml")
// A key the notebook has not written yet prints as a red placeholder, not an error.
#let v(key) = {
  let (section, name) = key.split(".")
  let found = generated.at(section, default: (:)).at(name, default: none)
  if found == none { text(fill: rgb("#b42318"))[\[#key\]] } else { found }
}

// The benchmark table (Table 1 and its test-B twin), from a CSV the figure notebook
// writes: "mean (low–high)" cells put the interval on a smaller line under the mean.
#let benchmark-table(path) = {
  let rows = csv(path).slice(1)
  set text(size: 7pt)
  table(
    columns: 10,
    align: (left,) + (right,) * 9,
    fill: (_, y) => if y < 2 { colors.table-head } else { colors.table-row },
    toprule,
    table.header(
      [], [], [], table.cell(colspan: 4, align: center)[*Breathing rate (10-s windows)*],
      table.cell(colspan: 3, align: center)[*Waveform and events*],
      [*Method*], [*Splits*], [*Inference (s)*], [*MAE ↓*], [*RMSE ↓*], [*MAPE (%) ↓*],
      [*r ↑*], [*Trace r ↑*], [*Inhale F1 ↑*], [*Exhale F1 ↑*],
    ),
    midrule,
    // "mean (low–high)": the interval goes on a smaller line under the mean.
    ..rows
      .flatten()
      .map(c => {
        let m = c.match(regex("^(\S+) \((.+)\)$"))
        if m == none { [#c] } else {
          [#m.captures.at(0) \ #text(size: 5.5pt, fill: luma(90))[#m.captures.at(1)]]
        }
      }),
    bottomrule,
  )
}

#let tbd(body) = highlight(fill: rgb("#fff3b0"), body)
#let note(body) = text(fill: rgb("#b42318"), size: 8.5pt)[\[#body\]]

// Figure reference with a panel letter: #fig(<fig:corpus>, "b") gives "Fig. 2b".
#let fig(label, panel) = [#ref(label)#panel]
// Placeholder for a figure that is not drawn yet; `source` says what will draw it.
#let mock(panels, source) = rect(
  width: 100%,
  inset: 10pt,
  fill: luma(246),
  stroke: (paint: luma(160), dash: "dashed"),
)[
  #set text(size: 8.5pt, fill: luma(70))
  #set align(left)
  #panels
  #v(2pt)
  _Source: #source _
]

#show: preprint.with(
  title: [An open benchmark and framework for breath-by-breath sniffing from video in mice],
  short-title: [Zephyr],
  authors: ((name: "VrForaging Platform", affiliations: ("aind",)),),
  affiliations: (
    aind: [Allen Institute for Neural Dynamics, Seattle, WA, USA #note[authors]],
  ),
  abstract: [
    Rodents rely on olfaction to survive. Odors guide them to find food, avoid predators
    and recognize mates and kin. Mice sample odors actively by sniffing, so the timing of
    each breath shapes odor perception. Ground-truth methods for measuring sniffing in mice
    typically rely on sensors implanted in the nasal cavity, which require invasive surgery
    and often fail to yield a usable signal. Here we show that the breathing waveform and
    the timing of each inhalation can be inferred from ordinary video. We assembled a corpus
    of #v("corpus.sessions") curated sessions from #v("corpus.animals") head-fixed mice
    performing an olfactory foraging task, in which face, side and front video are
    synchronized with an intranasal thermistor. Signals that current tools extract from face video did not track
    breathing, whereas networks trained on the corpus did. In animals held out from
    training (#v("benchmark.animals_a") animals, each held out once in a five-fold
    cross-validation), our network, Zephyr, detected inhalations within 25 ms with an F1
    score of #v("benchmark.f1") and recovered the breathing waveform with a correlation of
    #v("benchmark.r"). Its accuracy grew with the number of animals it was trained on rather
    than with the amount of video, and it also recovered breathing, less accurately, from the side and front cameras. We release the corpus, the trained
    models, quality-control measures computed from the network's output alone, and
    software with which labs can build their own inference pipelines and measure
    breath-by-breath sniffing without a nasal implant.
  ],
)

= Introduction

Sniffing is a form of active sensing. Each inhalation draws odor into the nose, so the
timing of sniffing sets when olfactory information is sampled @kepecs2006 @verhagen2007
@wachowiak2011. Mice breathe at about 2–4 Hz at rest and sniff at up to 12–15 Hz when they
explore or anticipate an odor, and they switch between the two within a fraction of a
second @wesson2008. Olfactory bulb neurons respond at precise phases of this cycle
@shusterman2011. Breathing also reports the animal's state. The rhythm originates in the
brainstem @smith1991 and changes with arousal and emotion @homma2008 @delnegro2018; neurons
of the rhythm generator themselves promote arousal @yackle2017, and fear-conditioned mice
breathe at a steady 4 Hz while they freeze @bagur2021. The inhalation is therefore a
natural reference event for olfactory coding, and breathing is worth recording in many
experiments on awake mice, not only in studies of olfaction.

Measuring breathing with this precision in a behaving mouse, however, requires an
implanted sensor. The standard methods place a thermistor or a pressure cannula in the
nasal cavity @grimaud2018 @wesson2008 @reisert2014 @mcafee2016; plethysmography,
piezoelectric sensors and electromyography each give up timing precision, freedom of
movement or invasiveness @grimaud2018. Nasal implants require surgery, can obstruct airflow
and damage the nasal epithelium, and cannulae clog and must be cleared @mcafee2016
@grimaud2018. Their signals also degrade: chronic methods "are invasive, and none are
perfectly robust" @rafilson2025, and recordings are routinely discarded after inspection
@weinreb2024. A measure of breathing from video would spare animals this surgery, a
refinement in the sense of the 3Rs @tannenbaum2015. It would also add breathing to the
many experiments that already film the face.

The face should carry sniffing. Breathing and orofacial movements are coordinated
@moore2013 @kleinfeld2014, and machine vision now measures behaviour @mathis2018
@pereira2022 @weinreb2024 and uses face video to explain neural activity @stringer2019
@musall2019 @syeda2024. Yet facial motion is not breathing. The nose, whisker pad and jaw
also whisk, lick, chew and groom, and the nose has its own rhythm that can run between
breaths @kurnikova2017. Current tools label a "sniffing" component of facial motion without
checking it against a breathing sensor @syeda2024 @hakim2025, and studies that recorded
both measured how the two align rather than predicting one from the other @kurnikova2017
@findley2021 @weinreb2024. Thermal imaging detects inhalations in head-restrained mice but
requires a thermal camera @esquivelzeta2017 @mutlu2018, and conventional cameras have
yielded only breathing rate, in restrained, anaesthetized or sleeping animals @kunczik2019
@breuer2023 @geuther2022. Networks that recover pulse and breathing from human face video
@chen2018 @liu2020 @yu2019 @yu2022 @zou2025 @liu2023toolbox were built for slow, steady
signals, whereas a mouse changes its breathing rate several-fold within a second. No method
has been shown to recover the breathing waveform and the timing of each inhalation from
conventional video in animals it was not trained on, and no public dataset exists to
develop and test one.

We assembled a corpus that pairs face, side and front video with synchronized intranasal
thermistor recordings from #v("corpus.sessions") sessions in #v("corpus.animals")
head-fixed mice on #v("corpus.rigs") rigs. Every session passed the same quality rules.
Splitting the corpus to hold out whole animals, we trained and scored, under one protocol,
the signals that current tools extract, networks developed for human camera physiology and our own
network, Zephyr. Zephyr recovered the timing of each inhalation in animals it had never
seen, gained more from training on more animals than from more video of the same
animals, and also recovered breathing from the side and front cameras. A method meant to replace a sensor will be used
where there is no sensor, so we also developed quality-control measures, computed from the
network's output alone, that identify unreliable predictions. We release the corpus, the
trained models, an evaluation that reproduces our scores and an application that takes a
lab's own videos from crop placement to predictions and quality-control reports.

= Results

== A corpus of orofacial video paired with intranasal thermistor recordings

Head-fixed mice performed an olfactory virtual-reality foraging task on a circular
treadmill. They ran along a virtual corridor of odor-cued sites and could stop at a site to
collect a probabilistic reward (#fig(<fig:overview>, "a")). Because running, stopping and
odor sampling alternate throughout a session, breathing ranges from rest to fast sniffing
(#fig(<fig:overview>, "b,c") and #fig(<fig:corpus>, "b")). Three high-speed cameras filmed
the animal: a snout-facing camera captured the nose and whiskers, and side and front
cameras captured posture and locomotion, as on many behavioural rigs. A thermistor
implanted in the nasal cavity recorded breathing. Inhalation draws cool air over it and
exhalation warm air, and we detected the onset of each inhalation and exhalation on its
signal (#fig(<fig:overview>, "e")\; Methods). A custom open-source board conditioned and
digitized the signal at 250 Hz #note[cite sniff-detector repo / RRID].

Because the thermistor is the training target, we kept only sessions in which it gave a
clean signal. We started from every recorded session of this task with a thermistor
(#v("selection.recorded") sessions) and applied one quality-control pipeline, whose rules
and thresholds are fixed in a single specification (#fig(<fig:corpus>, "a")\; @tab:qc;
Methods). A session passed if the thermistor was clean for long enough to cut two 300-s
clips. Sessions that failed are not necessarily unusable, but they would be weaker targets
#note[check whether clips the rules left out are predicted worse, and add the analysis to
  Methods]. Of the #v("selection.clips") sessions that passed every check, we kept
#v("selection.corpus"), capped per animal and rig so that no animal or rig dominates.

The corpus holds #v("corpus.sessions") sessions with two 300-s clips each
(≈#v("corpus.hours") h of video per camera), from #v("corpus.animals") mice on
#v("corpus.rigs") rigs over #v("corpus.months") months. Animals contributed a median of
#v("corpus.median_sessions") sessions over a median of #v("corpus.median_days") days, and
#v("corpus.multi_rig") were recorded on more than one rig (#fig(<fig:corpus>, "c")).
Breathing covers the range of the mouse: #v("rates.slow")% of breaths were at 2–4 Hz,
#v("rates.medium")% at 4–8 Hz and #v("rates.fast")% at 8–15 Hz (#fig(<fig:corpus>, "b")).

To measure how well methods generalize, we held out whole animals, and blocks of sessions
of training animals, from training (Methods, Data splits). We release these splits so that
our scores can be reproduced, but they are one use of the corpus, which can be divided
differently for other questions.

#figure(
  fit(fig1),
  caption: [*Measuring breathing from face video.* *a,* A head-fixed mouse runs on a wheel
    in front of the virtual-reality screens. The cameras are shown with example frames,
    together with the odor delivery line. *b,* Breathing during the task over one whole session, measured by the
    thermistor. Inhalations were detected on the whole recording with the same detector
    as the corpus clips. The top panels show breathing rate and the bottom panels running
    speed around odor onset, mean ± s.e.m. over sites. Sites are split by whether the
    animal stopped and was rewarded (n = #v("task.sites_rewarded")), stopped and was not
    (n = #v("task.sites_unrewarded")) or ran past without stopping
    (n = #v("task.sites_skipped")). The panel below shows breathing rate against running
    speed over every 60-Hz sample of the session, as the distribution of rates in each
    speed bin. *c,* One minute of a clip from the same session, an animal Zephyr never saw. The clip
    was predicted by the cross-validation fold that holds the animal out. Breathing rate was calculated as 1 over each
    breath's interval, from Zephyr (blue) over the thermistor (black). Running speed, licks (ticks) and reward deliveries (triangles), and odor sites
    are shaded by kind as in *b*. *d,* Zephyr network architecture (see methods). *e,* A representative eight seconds of a test animal. The face frame beside it shows the crop the network sees
    (orange). The predicted breathing is drawn over the thermistor, above the onset
    probability. Ticks mark the thermistor's inhalations (black) and exhalations (grey) and
    Zephyr's inhalations (blue).],
) <fig:overview>

#figure(
  fit(fig2),
  caption: [*Summary of the benchmarking corpus.* *a,* Selection process of the recorded sessions. Each step
    removes the sessions that failed a rule, labelled by the first rule they failed. The
    rules and thresholds are in @tab:qc. *b,* The distribution of breathing rate over
    thermistor breaths at 2–15 Hz. The black line shows the whole corpus
    (n = #v("rates.breaths") breaths) and grey lines each animal
    (n = #v("rates.animals")). *c,* The number of sessions per animal, coloured by the
    number of rigs the animal was recorded on (n = #v("corpus.animals") animals,
    #v("corpus.sessions") sessions).],
) <fig:corpus>

== Neural networks recover sniffing from video

We scored every method on held-out animals, never seen in training, in a five-fold
cross-validation in which each of #v("benchmark.animals_a") animals is held out once
(Methods, @tab:split). Scores were averaged within each animal on one rig before
comparison (#v("benchmark.units_a") animal × rig units; Methods, Statistics).

We first asked whether the signals that current tools extract from face video track
breathing. Motion energy, the mean absolute difference between frames, is the most common
summary of facial movement @stringer2019 @musall2019. Even with a sign and lag fitted on
the training clips, it was uncorrelated with the thermistor on new animals
($r = #v("benchmark.motion_r")$; #fig(<fig:benchmark>, "a,b")\; @tab:benchmark). So were
two other pixel methods that select the components of the crop with most power in the
breathing band ($r <= #v("benchmark.pixel_r_max")$ each), and all three detected few
inhalations (F1 #v("benchmark.pixel_f1")) #note[random-trace control]. A linear readout of
the leading facial motion components, as in Facemap @syeda2024, recovered part of the
signal ($r = #v("benchmark.facemap_r")$, inhalation F1 #v("benchmark.facemap_f1")).

We next asked whether networks trained against the thermistor do better. We started
from the state of the art in camera physiology: four networks developed to recover pulse
and breathing from human face video, TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022
and RhythmFormer @zou2025. We trained each on the same clips under the same protocol, reading
a 96 × 96 grayscale crop over the snout and predicting the thermistor trace (Methods).
These networks are built for long, slowly varying signals, and the two transformers are
large and slow to run. We therefore asked whether a smaller, cheaper architecture would
suffice. Zephyr encodes each frame with a small convolutional network and models breathing
over time with a temporal convolutional network. The comparison networks predict only the
breathing trace. Zephyr predicts the trace and, through a second output head, the
probability at each time step that an inhalation starts (the onset output;
#fig(<fig:overview>, "d")\; Methods).

Every network recovered breathing on new animals (#fig(<fig:benchmark>, "b")\;
@tab:benchmark\; an example in #fig(<fig:overview>, "e")), so breathing can be learned from
face video. Zephyr detected inhalations best, with an F1 score of #v("benchmark.f1") (95%
interval #v("benchmark.f1_ci")) within 25 ms, or 1.5 frames at 60 Hz. PhysNet, the closest,
reached #v("benchmark.physnet_f1") (paired difference #v("benchmark.physnet_f1_diff"),
#v("benchmark.physnet_f1_diff_ci")). Zephyr's trace correlation, the correlation of its
predicted trace with the thermistor, was #v("benchmark.r") (#v("benchmark.r_ci")), against
#v("benchmark.physnet_r") for PhysNet (difference #v("benchmark.physnet_r_diff"),
#v("benchmark.physnet_r_diff_ci")) and #v("benchmark.others_r") for TS-CAN, RhythmFormer and
PhysFormer. Zephyr matched the best of these networks while running faster. It predicted a
5-min clip in #v("benchmark.seconds") s on a consumer GPU, #v("benchmark.physnet_saved")%
less time than PhysNet (#v("benchmark.physnet_seconds") s) and about
#v("benchmark.transformer_times") times less than the two transformers
(#v("benchmark.transformer_seconds") s; #fig(<fig:benchmark>, "b")\; @tab:nets). Thus, neural networks can recover breathing waveforms accurately and efficiently from face video.

== Accurate and precise extraction of single sniffing events

Mice breathe over a wide range of rates, so we asked whether accuracy held across that
range (#fig(<fig:benchmark>, "c")). Zephyr found almost every inhalation between 6 and
12 Hz (F1 #v("rate.f1_middle")) and slightly fewer above 12 Hz (#v("rate.f1_fast")). It was
least accurate in slow breathing, at 2–3 Hz (#v("rate.f1_slow")). The other networks lost
more accuracy at both ends of the range.

The ability to recover inhalations events with precise timing is crucial for behavior and physiology studies. Over the new animals of all
folds, Zephyr's inhalations lagged the thermistor's by #v("rate.lag") ms on average (95%
interval over folds #v("rate.lag_ci") ms), well within one breath at the fastest sniffing
rates (#fig(<fig:benchmark>, "d")). The lag hardly changed with breathing rate, from
#v("rate.lag_slow") ms at 2–3 Hz to #v("rate.lag_fast") ms at 11–12 Hz.
We also compared breathing rate, counted in 10-s windows, with the thermistor's, using
the measures reported for camera-based breathing in humans @villarroel2019
@liu2023toolbox. Zephyr's rate was off by #v("rate.mae") breaths per minute on average, or
#v("rate.mape")% of the thermistor's rate, with almost no overall bias
(#v("rate.bias") breaths per minute). In #v("rate.within_10")% of windows, it was within
10% of the thermistor's rate (#fig(<fig:benchmark>, "e")).

#figure(
  fit(fig3),
  caption: [*Neural network methods faithfully recover sniffing dynamics.* Panel *a*
    shows one clip of a held-out animal in the first cross-validation fold. Panels *b–e* show the held-out animals of the five
    folds, #v("rate.clips") clips from #v("rate.sessions") sessions of #v("rate.animals")
    animals. *a,* Six seconds of one clip of a held-out animal, predicted by each method
    trained on that fold #note[PhysFormer and RhythmFormer: networks of the published
      split, which also holds this animal out; replace once retrained on the folds]. The
    thermistor is at the top and each method's trace is
    drawn over it in grey. Vertical
    lines mark the thermistor's inhalations. *b,* Inhalation F1 scores on new animals against
    the inference time for a 5-min clip from the preprocessed crops (median of 8 clips after warm-up).
    Each dot is one fold, the mean over its animal × rig units. The mean over the
    #v("benchmark.folds") folds is shown with its 95% interval. RhythmFormer
    and PhysFormer were run on the published split only. Filled symbols are networks on a
    consumer GPU and open symbols are methods on a CPU. *c,* Inhalation F1 on new animals
    by breathing rate for each network, mean over folds with its 95% interval. Colours
    are as in *b*. The bottom panel shows the share of breaths at each rate
    (n = #v("rate.breaths") thermistor breaths at 2–15 Hz over the folds;
    #v("rate.breaths_published") in the published split). *d,* Zephyr's onset lag, its
    inhalation time minus the thermistor's, against the breath's rate. Every thermistor
    inhalation with a prediction within 50 ms is included, pooled over the new animals of
    every fold (n = #v("rate.lags")). Colour shows the share of each rate bin's lags. The
    line shows each rate bin's mean lag, averaged over folds, and the band its 95%
    interval over the folds. Dotted lines mark the 25-ms tolerance. *e,* Breathing rate in 10-s windows, pooled over the new
    animals of every fold (n = #v("rate.windows") windows). The difference between
    Zephyr's rate and the thermistor's is plotted against their mean, coloured by the
    number of windows. The solid line is the bias and the dotted lines the 95% limits of
    agreement.],
) <fig:benchmark>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (new animals, over folds, seed 17).
  benchmark-table("tables/benchmark.csv"),
  caption: [*Method comparison on held-out animals.* Each score was averaged within each
    animal × rig unit, then over the new-animal units of a split, then over splits. Values
    are the mean with its 95% interval over the #v("benchmark.folds") cross-validation
    folds (#v("benchmark.units_a") units of #v("benchmark.animals_a") animals, each animal
    new in one fold). Methods run on one split show the published split alone
    (#v("benchmark.published_units") units of #v("benchmark.published_animals") animals),
    without an interval. Breathing rate is counted in 10-s windows. MAE and RMSE are in
    breaths per minute, MAPE is relative to the thermistor's rate, and r is the Pearson
    correlation of the windowed rates. Trace r is the correlation of the predicted
    breathing trace with the thermistor. F1 counts events within 25 ms. Inference is the
    time in seconds for a 5-min clip from the preprocessed crops on a consumer GPU, or on
    a CPU where marked (\*).],
) <tab:benchmark>

== Zephyr generalizes to unseen sessions and animals

We next asked whether the network generalizes differently to new sessions of seen animals
than to entirely new animals. Besides whole animals, each fold held out blocks of consecutive
sessions of its training animals (#v("benchmark.units_b") units of
#v("benchmark.animals_b") animals). Every network scored higher on these new sessions of
known animals than on new animals (#fig(<fig:general>, "a")\; every measure in
@tab:benchmark-b). Zephyr's inhalation F1 was #v("benchmark.f1_b") on held-out sessions against
#v("benchmark.f1") on held-out animals, and its trace correlation #v("benchmark.r_b") against
#v("benchmark.r"). The timing of inhalations thus carried over almost fully to new animals,
whereas the shape of the waveform carried over less well.

Among new animals, scores varied between the #v("benchmark.units_a") animal × rig units of
the five folds (#fig(<fig:general>, "b")). Two rigs of one animal differed about two-thirds
as much as two animals on one rig (@fig:rigs), so both the animal and the rig contribute. Zephyr therefore generalizes
to new recording days of an animal with little loss, and to new animals with a modest one
that falls mostly on the waveform rather than on inhalation timing. This may reflect inherent variability between animals, and variability between implants can also contribute to differences in the recorded signals that may affect the recovery of animal-specific waveform features.

== Timing errors rather than missed or hallucinated inhalations.

Most of the remaining errors were inhalations detected slightly too early or too late, not
inhalations lost or invented (#fig(<fig:general>, "d")). Relaxing the tolerance from 25 to
100 ms raised Zephyr's inhalation F1 on new animals from #v("tolerance.f1_25") to
#v("tolerance.f1_100") (mean over folds), and the errors left fell from
#v("tolerance.errors_25")% to #v("tolerance.errors_100")% of breaths
(#fig(<fig:general>, "c")).

We then asked whether a poor waveform means lost inhalations. For this and the analyses
of a single trained network below, we used the network trained on the first
cross-validation fold. Across its clips of held-out animals, trace correlation and
inhalation F1 were moderately related (Spearman $rho = #v("clips.spearman")$). In the
#v("clips.low_clips") clips with the lowest tenth of correlations
($r < #v("clips.low_r")$), median F1 fell to #v("clips.low_f1"), so a poor waveform usually
came with missed or false inhalations. It did not always: in the clip shown (trace
correlation #v("clips.quiet_r")), Zephyr found every inhalation over 2 s although its trace
departed in shape from the thermistor's ($r = #v("clips.quiet_stretch_r")$ over the 2 s;
#fig(<fig:general>, "e")). Clips with a poor waveform are therefore the ones to check,
which the quality-control measures below are meant to flag.

#figure(
  fit(fig4),
  caption: [*Generalization and its limits.* *a,* Scores of each method on held-out
    animals (test-A; n = #v("benchmark.units_a") animal × rig units) and on held-out sessions
    of training animals (test-B; n = #v("benchmark.units_b")). Each fold's score
    is the mean over its units. Points show the mean over the #v("benchmark.folds") folds
    with its 95% interval. Colours as in @fig:benchmark. *b,* Inhalation F1 of each held-out animal × rig unit of the five
    folds, ranked by Zephyr (n = #v("benchmark.units_a")). Each unit's score is the mean
    of its recordings. The inset ranks the #v("benchmark.animals_a") animals by their mean,
    and lines join the units of one animal on different rigs. *c,* Zephyr's
    inhalation F1 on new animals as the matching tolerance is relaxed up to 100 ms (top),
    and the misses (FN) and false inhalations (FP) left, as a share of thermistor breaths
    (bottom). Lines show the mean over the #v("benchmark.folds") folds with its 95%
    interval (n = #v("tolerance.breaths") thermistor breaths). Dotted lines mark 12.5 ms,
    25 ms (the tolerance used throughout), 50 ms and 100 ms. A prediction just outside the
    tolerance counts as both a miss and a false inhalation, so the two shares nearly
    coincide. *d,* An example of near misses. Among the first
    fold's clips of held-out animals with at least the median correlation, this clip has the most
    predictions 25–100 ms from a thermistor inhalation (r = #v("clips.near_r"),
    F1 #v("clips.near_f1")). The window shown holds its densest near misses. *e,* Inhalations found where the trace is poor. Among the
    #v("clips.low_clips") clips of held-out animals in the first fold in the lowest tenth of
    correlation, this is the 2-s stretch, with every inhalation found and at least 4 of
    them, where the onset probability at Zephyr's inhalations was lowest (clip
    r = #v("clips.quiet_r"); over the 2 s, r = #v("clips.quiet_stretch_r")). In *d* and *e*, crops are shown at the top. Below, Zephyr's
    trace is drawn over the thermistor with the thermistor's (black) and Zephyr's (blue)
    inhalations, misses (FN) and false inhalations (FP) marked, above the onset
    probability.],
) <fig:general>

== What it takes to use the network on new data

A lab that wants to measure sniffing from video needs to know how many animals it must
record with a sensor, and how far its camera may differ from ours. To estimate the first,
we retrained Zephyr on one cross-validation fold with nested subsets of
#v("size.animals") of its #v("size.all_animals") training animals, with all their sessions
(#v("size.sessions") of #v("size.all_sessions")), and scored each on the fold's held-out
animals (#fig(<fig:use>, "a")). While accuracy grew with the number of training animals, it did so modestly.
Inhalation F1 was #v("size.nested_f1") with #v("size.animals") animals, against
#v("size.f1") with all #v("size.all_animals") (#v("size.fewest") animals minus all: paired
difference #v("size.diff"), 95% interval #v("size.diff_ci")), and trace correlation fell
from #v("size.r") to #v("size.fewest_r"). The amount of video mattered less than the
number of animals: training on all #v("size.all_animals") animals but a quarter of their
sessions (#v("size.partial_sessions")) matched the full training set (F1
#v("size.partial_f1"), $r$ #v("size.partial_r")), and outscored 8 animals with the same
number of sessions.

To test if our network architecture can extract breathing signals from different views, we trained Zephyr the same way on the side and front
cameras, which film the whole animal from farther away and at different angles (@fig:views). Both recovered
breathing on new animals, though less accurately than the face camera. On the recording
spans all three cameras share, inhalation F1 was #v("cameras.f1_side") (side) and
#v("cameras.f1_front") (front) against #v("cameras.f1_face") (face), and trace correlation
#v("cameras.r_side") and #v("cameras.r_front") against #v("cameras.r_face")
(#fig(<fig:use>, "b")\; @fig:cameras, @tab:cameras).

Next, we tested if a lower frame rate would affect performance. Halving the frame
rate to 30 Hz changed inhalation F1 and trace correlation on new animals little
(@fig:input-rate).

#note[Fig. 5c, to run: adapting to a new view or rig. Fine-tune the face network on the
  side and front cameras with 1, 2, 4 and 8 labelled animals,
  against training from scratch; this gives the labelling cost of a new rig.]
The released application takes a lab's own videos through these steps: it places a crop
box on each recording, predicts breathing, and reports the quality-control measures
described below (Methods).

#figure(
  fit(fig5),
  caption: [*Training on new data.* *a,* Inhalation F1
    (top) and trace correlation (bottom) of Zephyr retrained on fewer animals, scored on
    the held-out animals of fold #v("size.fold"), against Zephyr trained on all of that fold's
    training animals. Filled points, joined by a line, are nested
    subsets of the training animals with all their sessions. The open point, drawn beside
    #v("size.all_animals"), uses all #v("size.all_animals") animals with a quarter of their
    sessions (#v("size.partial_sessions")). Points show the mean over the same
    #v("size.units") units with a 95% bootstrap interval. Each run used one seed. *b,* The
    same network trained on the face, side or front camera. Trace correlation and
    inhalation F1 are shown on held-out animals (A) and held-out sessions (B), on the recording
    spans all three cameras share. Points show the mean over animal × rig units with a 95%
    bootstrap interval, over the units of all #v("cameras.folds") folds (@fig:cameras). *c,* Adapting to a new view or rig with few labelled animals
    #note[to run].],
) <fig:use>

== Knowing when to trust a prediction

A method that replaces a sensor will be used where no sensor can check it, so its errors
must be predictable. Of the #v("errors.misses") missed and #v("errors.false") false
inhalations on the new animals of the five folds, #v("errors.misses_near")% and
#v("errors.false_near")% were near misses (mean over folds): a predicted and a thermistor
inhalation within 100 ms of each other that the 25-ms tolerance did not pair
(#fig(<fig:trust>, "a")). The remaining, gross errors were not concentrated where one might
expect. Fast breathing, above 8 Hz, accounted for #v("errors.fast_misses")% of gross misses
and no gross false inhalations, against #v("errors.fast_breaths")% of all breaths, and face
motion was no more common around errors than around breaths in general
(#v("errors.motion")% against 10%). Only shallow breaths were over-represented, at
#v("errors.shallow")% of gross errors against 25% of breaths (Methods).

We therefore asked whether the network's output alone can identify unreliable stretches of
a recording. We computed quality-control measures on each 30-s window of a prediction,
without the thermistor, and set their bounds on validation animals (Methods). On new
animals, which played no part in setting the bounds, the best measures ranked videos by
their true inhalation F1 (#fig(<fig:trust>, "b")), and windows that passed every check were
predicted more accurately than those that failed (#fig(<fig:trust>, "c")) #note[quality
  control is in progress: rerun the reports on the fold-cv_0 network at its own onset
  threshold and fill in the numbers]. The onset output is also calibrated: on validation
animals, the probability Zephyr gave a predicted inhalation matched the share that the
thermistor confirmed (#fig(<fig:trust>, "d")) #note[calibrate the fold-cv_0 network].
On a session without a thermistor, the quality-control measures mark which 30-s windows to
keep (#fig(<fig:trust>, "e")) #note[to draw].

#figure(
  fit(fig6),
  caption: [*When to trust a prediction.*
    *a,* Zephyr's errors on new animals (#v("errors.misses") misses, FN;
    #v("errors.false") false inhalations, FP). Bars show the mean over the
    #v("benchmark.folds") folds with its 95% interval. The first bars give the
    share of errors that are near misses, where a predicted and a thermistor inhalation lie
    within 100 ms. The other bars consider only the remaining errors (#v("errors.misses_gross")
    FN, #v("errors.false_gross") FP). They give the share in fast breathing (above 8 Hz),
    in face motion (the top 10% of the clip) and on shallow breaths (the bottom 25% of the
    clip). Grey ticks give the share of all #v("tolerance.breaths") thermistor breaths
    with each property. *b,* The quality-control measures that best rank new-animal
    videos by their true inhalation F1 #note[to redraw on the fold-cv_0 network]. *c,* Mean
    true inhalation F1 of the new-animal windows kept, keeping the best-judged first
    #note[to redraw]. *d,* Calibration of the onset output on the validation animals
    #note[to redraw]. *e,* Quality control of a session without a thermistor
    #note[to draw].],
) <fig:trust>

== Sniffing in experiments without a thermistor

#note[To do: sniffing aligned to behaviour, as in #fig(<fig:overview>, "b"), in task
  sessions that have no usable thermistor, over many more animals than the corpus.]

#figure(
  fit(fig7),
  caption: [*Sniffing measured from video alone.* Sniffing around odor onset in task
    sessions without a thermistor, by site outcome #note[to run].],
) <fig:application>

= Discussion

We provide a corpus that pairs orofacial video with an intranasal breathing sensor in
behaving mice, splits and scoring that test generalization to new animals, and a network,
Zephyr, that recovers each inhalation in animals it has never seen. Two results bear on how labs can use them.

First, in head-fixed mice, neural networks can recover the time of each inhalation from
video with high temporal accuracy. In animals it had never seen, Zephyr found inhalations
with an F1 of #v("benchmark.f1") at a 25-ms tolerance, its inhalations lagged the
thermistor's by #v("rate.lag") ms on average (95% interval #v("rate.lag_ci") ms), well
within one 16.7-ms frame at 60 Hz, and its breathing rate was within #v("rate.mape")% of the
thermistor's. This accuracy goes beyond the average breathing rate: it is enough to align
behaviour and neural activity to single sniffs, at the temporal resolution the camera
allows. The signals that current behavioural tools extract from facial motion did not
recover breathing, so a "sniffing" component of facial motion should not be read as
breathing without a sensor to check it.

Second, we provide a framework to train such networks and to predict with them
efficiently. Zephyr matched larger networks from camera physiology in accuracy and took
#v("benchmark.physnet_saved")% less time than PhysNet to predict. To our knowledge, no other dataset yet
exists on which to test generalization beyond our rigs, but within the corpus the method
worked from other views: trained on the side and front cameras, it found inhalations with
an F1 of #v("cameras.f1_side") and #v("cameras.f1_front"), against #v("cameras.f1_face")
for the face. Accuracy depended more on the number of training animals than on the amount
of video, and #v("size.fewest") animals already gave an F1 of #v("size.fewest_f1"), so a lab
should need only a handful of animals implanted with a sensor to train the network on its
own rig and view. A sensor remains necessary where the breathing waveform itself matters,
and in the slowest breathing, where Zephyr was least accurate.

Reading breathing from video also bears on animal welfare. A nasal implant requires
surgery on the snout, can obstruct airflow and damage the nasal epithelium, and can fail
before an experiment ends @mcafee2016 @grimaud2018, so that some animals are implanted
whose recordings are later discarded @weinreb2024; we too discarded sessions in which the
thermistor's signal had degraded. Once a network has been trained for a
lab's rig, only the few animals that train and check it need the implant, and the rest of
a study can measure breathing from the camera alone. This is a refinement for those
animals and, where implants would otherwise fail, a reduction in the number of animals
used @tannenbaum2015. Face videos already recorded in past experiments can also be read
for breathing without any new animals.

A lab with its own rig should expect to record a few animals with a sensor before relying
on video. The rigs in our corpus share one design and place the cameras in similar
positions, yet their lighting and the cameras' exact position, angle and lens still
differ (@fig:views), and the network scored lower on some rigs and animals than on others
(@fig:rigs). A camera placed at another angle, a different lens or lighting, or another
strain may move the face further from what the network has seen. The side and front cameras, which see the
snout from farther away and at other angles, still yielded breathing #note[quantify with
  Fig. 5c: the labelled animals a new view or rig needs]. The
quality-control measures and the calibrated onset output are meant for the step after:
once a lab has adapted the network, they flag the windows of each new recording that
should not be trusted, without a sensor.

The corpus also limits what we can claim. All mice were head-fixed, so the camera saw the
face from a fixed distance and angle, and the animal could not turn away, rear or groom
freely. Whether breathing can be read from video in freely moving mice remains untested.
Head-mounted cameras, which already film the eyes and face of freely moving mice during
natural behaviour and electrophysiology @meyer2018 @sattler2021 @michaiel2020 @parker2022,
keep the face at a fixed distance and angle, and are a natural setting in which to test
it. The
thermistor measures the temperature of the airflow, not the airflow itself, so its own
delay relative to nasal pressure bounds the timing error we can measure.

Breathing from animal video needs more paired recordings than any one lab can collect.
Camera physiology in humans advanced on shared datasets and benchmarks recorded across
groups @liu2023toolbox, and networks that excel on one dataset often fall on another. The
same is likely for mice: the corpus spans #v("corpus.rigs") rigs of one design and one
task. We release the corpus with the splits and scoring we used, in a format to which
other labs' recordings can be added, and we hope that groups who record breathing with a
sensor will also film the face, so that a benchmark spanning rigs, tasks, strains and
freely moving animals can be built.

Future methods may also need fewer labels. Thermistor recordings are costly to collect,
whereas video of the face is recorded in many experiments that measure no breathing.
Self-supervised methods, which in humans recover pulse from unlabelled face video by
exploiting the periodicity of the signal @gideon2021 @sun2022contrast, could learn from
such video and need a sensor only to check the result. Networks pre-trained on many rigs
could also become a starting point that each lab refines. In pose estimation, models
trained on data pooled from many laboratories already track animals in new settings with
little or no further labelling @ye2024superanimal. A breathing network pre-trained on a
corpus that spans rigs, views and tasks could likewise be fine-tuned on a lab's few
animals with a sensor rather than trained from scratch, and general-purpose visual
foundation models @oquab2023dinov2 may provide features from which breathing can be read
with fewer labels.

#set par.line(numbering: none)

= Data and code availability

We release the corpus (face, side and front video, frame timestamps and the thermistor,
on one clock) with the splits we used, the code, the trained weights, an evaluation that
reproduces the scores reported here, and an application that runs the network on new
recordings #note[repository URL, data DOI (DANDI or the Registry of
  Open Data), licence; confirm the corpus can be released; usage: `zephyr preprocess`,
  `zephyr evaluate`, `zephyr quality-control run`, `zephyr gui`].

#pagebreak()
#references(bibliography("references.bib"))

#pagebreak()
= Methods

== Data collection

_Surgery._ All procedures were approved by the Allen Institute Institutional Animal Care
and Use Committee #note[protocol number; mice: strain, sex, age at surgery, housing and
  light cycle; analgesia and post-operative care]. Thermistors were implanted over the
nasal epithelium as described @mcafee2016. Briefly, under isoflurane anaesthesia (1–5%), a
0.5-mm burr hole was thinned through the nasal bone #tbd[3.1] mm anterior to the
frontonasal suture and #tbd[0.5] mm right of the midline, and the thermistor tip was
lowered onto the epithelium without pressing on it. The hole was sealed with silicone
(Kwik-Cast), and the probe and a head bar were fixed with dental cement (Metabond).

_Data acquisition._ Data were acquired on the VrForaging platform #note[GitHub link],
whose devices are synchronized by the Harp protocol: a Behavior board #note[ref]
triggered the 240- or 360-Hz FLIR cameras, an olfactometer #note[ref] delivered odors and
a sniff-detector board (below) recorded the thermistor. Video was encoded in real time
with H.264 on the GPU (NVENC, constant quality 18) and re-encoded offline on the CPU
(libx264, CRF 18) #note[second-pass compression: purpose and size saving]\; clips were cut
from these files by stream copy.

== Sniff-detector board
TODO SONYA

== Corpus construction

Sessions were selected with the rules of @tab:qc, in order. Thermistor quality was judged
in 30-s windows after removing large artifacts (and #tbd[2] s around them), and clips were
cut only from stretches of at least 300 s in which every window passed. Detector settings
and thresholds were set by eye on #tbd[which sessions].

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, left, right),
    toprule,
    table.header([*Stage*], [*Rule*], [*Threshold*], [*Removed*]),
    midrule, [Window (30 s)], [Signal amplitude (5th–95th percentile range)], [≥ #tbd[100] raw units],
    [#tbd[2,038]], [], [Clipped samples], [#tbd[0]% at ≤ #tbd[20] or ≥ #tbd[4,090]],
    [—], [], [Breaths outside #tbd[2–15] Hz], [≤ #tbd[10]%],
    [—], [], [Repeated events (I→I or E→E)], [≤ #tbd[5]%],
    [—], [Session], [Repeated events over the session], [≤ #tbd[0.3]%],
    [#tbd[257]], [], [Longest run of good windows], [≥ #tbd[300] s],
    [#tbd[229]], [], [Event amplitude within expected range], [robust $z <= #tbd[3]$],
    [#tbd[1]], [], [Dead sensor (low SNR)], [modes ≥ #tbd[3] dB apart],
    [—], [Clips], [Room for two clips in good stretches], [#tbd[2] × #tbd[300] s],
    [#tbd[73]], [Cameras], [A camera missing], [],
    [#tbd[3]], [], [Timestamp alignment issues], [],
    [#tbd[13]], [], [Low exposure], [],
    [#tbd[6]], [Manual review], [], [],
    [#tbd[4]], midrule, [Front camera], [Not recorded in the session],
    [], [#v("optional_cameras.front_not_recorded")], [], [Not on the animal (manual review)],
    [], [#v("optional_cameras.front_reviewed")], bottomrule,
  ),
  caption: [*Corpus selection.* Rules are listed in the order they were applied,
    with the number of sessions each removed. Of #v("selection.recorded") recorded
    sessions, #v("selection.thermistor") passed the thermistor rules and
    #v("selection.clips") the clip and camera rules. Window rules mark 30-s windows as good
    or bad and remove no session on their own ("—"). The exception is amplitude, which is
    also applied to the session as a whole and removed #tbd[2,038] sessions. A session
    without a long enough run of good windows is removed. A repeated event is an
    inhalation followed by another inhalation, or an exhalation by another exhalation.
    Timestamps with a single one-frame rollback were repaired (#tbd[73]) rather than
    removed. Low exposure and unplugged sync cables were found after the sets were drawn,
    and those sessions were removed from them, leaving the #tbd[560] sessions of the
    corpus. The front camera is optional. A session without its video, because it was
    never recorded or pointed away from the animal, keeps its face and side clips, so the
    last two rows count front-camera videos left out, not sessions. The face and side
    cameras were recorded in every session #note[check the dead-sensor and window-rule
      counts].],
) <tab:qc>

_Thermistor events._ Inhalation and exhalation onsets were detected on the thermistor at
its own rate (250 Hz) with an adaptive peak detector. The signal was filtered (zero-phase
second-order Butterworth, high-pass #tbd[1] Hz, low-pass #tbd[40] Hz) and z-scored over
the recording. Maxima at least #tbd[50] ms apart were candidate inhalation onsets, and
minima candidate exhalation onsets. A candidate was kept when its prominence was at least
#tbd[0.2] of the local range of the signal (5th to 95th percentile over a centred
#tbd[1.5]-s window), so that a shallow breath next to deep ones is kept while small
ripples on a flat stretch are not. These events were used for training, quality control
and scoring.

== Data splits

Models were evaluated under two protocols. Test set A holds out whole animals and asks
whether a model works on an animal it has never seen; test set B holds out blocks of
consecutive sessions of training animals and asks whether it works on a new day of a known
animal. Validation animals, also held out whole, were used only for early stopping and
model selection, and test sessions only for final scoring. We drew five such splits, the
folds of a cross-validation in which each of the #v("benchmark.animals_a") animals outside the #tbd[359]-Hz
group is a test-A animal once (@tab:split); analyses of a single trained network use the
first fold. One more split drawn the same way, the published split, holds only the
comparisons not yet retrained on the folds (PhysFormer, RhythmFormer, the
2-animal draws and test set C). Held-out animals were
the best of many seeded draws covering the corpus's rigs, months and breathing rates, and
sessions per animal and rig were capped. Animals recorded at #tbd[359] Hz were split by
session; their remaining #tbd[5] sessions form test set C (@tab:testc). The cross-validation folds are described in Supplementary Note 1.

_Training-set size._ We retrained Zephyr on parts of the training set of fold
#v("size.fold") (#fig(<fig:use>, "a")): nested subsets of #v("size.animals") of its
#v("size.all_animals") training animals with all their sessions (#v("size.sessions")
sessions), and #v("size.partial_sessions") sessions drawn from all #v("size.all_animals")
animals. Three draws of 2 animals (#v("size.draw_sessions") sessions) were trained on the
published split only.

_Rig coverage._ The #tbd[4]- and #tbd[8]-animal runs trained on only some test-A rigs.
Test-A units on a rig seen in training scored F1 #tbd[0.94] and #tbd[0.96], against
#tbd[0.91] for units on other rigs (#tbd[10] against #tbd[11] units with #tbd[4] animals,
#tbd[15] against #tbd[6] with #tbd[8]). The groups hold different units, so the
comparison is not paired #note[one seed; repeat with the other seeds and the
  cross-validation folds].

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    toprule,
    table.header([*Set*], [*Held out as*], [*Animals*], [*Sessions*]),
    midrule, [Train], [—], [#tbd[31–33]],
    [#tbd[213–295]], [Validation], [whole animals], [#tbd[9]],
    [#tbd[67–144]], [Test A], [whole animals], [#tbd[9–10]],
    [#tbd[75–128]], [Test B], [sessions of training animals], [#tbd[26–30]],
    [#tbd[70–119]], [Test C], [#tbd[359]-Hz sessions], [#tbd[2]],
    [#tbd[5]], midrule, [Total], [],
    [#tbd[50]], [#tbd[560]], bottomrule,
  ),
  caption: [*Split sizes.* Values are the minimum and maximum over the #tbd[6]
    splits, the published split and the #tbd[5] cross-validation folds. In the folds,
    each of the #tbd[48] animals outside the #tbd[359]-Hz group is a test-A animal once.
    Validation includes #tbd[2] sessions of the #tbd[359]-Hz animals. Test B animals are
    also training animals, so animals do not add up across sets.],
) <tab:split>

== Preprocessing

Frames were downsampled to 360 × 270 pixels, cropped to a 96 × 96-pixel box over the
snout (one per session), converted to gray and selected on a 60-Hz grid by their
timestamps, and standardized over the training clips. The target was the thermistor
filtered as above but high-passed at #tbd[0.2] Hz, resampled to 60 Hz and z-scored per
clip.

== Network architecture

A convolutional encoder, whose first layer spans three frames, embeds each crop
(@tab:arch); embeddings are interpolated by their timestamps onto a uniform 60-Hz grid, and
a non-causal temporal convolutional network (receptive field #tbd[4.2] s) outputs the
breathing trace and an onset logit at each step (#tbd[0.86] million parameters).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    toprule,
    table.header([*Stage*], [*Layer*], [*Output*]),
    midrule, [Input], [gray crop, three consecutive frames],
    [3 × 96 × 96], [Encoder], [Conv3D 3×5×5, stride (1, 2, 2), BN, GELU],
    [32 × 48 × 48], [], [Conv 3×3, stride 2, BN, GELU],
    [64 × 24 × 24], [], [Conv 3×3, stride 2, BN, GELU],
    [96 × 12 × 12], [], [Conv 3×3, stride 2, BN, GELU],
    [128 × 6 × 6], [], [average pool to 2×2, flatten, linear, GELU],
    [128 per frame], [Resampling], [linear interpolation by frame timestamps],
    [128 at 60 Hz], [Temporal], [Conv 1×1],
    [128], [], [6 residual blocks of 2 × (Conv 3, dilation $d$, BN, GELU) and dropout; $d = 1, 2, …, 32$],
    [128], [Heads], [Conv 1×1 (trace), Conv 1×1 (onset logit)],
    [1 + 1], bottomrule,
  ),
  caption: [*Network architecture.* BN, batch normalization.],
) <tab:arch>

== Training

The loss was $1 - r$ between the predicted and target traces, averaged over three time
scales (the trace, and copies average-pooled by 4 and 16 samples), plus #tbd[0.5] times a
binary cross-entropy on onsets against Gaussians (s.d. #tbd[20] ms) centred on the
thermistor inhalations, with positives weighted by the inverse of their share in each
batch. Other settings are in @tab:train; the checkpoint with the highest mean
correlation on the validation clips was kept #note[seeds, GPU, time per run]. The
comparison networks were trained the same way, each with the loss of its original paper
(Supplementary Note 2).

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    toprule,
    table.header([*Setting*], [*Value*]),
    midrule, [Input],
    [gray, 96 × 96 crop, 60-Hz selection grid], [Window, batch],
    [#tbd[512] samples (#tbd[8.5] s), #tbd[16]], [Optimizer],
    [AdamW, learning rate #tbd[6 × 10#super[−4]], weight decay #tbd[10#super[−4]]], [Schedule],
    [#tbd[200]-step linear warm-up, cosine decay to 1%], [Regularization],
    [dropout #tbd[0.1] (temporal network), gradient clip #tbd[1], weight EMA #tbd[0.999]], [Loss],
    [multi-scale $1 - r$ (scales 1, 4, 16) + #tbd[0.5] × onset BCE], [Onset target],
    [Gaussians, s.d. #tbd[20] ms, per-batch positive weight], [Augmentation],
    [frame jitter ±0.25 frame, shift ±#tbd[4] px, zoom ±#tbd[10]%, brightness ±#tbd[15]%, contrast ±#tbd[20]%, noise #tbd[2] grey levels],
    [Selection],

    [mean correlation on validation animals, every #tbd[4] epochs], [Stopping],
    [patience #tbd[12] scorings, #tbd[40]–#tbd[200] epochs of #tbd[100] steps], [Precision],
    [bf16 mixed precision], bottomrule,
  ),
  caption: [*Training settings of Zephyr.*],
) <tab:train>

== Inference and event detection

Clips were predicted in overlapping windows of #tbd[1,024] samples, discarding #tbd[126]
samples (half the receptive field) at each edge, so that the stitched trace equals a
single pass over the clip. Inhalation onsets were the peaks of the onset probability above
a threshold, at least #tbd[50] ms apart. The threshold was chosen for each trained network
on its own validation animals, never on a test set: of #tbd[0.20–0.85] in steps of
#tbd[0.05], the one with the best mean inhalation F1 over the validation clips. It was
#v("threshold.folds") for the five folds' networks and #v("threshold.published") for the published split's. Exhalations, and the inhalations of methods without
an onset output ("DSP"), were found on the predicted trace by the detector used on the
thermistor, with the same settings (the 40-Hz low-pass does not apply at 60 Hz).

_Timing._ Inference time was the wall-clock time from preprocessed crops in memory to the
60-Hz trace, median over #tbd[8] 5-min clips after a warm-up, at batch size 1 in bf16 on an
NVIDIA RTX 4060 Ti (non-network methods on the CPU).

== Benchmarked methods

Every method read the same 96 × 96 gray crops on the 60-Hz grid and was scored by the same
rules (Scoring). The three pixel methods compute a trace from each clip's pixels with no
training, so the trace has no set polarity or timing relative to breathing. Each trace was
therefore band-passed to the breathing band (1–15 Hz) and turned the right way up from its
own shape: inhalation is shorter than exhalation, so a correct trace falls faster than it
rises. A single sign and lag (at most 0.5 s), the same for every clip, were then chosen to
maximise the mean correlation with the thermistor over the training clips, and applied
unchanged to the test clips.

The methods, their sources and what we changed are in @tab:methods. The four networks
were trained in our training loop on the losses of their own papers (Supplementary
Note 2); none has an onset output, so their inhalations were found on the predicted trace.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    toprule,
    table.header([*Method*], [*Source*], [*Details*]),
    midrule, [Pixel (motion)], [@stringer2019 @musall2019],
    [Motion energy: mean absolute
      difference between consecutive frames over the crop],
    [Pixel (PCA)],
    [@poh2010],

    [Of the top #tbd[10] principal components of the
      band-passed pixels, the one most concentrated in a single spectral peak],
    [Pixel (SNR)],
    [this work],

    [Band-passed pixels weighted by their share of variance in
      the breathing band, signed to agree, averaged],
    [Facemap],
    [@stringer2019 @syeda2024],

    [Top #tbd[100] principal components of the
      crop's motion energy and raw frames, at #tbd[11] lags (#tbd[±250] ms); ridge
      readout, strength chosen leaving one training recording out],
    [TS-CAN],
    [@liu2020],

    [One gray channel; output band-passed to 1–15 Hz], [PhysNet], [@yu2019],
    [One gray channel; 96 × 96 crop], [PhysFormer], [@yu2022, authors' code],
    [One gray channel; crop upsampled to 128
      pixels; rate spectrum over 1–15 Hz],
    [RhythmFormer],
    [@zou2025, authors' code],

    [One gray channel; crop upsampled to 128
      pixels; rate spectrum over 1–15 Hz],
    bottomrule,
  ),
  caption: [*Benchmarked methods.* For methods that are not networks, the details
    give their settings. For the networks, they give only what we changed; otherwise the
    networks follow their papers. All networks read one gray channel instead of colour,
    and a breathing band replaces the heart-rate band.],
) <tab:methods>

== Scoring

Predictions and thermistor were compared by time, over the span both cover. Traces were
scored by their Pearson correlation with the target. A predicted event matched a
thermistor event within 25 ms (#tbd[1.5] frames at 60 Hz), each event matching at most
once, and F1 was computed per clip. Breathing rate was counted in 10-s windows.

_Error analysis._ An error was a near miss when an error of the other kind lay within
100 ms (#fig(<fig:trust>, "a")). For the others we recorded the breathing rate, face motion
(mean absolute frame difference within 0.25 s) and breath size (thermistor peak-to-peak
within 0.25 s), the last two as quantiles over the clip's breaths.

== Statistics

The unit of analysis is an animal on one rig, whose sessions share the implant, camera
position and lighting; two rigs of one animal differ in score about two-thirds as much as
two animals on one rig (@fig:rigs). Scores were averaged over a unit's clips, then over a fold's units,
and reported as the mean over the #tbd[5] folds with a 95% t-interval; differences between
methods are paired by fold. Analyses of a single fold, or of the published split,
use a percentile bootstrap over units (#tbd[2,000] draws). F1 by rate, F1 against
tolerance and the error profile pool each fold's breaths before averaging over folds; the
onset lag and rate agreement pool all folds and carry no interval. #note[units of one animal are not independent of each other; check intervals
  with a bootstrap that resamples animals and then units within them]

#pagebreak()
#show: supplement
= Supplementary information

#let csv-table(path, columns, header, align: auto) = {
  let rows = csv(path).slice(1)
  table(
    columns: columns,
    align: if align == auto { (left,) + (right,) * (columns - 1) } else { align },
    toprule,
    table.header(..header.map(h => [*#h*])),
    midrule,
    ..rows.flatten(),
    bottomrule,
  )
}

== Supplementary Note 1: Cross-validation folds

The five folds were drawn over the #tbd[560] sessions of the corpus, beside the
published split and under the same rules (Methods, Data splits). The #tbd[48] animals
outside the #tbd[359]-Hz group were divided into five test-A groups, the most balanced of
#tbd[20,000] seeded draws judged by the worst group, and fold $k$ tests group $k$ whole,
so that each animal is a new animal once. Each fold draws #tbd[8] whole validation animals
from the animals it trains on. Every animal with at least #tbd[4] sessions is cut by date
into one block of sessions for each fold it trains in, and that fold tests the block as
test set B, so each such session is in test set B once. The #tbd[359]-Hz animals keep
their sets of the published split in every fold. Every set of every fold is within about
#tbd[2] percentage points of the corpus's share of time at each breathing rate, and every
fold passes the checks of the published split (no test or validation animal in training).
@tab:folds gives the sets of each fold; each fold's runs were scored on exactly its
test-A animals.

#figure(
  kind: table,
  csv-table("tables/cv_folds.csv", 6, ([Set], [cv_0], [cv_1], [cv_2], [cv_3], [cv_4])),
  caption: [*Cross-validation folds.* Each cell gives animals / sessions in one set
    of one fold. Test B animals also train, so animals do not add up across sets.],
) <tab:folds>

== Supplementary Note 2: Comparison networks

TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025 were
reimplemented from their papers and the authors' code, with their published layers, and
trained through the same training loop as Zephyr: the same gray 96 × 96 crops on the
60-Hz grid, optimizer, warm-up and cosine schedule, weight averaging, augmentation
(without the frame jitter of Zephyr) and early stopping on the validation animals.
Each was trained on the loss of its own paper. TS-CAN, whose input is resized to
#tbd[36] × #tbd[36] pixels, is trained with the mean squared error against the
standardized first difference of the target, and PhysNet with the negative Pearson
correlation. PhysFormer and RhythmFormer read the crops upsampled to the 128 pixels they
were built for and are trained with a negative Pearson term plus a cross-entropy over
each window's spectrum (one bin per breath per minute over 1–15 Hz) against the
thermistor's rate in that window: $0.1 r + b ("CE" + "KL")$, with $b$ rising from 1 to 5
over training, for PhysFormer, and $0.2 r + "CE"$ for RhythmFormer. Their attention spans
a whole window, so they were also run in windows of their training length. Window,
batch, steps per epoch and learning rate are in @tab:nets #note[confirm how the learning
  rates were chosen: on the validation animals, or from the papers (PhysFormer 10#super[−4],
  RhythmFormer 9 × 10#super[−3])].

#figure(
  kind: table,
  csv-table(
    "tables/comparison_nets.csv",
    9,
    ([Network], [Window], [Batch], [Steps], [LR], [Inference window], [Scales], [Params (M)], [GPU (s)]),
  ),
  caption: [*Networks.* Training and inference windows are in samples at 60 Hz.
    Batch size is the number of training windows per optimizer step. The table also gives
    steps per epoch, the peak learning rate, and the time scales of the correlation loss,
    used by Zephyr and PhysNet; the others use their own losses. The last columns give the
    number of parameters and the inference time for a 5-min clip from the preprocessed
    crops (median of #tbd[8] clips) on a consumer GPU (NVIDIA RTX 4060 Ti, bf16).],
) <tab:nets>

#figure(
  kind: table,
  csv-table("tables/test_c.csv", 4, ([Method], [Trace r], [Inhale F1], [Exhale F1])),
  caption: [*Test set C.* Scores on the #tbd[5] held-out sessions of the #tbd[2]
    animals recorded with the #tbd[359]-Hz camera (published split). Each score was
    averaged within each animal × rig unit, then over the #tbd[2] units, without an
    interval.],
) <tab:testc>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (test set B, over folds, seed 17).
  benchmark-table("tables/benchmark_b.csv"),
  caption: [*Benchmark on new sessions of known animals (test set B).* The table
    follows @tab:benchmark, on the held-out sessions of training animals. Each score was
    averaged within each animal × rig unit, then over the test-B units of a split, then
    over splits. Values are the mean with its 95% interval over the
    #v("benchmark.folds") cross-validation folds (#v("benchmark.units_b") units of
    #v("benchmark.animals_b") animals). Methods run on one split show the published split
    alone. Inference is as in @tab:benchmark.],
) <tab:benchmark-b>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (the folds every camera has, seed 17).
  csv-table(
    "tables/cameras.csv",
    7,
    ([Set], [Camera], [Units], [Trace r], [Inhale F1], [Exhale F1], [Rate MAE]),
  ),
  caption: [*Zephyr on each camera.* The same network was trained the same way on
    the face, side and front cameras. Each was scored on the recording spans all three
    cameras share, against the same thermistor events. Scores are on test set A (new
    animals, #v("cameras.clips_a") clips) and test set B (new sessions,
    #v("cameras.clips_b") clips) of the #v("cameras.folds") cross-validation folds. Each
    score was averaged within each animal × rig unit, then over units, and is shown with
    its 95% bootstrap interval over units. Rate MAE is the error in breathing rate over
    10-s windows, in breaths per minute. Each network detected inhalations at its own
    onset threshold, chosen on its validation animals.],
) <tab:cameras>

#pagebreak()
= Supplementary figures

#figure(
  fit(figs1),
  caption: [*Rig and animal as sources of score differences.* Zephyr on new
    animals, from test set A of the five folds, in which each animal is new once. A unit
    is an animal on one rig, with its score averaged over its clips. We compared pairs of
    units that share the animal but not the rig (#tbd[159] pairs, #tbd[22] animals
    recorded on more than one rig) with pairs that share the rig but not the animal
    (#tbd[520] pairs, #tbd[48] animals). Histograms show the share of each kind's pairs in
    each bin of the absolute difference between the two units, in inhalation F1 (left)
    and trace correlation (right), over 30 bins shared by both kinds. Points above give
    each kind's mean with a 95% bootstrap interval resampling animals (#tbd[2,000]
    draws).],
) <fig:rigs>

#figure(
  fit(figs2),
  caption: [*Zephyr on 30-Hz video.* The same network was trained the same way
    on every other frame (30 Hz), still predicting on the 60-Hz grid, and retrained on each
    of the five folds. *a,* The 30-Hz network minus the 60-Hz network on each test set, in
    trace correlation, inhalation F1 (onset output) and exhalation F1. Grey points are
    single folds, shown with their mean and its 95% t-interval over the #tbd[5] folds. *b,* Inhalation F1 on new animals (test set A) by breathing rate, with each
    fold's breaths pooled. Lines show the mean over folds with its 95% interval. The
    bottom panel shows the share of breaths in each bin.],
) <fig:input-rate>

#figure(
  fit(figs3),
  caption: [*Zephyr on the side and front cameras.* The same network was trained
    the same way on each camera's video, in each of the #v("cameras.folds")
    cross-validation folds. Each was scored on the recording spans all three cameras share, against the same
    thermistor events. Test set A holds #v("cameras.clips_a") clips from
    #v("cameras.units_a") units of #v("cameras.animals_a") animals, and test set B
    #v("cameras.clips_b") clips from #v("cameras.units_b") units. Every score is in
    @tab:cameras, and trace correlation and inhalation F1 on both sets are in
    #fig(<fig:use>, "b"). *a,* Inhalation F1 of each new-animal unit, ranked by the face camera, as in
    #fig(<fig:general>, "b"). *b,* Inhalation F1 on new animals by breathing rate
    (#v("cameras.breaths") thermistor breaths). The bottom panel shows the share of breaths
    in each bin. *c,* Inhalation F1 of each new-animal clip from the side and front
    cameras, against the face camera. The dotted line marks equality, and the legend gives
    the share of clips scored higher than by the face camera. Each network detected
    inhalations at its own onset threshold, chosen on its validation animals.],
) <fig:cameras>

#figure(
  fit(figs4),
  caption: [*What each camera sees.* Frames from a representative subset of
    #v("views.sessions") sessions, each labelled by its session name. For each session,
    the first frames of the face, side and front cameras are shown side by side. The crop
    box that the network on each camera reads is outlined. Boxes are placed once per
    recording (Methods). The cameras' angle, zoom and lighting, the implant and the
    animal's posture differ between rigs and days.],
) <fig:views>
