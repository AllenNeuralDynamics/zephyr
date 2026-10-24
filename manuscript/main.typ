// Manuscript draft, in the Allen Institute preprint template (the submodule template/:
// `git submodule update --init`). Build from the repository root:
//   typst compile --font-path manuscript/template/fonts manuscript/main.typ
// Numbers in yellow are provisional; red notes are for the authors and are removed
// before submission.
#import "template/preprint.typ": *
#import "figures/fig1.typ": body as fig1
#import "figures/fig2.typ": body as fig2
#import "figures/fig3.typ": body as fig3
#import "figures/fig4.typ": body as fig4
#import "figures/fig5.typ": body as fig5
#import "figures/figs1.typ": body as figs1
#import "figures/figs2.typ": body as figs2

// A figure assembled at its printed width (183 mm), scaled to the text width.
#let fit(body) = layout(size => scale(reflow: true, size.width / 189mm * 100%, body))

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
  title: [Inferring sniffing from orofacial video in mice],
  short-title: [Sniffing from orofacial video],
  authors: ((name: "VrForaging Platform", affiliations: ("aind",)),),
  affiliations: (
    aind: [Allen Institute for Neural Dynamics, Seattle, WA, USA #note[authors]],
  ),
  abstract: [
    Rodents rely on olfaction to survive. Odors guide them to find food, avoid predators and recognize mates and
    kin. Mice sample odors actively by sniffing, so the timing of each breath shapes odor perception. Ground-truth methods for measuring sniffing in mice typically rely on sensors implanted in the nasal cavity which require invasive surgery and
    often fail to yield a usable signal.
    Here we show that the breathing waveform and the
    timing of each inhalation can be inferred from ordinary video. We assembled a
    corpus of #tbd[560] curated sessions from #tbd[50] head-fixed mice that pairs face video with a
    synchronized intranasal thermistor. On animals never seen in training (#tbd[48] animals,
    each held out once in a five-fold cross-validation), our network detected inhalation
    onsets with an F1 score of #tbd[0.96] (within 25 ms of the real onset) and recovered the
    waveform with an average correlation of #tbd[0.81]. Video can thus provide
    breath-by-breath timing in head-fixed mice without a nasal implant.
  ],
)

= Introduction

Breathing is more than a vital function in rodents. Sniffing is a form of active sensing: a movement the animal makes to sample its
surroundings. Each inhalation draws odor into the nose, so sniffing sets when and how
olfactory information is sampled @kepecs2006 @verhagen2007 @wachowiak2011, and rodents
adjust it constantly. Mice breathe at about 2–4 Hz at rest and sniff at up to about 12-15 Hz
when they explore or anticipate an odor, switching between the two within a fraction of
a second @wesson2008. Human breathing, by contrast, stays near 0.2–0.3 Hz, so
a rodent's breathing rate varies over a range several times wider and changes far faster.
Responses of olfactory bulb neurons are locked to precise phases of the sniff cycle
@shusterman2011. The respiratory rhythm
originates in the brainstem @smith1991, and it also reports the animal's state. Breathing
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
thermistor recordings from #tbd[560] sessions in #tbd[50] head-fixed mice on different rigs, applied
the same quality rules to every session, and drew a split in which test and validation
animals are held out whole. On this benchmark we compared the signals that current tools
extract (motion energy and facial motion components with a linear readout) with networks
developed for human camera physiology and with our own network, all trained and scored
under one protocol. The simple signals failed, and every trained network recovered
breathing on new animals. Our network, zephyr, detected inhalation onsets with an F1 score of
#tbd[0.96] in #tbd[48] animals it had never seen, more accurately than any other method. Because a method meant
to replace a sensor will be used where there is no sensor, we also developed measures,
computed from the network's output alone, that flag unreliable predictions, and checked
them against held-out animals. We release the corpus, trained models and an evaluation that reproduces our scores.

= Results

== A corpus of face video paired with ground-truth sniffing recordings

Head-fixed mice performed an olfactory virtual-reality foraging task on a circular
treadmill: they ran along a virtual corridor of odor-cued sites and could stop at a site
to collect a probabilistic reward. Running, stopping and odor sampling alternate throughout a session, so breathing ranges from rest to fast sniffing (#fig(<fig:overview>, "b,c") and #fig(<fig:corpus>, "b")). While the mice performed the task, high-speed video was recorded from two angles (#fig(<fig:overview>, "a")). A snout-facing camera captured detailed movements of the nose and whiskers, while a side camera recorded the overall posture and locomotion. For all analysis in this study, we only used the face camera view, but we are including side-camera video for every session with the corpus for future work and benchmarks. All mice were additionally implanted with a small thermistor in the nasal cavity to record temperature changes as a proxy for breathing. The signal was conditioned and digitized using a custom, open-source, board #note[cite
  sniff-detector repo / RRID] at 250 Hz.
As inhalation draws cool air, temperature in the thermistor drops, and it rises again during exhalation as the subject expells warm air. This periodic signal reflects the breathing cycle that can be readily extracted with simple digital-signal processing (DSP) peak detection algorithms (#fig(<fig:overview>, "e")\; Methods).

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

We scored every method on two test sets: new animals, held out whole (test set A), and
new sessions of training animals (test set B). Scores come from a five-fold
cross-validation in which each of the #tbd[48] animals outside the #tbd[359]-Hz group is a
new animal once (Methods, @tab:split), and are compared between animals on one rig
(animal × rig units; Methods, Statistics): over the folds, test set A holds #tbd[107]
units from #tbd[48] animals, and test set B #tbd[86] from #tbd[41]. Analyses of a single
trained network (the training-set size, quality control and example clips) use the
published split, a separate draw made the same way.

#figure(
  fit(fig1),
  caption: [*Measuring breathing from face video.* *a,* The rig: a head-fixed mouse on a wheel facing the virtual-reality screens, the
    cameras with example frames, and the odor delivery line. *b,* Breathing during the
    task, in one session of an animal the network never saw (both 300-s clips, predicted
    by the cross-validation fold that holds the animal out): breathing rate (top) and
    running speed (bottom) around odor onset at sites where the animal stopped and was
    rewarded (n = #tbd[28]), stopped and was not (n = #tbd[38]) or ran past without
    stopping (n = #tbd[14]), mean ± s.e.m. over sites, the rate from the network's
    inhalations (coloured) and from the thermistor's (dotted); below, the network's
    breathing rate against running speed over every 60-Hz sample (each speed bin's
    distribution of rates; rate bins hold whole numbers of output frames). *c,* A
    minute of the same session: running speed, and breathing rate (1 / each breath's
    interval) from the network (blue) over the thermistor's (black), with licks (ticks),
    reward deliveries (triangles) and odor sites shaded by kind as in *b*. *d,* Network. *e,* Eight seconds of one
    test-A clip as the animal starts sniffing, beside the face frame with the crop the
    network sees (orange): the predicted breathing over the thermistor, and the onset
    probability, with the thermistor's inhalations (black) and exhalations (grey) and
    the network's inhalations (blue).],
) <fig:overview>

#figure(
  fit(fig2),
  caption: [*The corpus.* *a,* Selection: the recorded sessions, step by step, to the
    corpus; each loss is labelled with the first rule the session failed (rules and
    thresholds in @tab:qc). *b,* Distribution of breathing rate (thermistor breaths,
    2–15 Hz) over the corpus (black; n = #tbd[1,614,211] breaths) and for each animal (grey;
    n = #tbd[50]). *c,* Sessions per animal, by the number of rigs the animal was recorded
    on (n = #tbd[50] animals, #tbd[560] sessions).],
) <fig:corpus>

== Simple video signals do not recover breathing; trained networks do

We first asked whether the signals that current tools extract from face video carry
breathing. Motion energy, the mean absolute difference between frames, is the most common
summary of facial movement @stringer2019 @musall2019. Fitted with a global sign and lag on
the training clips, it was uncorrelated with the thermistor on new animals
($r = #tbd[0.03]$), as were two other pixel methods that select the components of the crop
with most power in the breathing band ($r <= #tbd[0.01]$ each;
#fig(<fig:benchmark>, "a,b")\; @tab:benchmark), and their inhalation F1 scores were low
(#tbd[0.26–0.32]) #note[random-trace control].
A linear readout of the leading facial motion components, as in Facemap @syeda2024, did
better, but recovered only part of the signal ($r = #tbd[0.48]$, inhalation F1
#tbd[0.51]).

We then trained networks to predict the thermistor trace from a 96 × 96 grayscale crop
over the snout. Ours encodes each frame and models breathing over time with a temporal
convolutional network; it outputs the breathing trace and the probability of an
inhalation onset at each time step (#fig(<fig:overview>, "d")\; Methods). For comparison
we trained four networks developed for physiology from human face video, TS-CAN @liu2020,
PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025, on the same clips under the
same protocol (Methods).

Every network recovered breathing on new animals (#fig(<fig:benchmark>, "b")\;
@tab:benchmark\; an example in #fig(<fig:overview>, "e")). Ours
detected inhalation onsets best on new animals, at F1 #tbd[0.96] (95% interval
#tbd[0.94–0.97]) within 25 ms (#tbd[1.5] frames at 60 Hz), against #tbd[0.94] for PhysNet
(paired difference #tbd[0.017], #tbd[0.002–0.031]), and recovered the waveform with a
correlation of #tbd[0.81] (#tbd[0.78–0.84]), against #tbd[0.79] for PhysNet (difference
#tbd[0.02], #tbd[0.00–0.04]); TS-CAN, RhythmFormer and PhysFormer reached correlations of
#tbd[0.64–0.77]. On the published split's ten test animals alone the two correlations
were tied (#tbd[0.79] against #tbd[0.80]). Our network counted breathing rate at least as well as
PhysNet (@tab:benchmark). It also infers faster than PhysNet:
a 5-min clip took #tbd[0.87] s on a consumer GPU (RTX 4060 Ti), against #tbd[1.58] s for
PhysNet and #tbd[5.8–6.0] s for the two transformers (#fig(<fig:benchmark>, "b"); @tab:nets). The onset output contributed modestly: detecting inhalations as
peaks of the predicted trace gave F1 #tbd[0.94], against #tbd[0.96] from the onset
probability. Exhalations, detected on the trace, were found less reliably (F1 #tbd[0.75]).

Accuracy depended on breathing rate (#fig(<fig:benchmark>, "c")). Our network was
least accurate on slow breathing (F1 #tbd[0.89] at 2–3 Hz) and near perfect from 6 to
12 Hz (F1 #tbd[0.99]), dropping only slightly above 12 Hz (#tbd[0.95]). PhysNet and
RhythmFormer matched it from 8 to 12 Hz but fell further at both ends (PhysNet
#tbd[0.86] at 2–3 Hz and #tbd[0.93] above 12 Hz), and PhysFormer reached only #tbd[0.51]
at 2–3 Hz.

Inhalation timing was precise. Over the new animals of all folds, matched onsets lagged
the thermistor by a median of #tbd[1.7] ms (interquartile range #tbd[−6.2 to 8.7] ms),
well within one breath at the fastest sniffing rates, and the lag hardly changed with
breathing rate, from a median of #tbd[+2.5] ms at 2–3 Hz to #tbd[+0.5] ms at 11–12 Hz
(#fig(<fig:benchmark>, "d")). Breathing rate, counted in 10-s windows, agreed with
the thermistor with a mean absolute error of #tbd[4.1] breaths per minute (over all windows;
#tbd[4.4] averaged within animal × rig units, @tab:benchmark) and a bias of
#tbd[+0.2] (95% limits of agreement #tbd[−16.0] to #tbd[+16.4]\; #fig(<fig:benchmark>, "e")), the measures
used for camera respiration in humans @villarroel2019. @tab:benchmark gives these rate
measures for every method, as reported for camera physiology in humans @liu2023toolbox.

#figure(
  fit(fig3),
  caption: [*Breathing from face video on animals held out from training.* *a,* Six
    seconds of one test-A clip (the clip of #fig(<fig:overview>, "e")): the thermistor (top) and the
    trace of each method over it (grey), with each method's correlation on the whole
    clip; vertical lines, thermistor inhalations. *b,* Inhalation F1 on new animals
    against inference time per 5-min clip from the preprocessed crops (median of 8 clips):
    one dot per fold (mean over its animal × rig units), and the mean over the #tbd[5]
    folds with its 95% interval; RhythmFormer and PhysFormer, the published split only;
    filled, networks on a consumer GPU; open, methods on a CPU. *c,* Inhalation F1 on new
    animals by breathing rate for each network, mean over folds with its 95% interval (top;
    colours as in *b*), and the share of breaths at each rate (bottom); n =
    #tbd[1,503,469] thermistor breaths at 2–15 Hz over the folds (published split,
    #tbd[221,626]). *d,* Our network's onset lag (predicted minus thermistor; every
    thermistor inhalation with a prediction within 50 ms) against the breath's rate, every
    fold's new animals pooled: the share of each rate bin's lags (colour), with the median
    (solid) and interquartile range (dashed); dotted lines, the 25-ms tolerance; n =
    #tbd[1,505,437] inhalations. *e,* Breathing rate in 10-s windows, every fold's new
    animals pooled: our network minus the thermistor against their mean (colour, number of
    windows), with the bias (solid) and 95% limits of agreement (dotted); n =
    #tbd[31,136] windows. *a,* test set A of the published split; *b–e,* test set A of
    the five folds, #tbd[1,038] clips from #tbd[519] sessions of #tbd[48] animals.],
) <fig:benchmark>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (new animals, over folds, seed 17).
  benchmark-table("tables/benchmark.csv"),
  caption: [*Benchmark on new animals.* Each score averaged within each animal × rig
    unit, then over the new-animal units of a split, then over splits: mean with its 95%
    interval over the #tbd[5] cross-validation folds (#tbd[107] units of #tbd[48] animals,
    each animal new in one fold), or, for the methods with one split, the published
    split alone (#tbd[21] units of #tbd[10] animals; no interval). Breathing rate is counted in 10-s windows;
    MAE and RMSE in breaths per minute, MAPE relative to the thermistor's rate, and r the
    Pearson correlation of the windowed rates. Trace r, correlation of the predicted
    breathing trace with the thermistor; F1 within 25 ms. Inference, seconds per 5-min
    clip from the preprocessed crops on a consumer GPU (\*, on a CPU). #tbd[Seed 17.]],
) <tab:benchmark>

== The method generalizes to new sessions and subjects

The two protocols separate two kinds of generalization. For every network, scores on new sessions
of known animals (test set B) were higher than on new animals (test set A): our network's
correlation was #tbd[0.91] on B against #tbd[0.81] on A, and PhysNet's #tbd[0.89] against
#tbd[0.79] (#fig(<fig:general>, "a")\; every measure in @tab:benchmark-b). A new recording day of a known animal costs
little; most of the error comes from animals the network has never seen. Scores varied between the #tbd[107] animal × rig units of the five folds
(#fig(<fig:general>, "b")) #note[range of per-unit F1; which units are worst and why]. In every fold, our network scored at least as high as PhysNet in inhalation F1 and correlation (in one fold the two were tied; #fig(<fig:benchmark>, "b")).


To ask how much data is necessary to generalize to new held-out animals, we retrained our
network on nested subsets of #tbd[4], #tbd[8] and #tbd[16] of the #tbd[32] training
animals, with all their sessions (#tbd[35], #tbd[74] and #tbd[148] of #tbd[295]), and on
three draws of #tbd[2] animals (#tbd[18–19] sessions), and scored each on the same test sets
(#fig(<fig:general>, "c")). On new animals (test set A), scores barely depended on the
number of training animals: inhalation F1 was #tbd[0.93] with #tbd[4] animals and
#tbd[0.94] with #tbd[32] (paired difference #tbd[0.01], 95% interval #tbd[−0.01 to
  0.04]), and the waveform correlation #tbd[0.77] against #tbd[0.79]. Keeping all #tbd[32]
animals but a quarter of their sessions (#tbd[74]) gave much the same scores (F1
#tbd[0.92], $r$ #tbd[0.77]). Below four animals the network depended on which animals
it saw: the three 2-animal draws reached F1 #tbd[0.92], #tbd[0.85] and #tbd[0.71]. A
network trained on a few animals thus generalizes to new ones about as well as one
trained on all 32, but not one trained on only two #note[one seed per size; repeat with
  the other seeds, and with the cross-validation folds (full-gray-data-size-cv4-v3)].

To see where the
network fails, we examined every missed (false negative, FN) and every false inhalation (false positive, FP) on new animals
(#fig(<fig:general>, "d")\; their causes are in #fig(<fig:qc>, "a")). Most of the failures, about three quarters, were small differences between the predicted and true inhalation times rather than gross misdetections (#fig(<fig:general>, "d")). Relaxing the tolerance from 25 to 100 ms raised inhalation F1 on
new animals from #tbd[0.96] to #tbd[0.99] (mean over folds) and cut the errors left from
#tbd[8.6%] to #tbd[1.9%] of breaths; halving it to 12.5 ms lowered F1 to #tbd[0.72]
(#fig(<fig:general>, "e")).

A low waveform correlation did not mean that the breaths were lost. Across the test-A
clips of the published split,
correlation and inhalation F1 were only loosely related (Spearman $rho = #tbd[0.41]$):
in the #tbd[16] clips with the lowest tenth of correlations ($r < #tbd[0.61]$), median F1
was still #tbd[0.76], and a quarter reached #tbd[0.89] or more. In the clip shown
(correlation #tbd[0.51]), every inhalation over the 2 s shown was found although the
onset output peaked at only #tbd[0.80–0.91] (median #tbd[0.87], against a threshold of
#tbd[0.75]) and the trace departed from the thermistor in shape ($r = #tbd[0.51]$ over the
2 s; #fig(<fig:general>, "f")). Onset times can thus be used even where the waveform is
poor and the onset output is far from certain.

#figure(
  fit(fig4),
  caption: [*Generalization and its limits.* *a,* Per method, on new animals (test set
    A; n = #tbd[107] animal × rig units) and new sessions of known animals (test set B;
    n = #tbd[86]): mean over the #tbd[5] folds with its 95% interval (each fold, the mean
    over its units). *b,* Inhalation F1 of each test-A animal × rig unit of the five folds
    (mean of its recordings; n = #tbd[107]), ranked by our network; inset, the same units
    by animal (#tbd[48] animals ranked by their mean), the units of one animal (its rigs)
    joined. *c,* Inhalation F1 (top) and
    correlation (bottom) of our network retrained on nested subsets of the training
    animals, with all their sessions (filled, joined) or with all #tbd[32] animals and a
    quarter of their sessions (#tbd[74]\; open, drawn beside #tbd[32]), on new animals
    (test set A of the published split): mean over the same #tbd[21] units with 95%
    bootstrap interval; at
    #tbd[2] animals, three draws of the animals (dots; the line joins their mean); one
    seed per run. *d,* Near misses: among the published split's test-A clips with a
    correlation at least the median,
    the one with the most predictions 25–100 ms from a thermistor inhalation (r =
    #tbd[0.90], F1 #tbd[0.95]), around its densest near misses. *e,* Our network's inhalation F1 on
    new animals as the matching tolerance is relaxed (pairs within 100 ms; top) and the
    misses (FN) and false inhalations (FP) left, as a share of thermistor breaths
    (bottom), mean over the #tbd[5] folds with its 95% interval; F1 and errors left at
    12.5 ms, 25 ms (the tolerance used throughout), 50 ms (dotted lines) and 100 ms. A
    prediction just outside the tolerance counts as both a miss and a false inhalation,
    so the two shares nearly coincide; n = #tbd[1,519,650] thermistor breaths over the
    folds. *f,* Inhalations found from an uncertain onset output: among the
    #tbd[16] test-A clips of the published split in the lowest tenth of correlation, the
    2-s stretch in which every inhalation is found (no error or excluded span, at least
    4 inhalations) with the lowest median onset probability at the network's events
    (clip r = #tbd[0.51]; over the 2 s, r = #tbd[0.51] and onset peaks #tbd[0.80–0.91]). In *d* and *f* (one clip each):
    crops (top), the thermistor and our trace with the thermistor's (black) and the
    network's (blue) inhalations, misses (FN) and false inhalations (FP) marked, and the
    onset probability.],
) <fig:general>

== Known failure modes and limitations

We first asked what our network's remaining errors are. Of the #tbd[67,027] missed and
#tbd[68,233] false inhalations on the new animals of the five folds, #tbd[79]% and
#tbd[77]% were near misses (mean over folds): a prediction and a thermistor inhalation
within 100 ms of each other that the 25-ms tolerance did not pair (#fig(<fig:qc>, "a")).
The other, gross errors did not gather where one might expect. Fast breathing, above
8 Hz, accounted for #tbd[9]% of gross misses and no gross false inhalations, against
#tbd[20]% of all breaths; face motion was no more common around errors than around
breaths in general (#tbd[9–10]% against #tbd[10]%). Only shallow breaths were
over-represented, at #tbd[36–37]% of gross errors against #tbd[25]% of breaths (Methods).

A method that replaces a sensor will be used where there is no sensor to check it. We
therefore computed measures from the network's output alone, which judge each 30-s
window of a prediction without the thermistor; their bounds were set on validation
animals (Methods). On new animals (test set A), which played no part in setting them,
the best measures ranked the published split's #tbd[156] videos by their true inhalation F1 with Spearman
correlations of #tbd[0.63–0.80] (#fig(<fig:qc>, "b")). The #tbd[88]% of windows that
passed every check had a mean inhalation F1 of #tbd[0.96], against #tbd[0.81] for the
windows that failed and #tbd[0.95] for all (#fig(<fig:qc>, "c")). #note[these quality-control reports were made at the 0.5 onset threshold; rerun them at each checkpoint's own] The onset output is
also calibrated: on validation animals, the probability the network gives a predicted
inhalation matched the share that were real (expected calibration error #tbd[0.002]\;
#fig(<fig:qc>, "d")), so each inhalation can carry its own confidence.

Finally, the network's accuracy did not hinge on how it was trained, but it did depend
on the crop it sees (#fig(<fig:qc>, "e")). Training with more augmentation, or with every
breathing-rate bin weighted equally, changed inhalation F1 and the waveform correlation on
test set A of the published split by less than #tbd[0.01] (paired intervals including
zero). Doubling the crop to 192 × 192 pixels lowered both: inhalation F1 by #tbd[0.055]
(95% interval #tbd[0.029–0.083]) and the correlation by #tbd[0.078] (#tbd[0.026–0.122])
#note[one seed per variant].

#note[Application, if one is added: run on sessions or animals with no usable thermistor
  and show one analysis that needs inhalation times.]

#figure(
  fit(fig5),
  caption: [*When to trust a prediction.* *a,* Our network's errors on new animals, mean
    over the #tbd[5] folds with its 95% interval (#tbd[67,027] misses, FN; #tbd[68,233] false
    inhalations, FP): the share that are near misses (a prediction and a thermistor
    inhalation within 100 ms), and, among the other errors (#tbd[14,089] FN, #tbd[15,137]
    FP), the share in fast breathing (> 8 Hz), in face motion (top 10% of the clip) and on
    shallow breaths (bottom 25% of the clip); grey ticks, the share of all
    #tbd[1,519,650] thermistor breaths with each property. *b–e,* the published split.
    *b,*
    Spearman correlation with the true inhalation F1 of the five measures that rank the
    #tbd[156] test-A videos best, signed so that a positive value means the measure ranks
    predictions correctly; colours, families of measures. *c,* Mean true inhalation F1 of
    the test-A windows kept (#tbd[1,560] 30-s windows), best-judged first, for the three
    measures that rank windows best and for the truth itself (oracle); star, the windows
    that pass every check; dotted, all windows. *d,* Calibration of the onset output on the
    validation animals: the predicted probability that an inhalation is real against the
    share matched to a thermistor inhalation. *e,* Our network trained differently, minus
    our network, on test set A: inhalation F1 and correlation, mean over the #tbd[21]
    units with a paired 95% bootstrap interval; one seed each.],
) <fig:qc>

#set par.line(numbering: none)

= Data and code availability

We release the code, trained weights, the corpus (face and side video, thermistor and
timestamps) with its fixed split, and an evaluation that reproduces the scores reported
here #note[repository URL, data DOI, licence; confirm the corpus can be released; usage:
  `zephyr preprocess`, `zephyr evaluate`, `zephyr quality-control run`].

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
and thresholds were set by eye on the electrophysiology sessions.

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
    [#tbd[4]], bottomrule,
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
folds of a cross-validation in which each of the #tbd[48] animals outside the #tbd[359]-Hz
group is a test-A animal once, and one more, the published split, used for analyses of a
single trained network (@tab:split). Held-out animals were
the best of many seeded draws covering the corpus's rigs, months and breathing rates, and
sessions per animal and rig were capped. Animals recorded at #tbd[359] Hz were split by
session; their remaining #tbd[5] sessions form test set C (@tab:testc). The cross-validation folds are described in Supplementary Note 1.

_Training-set size._ We retrained our network on parts of the published split's
training set (#fig(<fig:general>, "c")): nested subsets of #tbd[4], #tbd[8] and #tbd[16] of the #tbd[32] training animals with all
their sessions (#tbd[35], #tbd[74] and #tbd[148] sessions), three draws of #tbd[2]
animals (#tbd[18–19] sessions), and #tbd[74] sessions drawn from all #tbd[32] animals.

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
  caption: [*Split sizes.* Minimum and maximum over the #tbd[6] splits: the published
    split and the #tbd[5] cross-validation folds, in which each of the #tbd[48] animals
    outside the #tbd[359]-Hz group is a test-A animal once. Validation includes #tbd[2]
    sessions of the #tbd[359]-Hz animals; test B animals are training animals, so animals
    do not add up across sets.],
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
  caption: [*Training settings* of the network reported in the main text.],
) <tab:train>

== Inference and event detection

Clips were predicted in overlapping windows of #tbd[1,024] samples, discarding #tbd[126]
samples (half the receptive field) at each edge, so that the stitched trace equals a
single pass over the clip. Inhalation onsets were the peaks of the onset probability above
a threshold, at least #tbd[50] ms apart. The threshold was chosen for each trained network
on its own validation animals, never on a test set: of #tbd[0.20–0.85] in steps of
#tbd[0.05], the one with the best mean inhalation F1 over the validation clips. It was
#tbd[0.70–0.75] for the five folds' networks and #tbd[0.75] for the published split's. Exhalations, and the inhalations of methods without
an onset output ("DSP"), were found on the predicted trace by the detector used on the
thermistor, with the same settings (the 40-Hz low-pass does not apply at 60 Hz).

_Timing._ Inference time was the wall-clock time from preprocessed crops in memory to the
60-Hz trace, median over #tbd[8] 5-min clips after a warm-up, at batch size 1 in bf16 on an
NVIDIA RTX 4060 Ti (non-network methods on the CPU).

== Comparison methods

Motion energy and two pixel methods (the crop's first principal component, and a pixel
average weighted by power in the breathing band) were computed from the same crops, with
one sign and lag fitted on the training clips. The Facemap-style readout was a ridge
regression from lagged principal components of the crop's movie and motion energy
@syeda2024. TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer
@zou2025 were reimplemented from their papers (Supplementary Note 2).

== Scoring

Predictions and thermistor were compared by time, over the span both cover. Traces were
scored by their Pearson correlation with the target. A predicted event matched a
thermistor event within 25 ms (#tbd[1.5] frames at 60 Hz), each event matching at most
once, and F1 was computed per clip. Breathing rate was counted in 10-s windows.

_Error analysis._ An error was a near miss when an error of the other kind lay within
100 ms (#fig(<fig:qc>, "a")). For the others we recorded the breathing rate, face motion
(mean absolute frame difference within 0.25 s) and breath size (thermistor peak-to-peak
within 0.25 s), the last two as quantiles over the clip's breaths.

_Training variants._ Each variant changed one setting of @tab:train
(#fig(<fig:qc>, "e")): more augmentation (shifts of up to 8 pixels, zoom 0.15, rotation
up to 10° and blur), every breathing-rate bin weighted equally in each batch, or
192 × 192-pixel crops from frames downsampled to 720 × 540 (batch 4, 400 steps per epoch,
learning rate 3 × 10#super[−4]).

== Statistics

The unit of analysis is an animal on one rig, whose sessions share the implant, camera
position and lighting; two rigs of one animal differ in score about two-thirds as much as
two animals on one rig (@fig:rigs). Scores were averaged over a unit's clips, then over a fold's units,
and reported as the mean over the #tbd[5] folds with a 95% t-interval; differences between
methods are paired by fold. Analyses of the published split alone (#tbd[21] test-A units)
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
  caption: [*Cross-validation folds.* Animals / sessions in each set of each fold. Test B
    animals also train, so animals do not add up across sets.],
) <tab:folds>

== Supplementary Note 2: Comparison networks

TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025 were
reimplemented from their papers and the authors' code, with their published layers, and
trained through the same training loop as our network: the same gray 96 × 96 crops on the
60-Hz grid, optimizer, warm-up and cosine schedule, weight averaging, augmentation
(without the frame jitter of our network) and early stopping on the validation animals.
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
  caption: [*Networks.* Training window and inference window (samples at 60 Hz), batch
    size (training windows per optimizer step), steps per epoch, peak learning rate, the time scales of the correlation loss
    (our network and PhysNet; the others use their own losses), the number of parameters,
    and the inference time per 5-min clip from the preprocessed crops (median of #tbd[8]
    clips) on a consumer GPU (NVIDIA RTX 4060 Ti, bf16).],
) <tab:nets>

#figure(
  kind: table,
  csv-table("tables/test_c.csv", 4, ([Method], [Trace r], [Inhale F1], [Exhale F1])),
  caption: [*Test set C.* The #tbd[5] sessions of the #tbd[2] animals recorded with the
    #tbd[359]-Hz camera that were held out (published split): each score averaged within
    each animal × rig unit, then over the #tbd[2] units; no interval.],
) <tab:testc>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (test set B, over folds, seed 17).
  benchmark-table("tables/benchmark_b.csv"),
  caption: [*Benchmark on new sessions of known animals (test set B).* As @tab:benchmark,
    on the held-out sessions of training animals: each score averaged within each animal
    × rig unit, then over the test-B units of a split, then over splits: mean with its
    95% interval over the #tbd[5] cross-validation folds (#tbd[86] units of #tbd[41]
    animals), or, for the methods with one split, the published split alone. Inference
    as in @tab:benchmark.],
) <tab:benchmark-b>

#pagebreak()
= Supplementary figures

#figure(
  fit(figs1),
  caption: [*Rig and animal as sources of score differences.* Our network on new animals
    (test set A of the five folds, each animal new once). Pairs of units (an animal on one
    rig, its score averaged over its clips) that share the animal but not the rig
    (#tbd[159] pairs, #tbd[22] animals recorded on more than one rig) or the rig but not
    the animal (#tbd[520] pairs, #tbd[48] animals): the share of each kind's pairs in each
    bin of absolute difference between the two units of a pair, in inhale F1 (left) and
    trace correlation (right), 30 bins shared by the two kinds; above, each kind's mean with a
    95% bootstrap interval resampling animals (#tbd[2,000] draws).],
) <fig:rigs>

#figure(
  fit(figs2),
  caption: [*Our network on 30-Hz video.* The same network and training, fed every other
    frame (30 Hz) and still predicting on the 60-Hz grid, retrained on each of the five
    folds. *a*, The 30-Hz network minus the 60-Hz network on each test set, in trace
    correlation, inhale F1 (onset head) and exhale F1: each fold's difference (gray) and
    their mean with a 95% t-interval over the #tbd[5] folds. *b*, Inhale F1 on new animals
    (test set A) by breathing rate, each fold's breaths pooled, mean over folds with its
    95% interval; below, the share of breaths in each bin.],
) <fig:input-rate>
