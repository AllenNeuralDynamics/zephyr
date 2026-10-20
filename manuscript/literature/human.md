# Human and clinical background: camera-based respiration

Survey for the zephyr methods paper. It covers the human and clinical work on breathing from video,
whether our baselines (TS-CAN, PhysNet, PhysFormer, RhythmFormer, optical flow, PCA, pixel SNR,
motion energy) are the right ones, and how the field evaluates.
Every reference below was checked against PubMed, Crossref, arXiv or the official proceedings
on 2026-10-19 (authors, year, title, venue, identifier). Anything that could not be checked is
marked UNVERIFIED.

Our corpus for scale: 560 sessions from 50 mice, about 2 × 300 s of face video per session
(≈93 h), with an implanted-thermistor reference.

---

## 1. Non-contact respiration from video in humans

**Synthesis.** Camera respiration grew out of two strands. *Colour* (rPPG): ambient-light video
picks up both pulse and respiration in skin colour (Verkruysse 2008). Blind-source separation
(ICA, Poh 2010) and auto-regressive pole analysis (Tarassenko 2014) made this automatic.
*Motion*: breathing moves the chest, abdomen and head, and Eulerian video magnification
(Wu 2012) made such sub-pixel periodic changes visible. Chest-motion pipelines use optical flow
or profile tracking (Bartula 2013) and motion factorisation that finds the respiratory region
by itself (Janssen 2016); head motion alone carries usable breathing signal (Gwak 2023).
*Thermal* imaging of the nostrils measures airflow directly, as the temperature change between
inhalation and exhalation (Murthy & Pavlidis 2006; Abbas 2011; Pereira 2015; Cho 2017). Depth
cameras measure chest volume and give tidal volume as well as rate (Addison 2019, 2021).
Nostril thermography is the closest human analogue to our thermistor, since both sense airflow
temperature. In humans it is usually the most accurate modality, with MAE well under
1 breath/min at rest. Our task is harder: an RGB/IR camera has to infer airflow-related motion
of the snout without seeing temperature. Recent reviews agree on what goes wrong: accuracy
falls with motion and low light, datasets are small and controlled, and no shared benchmark
exists (Massaroni 2021; Molinaro 2022; Selvaraju 2022; Srestha & Kim 2025).

- Verkruysse, Svaasand & Nelson 2008, *Opt Express* 16:21434, doi:10.1364/oe.16.021434, PMID 19104573: the first ambient-light camera study; heart and respiration rates appear in skin colour.
- Poh, McDuff & Picard 2010, *Opt Express* 18:10762, doi:10.1364/OE.18.010762, PMID 20588929: ICA on face RGB traces, the origin of our "PCA/ICA pixel" family; validated with Bland–Altman.
- Wu et al. 2012, "Eulerian video magnification for revealing subtle changes in the world", *ACM TOG* 31(4) (SIGGRAPH), doi:10.1145/2185520.2185561: temporal band-pass amplification of pixel changes; it underlies the "pixel-intensity, motion-magnified" baselines.
- Bartula, Tigges & Muehlsteff 2013, *EMBC*, doi:10.1109/embc.2013.6610090: camera respiration from chest-motion profiles.
- Tarassenko et al. 2014, *Physiol Meas* 35:807, doi:10.1088/0967-3334/35/5/807, PMID 24681430: AR-model maps of HR/RR in dialysis patients; camera RR was at least as accurate as a chest belt.
- Janssen et al. 2016, *Physiol Meas* 37:100, doi:10.1088/0967-3334/37/1/100, PMID 26640970: optical-flow motion factorisation finds the breathing region automatically; limits of agreement −2.67 to 2.81 bpm. The closest human analogue to our optical-flow baseline.
- Gwak et al. 2023, *EMBC*, doi:10.1109/EMBC40787.2023.10340590, PMID 38082888: RR and breathing absence from head movements alone (MAE 1.91 bpm, 30 subjects).
- Murthy & Pavlidis 2006, "Noncontact measurement of breathing function", *IEEE EMB Mag* 25(3), doi:10.1109/memb.2006.1636352: nostril thermal imaging of airflow.
- Abbas et al. 2011, *Biomed Eng Online* 10:93, doi:10.1186/1475-925X-10-93, PMID 22243660: nostril thermography in NICU neonates (0.3–0.5 °C inhale/exhale contrast).
- Pereira et al. 2015, *Biomed Opt Express* 6:4378, doi:10.1364/BOE.6.004378, PMID 26601003: nose-tracking thermography against a piezo belt; MAE 0.33 bpm at rest and 0.55 bpm with head motion (11 subjects).
- Cho et al. 2017, *Biomed Opt Express* 8:4480, doi:10.1364/BOE.8.004480, PMID 29082079: mobile thermal nostril tracking, robust to ambient temperature changes.
- Addison et al. 2019, *J Clin Monit Comput* 34:1025, doi:10.1007/s10877-019-00417-6, PMID 31701371: depth camera against capnography during hypoxic challenge; bias 0.04 bpm, RMSD 0.66 bpm (14 volunteers).
- **Reviews:** Massaroni et al. 2021, *IEEE Sensors J* 21:12821, doi:10.1109/jsen.2020.3023486 (all contactless RR modalities); Molinaro et al. 2022, *Front Physiol* 13:801709, doi:10.3389/fphys.2022.801709, PMID 35250612 (digital-camera vital signs); Selvaraju et al. 2022, *Sensors* 22:4097, doi:10.3390/s22114097, PMID 35684717 (PRISMA review of 104 studies; RR RMSE ≈2.2 breaths/min under ideal conditions); Srestha & Kim 2025, *Physiol Meas* 46(9), doi:10.1088/1361-6579/adfc24, PMID 40816317 (RGB-only RR, public datasets, and the performance drop under motion and low light); McDuff 2023, "Camera measurement of physiological vital signs", *ACM Comput Surv*, doi:10.1145/3558518 (the broad survey, with resources).

## 2. Clinical uses and the accuracy they report

**Synthesis.** The best-developed clinical setting is the NICU. The Oxford group recorded 426.6 h
from 30 preterm infants and reached RR MAE 3.5 breaths/min for over 82 % of valid time
(Villarroel 2019). The reference was the bedside monitor, whose RR comes from impedance
pneumography, and they reported a signal-quality gate and time coverage alongside the error.
Smaller NICU studies show wide limits of agreement (Gibson 2019: RR LoA −22 to +23.6/min against
ECG-derived RR) and target apnoea detection from motion (Cattani 2017) or thermal airflow (Lorato
2021). In sleep medicine, NIR cameras tracked RR within 2 breaths/min 91 % of the time against
polysomnography (van Gastel 2021), and IR video detected apnoea episodes (Wang 2014). The
Covid-19 pandemic pushed remote and telehealth uses: smartphone rPPG validated against manual
RR counting in about 8,500 people (Heiden 2022), depth cameras against ventilators (Addison 2021),
and screening for abnormal breathing patterns (Wang 2020). References range from manual counting
and chest belts to impedance pneumography and capnography or ventilator flow. Most report RR
error in breaths/min, Bland–Altman bias and LoA, and sometimes coverage. None scores individual
breath timing.

- Villarroel et al. 2014, *Healthc Technol Lett* 1:87, doi:10.1049/htl.2014.0077, PMID 26609384: the first continuous NICU camera monitoring of HR, RR and SpO2.
- Villarroel et al. 2019, *npj Digit Med* 2:128, doi:10.1038/s41746-019-0199-5, PMID 31872068: 90 sessions from 30 infants; multi-task CNN skin segmentation; RR MAE 3.5/min; reports coverage. The model for reporting accuracy together with time-valid.
- Chaichulee et al. 2019, *Physiol Meas* 40:115001, doi:10.1088/1361-6579/ab525c, PMID 31661680: CNNs for patient detection and intervention detection, used to exclude unusable periods (cf. our QC).
- Gibson et al. 2019, *Pediatr Res* 86:738, doi:10.1038/s41390-019-0506-5, PMID 31351437: Bland–Altman against ECG in 10 preterm infants; very wide RR LoA.
- Cattani et al. 2017, *Comput Biol Med* 80:158, doi:10.1016/j.compbiomed.2016.11.010, PMID 27940321: motion-periodicity apnoea detection in newborns, using motion magnification.
- Lorato et al. 2021, *Sensors* 21:6306, doi:10.3390/s21186306, PMID 34577513: separates respiratory flow from motion in infant thermal video for obstructive apnoea, an analogue of airflow versus movement in our clips.
- Addison et al. 2025, *BMC Pediatr* 25:134, doi:10.1186/s12887-024-05238-4, PMID 39994574: depth-camera neonatal activity detection, which flags motion-corrupted respiratory segments.
- van Gastel et al. 2021, *IEEE JBHI* 25:1409, doi:10.1109/JBHI.2020.3045859, PMID 33338025: 46.5 h of NIR video with PSG in 8 suspected-OSA patients; RR within 2/min 91 % of the time.
- Wang et al. 2014, *IEEE TBME* 61:396, doi:10.1109/TBME.2013.2280132, PMID 24001952: IR-video breathing anomaly and apnoea recognition (94 % on clinical data).
- van der Linden et al. 2024, *Pediatr Pulmonol* 59:1871, doi:10.1002/ppul.27028, PMID 38661255: scoping review of contactless respiration in children; no modality clearly superior.
- Heiden et al. 2022, *JMIR Form Res* 6:e36340, doi:10.2196/36340, PMID 36374541: Lifelight smartphone rPPG; RR mean error 0.3 (SD 3.6) against manual counting.
- Addison et al. 2021, *J Clin Monit Comput* 36:657, doi:10.1007/s10877-021-00691-3, PMID 33743106: depth-camera RR and tidal volume against a ventilator (RMSD 0.51 bpm, single volunteer).
- Addison et al. 2021, *Sensors* 21:1135, doi:10.3390/s21041135, PMID 33561970: review of depth-camera respiratory monitoring, framed by the pandemic.
- Wang Y, Hu M, Li Q, et al. 2020, arXiv:2002.05534: depth-camera classification of abnormal breathing patterns for Covid-19 screening. The authors name an extended IEEE IoT J version (doi:10.1109/JIOT.2020.2991456), which was not checked here (UNVERIFIED).

## 3. Deep learning for remote physiological measurement

**Synthesis.** DeepPhys (Chen & McDuff 2018) introduced the two-branch design: normalised frame
differences for motion, plus an appearance-attention branch. It was the first end-to-end network
for heart *and breathing* rate. TS-CAN/MTTS-CAN (Liu 2020) added temporal shift modules and a
multi-task pulse and respiration head. On AFRL (25 subjects, five-fold subject-wise) it reached
respiration MAE ≈2.2–2.3 breaths/min, against 3.68 for the AR baseline. PhysNet (Yu 2019) uses
3D-CNNs with a negative-Pearson loss. PhysFormer (Yu 2022) is a temporal-difference transformer.
EfficientPhys (Liu 2023) is a single-branch, face-detector-free network. RhythmFormer (Zou 2025)
adds periodic sparse attention. Of these, only DeepPhys and MTTS-CAN target breathing explicitly,
along with BigSmall (Narayanswamy 2024), which multi-tasks PPG, breathing and facial action, and
AIRFlowNet (Manne 2023), an optical-flow-input network for infant respiration (MAE ≈2.9/min,
against 4.7–6.2 for adult-trained models). The rPPG-Toolbox (Liu 2023) is now the field's
reference implementation and benchmark. It implements TS-CAN, PhysNet, PhysFormer, DeepPhys,
EfficientPhys and BigSmall plus unsupervised POS/CHROM/ICA/GREEN/LGI/PBV, and it defines the
metrics (MAE, RMSE, MAPE, Pearson r, SNR, Bland–Altman) and cross-dataset protocols. The
toolbox's own respiration result is sobering. BigSmall on BP4D+ (3-fold, subject-wise) gives
breathing MAE 5.19/min, MAPE 26.3 % and **r = 0.14**. Face-video respiration remains far less
solved than pulse, even in humans.

**Public datasets with a respiration reference** (human):

| Dataset | Subjects | Size | Respiration reference |
| --- | --- | --- | --- |
| COHFACE (Heusch 2017) | 40 | 160 × 1 min (≈2.7 h) | chest belt |
| MAHNOB-HCI (Soleymani 2012) | 27 | emotion-elicitation trials | respiration amplitude (belt) |
| BP4D+ (Zhang 2016) | 140 | 1,400 tasks | BIOPAC respiration wave and rate |
| V4V (Revanur 2021, from BP4D+ plus new subjects) | 179 | 1,358 videos ≈44 s (≈17 h) | RR track |
| OBF (Li 2018) | 100 healthy plus AF patients | 200 × 5 min (≈17 h) | respiration at 32 Hz |
| SCAMPS (McDuff 2022), synthetic | avatars | 2,800 videos, 1.68 M frames | exact synthetic breathing waveform |
| AIR-125 (Manne 2023), infants | 8 | 125 videos | manual annotation |

UBFC-Phys (56 subjects, 168 videos) provides **only BVP and EDA, not respiration**. MMPD (33
subjects, 11 h, phone video) provides **only PPG**. Neither can be a respiration benchmark. Our
corpus (50 animals, ≈93 h, inhalation-resolved thermistor) is larger in hours than every public
human respiration set. It is comparable in subjects to COHFACE and MAHNOB but smaller than BP4D+
and V4V. Only the non-public NICU set of Villarroel 2019 (426.6 h, 30 infants) is longer.

- Chen & McDuff 2018, DeepPhys, *ECCV*, doi:10.1007/978-3-030-01216-8_22, arXiv:1805.07888: two-branch motion and appearance network for HR and breathing; its difference-frame input is shared by TS-CAN, PhysNet and EfficientPhys in the toolbox.
- Liu, Fromm, Patel & McDuff 2020, MTTS-CAN, *NeurIPS* 33, arXiv:2006.03790: on-device multi-task pulse and respiration; our TS-CAN baseline is the right one, and it already has a respiration output.
- Yu, Li & Zhao 2019, PhysNet, *BMVC*, arXiv:1905.02419: 3D-CNN rPPG (BMVC has no DOI).
- Yu et al. 2022, PhysFormer, *CVPR*, doi:10.1109/cvpr52688.2022.00415, arXiv:2111.12082: temporal-difference transformer.
- Liu et al. 2023, EfficientPhys, *WACV*, doi:10.1109/wacv56688.2023.00498, arXiv:2110.04447: no face detection or cropping, so it suits fixed-crop animal video.
- Zou et al. 2025, RhythmFormer, *Pattern Recognit*, doi:10.1016/j.patcog.2025.111511, arXiv:2402.12788: periodic sparse attention, the current transformer state of the art.
- Narayanswamy et al. 2024, BigSmall, *WACV*, doi:10.1109/wacv57701.2024.00773, arXiv:2303.11573: joint PPG, breathing and AU model on BP4D+; breathing correlation is low.
- Manne, Zhu, Ostadabbas & Wan 2023, AIRFlowNet and AIR-125, *MICCAI workshop (LNCS)*, doi:10.1007/978-3-031-45544-5_10, arXiv:2307.13110: optical flow fed to a spatiotemporal CNN with a spectral band-pass loss; the respiration-specific deep baseline closest to ours.
- Liu et al. 2023, rPPG-Toolbox, *NeurIPS 36 Datasets & Benchmarks*, arXiv:2210.00716: the reference benchmark; it fixes metrics and cross-dataset protocol.
- Boccignone et al. 2023, *Sensors* 23:3387, doi:10.3390/s23073387, PMID 37050444: on BP4D+, rPPG-derived respiratory variations beat a deep remote-respiration model; an argument for strong DSP baselines.
- Heusch, Anjos & Marcel 2017, COHFACE, arXiv:1709.00962: also notes that "none of the selected algorithms is precise enough" for real-world use.
- Soleymani et al. 2012, MAHNOB-HCI, *IEEE TAC* 3:42, doi:10.1109/t-affc.2011.25.
- Zhang et al. 2016, BP4D+, *CVPR*, doi:10.1109/cvpr.2016.374.
- Revanur et al. 2021, V4V challenge, *ICCVW*, doi:10.1109/iccvw54120.2021.00310, arXiv:2109.10471: the first shared RR track, scored frame-level (continuous MAE).
- Li et al. 2018, OBF, *FG*, doi:10.1109/fg.2018.00043, pp. 242–249.
- McDuff et al. 2022, SCAMPS, *NeurIPS Datasets & Benchmarks*, arXiv:2206.04197.
- Sabour et al. 2023, UBFC-Phys, *IEEE TAC* 14:622, doi:10.1109/taffc.2021.3056960 (no respiration channel).
- Tang et al. 2023, MMPD, *EMBC*, doi:10.1109/embc40787.2023.10340857, arXiv:2302.03840 (PPG only).

## 4. Lessons from the field

**Synthesis.** *Generalisation.* Cross-dataset error dwarfs within-dataset error. In the
rPPG-Toolbox, TS-CAN trained on PURE scores HR MAE 1.29 bpm on UBFC-rPPG but 13.9 bpm on MMPD.
Unsupervised POS often matches or beats supervised networks out of domain (Liu 2023). Synthetic
data (SCAMPS) and motion augmentation only partly close the gap. Performance also varies with
skin type and demographics (Nowara 2020; Dasari 2021) and with video compression (McDuff 2017).
For us the analogues are fur colour, camera, rig and laboratory. *Evaluation practice.* Folds
are subject-disjoint (TS-CAN's five folds of five people; BigSmall's three subject folds;
Heusch's 19/10 split on MAHNOB). Errors are MAE and RMSE in breaths/min with Pearson r. Clinical
papers add Bland–Altman bias and 95 % LoA (Bland & Altman 1986) and the fraction of time with a
valid estimate (Villarroel 2019). V4V scores frame-level continuous rate rather than per-clip
rate. *Pitfalls.* Motion is the dominant failure: it raises Pereira's MAE threefold, and the
NICU studies have to exclude interventions. Frequency-domain rate metrics hide waveform errors
(BigSmall's low r). We found no study dedicated to subject leakage in rPPG; the field relies on
subject-disjoint folds by convention. No human study scores individual breath onsets.

- Nowara, McDuff & Veeraraghavan 2020, *CVPRW*, doi:10.1109/cvprw50498.2020.00150: meta-analysis (73 people) showing errors rise for the darkest skin type; the analogue is coat colour or strain.
- Dasari et al. 2021, *npj Digit Med* 4:91, doi:10.1038/s41746-021-00462-z: rPPG bias across demographics and environments in India and Sierra Leone.
- McDuff, Blackford & Estepp 2017, *FG*, doi:10.1109/fg.2017.17: video compression degrades rPPG, relevant to our stream-copied H.264 clips.
- Bland & Altman 1986, *Lancet* 1:307, PMID 2868172: the agreement-analysis standard expected by clinical reviewers.

## 5. Animal-adjacent work

**Synthesis.** Video vital signs in animals are rare and mostly small feasibility studies.
Rodents: RGB/thermal video of rats and mice gave RR RMSE 0.32 and 1.42 breaths/min against
contact references, but in anaesthetised or restrained animals (Kunczik 2019; review in Pereira
2019). Nostril IR thermography in head-restrained mice detects inhalation onsets without
segmentation (Mutlu 2018); it is the closest prior work to ours and needs a thermal camera.
Primates: face iPPG and EVM in rhesus macaques give heart rate, not respiration (Unakafov 2018;
Froesel 2020). Livestock: RR from IR video of cattle correlated r = 0.87 with *visual counts*
(Jorquera-Chavez 2019). Zoo animals: camera motion signals were shown in nine species without any
reference (Al-Naji 2019). Dogs: a 2026 pilot detected respiratory motion only against visual
review (Ohnishi 2026). We found no prior work that infers breathing waveform and inhalation onsets
from ordinary orofacial video of awake, behaving mice against an implanted sensor, or that
benchmarks human rPPG networks on animals.

- Kunczik et al. 2019, *Biomed Opt Express* 10:4422, doi:10.1364/BOE.10.004422, PMID 31565499: camera RR and HR in rats and mice for severity assessment.
- Pereira et al. 2019, *J Biomed Opt* 24:070601, doi:10.1117/1.JBO.24.7.070601, PMID 31286726: review of optical imaging for lab-animal welfare; vital signs only in sedated animals so far.
- Mutlu et al. 2018, *J Neurosci Methods* 301:1, doi:10.1016/j.jneumeth.2018.02.017, PMID 29501561: thermal pixel-time-series inhalation onsets in head-restrained mice, validated against established standard techniques. The prior art for onset timing.
- Unakafov et al. 2018, *PLoS One* 13:e0202581, doi:10.1371/journal.pone.0202581, PMID 30169537: iPPG pulse rate in awake rhesus monkeys.
- Froesel et al. 2020, *Sci Rep* 10:17940, doi:10.1038/s41598-020-74954-5, PMID 33087832: EVM heart rate in anaesthetised and behaving macaques against ECG.
- Jorquera-Chavez et al. 2019, *Animals* 9:1089, doi:10.3390/ani9121089, PMID 31817620: cattle HR, RR and temperature from RGB and IR video.
- Al-Naji et al. 2019, *Sensors* 19:5445, doi:10.3390/s19245445, PMID 31835550: cardiopulmonary motion in zoo animals, no reference.
- Ohnishi et al. 2026, *Vet Sci* 13:781, doi:10.3390/vetsci13080781, PMID 42655801: IR-video respiratory motion in hospitalised dogs.

---

## Implications for our baselines and evaluation

1. **The learned baselines are the right ones.** TS-CAN, PhysNet and PhysFormer are the three
   most-used supervised rPPG models and all are in the rPPG-Toolbox. RhythmFormer is the
   recent state of the art, and TS-CAN (as MTTS-CAN) is the canonical respiration-capable
   network. The method set is fixed: DeepPhys, EfficientPhys and AIRFlowNet are not added.
   Cite them as related work. Be ready to answer in review why TS-CAN (DeepPhys's
   successor) and our optical-flow baseline cover them.
2. **The non-learned baselines cover the field's families.** Pixel PCA/ICA matches Poh 2010,
   optical flow matches Janssen 2016, and motion energy and SNR pixel selection match
   EVM-style approaches. Unsupervised rPPG colour methods
   (POS/CHROM) are not relevant for respiration through fur; one sentence saying why will do.
3. **Metrics to add or report explicitly:** (a) breathing-rate MAE and RMSE in breaths/min
   with Pearson r, per clip and per animal, comparable to every human paper; (b) **Bland–Altman
   bias and 95 % LoA** against the thermistor, which clinical reviewers expect; (c)
   **coverage / time-valid** together with any QC gating (Villarroel 2019); (d) waveform
   correlation, since BigSmall shows rate MAE can hide r ≈ 0.1. Our inhalation-onset F1 and
   timing error go *beyond* the field; no human study scores single breaths, so we should say so
   as a contribution.
4. **Protocol.** Hold out whole animals, as the human literature holds out subjects; we already
   do. Report a **cross-rig / cross-camera** or cross-lab test as the analogue of the field's
   cross-dataset evaluation, since that is where human networks fail most (HR MAE 1.3 → 13.9 bpm
   in the toolbox). Report motion-stratified errors (running vs still), the human field's
   main failure mode.
5. **Scale claim.** At ≈93 h and 50 animals with a breath-resolved implanted reference, our
   corpus exceeds every public human respiration video dataset in hours. It is the first
   animal benchmark of this kind; only one non-public clinical NICU set (426.6 h) is larger.
