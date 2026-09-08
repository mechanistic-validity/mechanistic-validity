---
title: "Probing Classifiers"
description: "Criterion audit of the probing classifiers claim, as submitted."
---

# Probing Classifiers

**Source.** *Probing Classifiers: Promises, Shortcomings, and Advances* ([Belinkov, 2022](https://doi.org/10.1162/coli_a_00422)).

**Description.** A classifier $g$ is trained on a model's intermediate output $f_l(x)$ to predict a property $z$, and its accuracy is reported as evidence that the model has learned information relevant to $z$. The audited object is the method rather than one experiment: the anchor is a review that runs no experiments of its own, so the evidence sits in the studies it formalizes, and its Figure 1 separates the nine components a probing result depends on from the ten controls the literature has added to them. Evaluated on: every architecture family the method has been applied to, from static word embeddings through recurrent and recursive networks to transformers, and outward to speech recognition and computer vision. Description mode: representational.

## Readings of the claim

**Probing classifiers.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A probing result licenses that property $z$ is extractable from the representation by a classifier of the stated capacity | Proposed | Necessity: the method performs no removal of its own, and where removal was performed the studies the anchor reports disagree four ways (I1, E1) |
| A good probe score shows that the model represents $z$ | Disconfirmed | Nothing — tested and failed: control tasks attribute most of a nonlinear probe's accuracy to the probe itself, and even random features decode the property (C4) |
| A good probe score shows that the model uses $z$ in producing its output | Disconfirmed | Nothing — tested and failed: a control dataset holds $z$ non-discriminative for the original task and the probe recovers it anyway, so decodability does not localize to the task (I4) |
| Interventions on a probe-identified direction show which features the model uses | Underdetermined | A result that decides the four-way disagreement among the intervention studies of §4.3 (I1, E1). This is the broad scope, under which **I2** and **E5** are unattempted; under probing as §2 defines it, a read-out with no path writing back in, they are not applicable |

## Verdict

**Probing classifiers.** Verdict: **Proposed**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Blocked. **I1** necessity — Standard probing performs no removal; removal studies disagree four ways |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked at Causally Suggestive. **E1** intervention reach — Probing has no intervention of its own; those applied to it disagree; **I4** specificity — A control dataset holds the property constant and probing fails on it |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Causally Suggestive. **C4** discriminant validity — Two neighbors the instrument must separate: probe memorization, random features; **I6** double dissociation — Needs two properties and two behaviors; the framework supplies one of each |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Causally Suggestive. **I3** minimality — The unit is a whole intermediate output; no operation removes part of one; **I10** rescue reversibility — Every catalogued intervention runs one way, projecting or training out; **I11** onset coupling — The original model is taken as given and trained once; **I12** offset coupling — The converse needs the training sequence onset coupling also lacks; **M1** reliability — A review reports no variance; the anchor states what it requires of studies |

## 36-criterion audit

**Probing classifiers.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | The prediction can fail and did ([Giulianelli et al., 2018](https://aclanthology.org/W18-5426/); Elazar et al., 2021) |
| C2 | Structural plausibility | PC | Every symbol a probing result depends on is named and typed |
| C3 | Convergent validity | PC | Probe families agree on accuracy, reverse under selectivity (Hewitt & Liang, 2019) |
| C4 | Discriminant validity | D | Two neighbors the instrument must separate: probe memorization, random features |
| C5 | Nomological validity | PC | Tied to mutual information, description length and information gain |
| C6 | Complementation validity | N/A | The audited object is a method, not a construct with named parts |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | U | A review reports no variance; the anchor states what it requires of studies |
| M2 | Baseline separation | C | The criterion the literature solved; three control families formalized |
| M3 | Stability | I | Ordering holds under accuracy and inverts under selectivity (Hewitt & Liang, 2019) |
| M4 | Calibration | D | The shortcomings section opens by denying the measurement has a scale |
| M5 | Sensitivity | PC | Skylines and floors exist; no planted property is ever recovered |
| M6 | Invariance | PC | Invariance is what probing measures, so the instrument is the question |
| M7 | Selection correction | U | Three control families catalogued; the omission from the catalogue is the gap |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | I | Standard probing performs no removal; removal studies disagree four ways |
| I2 | Sufficiency | N/A | Probing is a map out of the representation, with no path writing back in |
| I3 | Minimality | U | The unit is a whole intermediate output; no operation removes part of one |
| I4 | Specificity | D | A control dataset holds the property constant and probing fails on it |
| I5 | Rival mechanism exclusion | PC | Control tasks, solvable only by memorization, separate cleanly (Hewitt & Liang, 2019) |
| I6 | Double dissociation | U | Needs two properties and two behaviors; the framework supplies one of each |
| I7 | Confound control | PC | Probe capacity, information in a random baseline and task difficulty |
| I8 | Confounding sensitivity | U | The anchor answers confounding with designed controls, never with a bound |
| I9 | Epistatic interaction | U | Non-additivity needs two manipulable components; the object is one output |
| I10 | Rescue reversibility | U | Every catalogued intervention runs one way, projecting or training out |
| I11 | Onset coupling | U | The original model is taken as given and trained once |
| I12 | Offset coupling | U | The converse needs the training sequence onset coupling also lacks |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | I | Probing has no intervention of its own; those applied to it disagree |
| E2 | Prompt generalization | PC | One line of work reports the same property across several datasets |
| E3 | Cross-task generalization | PC | Transfers to any property with an annotated dataset, which is the appeal |
| E4 | Cross-model recurrence | C | Portability is established without qualification, from static embeddings on |
| E5 | Graded response | N/A | No intervention, so no strength to dial and no response to grade |
| E6 | Novel prediction | PC | Four design changes followed probing results and then worked |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | The formalism fixes which objects a probing result is about |
| V2 | Level-evidence match | D | The evidence supports extractability; the claim made is representation |
| V3 | Alternative level | C | Every rival reading of a good score is given a measure of its own |
| V4 | Unlicensed labeling | PC | “The model knows z” projects an epistemic relation onto decodability |
| V5 | Scope declaration | C | A scope declaration end to end, conceding limits before any content |
| **Total:** 5 Confirmed, 12 Partially confirmed, 3 Inconclusive, 9 Untested, 4 Disconfirmed, 3 Not applicable |   |   |   |
| **Verdict: Proposed** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on.

| Criterion | In tension | What would settle it |
|---|---|---|
| I6 | Untested or Not applicable | Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Probing Classifiers — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-probing/).
