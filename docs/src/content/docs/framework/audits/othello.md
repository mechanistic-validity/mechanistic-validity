---
title: "Othello Board State"
description: "Criterion audit of the othello board state claim, as submitted."
---

# Othello Board State

**Source.** *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task* ([Li et al., 2023](https://arxiv.org/abs/2210.13382)).

**Description.** A GPT variant trained from random initialization on Othello transcripts carries the board state in its activations. Nonlinear probes decode all 64 tiles where linear probes never dip below 20% error, and 2,000 interventions move the top-$N$ predictions onto the counterfactual legal-move set (errors 0.12 and 0.06 against null baselines of 2.68 and 2.59), half of them on boards unreachable by legal play. Evaluated on: one 8-layer architecture trained twice, on championship games and on uniformly sampled legal moves, with both runs probed across all eight layers. Description mode: representational.

## Readings of the claim

**Othello world model.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Othello-GPT carries a decodable representation of board state that its move predictions causally use | Causally Suggestive | Necessity by removal (I1), a second intervention operator (E1), a dissociation (I6), and a known-positive control, which the origin does not contain (M5) |
| The representation is a model of the process producing the sequences, which is the paper's own definition of a world model | Underdetermined | An experiment separating that from a decodable, causally-used state summary, which every result in the paper also satisfies (V4, V3); the rule composition that would distinguish them is measured once and fails on the OR across lines (V2, I9) |
| The board state is encoded nonlinearly | Underdetermined | A perturbation of the probe target, held at black/white/empty throughout and the one configuration the stability sweep never varied (M3, I5) |
| The account travels beyond Othello and beyond this architecture | Insufficient | The origin runs one task on one architecture trained twice, and lists other games and natural language as future work; E3 and E4 are scored on post-origin evidence |

## Verdict

**Othello world model.** Verdict: **Causally Suggestive**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked. **E1** intervention reach — One intervention operator, reported under three outcome metrics |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Mechanistically Supported. **I6** double dissociation — One representation, one behavior; no converse leg is available |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Mechanistically Supported. **C6** complementation validity — Sixty-four tiles, intervened on one at a time throughout; **I11** onset coupling — Training-step axis available and explicitly declined; **I12** offset coupling — Two models differ in competence; the coupling is never analyzed |

## 36-criterion audit

**Othello world model.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Intervention fixes the failing outcome before it is measured |
| C2 | Structural plausibility | PC | Layerwise argument and probe geometry; no weight-level account |
| C3 | Convergent validity | C | Probe, intervention and three metrics agree; two share one instrument |
| C4 | Discriminant validity | PC | Separated from memorization and from a random net, not from a summary |
| C5 | Nomological validity | PC | Named rival theory and a literature; no theory of what a world model is |
| C6 | Complementation validity | U | Sixty-four tiles, intervened on one at a time throughout |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | C | Probe accuracies re-run 100 times; deviations in Tables 4 and 5 |
| M2 | Baseline separation | C | Untrained net, constant guess and null intervention floors all reported |
| M3 | Stability | PC | Optimizer, $$ and probe capacity swept; the probe target never varied |
| M4 | Calibration | PC | Intervention error has a null floor; probe accuracy has no task scale |
| M5 | Sensitivity | PC | Known-negative at origin; the known-positive is later ([Vafa et al., 2024](https://arxiv.org/abs/2406.03689)) |
| M6 | Invariance | C | All eight layers, both datasets, both benchmarks, three metrics |
| M7 | Selection correction | U | Headline numbers are the maximum over swept Ls and $$, uncorrected |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | Substitution to a counterfactual state; nothing is removed |
| I2 | Sufficiency | PC | Editing the representation alone moves predictions, including off-distribution |
| I3 | Minimality | N/A | One activation vector holds all 64 tiles; no components to prune |
| I4 | Specificity | PC | Off-target flips raised, mitigation swept, effects never counted |
| I5 | Rival mechanism exclusion | PC | Memorization and off-distribution rivals excluded; basis choice was not |
| I6 | Double dissociation | U | One representation, one behavior; no converse leg is available |
| I7 | Confound control | PC | Memorization, distribution and probe capacity controlled; target is not |
| I8 | Confounding sensitivity | U | No sensitivity analysis, and no quantity that could serve as a bound |
| I9 | Epistatic interaction | U | Single-tile edits only; rule composition named as future work |
| I10 | Rescue reversibility | PC | Restoration observed as an obstacle, not designed as a rescue test |
| I11 | Onset coupling | U | Training-step axis available and explicitly declined |
| I12 | Offset coupling | U | Two models differ in competence; the coupling is never analyzed |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | U | One intervention operator, reported under three outcome metrics |
| E2 | Prompt generalization | C | 1000 natural and 1000 unnatural cases; both training distributions |
| E3 | Cross-task generalization | PC | Othello only at origin; chess and rule variants come later ([Karvonen, 2024](https://arxiv.org/abs/2403.15498)) |
| E4 | Cross-model recurrence | C | Absent at origin; seven architectures tested later ([Yuan & Sogaard, 2025](https://arxiv.org/abs/2503.04421)) |
| E5 | Graded response | PC | Two knobs swept with degradation outside the optimum; no dose-response |
| E6 | Novel prediction | C | Unnatural boards predicted to work and did; saliency maps split two models |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | Level named by the phrase used, and the phrase changes between sections |
| V2 | Level-evidence match | PC | Evidence matches decodable plus causal; the definition asks for more |
| V3 | Alternative level | PC | Memorization alternative posed and refuted; state summary never posed |
| V4 | Unlicensed labeling | PC | “World model” is defined, and only a state summary is ever measured |
| V5 | Scope declaration | C | Synthetic scope declared in the title, §1.1 and the conclusion |
| **Total:** 9 Confirmed, 18 Partially confirmed, 8 Untested, 1 Not applicable |   |   |   |
| **Verdict: Causally Suggestive** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on.

| Criterion | In tension | What would settle it |
|---|---|---|
| M5 | Partially confirmed or Untested | Whether one instrument's known-positive licenses a sensitivity verdict for another |
| I6 | Untested or Not applicable | Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Othello Board State — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-othello/).
