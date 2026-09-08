---
title: "Refusal Direction"
description: "Criterion audit of the refusal direction claim, as submitted."
---

# Refusal Direction

**Source.** *Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al., 2024).

**Description.** A difference in means between activations on harmful and harmless instructions yields one residual-stream direction per model. Projecting that direction out of every matrix that writes to the residual stream stops the model from refusing harmful instructions, and adding it at its extraction layer makes the model refuse harmless ones, down to a request for the benefits of yoga. The direction is chosen as the minimizer of bypass\_score over every post-instruction position crossed with every layer subject to three thresholds, and the result is reported for 13 open-weight chat models in 5 families over a 40x parameter range. Description mode: undeclared — representational in the title, causal-behavioral in the evidence.

## Readings of the claim

**Refusal direction.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Ablating one difference-in-means direction stops refusal on harmful instructions and adding it induces refusal on harmless ones, across 13 chat models | Mechanistically Supported | A second estimator for the direction, since the three implementations are consumers of one difference in means (C3); a dissociation from any other behavior's direction (I6); sensitivity of the extraction to its own three thresholds (M3) |
| The direction carries refusal rather than harmfulness | Underdetermined | The estimator is the axis along which mean harmful and harmless activations differ, which a harmfulness direction satisfies exactly (V4); the two constructs are separated by citation rather than by experiment (C4); base models that never refuse express the same direction (I11) |
| Refusal is organized along one dimension of the residual stream | Underdetermined | The interventions license a lever rather than a representational organization, which the authors concede by calling the work an existence proof (V2); no rank above one is ever ablated and the search enumerates single vectors only (I3); refusal is scored as a twelve-substring match, an output property a single direction would control either way (V3) |
| Orthogonalizing the direction removes refusal whatever the prompting condition | Disconfirmed | Nothing — tested and failed: with its default system prompt the orthogonalized LLAMA-2 70B yields 4.4% attack success against 62.9% without it, so refusal produced on instruction survives an edit that removes the learned propensity (I1, M6) |

## Verdict

**Refusal direction.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — One direction and one behavior, tested in both directions of effect |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **E3** cross-task generalization — Every behavior tested is refusal or its absence; **E5** graded response — Strength is promised in 2.4 and every intervention has a coefficient of 1; **I10** rescue reversibility — The inverse is known in closed form and the restore is never run; **M3** stability — Three thresholds and one objective, none of them moved |

## 36-criterion audit

**Refusal direction.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Two predictions in opposite directions, with the outcome measure fixed first |
| C2 | Structural plausibility | C | The edit is enumerated over the five matrix kinds that write to the stream |
| C3 | Convergent validity | PC | Two interventions agree on outcome; the estimate itself is never replicated |
| C4 | Discriminant validity | PC | The estimator is a harmfulness contrast, and separation is argued by citation |
| C5 | Nomological validity | PC | The linear representation hypothesis does not entail the central parameter |
| C6 | Complementation validity | N/A | One direction, and a single direction has no subdivisions to carve |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Sampling error over prompts throughout; nothing quantifies the instrument |
| M2 | Baseline separation | PC | Five attacks and a fine-tuning comparator, and no baseline for the estimate |
| M3 | Stability | U | Three thresholds and one objective, none of them moved |
| M4 | Calibration | PC | Both metrics' failure modes shown by example and never counted |
| M5 | Sensitivity | PC | LoRA is an independently established positive and the instrument detects it |
| M6 | Invariance | C | 13 models, 5 families, 40x parameters, including where the effect degrades |
| M7 | Selection correction | U | Positions crossed with layers, and the search size is never written down |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | 0.95 to 0.01 on all 13 models; unnecessary for refusal produced on instruction |
| I2 | Sufficiency | PC | Harmless prompts refused across 13 models, on a direction chosen for sufficiency |
| I3 | Minimality | PC | Nothing to prune inside rank one, and no higher rank is compared against |
| I4 | Specificity | PC | Six benchmarks and three corpora, with TruthfulQA's fall left unresolved |
| I5 | Rival mechanism exclusion | PC | Token suppression eliminated on one case; the second rival pre-empted by design |
| I6 | Double dissociation | U | One direction and one behavior, tested in both directions of effect |
| I7 | Confound control | PC | Disjoint train, validation and evaluation sets; nothing on the extraction axis |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | U | Eight contributors measured one at a time, with no joint test |
| I10 | Rescue reversibility | U | The inverse is known in closed form and the restore is never run |
| I11 | Onset coupling | I | Base models express the direction as strongly as chat models do |
| I12 | Offset coupling | D | Refusal fine-tuned away; the direction survives it (Zhao et al., 2025) |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | A projection and an additive shift, agreeing on outcome and differing on cost |
| E2 | Prompt generalization | C | Evaluation prompts disjoint from extraction by construction, across 10 categories |
| E3 | Cross-task generalization | U | Every behavior tested is refusal or its absence |
| E4 | Cross-model recurrence | C | 13 models, 5 families and a 40x range, not near-copies |
| E5 | Graded response | U | Strength is promised in 2.4 and every intervention has a coefficient of 1 |
| E6 | Novel prediction | PC | An unrelated attack shows up as suppression, with a matched random control |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | Three description modes appear in the first paragraph and none is declared |
| V2 | Level-evidence match | PC | The interventions license a lever; the title asserts representational organization |
| V3 | Alternative level | PC | The token-suppression reading is retired; the level-of-measure one is untouched |
| V4 | Unlicensed labeling | PC | The outcome measure is operationalized; the direction's identity is not |
| V5 | Scope declaration | C | Four limitations named, each pointing at what the evidence does not reach |
| **Total:** 6 Confirmed, 19 Partially confirmed, 1 Inconclusive, 8 Untested, 1 Disconfirmed, 1 Not applicable |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Refusal Direction — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-refusal-direction/).
