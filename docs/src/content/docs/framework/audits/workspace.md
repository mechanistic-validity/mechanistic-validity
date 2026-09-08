---
title: "Global Workspace"
description: "Criterion audit of the global workspace claim, as submitted."
---

# Global Workspace

**Source.** *Verbalizable Representations Form a Global Workspace* ([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)).

**Description.** A low-dimensional subspace of the residual stream, recovered as the expected Jacobian $J_$ of the final-layer residual with respect to layer $$, is proposed as the set of contents a model can verbally report and reason over. Five properties are enumerated in advance from global workspace theory — verbal report, directed modulation, internal reasoning, flexible generalization, capacity limits — and one is used to find the object rather than to score it afterwards. Evaluated on: four closed-weight Claude models with sizes undisclosed, plus base and model-organism checkpoints. Description mode: implementational-functional.

## Readings of the claim

**Global workspace / J-space.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A low-dimensional Jacobian subspace mediates which contents the model reports and reasons over | Mechanistically Supported | A converse dissociation (I6), an unmeasured-confounder bound (I8), and correction for how the token sets were chosen (M7) |
| Any direction encoding the concept would serve equally | Disconfirmed | Nothing — named twice and tested twice, and concept vectors do not reproduce the effect |
| The subspace is a global workspace in the sense the theory intends | Underdetermined | The architectural disanalogies are enumerated by the authors; no experiment separates a workspace from a bottleneck that behaves like one |
| The result holds beyond one model family | Insufficient | Four models, one laboratory, all closed-weight, effects scale-graded from 54% to 70% (E4) |

## Verdict

**Global workspace / J-space.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — Single dissociation run cleanly and repeatedly; no converse |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. |

## 36-criterion audit

**Global workspace / J-space.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Five properties operationalized before any result is reported |
| C2 | Structural plausibility | C | $J_$ derived from the architecture, not posited |
| C3 | Convergent validity | PC | Three cross-checks, but almost every intervention edits along J-lens |
| C4 | Discriminant validity | PC | Separated from “any direction encoding this concept” by design |
| C5 | Nomological validity | C | Stated inside a named theory with disanalogies enumerated |
| C6 | Complementation validity | N/A | One subspace, undivided, so the question does not attach |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Wilson intervals and per-prompt dots throughout; no seed variance |
| M2 | Baseline separation | C | The densest negative controls in the set: logit lens and tuned lens |
| M3 | Stability | PC | Method variants swept and robust; corpus size from 1 to 1000 |
| M4 | Calibration | PC | Ablations normalized to unablated Sonnet with a smaller-model floor |
| M5 | Sensitivity | C | A known-positive control is run and named as one |
| M6 | Invariance | PC | Four models, base and organism checkpoints, two further lenses |
| M7 | Selection correction | U | Unattempted, and one token set is chosen self-referentially |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | C | Four necessity demonstrations, three with controls |
| I2 | Sufficiency | PC | Swaps install counterfactual content at 54–70% across three models |
| I3 | Minimality | PC | Occupancy estimated from where marginal improvement flattens |
| I4 | Specificity | C | Tested three ways; a fourteen-task battery maps where it is not needed |
| I5 | Rival mechanism exclusion | PC | The concept-vector rival is named twice and tested twice |
| I6 | Double dissociation | I | Single dissociation run cleanly and repeatedly; no converse |
| I7 | Confound control | C | Five designed controls, each aimed at its experiment's alternative |
| I8 | Confounding sensitivity | U | Unattempted; specific alternatives answered, no unmeasured-confounder bound |
| I9 | Epistatic interaction | PC | Token-level exclusion at 0.09 vs 0.29; two held concepts co-occupy at chance |
| I10 | Rescue reversibility | PC | Clamp-to-clean restoration run twice as a mediation control |
| I11 | Onset coupling | PC | Present and load-bearing in a base model before any RLHF |
| I12 | Offset coupling | PC | Deception signal amplified by coding RL, then attenuated by safety training |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Many intervention forms, nearly all editing along the same directions |
| E2 | Prompt generalization | C | Five public benchmarks plus six prompt distributions |
| E3 | Cross-task generalization | C | A fourteen-task battery maps where the mechanism is required |
| E4 | Cross-model recurrence | PC | Four closed-weight models, one lab, sizes undisclosed; partly replicated externally |
| E5 | Graded response | C | Six dose-response designs, including injection strength against rank |
| E6 | Novel prediction | C | Five properties enumerated from theory, one used to find the object |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | Declared before evidence, and the level is named explicitly |
| V2 | Level-evidence match | PC | Three levels in play and two matched; the third is asserted |
| V3 | Alternative level | PC | Two alternatives addressed and refuted; the workspace framing is not |
| V4 | Unlicensed labeling | PC | Disanalogies enumerated and consciousness not adjudicated; hedged |
| V5 | Scope declaration | C | Eight limitations named in a dedicated section |
| **Total:** 14 Confirmed, 18 Partially confirmed, 1 Inconclusive, 2 Untested, 1 Not applicable |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on. At Partially confirmed this claim reaches Validated.

| Criterion | In tension | What would settle it |
|---|---|---|
| I6 | Inconclusive or Partially confirmed | Whether a crossing over layer bands meets a criterion asking for two mechanisms |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Global Workspace — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-global-workspace/).
