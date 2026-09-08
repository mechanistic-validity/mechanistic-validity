---
title: "Induction Heads"
description: "Criterion audit of the induction heads claim, as submitted."
---

# Induction Heads

**Source.** *In-context Learning and Induction Heads* ([Olsson et al., 2022](https://arxiv.org/abs/2209.11895)).

**Description.** A two-head composition: a previous-token head writes the identity of the token before a repeat into the residual stream, and an induction head reads it to predict what followed last time. Heads are identified by two behavioral scores on sequences of repeated uniform-random tokens, prefix matching and copying, and the mechanism is derived at the weight level for the two-layer attention-only case. Evaluated on: 34 decoder-only models from one to forty layers, with causal ablation confined to the twelve small models, six attention-only and six with MLPs. Description mode: implementational-functional.

## Readings of the claim

**Induction heads: token copying.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Induction heads implement in-context copying of a token that followed the same context earlier | Triangulated | Minimality (I3), complementation validity (C6), rescue reversibility (I10) and offset coupling (I12) |
| The mechanism is the source of in-context learning as measured by token-loss difference | Mechanistically Supported | Necessity above 345M survives a decorrelating control, but the metric is uncalibrated (M4) |
| The mechanism is the source of in-context learning generally | Disconfirmed | Nothing — tested and failed; the bridge from the small-model result to the 13B claim is analogy (V2) |
| The two heads compose in the stated order, each performing the role its name asserts | Underdetermined | Weight-level derivation covers the two-layer case only; role semantics exceed it elsewhere (C2) |

## Verdict

**Induction heads: token copying.** Verdict: **Triangulated**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Reached. |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked. **C6** complementation validity — Two named roles, ablated one at a time and never together; **I3** minimality — Unattempted; the authors name the obstacle in marginal effects; **I10** rescue reversibility — Unattempted; pattern-preserving ablation already caches the clean run; **I12** offset coupling — Unattempted at origin; no published offset test for induction heads |

## 36-criterion audit

**Induction heads: token copying.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Two numerical evaluators with a stated failure mode; two predictions |
| C2 | Structural plausibility | C | Weight-level: copying is the OV circuit's positive-eigenvalue property |
| C3 | Convergent validity | C | Five families: eigenvalues, evaluators, ablation, scaling, training |
| C4 | Discriminant validity | PC | Built to separate induction from repeated-token memorization |
| C5 | Nomological validity | C | Derived from the framework rather than fitted to it (Elhage et al., 2021) |
| C6 | Complementation validity | U | Two named roles, ablated one at a time and never together |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Repeated across seeds in two-layer models by later work; never for the origin measurements |
| M2 | Baseline separation | PC | Every head ablated at every snapshot, so the effect has a reference |
| M3 | Stability | PC | Token indices perturbed and the score shown robust to the choice |
| M4 | Calibration | PC | Evaluators calibrated against weight-level quantities in the appendix |
| M5 | Sensitivity | PC | A known-positive: the two-layer model with a derived circuit (Elhage et al., 2021) |
| M6 | Invariance | C | 34 models, four series, 1–40 layers, up to 13B, 200 snapshots in the small models |
| M7 | Selection correction | U | Unattempted; heads are scored, then the high scorers are ablated |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | C | Almost all in-context learning in small models comes from these heads |
| I2 | Sufficiency | PC | No experiment reconstructs copying from induction heads alone |
| I3 | Minimality | U | Unattempted; the authors name the obstacle in marginal effects |
| I4 | Specificity | PC | Every head is ablated, so off-target effects are measured throughout |
| I5 | Rival mechanism exclusion | PC | Basic copying heads named and argued away rather than tested |
| I6 | Double dissociation | C | Unattempted at origin; run later and it succeeds ([Feucht et al., 2025](https://arxiv.org/abs/2504.03022)) |
| I7 | Confound control | PC | Three exogenous confounds checked, including scheduled hyperparameters |
| I8 | Confounding sensitivity | U | Unattempted; no sensitivity analysis for an unmeasured confounder |
| I9 | Epistatic interaction | PC | Composition is the mechanism; the two-head circuit is non-additive |
| I10 | Rescue reversibility | U | Unattempted; pattern-preserving ablation already caches the clean run |
| I11 | Onset coupling | C | Across 34 models, heads and the loss bump appear at the same point |
| I12 | Offset coupling | U | Unattempted at origin; no published offset test for induction heads |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | One primitive at origin: pattern-preserving zero ablation of a head |
| E2 | Prompt generalization | C | The defining stimulus is out of distribution by construction |
| E3 | Cross-task generalization | PC | The same heads perform literal translation, verified against definition |
| E4 | Cross-model recurrence | C | 34 models across four series, plus two external replications |
| E5 | Graded response | PC | Both evaluators are continuous; the ablation is not interpolated |
| E6 | Novel prediction | C | Argument 2 predicts an architectural change will move the bump |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | Declared early and explicitly, by behavior on the defining stimulus |
| V2 | Level-evidence match | C | Close for the narrow claim; the abstract's narrow sentence is backed |
| V3 | Alternative level | PC | One alternative addressed; the authors offer their own caveat |
| V4 | Unlicensed labeling | PC | The label imports an epistemic category, and the import is declared |
| V5 | Scope declaration | C | The most thoroughly scoped claim in this audit set |
| **Total:** 14 Confirmed, 16 Partially confirmed, 6 Untested |   |   |   |
| **Verdict: Triangulated** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Induction Heads — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-induction-heads/).
