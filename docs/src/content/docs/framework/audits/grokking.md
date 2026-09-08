---
title: "Grokking / Modular Addition"
description: "Criterion audit of the grokking / modular addition claim, as submitted."
---

# Grokking / Modular Addition

**Source.** *Progress Measures for Grokking* ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)).

**Description.** A one-layer transformer trained on $a+b 113$ maps each input to $$ and $$ at five key frequencies, multiplies those components so the products encode $a+b$, and reads the sum off at the same five frequencies. The frequencies are identified from the embedding's Fourier spectrum; the account is validated by ablating in Fourier space and by two progress measures, restricted and excluded loss, tracked across training. Restricted and excluded loss move before test accuracy does, which is what makes the account predictive rather than descriptive. Description mode: algorithmic.

## Readings of the claim

**Fourier multiplication.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** The network computes $a+b 113$ by multiplying Fourier components at five key frequencies | Mechanistically Supported | Selection correction over the frequency pool (M7) and a positive control (M5) |
| The five frequencies are the mechanism at neuron granularity | Underdetermined | The leave-one-out runs at frequency level; 79 of 512 neurons fail the polynomial fit the account predicts (I3) |
| The account fixes which algorithm the network runs | Underdetermined | ([Zhong et al., 2023](https://arxiv.org/abs/2306.17844)) give two algorithms on the same frequencies; the metrics separating them are unrun here (I5) |
| The progress measures explain grokking generally | Insufficient | Three further tasks are run but only for whether grokking occurs, never for the mechanism (E3) |

## Verdict

**Fourier multiplication.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — Unattempted; no second mechanism shown intact under ablation |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **C6** complementation validity — Five frequencies treated as interchangeable, never compared in pairs; **E5** graded response — Unattempted; ablation is binary per component, with no interpolation; **I10** rescue reversibility — Unattempted; every intervention substitutes rather than restores; **I12** offset coupling — Unattempted; no test that behavior disappears as the circuit does; **M5** sensitivity — Unattempted; a known-negative control with no known-positive |

## 36-criterion audit

**Fourier multiplication.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Each key frequency makes a directional ablation prediction that could fail |
| C2 | Structural plausibility | C | Weight-level throughout; $W_E$ sparse in the Fourier basis at 6 frequencies |
| C3 | Convergent validity | PC | Three families at origin, but all descend from Fourier-space ablation |
| C4 | Discriminant validity | PC | Excluded loss rises during circuit formation while train loss stays flat |
| C5 | Nomological validity | PC | Placed inside progress measures and phase changes; links are analogical |
| C6 | Complementation validity | U | Five frequencies treated as interchangeable, never compared in pairs |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Five seeds at the mainline configuration, reported in full (Table 3) |
| M2 | Baseline separation | C | All 56 frequencies ablated individually; the five separate by orders |
| M3 | Stability | PC | Seed, data fraction, depth and modulus swept; analysis choices are not |
| M4 | Calibration | PC | Loss in natural units; the uniform reference is invoked but never computed |
| M5 | Sensitivity | U | Unattempted; a known-negative control with no known-positive |
| M6 | Invariance | C | Table 5 repeats the mechanism statistics for 27 models, five conditions |
| M7 | Selection correction | U | Unattempted, and circular: frequencies chosen from the spectrum tested |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | C | Key-frequency removal gives 6.5–11 in four seeds; nullspace gives 5.27 |
| I2 | Sufficiency | C | Non-key Fourier ablation improves loss 70%; polynomial substitution costs 3% |
| I3 | Minimality | PC | Every key frequency is load-bearing singly; 79 of 512 neurons miss the cutoff |
| I4 | Specificity | N/A | One task; the model has no off-target behavior to spare |
| I5 | Rival mechanism exclusion | PC | Memorization excluded; Clock and Pizza are not ([Zhong et al., 2023](https://arxiv.org/abs/2306.17844)) |
| I6 | Double dissociation | U | Unattempted; no second mechanism shown intact under ablation |
| I7 | Confound control | PC | Data fraction, modulus, depth, seed and weight decay all varied |
| I8 | Confounding sensitivity | U | Unattempted; no sensitivity analysis for an unmeasured confounder |
| I9 | Epistatic interaction | PC | Non-additivity demonstrated: single frequencies have sparse effects |
| I10 | Rescue reversibility | U | Unattempted; every intervention substitutes rather than restores |
| I11 | Onset coupling | C | Restricted and excluded loss move before test accuracy does |
| I12 | Offset coupling | U | Unattempted; no test that behavior disappears as the circuit does |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | C | Five intervention primitives at four loci, all agreeing |
| E2 | Prompt generalization | C | The input space is enumerable and enumerated: all $113^2$ pairs |
| E3 | Cross-task generalization | PC | Three further tasks run, but only for whether grokking occurs |
| E4 | Cross-model recurrence | PC | One- and two-layer transformers only at origin |
| E5 | Graded response | U | Unattempted; ablation is binary per component, with no interpolation |
| E6 | Novel prediction | C | The progress measures are derived, then shown to predict the curve |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | Level implied by usage rather than declared |
| V2 | Level-evidence match | PC | Algorithm-level claim backed at the weight level for most of the model |
| V3 | Alternative level | PC | Memorization addressed and refuted; the basis alternative is not |
| V4 | Unlicensed labeling | C | Four labels restate measurements; the algorithm name outruns what separates it |
| V5 | Scope declaration | C | §6 states the limits before any generalization is implied |
| **Total:** 12 Confirmed, 15 Partially confirmed, 8 Untested, 1 Not applicable |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Grokking / Modular Addition — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-grokking/).
