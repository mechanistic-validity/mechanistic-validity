---
title: "SAE Features"
description: "Criterion audit of the sae features claim, as submitted."
---

# SAE Features

**Source.** *Sparse Autoencoders Find Highly Interpretable Features* ([Cunningham et al., 2024](https://arxiv.org/abs/2309.08600)).

**Description.** Sparse autoencoders are trained on residual-stream activations to recover an overcomplete dictionary of directions, evaluated two ways: automated interpretability scoring of 150 features per method against six baselines (default basis, random directions, PCA, ICA, top-$K$ PCA, top-$K$ ICA), and activation patching on 50 IOI data points. Evaluated on: Pythia-70M and 410M at origin, GPT-2 small and Gemma 2 2B in the follow-up. Being a method-level claim, these are the systems the evidence was gathered on rather than one audited model. Description mode: representational.

## Readings of the claim

**SAE features.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Sparse dictionary learning recovers directions more interpretable than PCA, ICA and the neuron basis, and causally usable | Proposed | Baseline separation (M2), where the origin's within-model controls and the later random-network controls disagree; then specificity (I4), independent convergence (C3), separation from other decompositions (I6), and calibration of the scoring instrument (M1, M3, M5, M7) |
| SAE features are atomic, canonical units of the model | Disconfirmed | Nothing — tested and failed: meta-SAEs decompose latents further (C4), and the instrument cannot distinguish a trained transformer from a random one ([Heap et al., 2025](https://arxiv.org/abs/2501.17727)) |
| SAE features localize causation better than neurons | Disconfirmed | Nothing — tested and failed: (Mueller et al., 2025) report no causal-localization advantage over the neuron basis |

## Verdict

**SAE features.** Verdict: **Proposed**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Blocked. **M2** baseline separation — Separates from random directions; not from a random network when that control is run |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked at Causally Suggestive. **I4** specificity — Single-feature ablation moves 12,000 logits; left unanalyzed |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Causally Suggestive. **I6** double dissociation — Unattempted; no second decomposition shown intact |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Causally Suggestive. **C6** complementation validity — Feature splitting is the open question; no test of atomicity is run; **E3** cross-task generalization — Unattempted; generalization stated as an expectation; **I10** rescue reversibility — Unattempted, though patching is reversible by construction; **I11** onset coupling — Unattempted, and cheap: Pythia ships pretraining checkpoints ([Biderman et al., 2023](https://arxiv.org/abs/2304.01373)); **I12** offset coupling — Unattempted; the nearest result is at an extreme, not a trajectory ([Heap et al., 2025](https://arxiv.org/abs/2501.17727)) |

## 36-criterion audit

**SAE features.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Directional prediction against four named baselines; partly fails in layer 4 |
| C2 | Structural plausibility | PC | Predicted by superposition; MLP training fails, attention not attempted |
| C3 | Convergent validity | PC | Two evidence types; the concurrent-origin pairing comes later ([Leask et al., 2025](https://arxiv.org/abs/2502.04878)) |
| C4 | Discriminant validity | PC | Scored against the origin's operational construct, not the field's |
| C5 | Nomological validity | PC | Fits superposition; the theory does not underwrite the method |
| C6 | Complementation validity | U | Feature splitting is the open question; no test of atomicity is run |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | I | One run at origin; later seed studies disagree ([Paulo & Belrose, 2025](https://arxiv.org/abs/2501.16615)) |
| M2 | Baseline separation | I | Separates from random directions; not from a random network when that control is run |
| M3 | Stability | I | Sparsity sweep gives a smooth tradeoff, not a stable optimum |
| M4 | Calibration | PC | Interpretability score is a correlation with a meaningful zero |
| M5 | Sensitivity | D | Tested later on planted features and the recovery is poor ([Korznikov et al., 2026](https://arxiv.org/abs/2602.14111)) |
| M6 | Invariance | PC | All layers of Pythia-70M, five expansion ratios, two scoring regimes |
| M7 | Selection correction | U | A feature scoring fewer than 20 varying fragments is skipped; the count is unreported |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | One feature ablated for behavior; a full previous-layer sweep reads out to features |
| I2 | Sufficiency | PC | Interchange intervention on IOI; counterfactual features written in |
| I3 | Minimality | PC | ACDC orders features; the next feature usually adds little |
| I4 | Specificity | I | Single-feature ablation moves 12,000 logits; left unanalyzed |
| I5 | Rival mechanism exclusion | PC | Rival methods excluded: PCA, ICA, top-$K$, non-sparse dictionary |
| I6 | Double dissociation | U | Unattempted; no second decomposition shown intact |
| I7 | Confound control | PC | Three designed controls, including an $=0$ dictionary |
| I8 | Confounding sensitivity | U | Unattempted; no E-value or sensitivity analysis |
| I9 | Epistatic interaction | PC | Assumed away explicitly; ACDC treats features as a flat graph |
| I10 | Rescue reversibility | U | Unattempted, though patching is reversible by construction |
| I11 | Onset coupling | U | Unattempted, and cheap: Pythia ships pretraining checkpoints ([Biderman et al., 2023](https://arxiv.org/abs/2304.01373)) |
| I12 | Offset coupling | U | Unattempted; the nearest result is at an extreme, not a trajectory ([Heap et al., 2025](https://arxiv.org/abs/2501.17727)) |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Two forms: activation patching and less-than-rank-one ablation |
| E2 | Prompt generalization | PC | Cross-corpus by construction: trained on the Pile, scored on OpenWebText |
| E3 | Cross-task generalization | U | Unattempted; generalization stated as an expectation |
| E4 | Cross-model recurrence | C | Transfers everywhere tried: Pythia, GPT-2 small, Gemma 2 2B ([Leask et al., 2025](https://arxiv.org/abs/2502.04878)) |
| E5 | Graded response | PC | Two graded curves: KL against features patched, edit size against KL |
| E6 | Novel prediction | PC | The IOI experiment is a prediction in the weak sense |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | Formal object declared precisely; “feature” carries more than it |
| V2 | Level-evidence match | PC | Evidence supports the decomposition claim the origin makes |
| V3 | Alternative level | PC | Raised at origin and not pursued: no single correct decomposition |
| V4 | Unlicensed labeling | PC | “Monosemantic” projects semantics; the origin hedges consistently |
| V5 | Scope declaration | C | Both papers declare limits and quantify them ([Leask et al., 2025](https://arxiv.org/abs/2502.04878)) |
| **Total:** 3 Confirmed, 20 Partially confirmed, 4 Inconclusive, 8 Untested, 1 Disconfirmed |   |   |   |
| **Verdict: Proposed** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [SAE Features — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-sae-features/).
