---
title: "Copy Suppression"
description: "Criterion audit of the copy suppression claim, as submitted."
---

# Copy Suppression

**Source.** *Copy Suppression: Comprehensively Understanding an Attention Head* ([McDougall et al., 2023](https://arxiv.org/abs/2310.04625)).

**Description.** Negative Head L10H7 in GPT-2 Small suppresses tokens that earlier components have already predicted, in three steps: an early component predicts a token that already appears in context, the head attends back to that token, and its OV circuit writes a negative logit for it. The QK and OV halves are read off the weights before any ablation, with 84.70% of tokens carrying their OV diagonal among the ten most negative entries of its column and 95.72% of QK rows carrying the diagonal as the largest entry. A structured ablation that deletes every function of the head except those two preserves 76.9% of its effect on OpenWebText, and copy-suppression scores correlate head by head with anti-induction scores across GPT, Pythia and SoLU models. Description mode: implementational-functional.

## Readings of the claim

**Copy suppression.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Copy suppression is the main role of Negative Head L10H7 in GPT-2 Small, covering at least 76.9% of its direct effect | Mechanistically Supported | A dissociation from the backup-head mechanism the same ablation moves (I6), a known-strength positive control fixing the method's floor (M5), and correction for the top-5% slices the headline figures are computed on (M7) |
| The token the head suppresses is the source token itself | Disconfirmed | Nothing — tested and failed: in 42.00% of large-attention pairs the source token is suppressed without being the most suppressed, and a semantically related token takes first place in 90% of those (M6); the authors conclude the mechanism is better read as semantic copy suppression and keep the coarser name elsewhere (V3) |
| Copy suppression is what separates L10H7 from the other heads in its layers | Underdetermined | Either half of the instrument alone clears 50% recovered KL for many layer 9–11 heads, so the conjunction rather than the mechanism does the separating (I4, C4); the queryside direction perpendicular to the IO unembedding matters more than the parallel one and is left uncharacterized (I5) |
| Negative Heads do copy suppression across models | Proposed | The structured ablation is never run outside GPT-2 Small (E4); GPT-2 Medium recovers two of its three most negative heads, Pythia's copy suppression is weaker, and Stanford GPT-2 Small E's analogue attends to IO and S2 equally (M1) |

## Verdict

**Copy suppression.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — Materials present in Table 2, never assembled into a dissociation |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **C6** complementation validity — OV and QK split within one head; the across-head trans test is absent; **I10** rescue reversibility — Clean activations already in hand; the restore is never run; **I12** offset coupling — Pythia checkpoints already in use; no offset observed anywhere; **M5** sensitivity — Every control is a known-negative; no planted mechanism recovered |

## 36-criterion audit

**Copy suppression.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Three conjunctive conditions, 80% pass; weight predictions stated first |
| C2 | Structural plausibility | C | Read off the weights: 84.70% OV, 95.72% QK, before any ablation |
| C3 | Convergent validity | C | Four instruments differing in kind agree: weights, coding, ablation, OOD |
| C4 | Discriminant validity | PC | Three rivals separated; only the conjunction discriminates, not either half |
| C5 | Nomological validity | PC | Unifies negative name movers with anti-induction; formation unexplained |
| C6 | Complementation validity | U | OV and QK split within one head; the across-head trans test is absent |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Spread across data in 100 percentiles; no interval on any estimate |
| M2 | Baseline separation | PC | Mean ablation is the denominator; no null constructed for 76.9% itself |
| M3 | Stability | PC | Perturbations run and disclosed where they fail the idealization |
| M4 | Calibration | PC | KL's failure mode named; the effect is small and sign-unstable |
| M5 | Sensitivity | U | Every control is a known-negative; no planted mechanism recovered |
| M6 | Invariance | PC | All of OpenWebText under three metrics; no domain or frequency split |
| M7 | Selection correction | U | Selection disclosed at every step and corrected at none |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | Direct path localized cleanly; the absolute effect is a thousandth of loss |
| I2 | Sufficiency | PC | CSPA preserves 76.9%, but the figure spans 25–95% across metrics |
| I3 | Minimality | PC | Each half mild alone and the conjunction costs; nothing left to prune |
| I4 | Specificity | PC | CSPA discriminates L10H7; its components clear 50% for many heads |
| I5 | Rival mechanism exclusion | PC | Two rivals addressed; the perpendicular queryside driver left unknown |
| I6 | Double dissociation | U | Materials present in Table 2, never assembled into a dissociation |
| I7 | Confound control | PC | Tied embeddings and LayerNorm handled; data confounds untouched |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | PC | One interaction quantified, copy suppression on itself; no grid |
| I10 | Rescue reversibility | U | Clean activations already in hand; the restore is never run |
| I11 | Onset coupling | PC | No onset experiment at origin; run later in a list sorter ([Urdshals & Urdshals, 2025](https://arxiv.org/abs/2501.18666)) |
| I12 | Offset coupling | U | Pythia checkpoints already in use; no offset observed anywhere |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Four intervention forms, all activation edits on one head |
| E2 | Prompt generalization | C | Averaged over the pretraining distribution, not a template set |
| E3 | Cross-task generalization | C | IOI and random-token scores correlate across three model families |
| E4 | Cross-model recurrence | PC | Phenomenon replicates in three systems; the mechanism does not |
| E5 | Graded response | PC | A genuine dose-response at the query; every other intervention binary |
| E6 | Novel prediction | C | Anti-induction on random tokens predicted, then confirmed |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | Three numbered steps mapped to matrices, with the remainder named |
| V2 | Level-evidence match | PC | Matched on mechanism and coverage; when and how much it fires is not |
| V3 | Alternative level | PC | Appendix K favors semantic copy suppression over the paper's own name |
| V4 | Unlicensed labeling | C | Every label redescribes a measured object; no intentional vocabulary |
| V5 | Scope declaration | C | Scope declared where each claim is made, not collected at the end |
| **Total:** 9 Confirmed, 20 Partially confirmed, 7 Untested |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Copy Suppression — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-copy-suppression/).
