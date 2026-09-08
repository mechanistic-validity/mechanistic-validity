---
title: "Docstring Circuit"
description: "Criterion audit of the docstring circuit claim, as submitted."
---

# Docstring Circuit

**Source.** *A circuit for Python docstrings in a 4-layer attention-only transformer* ([Heimersheim & Janiak, 2023](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)).

**Description.** Eight attention heads in a 4-layer attention-only transformer compose across three levels to predict the next argument name in a Python docstring: fuzzy previous-token heads and a positional head set up an induction step, and argument movers carry the name from the definition line to the output position. Resample ablation under three corruption types establishes which heads matter, and resample-ablating everything outside the eight leaves 42% success against the full model's 56%. Evaluated on: one released toy model, 50 prompts per experiment, on prompts built to disable the line-number and repeat-inhibition algorithms at a measured cost ($$75% to 56%). Description mode: implementational-functional.

## Readings of the claim

**Docstring circuit.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Eight named heads compose to predict the next argument name in a Python docstring in this 4-layer model | Causally Suggestive | Specificity on any non-docstring behavior (I4), a leave-one-out over the eight heads (I3), a matched random head set to show the 42% belongs to these heads (M2), and dispersion on any reported number (M1) |
| The circuit is the model's implementation of Docstring Induction, an algorithm | Underdetermined | Apportionment between induction and the line-number algorithm, which the authors judge co-implemented and suppress by prompt design rather than refute (I5, V3); the algorithm is inferred from attention patterns rather than tested against its rivals (V2) |
| The circuit behaves like the model on the task it was found on | Disconfirmed | Nothing — tested and failed (E2), and on post-origin evidence alone; the post's own remark that Google-style docstrings give similar performance points the other way and carries no number (M3) |
| These heads, or the head types they instantiate, serve other tasks or recur in other models | Insufficient | Cross-task reuse is never asked (E3), cross-model transfer is named as future work with no model, head set or number reported (E4), and the circuit is never compared with one for a neighboring task (C4) |

## Verdict

**Docstring circuit.** Verdict: **Causally Suggestive**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked. **I4** specificity — Measured on one behavior; head 1.4 alone is run off-distribution |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Mechanistically Supported. **C4** discriminant validity — The circuit is never compared with one for a neighboring task; **E2** prompt generalization — Circuit and model diverge on a subclass of benign inputs ([uit de Bos & Garriga-Alonso, 2024](https://arxiv.org/abs/2407.15166)); **E4** cross-model recurrence — Transfer is named as future work and no result is reported; **I6** double dissociation — No second capability is shown intact while the docstring one fails |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Mechanistically Supported. **C6** complementation validity — Three named levels, each ablated alone and never in combination; **E3** cross-task generalization — Whether the components do anything for another task is never asked; **E4** cross-model recurrence — Transfer is named as future work and no result is reported; **I10** rescue reversibility — The restoring direction was run and not shown, by the author's own account; **I11** onset coupling — One pre-trained model off the shelf; no training trajectory in evidence; **I12** offset coupling — Needs the training sequence that onset coupling also lacks; **M1** reliability — Fifty prompts behind every plot; dispersion is never stated; **M3** stability — Head membership is varied upward only; nothing re-measures the claim; **M5** sensitivity — Nothing with a known answer is fed in to see whether it comes back; **M6** invariance — No conclusion is re-derived under a changed condition and shown to hold |

## 36-criterion audit

**Docstring circuit.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Each head assignment is a signed prediction that could have come out flat |
| C2 | Structural plausibility | PC | Argued at the activation level: patterns locate, patching confirms |
| C3 | Convergent validity | PC | One instrument carries it; patching and composition share a primitive |
| C4 | Discriminant validity | U | The circuit is never compared with one for a neighboring task |
| C5 | Nomological validity | PC | Placed inside an existing theory of head types, which does work |
| C6 | Complementation validity | U | Three named levels, each ablated alone and never in combination |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | U | Fifty prompts behind every plot; dispersion is never stated |
| M2 | Baseline separation | PC | 42% against a 17% chance level, with two further reference points |
| M3 | Stability | U | Head membership is varied upward only; nothing re-measures the claim |
| M4 | Calibration | PC | Logit difference against the highest wrong answer, recomputed throughout |
| M5 | Sensitivity | U | Nothing with a known answer is fed in to see whether it comes back |
| M6 | Invariance | U | No conclusion is re-derived under a changed condition and shown to hold |
| M7 | Selection correction | U | Components are chosen from the strongest cells of a patching grid |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | Every head is knocked out by resampling and the tests bite |
| I2 | Sufficiency | PC | Everything outside the eight heads is ablated and reported straight |
| I3 | Minimality | PC | Heads are added rather than removed; one member is undercut anyway |
| I4 | Specificity | U | Measured on one behavior; head 1.4 alone is run off-distribution |
| I5 | Rival mechanism exclusion | PC | Rivals are named first and the prompt set disables two of them |
| I6 | Double dissociation | U | No second capability is shown intact while the docstring one fails |
| I7 | Confound control | PC | Line counting and repetition are designed out rather than argued away |
| I8 | Confounding sensitivity | U | Nothing asks how strong an unmeasured confounder would need to be |
| I9 | Epistatic interaction | PC | The composition score is a two-head quantity, so interaction is the subject |
| I10 | Rescue reversibility | U | The restoring direction was run and not shown, by the author's own account |
| I11 | Onset coupling | U | One pre-trained model off the shelf; no training trajectory in evidence |
| I12 | Offset coupling | U | Needs the training sequence that onset coupling also lacks |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Four do-operators beyond resampling, and they agree with it |
| E2 | Prompt generalization | D | Circuit and model diverge on a subclass of benign inputs ([uit de Bos & Garriga-Alonso, 2024](https://arxiv.org/abs/2407.15166)) |
| E3 | Cross-task generalization | U | Whether the components do anything for another task is never asked |
| E4 | Cross-model recurrence | U | Transfer is named as future work and no result is reported |
| E5 | Graded response | PC | Circuit size tracks performance monotonically across two axes |
| E6 | Novel prediction | C | A mechanism proposed for one reason predicts something else, and it holds |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | A circuit is defined as a subset of components before any result |
| V2 | Level-evidence match | PC | Matches the lower declared level and thins at the upper one |
| V3 | Alternative level | PC | Line counting is raised as the simpler description, then accepted |
| V4 | Unlicensed labeling | C | Each label is introduced where its operation has just been measured |
| V5 | Scope declaration | C | The limit is declared before the argument, not conceded after it |
| **Total:** 5 Confirmed, 15 Partially confirmed, 15 Untested, 1 Disconfirmed |   |   |   |
| **Verdict: Causally Suggestive** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Docstring Circuit — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-docstring/).
