---
title: "Successor Heads"
description: "Criterion audit of the successor heads claim, as submitted."
---

# Successor Heads

**Source.** *Successor Heads: Recurring, Interpretable Attention Heads In The Wild* (Gould et al., 2024).

**Description.** One attention head maps an ordinal token to its successor, and the mechanism is a product of four weight matrices — embedding, $MLP_0$, the head's OV circuit, unembedding — evaluated directly, with no activation from any prompt entering it. Mod-10 features recovered from $MLP_0$ by sparse autoencoders, by a linear probe and by single-neuron ablation carry the index the head increments. Evaluated on: GPT-2, Pythia and Llama-2 from 31M to 12B for behavioral recurrence, with the feature-level case study on Pythia-1.4B L12H0 and two appendix replications. Description mode: implementational-functional.

## Readings of the claim

**Successor heads.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A single head's effective OV circuit increments an ordinal token to its successor, and mod-10 features carry the index it acts on | Mechanistically Supported | Necessity measured rather than ranked (I1); a known-positive control (M5); correction for the head, feature and $$ selections (M7); and the feature account does not reproduce the greater-than bias the same weights show, which the authors state as incomplete (V2) |
| The head is specific to succession | Disconfirmed | Nothing — tested and failed: on natural text acronym and greater-than behavior take 23.8% and 18.9% of winning cases, and the paper reframes the head as interpretably polysemantic (I4) |
| The mod-10 mechanism recurs across architectures and sizes, as the abstract states | Underdetermined | Mechanistic recurrence is one case study plus two appendix replications against broad behavioral recurrence (E4); Llama-7B fails the held-out Roman-numeral task where the Pythia models do not (E4, M6), and putting the head on synthesised Roman-numeral representations drops top-1 accuracy to 0.125 (E3, E6) |
| The emergence of successor heads during training explains when the incrementation behavior appears | Underdetermined | Successor scores are tracked across checkpoints in two model families and the heads are seen to emerge, but nothing is plotted against a behavior or a loss curve, and the phase change the induction-head comparison predicts is absent (I11); no checkpoint interval shows the mechanism lapsing (I12) |

## Verdict

**Successor heads.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — Neither arm run, and the polysemantic head makes the converse hard |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **C6** complementation validity — Eight task classes scored individually and never ablated in pairs; **I10** rescue reversibility — Both ablations reversible and the clean run in hand; restore never run; **I12** offset coupling — The checkpoint data exists and is read in the forward direction only; **M5** sensitivity — Every control a known-negative; no planted circuit recovered |

## 36-criterion audit

**Successor heads.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | A fixed token list and a committed threshold; some models have no head |
| C2 | Structural plausibility | C | A product of four matrices tested directly, with no prompt entering it |
| C3 | Convergent validity | C | A probe at 0.708 cosine and single neurons, sharing nothing with the SAE |
| C4 | Discriminant validity | PC | Top head eight times the runner-up; no neighboring construct attempted |
| C5 | Nomological validity | PC | Weak universality imported, tested, and partly returned ([Chughtai et al., 2023](https://arxiv.org/abs/2302.03025)) |
| C6 | Complementation validity | U | Eight task classes scored individually and never ablated in pairs |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Modes over a hundred autoencoders; the central score has no interval |
| M2 | Baseline separation | PC | Cutting the head leaves under one percent and returns bigrams instead |
| M3 | Stability | PC | Three perturbations, one aimed at the paper's own premise |
| M4 | Calibration | PC | A proportion against a stated threshold; importance gives rank, not magnitude |
| M5 | Sensitivity | U | Every control a known-negative; no planted circuit recovered |
| M6 | Invariance | C | Three families over three orders of magnitude, failures kept in |
| M7 | Selection correction | U | Selection at three levels, disclosed at each and corrected at none |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | Winning head on all 64 prompts, which ranks the head without measuring it |
| I2 | Sufficiency | C | The OV circuit alone ranks the successor above every alternative |
| I3 | Minimality | PC | Minimal by construction; the droppable component contributes under a percent |
| I4 | Specificity | PC | Off-target function measured across five ablation regimes and found present |
| I5 | Rival mechanism exclusion | PC | Artifact and pre-head computation both excluded by built experiments |
| I6 | Double dissociation | U | Neither arm run, and the polysemantic head makes the converse hard |
| I7 | Confound control | PC | Tokenization handled by construction; letter collisions explained, not tested |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | PC | Indirect effect bounded under both ablation values; no feature-level grid |
| I10 | Rescue reversibility | U | Both ablations reversible and the clean run in hand; restore never run |
| I11 | Onset coupling | I | Emergence observed across checkpoints; coupling to behavior is not |
| I12 | Offset coupling | U | The checkpoint data exists and is read in the forward direction only |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | C | Five intervention forms spanning removal and addition, and they agree |
| E2 | Prompt generalization | PC | A constructed token list and 128 natural contexts, the harder one genuinely hard |
| E3 | Cross-task generalization | C | Eight tasks scored individually; Roman numerals decode though never trained |
| E4 | Cross-model recurrence | PC | Behavior recurs across three families; the mechanism is single-model |
| E5 | Graded response | PC | A two-sided dose criterion, and every other ablation all-or-nothing |
| E6 | Novel prediction | C | Roman numerals fitted out of every training set and decoded anyway |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | The level is fixed by the object: a product of four weight matrices |
| V2 | Level-evidence match | PC | Weights tested on weights; the recurrence claim rests mostly on behavior |
| V3 | Alternative level | PC | Two alternatives closed by experiment; the paper's own third left open |
| V4 | Unlicensed labeling | C | Each label defined in the sentence that introduces its measurement |
| V5 | Scope declaration | PC | Three limits declared beside the results they bound; one gap remains |
| **Total:** 10 Confirmed, 18 Partially confirmed, 1 Inconclusive, 7 Untested |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Successor Heads — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-successor-heads/).
