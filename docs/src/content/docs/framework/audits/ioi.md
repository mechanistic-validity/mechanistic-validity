---
title: "IOI Circuit"
description: "Criterion audit of the ioi circuit claim, as submitted."
---

# IOI Circuit

**Source.** *Interpretability in the Wild* ([Wang et al., 2023](https://arxiv.org/abs/2211.00593)).

**Description.** Path patching over GPT-2 small identifies 26 attention heads in 7 classes—Previous Token, Duplicate Token, Induction, S-Inhibition, Name Mover, Negative Name Mover, Backup Name Mover—for sentences of the form “When Mary and John went to the store, John gave a drink to \_\_\_”. Name Mover roles are supported at the weight level (OV copy score $>$95%), the rest at the activation level. Component-level redundancy (Backup Name Movers) is recorded under minimality (I3) rather than necessity: a circuit whose parts compensate for one another is still necessary as a circuit. Description mode: implementational-functional.

## Readings of the claim

**IOI circuit.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A circuit comprising these 26 heads performs IOI in GPT-2 small on the origin templates | Causally Suggestive | Independent evidence (C3), separation from other circuits (I6), and robustness to method and prompt (M3, M6) |
| These 26 heads are *the* mechanism, to the exclusion of others | Underdetermined | A test that discriminates among rival head sets; naïve and greedy-search circuits reach comparable faithfulness (I5) |
| The mechanism generalizes across prompts and models | Disconfirmed | Nothing — tested and failed: model and circuit diverge at $10^6$ prompt pairs (E2), and faithfulness varies across the origin's own templates (M6) |
| The seven classes compose in the stated order, each performing the role its name asserts | Underdetermined | Weight-level accounts for two of seven classes (C2); role semantics currently exceed the evidence (V2) |

## Verdict

**IOI circuit.** Verdict: **Causally Suggestive**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked. **E1** intervention reach — One ablation value at origin, three granularities; methods disagree; **I4** specificity — Head overlap and task effect give opposite verdicts ([Merullo et al., 2024](https://arxiv.org/abs/2310.08744)) |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Mechanistically Supported. **C4** discriminant validity — 78% overlap ([Merullo et al., 2024](https://arxiv.org/abs/2310.08744)); faithfulness opposite ([Hanna et al., 2024](https://arxiv.org/abs/2403.17806)); **E2** prompt generalization — At $10^6$ clean/corrupted pairs, model and circuit diverge ([uit de Bos & Garriga-Alonso, 2024](https://arxiv.org/abs/2407.15166)); **I5** rival mechanism exclusion — Greedy search finds knockout sets carrying 87% of the behavior; **I6** double dissociation — No IOI test; the nearest design finds none across tasks ([Li & Subramani, 2026](https://arxiv.org/abs/2605.08348)) |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Mechanistically Supported. **M5** sensitivity — Unattempted; no planted circuit of known extent |

## 36-criterion audit

**IOI circuit.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | 26 heads named with roles; criteria stated as failable quantities |
| C2 | Structural plausibility | PC | Weight-level for two of seven classes; three omissions the authors name |
| C3 | Convergent validity | PC | Five evidence types; only three share the mean-ablation primitive ([Conmy et al., 2023](https://arxiv.org/abs/2304.14997)) |
| C4 | Discriminant validity | I | 78% overlap ([Merullo et al., 2024](https://arxiv.org/abs/2310.08744)); faithfulness opposite ([Hanna et al., 2024](https://arxiv.org/abs/2403.17806)) |
| C5 | Nomological validity | PC | Induction-head theory applied operationally; no formal derivation |
| C6 | Complementation validity | PC | Seven classes knocked out singly; two later crossed ([Gong et al., 2026](https://arxiv.org/abs/2607.01940)) |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | I | SDs plotted for attention probabilities; none on faithfulness |
| M2 | Baseline separation | PC | Naïve circuit 0.1 against the full 0.46; null supplied later (Shi et al., 2024) |
| M3 | Stability | D | Six choices move the same quantity below 0% to over 100% ([Miller et al., 2024](https://arxiv.org/abs/2407.08734)) |
| M4 | Calibration | PC | Logit difference has sign, null and baseline; faithfulness does not |
| M5 | Sensitivity | U | Unattempted; no planted circuit of known extent |
| M6 | Invariance | D | Faithfulness differs across the origin's own templates ([Miller et al., 2024](https://arxiv.org/abs/2407.08734)) |
| M7 | Selection correction | U | N is unstated for the post-knockout sweep, the selection step itself |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | Name Mover knockout costs only 5%; the paper says so itself |
| I2 | Sufficiency | PC | Mean-ablating outside the circuit leaves 87% of the logit difference |
| I3 | Minimality | I | Every node clears 1%; contested under other ablations (Li & Janson, 2024) |
| I4 | Specificity | I | Head overlap and task effect give opposite verdicts ([Merullo et al., 2024](https://arxiv.org/abs/2310.08744)) |
| I5 | Rival mechanism exclusion | I | Greedy search finds knockout sets carrying 87% of the behavior |
| I6 | Double dissociation | U | No IOI test; the nearest design finds none across tasks ([Li & Subramani, 2026](https://arxiv.org/abs/2605.08348)) |
| I7 | Confound control | PC | Sequence length controlled twice; name frequency is not |
| I8 | Confounding sensitivity | U | Unattempted; no E-value or sensitivity analysis |
| I9 | Epistatic interaction | PC | Backup Name Movers are a discovered non-additivity |
| I10 | Rescue reversibility | PC | Direction reversed at origin; corrupt-then-restore run later |
| I11 | Onset coupling | N/A | Untestable in GPT-2 small, a released checkpoint with no training run |
| I12 | Offset coupling | PC | A Pythia name-mover head loses the behavior late in training (Tigges et al., 2024) |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | I | One ablation value at origin, three granularities; methods disagree |
| E2 | Prompt generalization | D | At $10^6$ clean/corrupted pairs, model and circuit diverge ([uit de Bos & Garriga-Alonso, 2024](https://arxiv.org/abs/2407.15166)) |
| E3 | Cross-task generalization | PC | Repeated-random-token at origin; transfer shown later ([Merullo et al., 2024](https://arxiv.org/abs/2310.08744)) |
| E4 | Cross-model recurrence | PC | GPT-2 medium at origin; reproduced across Pythia (Tigges et al., 2024; Mueller et al., 2025) |
| E5 | Graded response | PC | Graded in circuit extent, never in intervention strength |
| E6 | Novel prediction | C | Duplicated-name prediction, with a matched control |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | Circuits formally defined in §2.1; description level not declared |
| V2 | Level-evidence match | PC | Topography and information flow supported; role semantics exceed it |
| V3 | Alternative level | I | The naïve circuit is simpler and not clearly worse |
| V4 | Unlicensed labeling | PC | “Hedge” for the Negative Name Movers, marked by the authors as speculation |
| V5 | Scope declaration | C | Limits declared in the abstract, §1, §3 and §5 |
| **Total:** 3 Confirmed, 18 Partially confirmed, 7 Inconclusive, 4 Untested, 3 Disconfirmed, 1 Not applicable |   |   |   |
| **Verdict: Causally Suggestive** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [IOI Circuit — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-ioi/).
