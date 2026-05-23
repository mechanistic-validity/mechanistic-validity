---
title: "MI Faithfulness Metrics"
description: "Circuit faithfulness metrics: graph fidelity, cross-prompt consistency, error fraction, and minimality."
---

# MI Faithfulness Metrics

This page documents metrics that evaluate whether proposed circuits or model-level mechanisms are faithful, necessary, sufficient, and robust. Includes CLT (Circuit Tracing) attribution graph validation and circuit-level faithfulness tests.

---

## CLT (Circuit Tracing) Evaluation

These metrics evaluate the validity of attribution graphs produced by Anthropic's Circuit Tracing / CLT framework, testing faithfulness, reliability, completeness, and sensitivity to pruning decisions.

---

### FH01 -- CLT Attribution Graph Faithfulness

**ID.** `FH01.clt_graph_faithfulness` | **File.** `EX29_clt_graph_faithfulness.py`

**What it computes.** Measures how well a pruned attribution graph (keeping only the top-k highest-attribution heads) preserves the model's behavior, computed as the ratio of pruned-model logit difference to full-model logit difference.

**Evidence family.** Internal (I2 Compositional Sufficiency, C2 Structural Plausibility)

**Pass threshold.** `graph_faithfulness > 0.8`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH02 -- CLT Cross-Prompt Consistency

**ID.** `FH02.clt_cross_prompt_consistency` | **File.** `EX30_clt_cross_prompt_consistency.py`

**What it computes.** Tests M1 reliability of circuit identification by computing pairwise Jaccard similarity of top-k causally important head sets across semantically equivalent prompts (paraphrases targeting the same answer).

**Evidence family.** Measurement (M1 Reliability, M2 Invariance)

**Pass threshold.** `cross_prompt_consistency > 0.4`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH03 -- CLT Error Node Fraction

**ID.** `FH03.clt_error_fraction` | **File.** `EX31_clt_error_fraction.py`

**What it computes.** Quantifies the replacement model gap by measuring what fraction of the model's behavior is NOT captured by individually attributable head contributions, approximating the "error node" concept from the CLT framework.

**Evidence family.** Measurement (M6 Construct Coverage), Internal (I5 Confound Control)

**Pass threshold.** `error_fraction < 0.2`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH04 -- CLT Missing Attention Quantification

**ID.** `FH04.clt_missing_attention` | **File.** `EX32_clt_missing_attention.py`

**What it computes.** Quantifies the systematic explanatory gap from CLT attribution graphs' exclusion of attention mechanisms (QK circuits) by measuring the fraction of a task's causal effect attributable to attention heads versus MLP layers.

**Evidence family.** Internal (I5 Confound Control, M6 Construct Coverage)

**Pass threshold.** `attention_gap_fraction < 0.3`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH05 -- CLT Graph Minimality Sensitivity

**ID.** `FH05.clt_minimality_sensitivity` | **File.** `EX33_clt_minimality_sensitivity.py`

**What it computes.** Tests whether the pruned attribution graph's size is stable across pruning thresholds, detecting fragile minimality where small threshold changes cause large jumps in graph size.

**Evidence family.** Construct (C4 Minimality), Measurement (M4 Sensitivity)

**Pass threshold.** `minimality_stability > 0.5`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

## Circuit-Level Faithfulness

These metrics evaluate whether proposed circuits are faithful, actionable, and robust at the circuit level.

---

### FH06 -- CoT Faithfulness

**ID.** `FH06.cot_faithfulness` | **File.** `108_cot_faithfulness.py`

**What it computes.** Detects unfaithful Chain-of-Thought reasoning by presenting paired contradictory comparison questions (A > B? and B > A?) and measuring the rate of logical contradictions, revealing post-hoc rationalization.

**Evidence family.** Internal (I2 Compositional Sufficiency)

**Pass threshold.** `contradiction_rate < 0.05`

**Reference.** Arcuschin et al. (2025), Google DeepMind, ICLR 2025.

---

### FH07 -- ModCirc Cross-Task Modularity

**ID.** `FH07.modcirc_modularity` | **File.** `122_modcirc.py`

**What it computes.** For each pair of tasks with defined circuits, evaluates whether a circuit discovered on task A maintains faithfulness when evaluated on task B's prompts, plus Jaccard overlap of circuit head sets.

**Evidence family.** Construct (C5 Convergent Validity), Internal (E2 Causal Sufficiency)

**Pass threshold.** `mean_cross_task_faithfulness > 0.4`

**Reference.** He et al. (2025), ICML 2025.

---

### FH08 -- Actionability Score

**ID.** `FH08.actionability` | **File.** `128_actionability.py`

**What it computes.** Measures whether circuit-level insights translate into actionable steering interventions, combining concreteness (norm ratio of circuit-derived vs. full-model steering vectors) with validation (fraction of prompts where circuit-derived steering shifts output toward the correct answer).

**Evidence family.** External (E1 Downstream Utility)

**Pass threshold.** `actionability > 0.1`

**Reference.** Orgad, Barez et al. (2026), ICML 2026.

---

### FH09 -- Surprise Reduction

**ID.** `FH09.surprise_reduction` | **File.** `129_surprise_reduction.py`

**What it computes.** Measures how much knowing the circuit reduces uncertainty about model outputs by comparing output entropy with and without the circuit's contribution (entropy increase upon circuit ablation divided by ablated entropy).

**Evidence family.** Measurement (M4 Construct Coverage)

**Pass threshold.** `surprise_reduction > 0.05`

**Reference.** ARC (2026); Hilton et al. (2026), AlignmentForum.

---

### FH10 -- Behavior vs. Capability Gap

**ID.** `FH10.behavior_capability_gap` | **File.** `130_behavior_capability_gap.py`

**What it computes.** Compares circuit faithfulness on prompts where the model gets the correct answer (capability) against prompts where the model errs (behavior), detecting circuits that only explain success but not failure.

**Evidence family.** External (E3 Generalizability)

**Pass threshold.** `behavior_gap < 0.3`

**Reference.** Steinhardt (2026), AlignmentForum.

---

## Summary Table

| ID | Name | File | Evidence Family | Threshold |
|---|---|---|---|---|
| FH01 | CLT Graph Faithfulness | `EX29_clt_graph_faithfulness.py` | Internal | `> 0.8` |
| FH02 | CLT Cross-Prompt Consistency | `EX30_clt_cross_prompt_consistency.py` | Measurement | `> 0.4` |
| FH03 | CLT Error Fraction | `EX31_clt_error_fraction.py` | Measurement | `< 0.2` |
| FH04 | CLT Missing Attention | `EX32_clt_missing_attention.py` | Internal | `< 0.3` |
| FH05 | CLT Minimality Sensitivity | `EX33_clt_minimality_sensitivity.py` | Construct | `> 0.5` |
| FH06 | CoT Faithfulness | `108_cot_faithfulness.py` | Internal | `< 0.05` |
| FH07 | ModCirc Modularity | `122_modcirc.py` | Construct | `> 0.4` |
| FH08 | Actionability | `128_actionability.py` | External | `> 0.1` |
| FH09 | Surprise Reduction | `129_surprise_reduction.py` | Measurement | `> 0.05` |
| FH10 | Behavior vs. Capability Gap | `130_behavior_capability_gap.py` | External | `< 0.3` |

---

## Relationship to Other Pages

For artifact quality metrics (SAE features, transcoders, crosscoders), see [MI Artifact Quality Metrics](/framework/lenses/core/mi-artifact-quality-metrics). For safety-relevant construct validation, see [MI Safety Metrics](/framework/lenses/core/mi-safety-metrics). For the overall MI lens framework, see [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability).

---

<!-- REDISTRIBUTION NOTE: The following metrics from mi-evaluation-metrics.md are cross-model or miscellaneous
     and belong on their natural evidence-family pages. To be redistributed in a follow-up:

     - EX12 (107 MAS) -> mi-representational-metrics.md (cross-model alignment)
     - EX14 (109 activation reasoning) -> mi-causal-metrics.md
     - EX16 (112 layer navigator) -> mi-representational-metrics.md
     - EX19 (121 functional localizer) -> mi-causal-metrics.md
     - EX21 (123 latent reasoning) -> mi-causal-metrics.md
     - EX22 (124 semantic hub) -> mi-representational-metrics.md
     - EX31 (130 NLA semantic validity) -> mi-behavioral-metrics.md
     - EX32 (131 weight-sparse circuit) -> mi-structural-metrics.md
     - EX26 (133 atlas alignment) -> mi-representational-metrics.md
     - EX27 (134 MOT alignment) -> mi-representational-metrics.md
     - EX6 (rule descriptions) -> mi-structural-metrics.md
-->
