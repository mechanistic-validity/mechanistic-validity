---
title: "MI Safety Metrics"
description: "Safety-relevant construct validation: safety subspaces, dual mechanisms, adversarial ablation, and claim reliability."
---

# MI Safety Metrics

This page documents metrics that evaluate validity properties specific to safety-relevant mechanistic claims -- reliability, construct stability, and the relationship between alignment and interpretability.

---

## Safety and Alignment Evaluation

---

### S01 -- Dual Mechanism Discriminant Validity

**ID.** `S01.dual_mechanism` | **File.** `125_dual_mechanism.py`

**What it computes.** Decomposes representation directions for a behavioral construct into intrinsic (baseline) and prompted (instruction-following) mechanisms, testing whether they are genuinely distinct and whether each steers independently after removing the shared component.

**Evidence family.** Construct (C4 Discriminant Validity)

**Pass threshold.** `discriminant_separation > 0.2; independent_steering_effect > 0.1`

**Reference.** Han et al. (2025), NeurIPS 2025 / ICML 2026.

---

### S02 -- Adversarial Ablation Verification

**ID.** `S02.adversarial_ablation_gap` | **File.** `127_adversarial_ablation.py`

**What it computes.** Tests whether circuit heads that appear necessary under standard (mean) ablation remain necessary under adversarial ablation, where non-circuit heads are replaced with maximally disruptive values to detect false necessity.

**Evidence family.** Internal (I2 Sufficiency)

**Pass threshold.** `mean adversarial_gap < 0.3`

**Reference.** Sharkey et al. (2026), Goodfire / Apollo Research.

---

### S03 -- Safety Claim Reliability

**ID.** `S03.safety_claim_reliability` | **File.** `131_safety_claim_reliability.py`

**What it computes.** Tests whether circuit-based behavioral claims are consistent across different prompt templates and ablation calibration seeds, measuring reliability via coefficient of variation.

**Evidence family.** Measurement (M1 Reliability)

**Pass threshold.** `safety_reliability > 0.7`

**Reference.** Nanda (2025), AlignmentForum.

---

### S04 -- Assistant Axis Causal Stability

**ID.** `S04.assistant_axis` | **File.** `132_assistant_axis.py`

**What it computes.** Tests three validity properties of a dominant representational direction extracted via persona contrasts: causal sufficiency (suppression degrades behavior), reliability (direction stable across extraction runs), and discriminant validity (direction specific to target construct).

**Evidence family.** Internal (E2 Causal Sufficiency), Measurement (M1 Reliability), Construct (C4 Discriminant)

**Pass threshold.** `causal_deficit > 0.3; direction_stability > 0.8; discriminant_ratio > 2.0`

**Reference.** MATS + Anthropic Fellows (2026).

---

### S05 -- Safety Subspace Causal Validation

**ID.** `S05.safety_subspace` | **File.** `135_safety_subspace.py`

**What it computes.** Identifies safety-relevant directions by contrasting activations on safe vs. unsafe prompts, constructs a low-rank subspace via PCA, then tests causal sufficiency (linear classifier accuracy on projected activations) and necessity (refusal behavior change upon subspace ablation).

**Evidence family.** Internal (E2 Causal Sufficiency, I1 Necessity)

**Pass threshold.** `sufficiency > 0.6; ablation_deficit > 0.3`

**Reference.** NCSU (2025), arXiv:2512.23260.

---

### S06 -- Single-Shot Safety Recovery

**ID.** `S06.safety_one_shot` | **File.** `137_safety_one_shot.py`

**What it computes.** Tests whether the safety construct is compact enough that a single safety example's gradient aligns with the safety subspace, validating that safety alignment occupies a low-rank gradient structure.

**Evidence family.** Internal (E2 Causal Sufficiency)

**Pass threshold.** `gradient_alignment > 0.5; recovery_rate > 0.3`

**Reference.** Anonymous (2026), arXiv:2601.01887.

---

### S07 -- Alignment-Interpretability Trade-off

**ID.** `S07.alignment_interpretability` | **File.** `138_alignment_interpretability.py`

**What it computes.** Quantifies the trade-off between interpretability (activation consistency of per-unit responses) and representational richness (effective rank of activation matrices), testing whether they are discriminant constructs and how alignment affects the balance.

**Evidence family.** Construct (C4 Discriminant Validity)

**Pass threshold.** Diagnostic (no binary pass/fail). Reports whether alignment improves interpretability at the cost of richness.

**Reference.** Colin, Oliver, Serre (2026), ICLR 2026 Re-Align Workshop.

---

## Summary Table

| ID | Name | File | Evidence Family | Threshold |
|---|---|---|---|---|
| S01 | Dual Mechanism | `125_dual_mechanism.py` | Construct | `> 0.2` separation |
| S02 | Adversarial Ablation | `127_adversarial_ablation.py` | Internal | `< 0.3` |
| S03 | Safety Claim Reliability | `131_safety_claim_reliability.py` | Measurement | `> 0.7` |
| S04 | Assistant Axis | `132_assistant_axis.py` | Internal / Measurement | `> 0.3` deficit |
| S05 | Safety Subspace | `135_safety_subspace.py` | Internal | `> 0.6` suff. |
| S06 | Safety One-Shot | `137_safety_one_shot.py` | Internal | `> 0.5` align. |
| S07 | Alignment-Interpretability | `138_alignment_interpretability.py` | Construct | diagnostic |

---

## Relationship to Other Pages

For artifact quality metrics (SAE features, transcoders, crosscoders), see [MI Artifact Quality Metrics](/framework/lenses/core/mi-artifact-quality-metrics). For circuit faithfulness metrics (CLT attribution, cross-prompt consistency, minimality), see [MI Faithfulness Metrics](/framework/lenses/core/mi-faithfulness-metrics). For the overall MI lens framework, see [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability).

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
