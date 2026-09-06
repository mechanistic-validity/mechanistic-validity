---
title: "Evidence Families"
description: "Four sources of signal crossed with two access modes — a 4×2 matrix classifying evidence by where it comes from and how it is obtained."
---

# Evidence Families

An evidence family classifies a metric's output by its **source** (where the signal comes from) and its **access mode** (whether the analyst observes or intervenes). The classification sits between metrics and criteria in the evaluation pipeline: metrics produce raw measurements, evidence families describe what *type* of epistemic content those measurements carry, and criteria then evaluate whether that content meets the bar for a validity claim.

## The 4×2 matrix

Evidence divides along two axes. The **source** is the part of the system being measured: its persistent organization (weights), its runtime state (activations), its input-output relation (behavior), or its developmental history (training). The **access mode** is whether the analyst reads the system passively (observational) or perturbs it under a do-operator (interventional).

| | Observational | Interventional |
|---|---|---|
| **[Weights](/mechanistic-validity/framework/evidence-families/weights/)** | SVD, effective rank, OV/QK composition, MDL | Circuit transplant, weight editing, weight knockout |
| **[Activations](/mechanistic-validity/framework/evidence-families/activations/)** | Probing, CKA, mutual information, PID, logit lens | Activation patching, DAS-IIA, ablation, steering |
| **[Behavior](/mechanistic-validity/framework/evidence-families/behavior/)** | Logit diff, behavioral profiling, task accuracy | Prompt perturbation, dose-response, KL under ablation |
| **[Training](/mechanistic-validity/framework/evidence-families/training/)** | Loss curves, checkpoint comparison, phase transitions | Ablate-then-retrain, fine-tuning, curriculum manipulation |

Most published mechanistic interpretability work lives in one cell: **Activations × Interventional**. Activation patching, DAS-IIA, ablation, and steering are all interventional measurements of internal state. The framework does not privilege this cell — it notes that a claim supported by evidence from multiple cells is stronger than one supported by multiple methods from the same cell, because different cells have structurally different failure modes.

## Why this replaces the old taxonomy

An earlier version of this framework used six evidence families (Causal, Structural, Representational, Behavioral, Information-theoretic, Measurement-theoretic). That taxonomy mixed source and method type: "Causal" spanned activations and behavior, "Information-theoretic" was a method applicable to any source, and "Measurement-theoretic" was a meta-level calibration applicable to all eight cells. The 4×2 matrix separates the axes cleanly.

The old families map approximately:
- **Structural** → Weights (observational)
- **Causal** → Activations (interventional) + Behavior (interventional)
- **Representational** → Activations (observational) + Activations (interventional)
- **Behavioral** → Behavior (both modes)
- **Information-theoretic** → cross-cutting method applicable to any source
- **Measurement-theoretic** → cross-cutting calibrations (bootstrap stability, seed variance, baseline separation) that apply to all 8 cells

## How evidence families enable convergent validity

The primary function of the family classification is to structure convergent validity assessments:

1. **How many cells support the claim?** A claim supported by three cells is stronger than one supported by three metrics from the same cell.
2. **Which cells are represented?** Weights × Observational combined with Activations × Interventional is powerful because their failure modes are complementary: runtime compensation cannot confound weight-space analysis, and weight-space structure that exists but is never activated is caught by interventional testing.
3. **Are there cells that *should* support the claim but do not?** Absence of expected evidence from a cell is informative.

## Relationship to the evaluation pipeline

Evidence families are step 2 of the [seven-step evaluation pipeline](/mechanistic-validity/framework/). After declaring the [description mode](/mechanistic-validity/framework/description-modes/) (step 1), you identify which evidence families are relevant to the claim. This determines which [metrics to run](/mechanistic-validity/framework/metrics/) in step 3, which in turn produces the measurements scored against the 36 [criteria](/mechanistic-validity/framework/criteria/) in steps 4–5.
