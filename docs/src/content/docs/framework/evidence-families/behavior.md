---
title: "Behavior"
description: "Evidence from the model's input-output relation — what the network does from the outside."
---

# Behavior

Behavioral evidence comes from the model's input-output relation: what it produces given what it receives, without reference to how it produces it. This is the evidence family closest to classical machine learning evaluation — accuracy, loss, logit distributions — but in a mechanistic validity context it serves a different role: behavioral measurements constrain and calibrate internal claims.

## Observational

Methods that measure input-output relations without modifying the model:

- **Logit difference** — the difference in log-probability between a correct and incorrect completion. The standard dependent variable in circuit-discovery work (e.g., IOI logit diff).
- **Task accuracy** — fraction of inputs on which the model produces the correct output. Coarser than logit diff but more interpretable.
- **Behavioral profiling** — systematic mapping of which inputs the model handles correctly and which it fails on, organized by structural features of the input (length, syntactic complexity, semantic category).
- **Distribution analysis** — full output distributions, entropy, calibration curves, and tail behavior.
- **Minimal pairs** — pairs of inputs differing in one feature, used to isolate which input properties drive which outputs.

### Behavioral evidence as constraint

Behavioral observations constrain mechanistic claims in two directions. A mechanism that does not reproduce the observed behavior fails [I2 Sufficiency](/mechanistic-validity/framework/criteria/internal/sufficiency/). A mechanism that reproduces the observed behavior but also predicts behavior that the model does not exhibit fails [I4 Specificity](/mechanistic-validity/framework/criteria/internal/specificity/). Both constraints are available purely from behavioral measurement, without opening the model.

### Necessary, never sufficient

Behavioral evidence alone cannot establish which internal mechanism produces an observed input-output relation. Two components, or two entirely different networks, can implement an identical logit-difference curve, task-accuracy score, or output distribution through different computations — this is the underdetermination of mechanism by behavior. Matching the observed behavior is a necessary condition for any mechanistic claim: a proposed mechanism that fails to reproduce it is immediately disconfirmed. Matching behavior is never a sufficient condition: the same match is consistent with every mechanism capable of producing it. Distinguishing among them requires internal evidence — [activations](/mechanistic-validity/framework/evidence-families/activations/) or [weights](/mechanistic-validity/framework/evidence-families/weights/) — that behavior alone cannot supply.

## Interventional

Methods that perturb inputs or the environment and measure behavioral changes:

- **Prompt perturbation** — systematically varying input features (word order, entity names, syntactic structure) and measuring output changes. Establishes which input features the model's behavior is sensitive to.
- **Dose-response** — graded perturbations producing graded behavioral changes. Linearity, threshold effects, and saturation are all informative about mechanism type.
- **KL divergence under ablation** — measuring the full distributional shift in outputs when an internal component is ablated, rather than just the logit diff on the correct answer.
- **Adversarial inputs** — inputs designed to break the claimed mechanism, testing whether the behavior is robust or fragile.
- **Counterfactual prompts** — inputs where the mechanism should fire but the expected behavior should differ, testing whether the mechanism's predictions are specific.

### Behavioral evidence and validity types

Behavioral evidence bears most directly on [External validity](/mechanistic-validity/framework/validity-types/external/) — does the mechanism explain behavior across diverse inputs? — and on [Measurement validity](/mechanistic-validity/framework/validity-types/measurement/) — is the behavioral metric sensitive enough to detect the effects the mechanism predicts?

A common failure mode is reporting behavioral evidence only for the inputs used to discover the mechanism. [E2 Prompt generalization](/mechanistic-validity/framework/criteria/external/prompt-generalization/) requires testing on held-out inputs that were not part of the discovery set.

## Relevant criteria

Behavioral evidence is most directly relevant to:

| Criterion | How behavioral evidence bears on it |
|---|---|
| [I2 Sufficiency](/mechanistic-validity/framework/criteria/internal/sufficiency/) | Does the mechanism reproduce the observed behavior? |
| [I4 Specificity](/mechanistic-validity/framework/criteria/internal/specificity/) | Does the mechanism predict behavior only on target tasks? |
| [E2 Prompt generalization](/mechanistic-validity/framework/criteria/external/prompt-generalization/) | Does the behavior hold on diverse, held-out prompts? |
| [E5 Graded response](/mechanistic-validity/framework/criteria/external/graded-response/) | Does partial perturbation produce partial behavioral change? |
| [M5 Sensitivity](/mechanistic-validity/framework/criteria/measurement/sensitivity/) | Can the behavioral metric detect known-true effects? |
