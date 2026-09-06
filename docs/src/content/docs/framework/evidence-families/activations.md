---
title: "Activations"
description: "Evidence from the model's internal state during a forward pass — what the network does while it runs."
---

# Activations

Activation-based evidence comes from the model's internal state during computation: residual-stream vectors, attention patterns, MLP outputs, and any derived quantity (probes, SAE features, logit lens). This is the most heavily populated evidence family in current mechanistic interpretability — most published circuit-discovery and feature-analysis work lives here.

## Observational

Methods that read internal state without modifying it:

- **Probing** — training a classifier on intermediate representations to detect whether information is linearly (or nonlinearly) decodable at a given layer. Establishes that information is *present*, not that it is *used*.
- **CKA / representational similarity** — centered kernel alignment and related measures compare representation geometry across layers, models, or conditions.
- **Mutual information / PID** — information-theoretic measures quantify how much of the input or output is captured by an intermediate representation. Partial information decomposition separates redundant, unique, and synergistic contributions.
- **Logit lens / tuned lens** — projecting intermediate representations through the unembedding matrix to read what the model "would predict" at each layer.
- **SAE feature activations** — running a sparse autoencoder on activations and reading which features fire. Observational when the features are read; interventional when they are clamped or steered.
- **Attention pattern analysis** — reading which positions attend to which, though attention weights alone are not causal evidence of information flow.

### The probe wars

The question "do probes find features the model actually uses?" is a construct-level question ([C4 Discriminant validity](/mechanistic-validity/framework/criteria/construct/discriminant-validity/)). A probe that achieves high accuracy on a classification task demonstrates that the information is *linearly accessible* in the representation. Whether the model's own computation *accesses* that information requires interventional evidence — typically activation patching or DAS-IIA. The observational finding motivates the interventional test; neither replaces the other.

## Interventional

Methods that modify internal state and observe consequences:

- **Activation patching** — replacing one activation with another (from a different input, a mean, or zero) and measuring the effect on output. The workhorse of causal circuit discovery.
- **DAS-IIA (Distributed Alignment Search – Interchange Intervention Accuracy)** — learning a linear subspace and intervening on the projection of activations onto that subspace.
- **Ablation** — zeroing, mean-ablating, or resampling activations at specific sites. Establishes necessity when behavior degrades; combined with sufficiency tests, supports causal claims.
- **Steering** — adding a direction vector to activations at inference time to shift model behavior. Establishes that a direction has a causal effect, though the mechanism by which the shift works may be unclear.
- **SAE feature clamping** — setting specific SAE features to fixed values and observing behavioral effects.

### The activation-patching ecosystem

Most published causal evidence in mechanistic interpretability comes from variants of activation patching: path patching, causal tracing, causal scrubbing, ACDC, EAP, and attribution patching all belong to this cell. Different variants make different assumptions about what constitutes a "clean" counterfactual, how to handle indirect effects, and what granularity to patch at. Disagreements between methods ([E1 Intervention reach](/mechanistic-validity/framework/criteria/external/intervention-reach/)) are common and informative.

## Relevant criteria

Activation evidence is most directly relevant to:

| Criterion | How activation evidence bears on it |
|---|---|
| [I1 Necessity](/mechanistic-validity/framework/criteria/internal/necessity/) | Ablation → behavior degrades |
| [I2 Sufficiency](/mechanistic-validity/framework/criteria/internal/sufficiency/) | Activation patching from clean → corrupt restores behavior |
| [I4 Specificity](/mechanistic-validity/framework/criteria/internal/specificity/) | Same intervention on matched control task → no effect |
| [E1 Intervention reach](/mechanistic-validity/framework/criteria/external/intervention-reach/) | Multiple activation-level methods agree |
| [C3 Convergent validity](/mechanistic-validity/framework/criteria/construct/convergent-validity/) | Activation evidence converges with weight or behavioral evidence |
