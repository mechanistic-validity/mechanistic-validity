---
title: "Tier 1: Proposed"
description: "A mechanistic claim has been stated with enough precision to be evaluated, but the evidence does not yet establish causal relevance."
---

# Verdict Tier 1: Proposed

| | |
|---|---|
| Tier | 1 of 5 (progressive) |
| What it means | A claim is falsifiable and measured, but not yet causally tested |
| Minimum evidence | Defined construct + falsifiable prediction + at least one measurement |
| Upgrade to Causally suggestive | At least one well-controlled causal experiment demonstrating necessity (I1) |
| Downgrade | N/A (lowest progressive tier) |

## What this tier establishes

A Proposed claim has crossed the threshold from speculation into science: the entity is named, its boundaries are stated, and at least one quantitative measurement exists. What it has *not* done is demonstrate causal relevance. The evidence is correlational, structural, or statistical — probing accuracy, cosine similarity, activation patterns — but no intervention has been performed.

This is not a criticism. Many important findings begin here, and many remain here because the relevant causal experiments are expensive or technically difficult. The tier exists to distinguish "well-posed but untested" from "causally established," preventing the conflation of correlation with mechanism.

A claim can remain at Proposed indefinitely without being wrong or uninteresting. What it cannot do is claim mechanistic status without causal evidence.

## Example verdict statement

> **Verdict:** Proposed — `[representational-statistical]`
> **Claim:** SAE feature $f_{42}$ in GPT-2 Small layer 8 represents noun-hood.
> **Met:** Defined construct (noun-hood), falsifiable prediction (feature activates selectively on nouns), measurement (cosine similarity = 0.82 with probing direction, top-20 contexts are 18/20 nouns)
> **Open:** I1 (necessity), I2 (sufficiency), I3 (specificity vs. word frequency)
> **Scope:** GPT-2 Small, residual stream layer 8, Pile-10k distribution

## Minimum reporting for this tier

- Name of the construct and its operational definition
- The specific measurement metric (probe architecture, SAE variant, similarity metric)
- At least one quantitative result with confidence interval or equivalent
- Statement of what causal experiment would move the claim to Tier 2

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| → Causally suggestive | At least one well-controlled ablation or patching experiment showing the mechanism is necessary (I1), with random-component control and named method |
| → Disconfirmed | The measurement is shown to be artifactual (e.g., the probe achieves equal accuracy on a random feature direction) |

## Characteristic occupants

- **SAE feature descriptions** ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)) — monosemantic features identified by activation patterns, prior to causal intervention
- **Linear probing claims** for Othello board state, syntactic number, or sentiment — high accuracy without interchange intervention
- **Weight-space structural analyses** — SVD-based role identification, OV/QK decompositions — without behavioral confirmation
- **Superposition geometry** ([Elhage et al., 2022](https://arxiv.org/abs/2209.10652)) in real models — geometric structure identified but causal role untested

## Key references

- Bricken et al. (2023). *Towards Monosemanticity.* [Transformer Circuits](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- Elhage et al. (2022). *Toy Models of Superposition.* [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Hill, A. B. (1965). *The Environment and Disease: Association or Causation?* [doi:10.1177/003591576505800503](https://doi.org/10.1177/003591576505800503)
- GRADE Working Group (2004). *Grading quality of evidence.* [doi:10.1136/bmj.328.7454.1490](https://doi.org/10.1136/bmj.328.7454.1490)

<details class="worked-example">
<summary>Worked example: SAE features at Tier 1</summary>

A sparse autoencoder trained on GPT-2 Small residual stream activations produces a feature $f_{42}$ whose decoder direction has high cosine similarity with the "is_noun" probing direction, whose top-activating contexts are predominantly nouns, and whose activation magnitude correlates with the model's confidence on syntactic tasks.

This is a Proposed claim. The evidence is correlational and structural: the feature *looks like* it represents noun-hood. But no intervention has been performed. We do not know whether the feature is *causally relevant* to noun-related computation (I1), whether it is *sufficient* (I2), or whether it is *specific* to noun-hood rather than a correlated property like word frequency (I3).

The claim is well-posed (falsifiable, with a defined construct and quantitative measurements). It simply hasn't been causally tested. Moving to Tier 2 requires ablating or patching the feature and demonstrating a noun-specific behavioral change.
</details>
