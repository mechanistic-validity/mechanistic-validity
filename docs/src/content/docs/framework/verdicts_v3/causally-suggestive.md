---
title: "Tier 2: Causally Suggestive"
description: "There is causal evidence that the claimed mechanism is involved in the behavior — necessity but not yet sufficiency, specificity, or convergence."
---

# Verdict Tier 2: Causally Suggestive

| | |
|---|---|
| Tier | 2 of 5 (progressive) |
| What it means | Necessity established — removing the mechanism changes behavior — but sufficiency and specificity remain open |
| Minimum evidence | I1 (necessity) + E4 (effect magnitude) + V1 (level declaration) |
| Upgrade to Mechanistically supported | Sufficiency (I2) + Specificity (I3) + Consistency (I4) + Measurement reliability (M1) |
| Downgrade to Proposed | If the causal effect is shown to be an artifact of the ablation method or indistinguishable from random controls |

## What this tier establishes

A Causally Suggestive claim has crossed the most important threshold in mechanistic interpretability: from correlation to causation. At least one intervention (ablation, patching, or editing) has demonstrated that the claimed mechanism is *necessary* for the target behavior. Removing it changes the output in a way that exceeds random-component baselines.

What this tier does *not* establish is whether the mechanism is the whole story. Most circuits at this tier demonstrate necessity without sufficiency — ablating them hurts, but the circuit alone does not reproduce the behavior. Specificity is also typically untested: the ablation hurts the target behavior, but does it also degrade unrelated behaviors? If so, the component may be a general bottleneck rather than a specific mechanism.

This is where the majority of published mechanistic interpretability findings currently reside.

## Example verdict statement

> **Verdict:** Causally suggestive — `[implementational-topographic]`
> **Claim:** The IOI circuit (26 heads across layers 0-11) is necessary for indirect object identification in GPT-2 Small.
> **Met:** I1 (mean ablation of circuit heads reduces logit diff by 0.73, vs. 0.12 for size-matched random set), E4 (absolute effect = 2.1 logits), V1 (claim stated at head level)
> **Open:** I2 (sufficiency under resample ablation), I3 (specificity vs. general language tasks), C5 (multi-method convergence)
> **Scope:** GPT-2 Small, IOI task, ABBA template distribution

## Minimum reporting for this tier

- Ablation method named explicitly (zero, mean, resample, activation patching)
- Effect size on target behavior with confidence interval
- Random-component control (same number of components, randomly selected) with its effect size
- Baseline metric value (full model performance on the target behavior)
- Statement of whether sufficiency or specificity has been tested

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| → Mechanistically supported | Sufficiency (I2): circuit alone recovers target behavior. Specificity (I3): selectivity index SI > 10 or meaningful task separation. Consistency (I4): replication across templates/methods/seeds. Reliability (M1): bootstrap $\rho_{XX'} \geq 0.7$ |
| → Proposed (downgrade) | The causal effect disappears under a more appropriate ablation method (e.g., mean → resample), or the random-component control produces equal effect |
| → Underdetermined | Multiple non-overlapping circuits produce equivalent necessity effects and cannot be distinguished |

## Characteristic occupants

- **IOI circuit** ([Wang et al., 2022](https://arxiv.org/abs/2211.00593)) under its original mean-ablation evaluation — necessity demonstrated, but sufficiency challenged by [Miller et al. (2024)](https://arxiv.org/abs/2407.08734)
- **Knowledge neuron editing** ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)) — ROME demonstrates necessity of specific MLPs for factual recall, sufficiency partially established
- **Individual head ablation studies** — most activation-patching-based claims in the literature
- **ACDC-discovered circuits** ([Conmy et al., 2023](https://arxiv.org/abs/2304.14997)) — automated discovery with necessity verification but limited specificity testing

## Key references

- Wang et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT.* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- Conmy et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
