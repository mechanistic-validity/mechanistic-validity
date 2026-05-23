---
title: "Tier 3: Mechanistically Supported"
description: "The mechanism is both necessary and sufficient, with specificity evidence and measurement reliability established."
---

# Verdict Tier 3: Mechanistically Supported

| | |
|---|---|
| Tier | 3 of 5 (progressive) |
| What it means | Necessity, sufficiency, and specificity all established under at least one method |
| Minimum evidence | I1 (necessity) + I2 (sufficiency) + I3 (specificity) + I4 (consistency) + M1 (reliability $\geq$ 0.7) |
| Upgrade to Triangulated | Multi-method convergence (C5) + external robustness (E5) + cross-procedure agreement (V2) |
| Downgrade to Causally suggestive | If specificity (I3) fails, or if sufficiency (I2) is shown to be method-conditional |

## What this tier establishes

A Mechanistically Supported claim has demonstrated that the mechanism is both necessary and sufficient for the target behavior, and that this effect is specific to the mechanism rather than reflecting general model degradation. The claim has moved from "this is involved" to "this is specifically and sufficiently responsible."

The key transition from Tier 2 is the conjunction of sufficiency and specificity. Either alone is insufficient: a mechanism can be sufficient but non-specific (a large enough chunk of any model reproduces any behavior), or specific but not sufficient (the component does exactly one thing, but other components also contribute). Mechanistically Supported requires both.

Sufficiency is method-dependent. The complement ablation method (zero, mean, resample) is part of the claim. Miller et al. (2024) demonstrated that IOI's recovery ratio $R \approx 0.87$ under mean ablation drops below 0.50 under resample ablation — the same circuit changes tier depending on the method declared.

## Example verdict statement

> **Verdict:** Mechanistically supported — `[implementational-topographic]`
> **Claim:** Heads L9H9, L9H6, L10H0 are necessary and sufficient for name-mover behavior in IOI.
> **Met:** I1 (necessity, $\Delta$ logit diff > 0.7 under zero + mean ablation), I2 (sufficiency, 87% recovery), I3 (SI = 14.2 vs. SVA task), I4 (consistent across 3 prompt templates), M1 ($\rho_{XX'} = 0.84$)
> **Open:** E5 (cross-model), C5 (multi-method convergence), V2 (cross-procedure agreement)
> **Scope:** GPT-2 Small, IOI task, Wang et al. prompt distribution

## Minimum reporting for this tier

- Sufficiency metric: recovery ratio $R = M(C) / M(\text{full})$ with stated threshold $\tau$
- Specificity metric: selectivity index or cross-task comparison showing selective effect
- Complement ablation method named as part of the claim
- Replication across at least two of: prompt templates, ablation methods, random seeds
- Bootstrap confidence interval on the principal metric
- At least one published reference point for calibration

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| → Triangulated | At least two methods with non-overlapping assumptions confirm the same mechanism (C5). External robustness across distributions or model sizes (E5). Cross-procedure agreement characterized quantitatively (V2). |
| → Causally suggestive (downgrade) | Specificity (I3) fails: the ablation equally impairs unrelated tasks. Or sufficiency (I2) is method-conditional: recovery drops below threshold under a more appropriate ablation method. |

## Characteristic occupants

- **Induction heads** ([Olsson et al., 2022](https://arxiv.org/abs/2209.11895)) — necessity, sufficiency, and specificity all demonstrated for in-context copying behavior
- **Greater-Than circuit** ([Hanna et al., 2023](https://arxiv.org/abs/2305.00586)) — strong structural plausibility with specificity evidence across related numerical tasks
- **Copy suppression heads** ([McDougall et al., 2023](https://arxiv.org/abs/2310.04625)) — unusually clean specificity: the heads suppress repeated tokens specifically, with minimal off-target effects

## Key references

- Olsson et al. (2022). *In-context Learning and Induction Heads.* [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hanna et al. (2023). *How does GPT-2 compute greater-than?* [arXiv:2305.00586](https://arxiv.org/abs/2305.00586)
- McDougall et al. (2023). *Copy Suppression.* [arXiv:2310.04625](https://arxiv.org/abs/2310.04625)
- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
- Wang et al. (2022). *Interpretability in the Wild.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
