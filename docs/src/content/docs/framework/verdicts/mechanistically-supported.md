---
title: "Tier 3: Mechanistically Supported"
description: "The mechanism is both necessary and sufficient, with a specificity test conducted and the result reproduced across two intervention families."
---

# Verdict Tier 3: Mechanistically Supported

| | |
|---|---|
| Tier | 3 of 5 (progressive) |
| What it means | Necessity, sufficiency, and specificity all established under at least one method |
| Requires | Internal: I2 (sufficiency), I4 (specificity). External: E1 (intervention reach). Plus everything Causally Suggestive requires |
| Upgrade to Triangulated | C3--C4 (convergent and discriminant validity) + I5--I7 (rival exclusion, double dissociation, confound control) + E2, E4 (prompt and cross-model generalization) |
| Downgrade to Causally suggestive | If specificity (I4) fails, or if sufficiency (I2) is shown to be method-conditional |

## What this tier establishes

A Mechanistically Supported claim has demonstrated that the mechanism is both necessary and sufficient for the target behavior, and that this effect is specific to the mechanism rather than reflecting general model degradation. The claim has moved from "this is involved" to "this is specifically and sufficiently responsible."

The key transition from Tier 2 is the conjunction of sufficiency and specificity. Either alone is insufficient: a mechanism can be sufficient but non-specific (a large enough chunk of any model reproduces any behavior), or specific but not sufficient (the component does exactly one thing, but other components also contribute). Mechanistically Supported requires both.

Sufficiency is method-dependent. The complement ablation method (zero, mean, resample) is part of the claim. Miller et al. (2024) demonstrated that IOI's recovery ratio $R \approx 0.87$ under mean ablation drops below 0.50 under resample ablation — the same circuit changes tier depending on the method declared.

## Example verdict statement

> **Verdict:** Mechanistically supported — `[implementational-topographic]`
> **Claim:** Heads L9H9, L9H6, L10H0 are necessary and sufficient for name-mover behavior in IOI.
> **Met:** I1 (necessity, $\Delta$ logit diff > 0.7 under zero + mean ablation), I2 (sufficiency, 87% recovery), I4 (specificity, SI = 14.2 vs. SVA task), E1 (intervention reach, zero and mean ablation agree), E2 (prompt generalization, consistent across 3 templates), M1 (reliability, $\rho_{XX'} = 0.84$)
> **Open:** E4 (cross-model generalization), C3 (convergent validity across methods), I5 (rival mechanism exclusion)
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
| → Triangulated | Convergent evidence from independent evidence families (C3) and discriminant validity (C4). Rival mechanism exclusion (I5), double dissociation attempted (I6), confound control (I7). Cross-distribution replication: E4 where the claim asserts reach beyond the systems tested, E2 where it does not. |
| → Causally suggestive (downgrade) | Specificity (I4) fails: the ablation equally impairs unrelated tasks. Or sufficiency (I2) is method-conditional: recovery drops below threshold under a more appropriate ablation method. |

## Characteristic occupants

- **Modular addition** ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)) -- necessity, sufficiency and specificity established for the Fourier algorithm on $(a+b) \bmod p$; double dissociation (I6) caps it
- **Greater-Than circuit** ([Hanna et al., 2023](https://arxiv.org/abs/2305.00586)) — strong structural plausibility with specificity evidence across related numerical tasks
- **Copy suppression heads** ([McDougall et al., 2023](https://arxiv.org/abs/2310.04625)) — unusually clean specificity: the heads suppress repeated tokens specifically, with minimal off-target effects

## Key references

- Olsson et al. (2022). *In-context Learning and Induction Heads.* [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hanna et al. (2023). *How does GPT-2 compute greater-than?* [arXiv:2305.00586](https://arxiv.org/abs/2305.00586)
- McDougall et al. (2023). *Copy Suppression.* [arXiv:2310.04625](https://arxiv.org/abs/2310.04625)
- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
- Wang et al. (2022). *Interpretability in the Wild.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
