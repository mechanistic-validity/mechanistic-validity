---
title: "Disconfirmed"
description: "The evidence actively contradicts the mechanistic claim — not insufficient evidence but evidence against."
---

# Verdict: Disconfirmed

| | |
|---|---|
| Label | Diagnostic (replaces the tier) |
| What it means | Evidence actively contradicts the claimed mechanism — a specific prediction has failed or the finding is shown to be artifactual |
| When to assign | A prediction of the mechanism has been tested and refuted, OR the mechanism is demonstrated to be a measurement artifact |
| Relationship to progressive tiers | Any claim at any progressive tier can be moved to Disconfirmed when contradicting evidence emerges |
| Scientific value | High — disconfirmation narrows the hypothesis space and is informative |

## What this verdict establishes

Disconfirmed is not "failed research." It is a positive scientific conclusion: the evidence actively contradicts the mechanistic claim. A field that never disconfirms is not doing science. The lateral position (rather than placing it below Proposed) reflects this: disconfirmation is a *different kind of conclusion*, not a worse one.

Disconfirmation can take three forms: prediction failure (the mechanism predicts X, the model does not-X), artifact demonstration (the finding disappears under improved methodology), or construct dissolution (the named entity is not a coherent construct separable from other processing).

Each form is informative. Prediction failure narrows the space of viable mechanisms. Artifact demonstration improves methodology for the whole field. Construct dissolution reveals that the question was ill-posed, redirecting inquiry.

## Example verdict statement

> **Verdict:** Disconfirmed — `[implementational-topographic]`
> **Claim:** The IOI circuit is sufficient for indirect object identification under distribution-respecting ablation.
> **Disconfirming evidence:** Miller et al. (2024) demonstrated that sufficiency ($R = 0.87$ under mean ablation) drops to $R < 0.50$ under resample ablation. The original sufficiency claim is an artifact of mean ablation's distributional assumptions.
> **Type:** Artifact demonstration — the finding is method-conditional, not mechanism-intrinsic.
> **Remaining valid claims:** Necessity of the circuit components remains established. Sufficiency under mean ablation remains a true statement (with method qualification).
> **Scope:** GPT-2 Small, IOI task, sufficiency specifically (not the full circuit claim)

## Types of disconfirmation

| Type | Definition | Example |
|---|---|---|
| Prediction failure | Mechanism predicts behavior $X$; model produces $\neg X$ | A claimed "gender circuit" predicts male bias; model shows no gender preference on the test distribution |
| Artifact demonstration | Finding disappears under improved methodology | Patching result vanishes when mean ablation is replaced by resample ablation |
| Construct dissolution | Named entity is not separable from other processing | "The bias circuit" is indistinguishable from "the gender knowledge circuit" — the construct has no independent existence |

## Minimum reporting for this label

- The original claim stated precisely (what was predicted)
- The disconfirming evidence (what was observed instead)
- The type of disconfirmation (prediction failure, artifact, or dissolution)
- What remains valid from the original work (disconfirmation is usually partial)
- Whether the disconfirmation is total (mechanism is wrong) or scoped (mechanism is method-conditional or distribution-limited)

## Relationship to other verdicts

| Transition | Meaning |
|---|---|
| Any tier → Disconfirmed | New evidence contradicts the claim |
| Disconfirmed → Proposed (rare) | The disconfirming evidence is itself shown to be flawed; the original claim is reopened |
| Disconfirmed → refined claim at Tier 1+ | The original claim is revised to accommodate the disconfirming evidence — the revised claim is a new entity |

## Characteristic occupants

Two of the sixteen audited claims reach this tier, both on evidence that was collected rather than missing.

- **Knowledge neurons** ([Dai et al., 2022](https://arxiv.org/abs/2104.08696)) — the claim that roughly four feed-forward neurons *store* a relational fact. All three of the paper's own summaries report a correlation between activation and expression while the title claims storage (V2), and the same editing machinery moves non-factual linguistic patterns, so the construct never separates from its neighbor (C4).
- **Induction heads as the source of general in-context learning** ([Olsson et al., 2022](https://arxiv.org/abs/2209.11895)) — the broad reading. No ablation runs above the twelve small models, so nothing at scale separates induction heads from whatever else forms alongside them (E4, I5), and the adopted measure does not separate general in-context learning from the few-shot accuracy the field reads it as (C4). The narrow reading, that induction heads implement prefix-matching token copying, reaches Triangulated.

Individual criteria are Disconfirmed more often than whole claims. IOI's stability (M3) and invariance (M6) are both Disconfirmed by [Miller et al. (2024)](https://arxiv.org/abs/2407.08734), who move the same faithfulness quantity from below 0% to over 100% across six methodological choices, and the greater-than circuit's cross-model recurrence (E4) is Disconfirmed post-origin — without either claim as a whole reaching this tier.

## Key references

- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT.* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- GRADE Working Group (2004). *Grading quality of evidence and strength of recommendations.* [doi:10.1136/bmj.328.7454.1490](https://doi.org/10.1136/bmj.328.7454.1490)
- Hill, A. B. (1965). *The Environment and Disease: Association or Causation?* [doi:10.1177/003591576505800503](https://doi.org/10.1177/003591576505800503)
