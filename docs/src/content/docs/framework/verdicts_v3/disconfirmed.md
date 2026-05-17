---
title: "Disconfirmed"
description: "The evidence actively contradicts the mechanistic claim — not insufficient evidence but evidence against."
---

# Verdict: Disconfirmed

| | |
|---|---|
| Tier | Lateral (outside progression) |
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

## Minimum reporting for this tier

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

- **IOI sufficiency under resample ablation** — [Miller et al. (2024)](https://arxiv.org/abs/2407.08734) demonstrated method-conditionality of the sufficiency result
- **Early "knowledge neuron" localization claims** — initial claims that single neurons store facts were partially disconfirmed by distributed representation evidence
- **Induction head toxicity claims** — [Wang et al. (2025)](https://arxiv.org/abs/2505.13514) self-withdrawn after methodological concerns

## Key references

- Miller et al. (2024). *Faithfulness Metrics for Circuit Discovery.* [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT.* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- GRADE Working Group (2004). *Grading quality of evidence and strength of recommendations.* [doi:10.1136/bmj.328.7454.1490](https://doi.org/10.1136/bmj.328.7454.1490)
- Hill, A. B. (1965). *The Environment and Disease: Association or Causation?* [doi:10.1177/003591576505800503](https://doi.org/10.1177/003591576505800503)
