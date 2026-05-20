---
title: "A04 — Woodward Interventionism"
description: "Woodward's interventionist theory of causation applied to circuit ablation: what makes an intervention 'surgical' and when do ablation results support causal claims."
---

# A04 — Woodward Interventionism

This framework asks: **is the ablation method a valid intervention in the Woodward sense — surgical, targeted, and invariant under the right range of changes?**

Woodward's interventionist theory provides the philosophical foundation for what counts as a legitimate causal intervention. Not all ways of setting a variable to a value are equally informative: an intervention must be "surgical" (affecting only the target variable), must not introduce confounders, and the causal relationship must be *invariant* — holding stable across a range of intervention values rather than being an artifact of a single perturbation. This framework directly addresses the concern raised by Miller et al. (2024) that different ablation methods (zero, mean, resample) yield different circuit discoveries, asking which methods satisfy Woodward's criteria.

The practical consequence: sigma ablation (calibrated noise injection) and resample-complement methods are designed to satisfy Woodward's invariance criterion more robustly than zero or mean ablation, which may violate the "no confounders introduced" requirement by pushing activations off the natural data manifold.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Woodward, *Making Things Happen*](https://global.oup.com/academic/product/making-things-happen-9780195189537) | 2003 | Interventionist theory: causation as invariance under intervention |
| [Miller et al., arXiv 2407.08734](https://arxiv.org/abs/2407.08734) | 2024 | Ablation method dependence; different interventions yield different circuits |
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC uses resample ablation as the intervention method |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | Mean ablation as the default intervention in IOI circuit discovery |

## Core concept: invariance and surgical intervention

Woodward defines \( X \) as a cause of \( Y \) if there exists an intervention \( I \) on \( X \) such that: (1) \( I \) changes \( X \), (2) the change in \( X \) changes \( Y \), and (3) the relationship between \( X \) and \( Y \) is *invariant* — it holds across a range of values of \( I \), not just one specific setting. Additionally, the intervention must be "surgical": it must not change \( Y \) through any path except through \( X \).

In MI terms: zero ablation might violate surgicality because setting a component to exactly zero is unnatural and may trigger nonlinear interactions downstream (LayerNorm, softmax) that constitute a confounding path. Mean ablation is better (the replacement value is on-manifold) but still represents a single point intervention. Resample ablation from the empirical distribution satisfies invariance better because it tests the relationship across many intervention values drawn from the natural distribution.

## Metrics under A04

### C3 — Sigma Ablation (`03_sigma_ablation.py`)

Injects calibrated Gaussian noise at multiple standard-deviation scales rather than replacing with a fixed value. Tests whether the causal relationship is invariant across noise magnitudes:

\[
\text{SigmaAbl}(c, \sigma) = LD_{\text{clean}} - LD_{\text{noised at } \sigma}
\]

By sweeping \( \sigma \in \{0.5, 1, 2, 4\} \) standard deviations of the component's activation distribution, sigma ablation produces a dose-response curve. A component that shows a smooth, monotonic relationship between noise magnitude and output degradation satisfies Woodward's invariance criterion. A component that shows threshold effects or non-monotonicity suggests the causal relationship is fragile or confounded.

**What it establishes:** Whether the component's causal effect is robust across intervention magnitudes (invariance).

**What it does not establish:** The exact mechanism by which the component contributes; only that the contribution is stable.

**Usage:**
```
uv run python 03_sigma_ablation.py --tasks ioi sva --n-prompts 40
```

### C35 — Resample Complement (`35_resample_complement.py`)

Tests the complement of the circuit: ablates everything *outside* the hypothesized circuit by resampling from the empirical distribution, checking whether the circuit alone is sufficient. This is the Woodward-compliant version of the sufficiency test.

**What it establishes:** Sufficiency of the circuit under distribution-preserving interventions.

**What it does not establish:** Whether the circuit is minimal (no redundant components).

**Usage:**
```
uv run python 35_resample_complement.py --tasks ioi sva --n-prompts 40
```

### C37 — Misalignment Score (`37_misalignment_score.py`)

Quantifies disagreement between different ablation methods (zero, mean, resample, sigma) on the same circuit. High misalignment indicates that causal conclusions are method-dependent — a violation of Woodward's invariance criterion at the meta-level.

\[
\text{Misalign} = \text{Var}_{\text{methods}}\left[ \text{rank}(c) \right]
\]

**What it establishes:** Whether circuit discovery results are robust to methodological choices.

**What it does not establish:** Which method is "correct" — only that they disagree.

**Usage:**
```
uv run python 37_misalignment_score.py --tasks ioi sva --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Smooth dose-response in sigma ablation | Causal effect satisfies Woodward invariance |
| Threshold/non-monotonic sigma response | Fragile or confounded causal relationship |
| Low misalignment across methods | Robust causal conclusion; method-independent |
| High misalignment | Causal claim is method-dependent; interpret with caution |

