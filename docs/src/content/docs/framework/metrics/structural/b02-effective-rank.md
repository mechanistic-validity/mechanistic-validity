---
title: "B02 — Effective Rank"
description: "Entropy-based dimensionality of weight matrices as a scalar summary of spectral concentration."
---

# B02 — Effective Rank

This framework asks: **how many dimensions of the weight matrix are actively used, and does the circuit occupy a lower-dimensional subspace than the full model?**

Effective rank distills the entire singular value spectrum into a single number that quantifies the intrinsic dimensionality of a linear transformation. A matrix with effective rank 1 acts as a rank-one projection; a matrix with effective rank equal to its ambient dimension uses all available capacity uniformly. In circuit analysis, comparing effective rank between circuit heads and non-circuit heads tests whether the identified circuit is structurally simpler — a hallmark of clean mechanistic explanations.

The local learning coefficient (LLC) from singular learning theory provides a complementary spectral measure: it quantifies the effective dimensionality of the loss landscape near a parameter, which correlates with the functional complexity of the learned computation. Together, effective rank and LLC bound both the *capacity* (what the matrix can represent) and the *complexity* (how many degrees of freedom the model uses).

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Roy & Bhatt, "Exponential bounds on curvature"](https://doi.org/10.1109/18.556667) | 2007 | Entropy-based effective rank definition |
| [Lau et al., arXiv 2308.12108](https://arxiv.org/abs/2308.12108) | 2023 | Local learning coefficient for neural network complexity |
| [Elhage et al., "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) | 2022 | Relationship between capacity, dimensionality, and feature packing |
| [Feng & Tu, arXiv 2301.02827](https://arxiv.org/abs/2301.02827) | 2023 | Rank collapse in attention heads during training |

## Core concept

Given singular values \( \sigma_1, \ldots, \sigma_r \) of a matrix \( W \), define the normalized distribution:

\[
p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}
\]

The effective rank is the exponential of the Shannon entropy of this distribution:

\[
\text{erank}(W) = \exp\left( -\sum_{i=1}^{r} p_i \log p_i \right)
\]

This satisfies \( 1 \leq \text{erank}(W) \leq r \), equals 1 when all energy is in one singular value, and equals \( r \) when energy is uniformly distributed. Unlike hard thresholding (count singular values above some cutoff), effective rank is continuous and differentiable with respect to the spectrum.

The local learning coefficient \( \hat{\lambda} \) from `10_llc.py` measures the effective dimensionality of the loss landscape via MCMC sampling near the trained parameters. High LLC indicates the parameter is in a high-dimensional valley (complex computation); low LLC indicates a degenerate critical point (simple, low-rank computation).

## Metrics under B02

### Effective Rank (`18_weight_extended.py`)

Computes \( \text{erank}(W_{QK}) \) and \( \text{erank}(W_{OV}) \) for every attention head. Reports per-head values and the mean difference between circuit and non-circuit heads.

**What it establishes:** Whether circuit heads occupy a structurally lower-dimensional subspace than non-circuit heads.

**What it does not establish:** Causal relevance — a low-rank head may be irrelevant, and a high-rank head may be critical.

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva
```

### Local Learning Coefficient (`10_llc.py`)

Estimates \( \hat{\lambda} \) for circuit-relevant parameters using SGLD sampling around the trained checkpoint.

**What it establishes:** Functional complexity of the learned computation at each component.

**What it does not establish:** Interpretability or causal role.

**Usage:**
```
uv run python 10_llc.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Circuit heads have lower erank than non-circuit | Circuit implements a simpler linear transformation |
| erank close to \( d_{\text{head}} \) | Head uses full capacity — may be superposing multiple features |
| Low LLC in circuit heads | Simple functional form — amenable to mechanistic description |
| High LLC in circuit heads | Complex computation — may resist clean narrative explanation |

## Connection to other frameworks

Effective rank is a scalar summary of B01 (SVD spectral analysis). It connects to B07 (polysemanticity): high effective rank predicts superposition, which B07 tests via intervention specificity. The LLC measure bridges to A01 (causal): if a structurally simple component (low LLC) is also causally necessary (high activation patching), the circuit claim is jointly supported by both weight-level and activation-level evidence.
