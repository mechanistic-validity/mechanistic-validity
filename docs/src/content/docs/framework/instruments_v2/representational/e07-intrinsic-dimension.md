---
title: "E07 — Intrinsic Dimension"
description: "Estimates the true manifold dimensionality of activations, connecting to the local learning coefficient and geometric complexity."
---

# E07 — Intrinsic Dimension

This framework asks: **What is the true geometric dimensionality of the manifold on which activations live?**

Intrinsic dimension (ID) goes beyond PCA by measuring the dimensionality of the data *manifold* rather than its linear span. A set of activations might occupy a high-dimensional ambient space but actually lie on a low-dimensional curved surface. ID estimation reveals this hidden structure — connecting to model complexity, generalization, and the local learning coefficient.

The local learning coefficient (LLC) from singular learning theory provides a related measure: it quantifies the effective number of parameters the model uses near a given point in weight space. Low LLC relative to parameter count indicates the model has found a low-dimensional solution.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Ansuini et al., "Intrinsic dimension of data representations in deep neural networks"](https://arxiv.org/abs/1905.12784) | 2019 | Two-NN estimator for layer-wise ID |
| [Facco et al., "Estimating the intrinsic dimension of datasets"](https://doi.org/10.1038/s41598-017-11873-y) | 2017 | Two-NN method foundation |
| [Watanabe, "Algebraic Geometry and Statistical Learning Theory"](https://doi.org/10.1017/CBO9780511800474) | 2009 | Singular learning theory and RLCT |
| [Lau et al., "Quantifying Degeneracy in Singular Models via the Learning Coefficient"](https://arxiv.org/abs/2308.12108) | 2023 | LLC estimation for neural networks |

## Core concept

The Two-NN estimator computes ID from the ratio of distances to second and first nearest neighbors. For each point \( x_i \), let \( r_1(i) \) and \( r_2(i) \) be the distances to its nearest and second-nearest neighbors. Then:

\[
\mu_i = \frac{r_2(i)}{r_1(i)}, \qquad \hat{d} = \frac{n}{\sum_{i=1}^n \log \mu_i}
\]

This maximum-likelihood estimator assumes locally uniform density on a \( d \)-dimensional manifold. The LLC provides a complementary perspective from weight space:

\[
\lambda = \text{LLC}(\theta^*) \approx \frac{\text{effective parameters near } \theta^*}{2}
\]

Low \( \lambda \) indicates the loss landscape near the solution is highly degenerate — the model found a low-complexity representation.

## Instruments under E07

### LLC Estimation (`10_llc.py`)

Estimates the local learning coefficient via MCMC sampling around the trained weights, providing a complexity measure for each circuit component.

**What it establishes:** The effective geometric complexity of the learned solution — how many dimensions the model truly uses in weight space.
**What it does not establish:** Which specific directions are "active" (combine with E06/E08 for that).

**Usage:**
```
uv run python 10_llc.py --tasks ioi sva
```

### Two-NN Intrinsic Dimension

Estimates activation manifold dimensionality at each layer using the Two-NN method on cached activations.

**What it establishes:** The manifold dimensionality of representations — how many independent axes of variation exist.
**What it does not establish:** Whether those axes correspond to interpretable variables.

**Usage:**
```
uv run python 10_llc.py --tasks ioi sva --two-nn
```

## Reading the scores

| Pattern | What it means |
|---|---|
| ID \( \ll \) ambient dimension | Activations on a low-dimensional manifold; compact structure |
| ID increases through layers | Network progressively unfolds compressed representations |
| LLC \( \ll \) parameter count | Model solution is highly degenerate — simpler than capacity allows |
| LLC matches task variable count | Model complexity aligns with task complexity |
| ID spike at specific layer | Representational expansion — new dimensions computed at that layer |

