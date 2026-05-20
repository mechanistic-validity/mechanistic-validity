---
title: "E05 — Subspace Alignment"
description: "Measures cosine alignment between SVD-derived principal directions of different weight matrices or circuit components."
---

# E05 — Subspace Alignment

This framework asks: **Do two circuit components operate in the same subspace of the residual stream?**

Subspace alignment quantifies the geometric overlap between the principal directions of different weight matrices. By computing the SVD of each weight matrix and measuring the cosine similarity between their top singular vectors, we can determine whether two components read from or write to shared directions — indicating potential information flow or redundancy.

This is a purely weight-space measurement: no activations required. It reveals structural relationships between circuit components that hold regardless of input distribution, providing a static skeleton that activation-based methods can then validate.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Ding & Li, "An Analysis of SVD for Deep Rotation Estimation"](https://arxiv.org/abs/2006.14616) | 2020 | SVD geometry in deep networks |
| [Li et al., "Convergent Learning: Do different neural networks learn the same representations?"](https://arxiv.org/abs/1511.07543) | 2015 | Subspace overlap as convergence measure |
| [Gur-Ari et al., "Gradient Descent Happens in a Tiny Subspace"](https://arxiv.org/abs/1812.04754) | 2018 | Low-rank structure in gradient subspaces |
| [Martin & Mahoney, "Implicit Self-Regularization in Deep Neural Networks"](https://arxiv.org/abs/1810.01075) | 2019 | Spectral analysis of weight matrices |

## Core concept

Given weight matrices \( W_1 \in \mathbb{R}^{m \times d} \) and \( W_2 \in \mathbb{R}^{n \times d} \) with SVDs \( W_1 = U_1 \Sigma_1 V_1^T \) and \( W_2 = U_2 \Sigma_2 V_2^T \), the subspace alignment between their top-\( k \) right singular vectors is:

\[
\text{Align}(W_1, W_2, k) = \frac{1}{k} \sum_{i=1}^{k} \max_j |\langle v_i^{(1)}, v_j^{(2)} \rangle|
\]

Alternatively, the Grassmann distance between the \( k \)-dimensional subspaces \( \text{span}(V_1[:k]) \) and \( \text{span}(V_2[:k]) \) gives a single scalar. Values near 1 indicate the components share input subspace; values near 0 indicate orthogonality.

The metric from script 18 (weight extended, metric #63) specifically computes cosine alignment of the top SVD directions between OV circuits of different heads, revealing shared read/write subspaces.

## Metrics under E05

### Weight Subspace Alignment (`18_weight_extended.py`)

Computes pairwise SVD alignment between all circuit heads' OV and QK matrices (metric #63).

**What it establishes:** Which heads read from or write to overlapping subspaces — structural redundancy or composition.
**What it does not establish:** Whether the shared subspace carries task-relevant information (combine with E01/E02).

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva --metric subspace_alignment
```

### Cross-Component Alignment

Measures alignment between Q/K/V/O subspaces within a single head, revealing internal geometric organization.

**What it establishes:** Whether a head's query and key spaces are aligned (low-rank attention) or orthogonal (full-rank).
**What it does not establish:** The computational role of the alignment pattern.

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva --metric cross_component
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Alignment > 0.8 between heads | Heads share principal input/output directions — potential redundancy |
| Alignment < 0.2 between heads | Heads operate in orthogonal subspaces — complementary roles |
| Q-K alignment high within a head | Head performs approximate "matching" in a shared subspace |
| OV alignment across layers | Information highway — later head reads what earlier head writes |

