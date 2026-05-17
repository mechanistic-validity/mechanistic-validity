---
title: "B01 — SVD / Spectral Analysis"
description: "Singular value decomposition of weight matrices to identify dominant computational directions."
---

# B01 — SVD / Spectral Analysis

This framework asks: **what are the principal directions of computation in a circuit's weight matrices, and how concentrated is their energy?**

The singular value decomposition (SVD) of a weight matrix reveals which linear subspaces carry most of the matrix's computational energy. In mechanistic interpretability, applying SVD to attention weight matrices (W_QK, W_OV) decomposes multi-dimensional transformations into rank-one contributions ordered by importance. A circuit claim that identifies specific heads as load-bearing should predict that those heads have spectrally distinct weight structure — concentrated singular values, interpretable singular vectors, or spectral gaps separating signal from noise.

SVD is the workhorse of structural circuit analysis because it provides a canonical, rotation-invariant decomposition. Unlike probing (which requires labeled data) or activation patching (which requires paired inputs), SVD operates directly on weights and reveals capacity-level structure: what the circuit *could* compute, not merely what it does compute on a given distribution.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | SVD of W_OV and W_QK as interpretive decomposition |
| [Millidge & Black, arXiv 2202.05924](https://arxiv.org/abs/2202.05924) | 2022 | SVD-based analysis of singular value spectra in GPT-2 heads |
| [Eckart & Young, "The approximation of one matrix by another"](https://doi.org/10.1007/BF02288367) | 1936 | Optimal low-rank approximation theorem |
| [Henighan et al., arXiv 2309.14322](https://arxiv.org/abs/2309.14322) | 2023 | Superposition and spectral structure of MLP weight matrices |

## Core concept

Given a weight matrix \( W \in \mathbb{R}^{m \times n} \), the SVD factorizes it as:

\[
W = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i \, u_i \, v_i^T
\]

where \( \sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0 \) are the singular values, \( u_i \) are left singular vectors (output directions), and \( v_i \) are right singular vectors (input directions). The effective rank measures how many of these directions carry meaningful energy.

For attention heads, the composed matrices \( W_Q^T W_K \in \mathbb{R}^{d_h \times d_h} \) (QK circuit) and \( W_O W_V \in \mathbb{R}^{d_m \times d_m} \) (OV circuit) are the natural objects to decompose. A head performing a single interpretable operation (e.g., "copy the previous token") will have a rapidly decaying spectrum dominated by one or two singular values. A head performing multiple superposed operations will have a flatter spectrum.

## Instruments under B01

### Effective Rank of W_QK (`18_weight_extended.py`)

Computes the SVD of the composed QK matrix for each attention head and reports the effective rank (see B02 for the entropy-based definition) alongside the top-k singular value ratios \( \sigma_1 / \sigma_k \).

**What it establishes:** Whether circuit heads have spectrally concentrated attention patterns — low effective rank indicates a head is implementing a small number of attention motifs.

**What it does not establish:** Whether those motifs are *correct* for the task, or whether the head is necessary (that requires causal validation from Pillar A).

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Low effective rank in circuit heads | Head implements a concentrated, potentially interpretable operation |
| High \( \sigma_1 / \sigma_2 \) ratio | Dominated by a single rank-one term (e.g., copy or induction) |
| Flat spectrum across all heads | No structural differentiation — circuit claim lacks weight-level support |
| Circuit heads spectrally distinct from non-circuit | Structural signature corroborates causal findings |

## Connection to other frameworks

SVD provides the substrate for B02 (effective rank as a scalar summary), B03 (OV/QK decomposition interprets the singular vectors), and B04 (weight alignment compares top singular directions across heads). Causal findings from A01 (activation patching) identify *which* heads matter; B01 then asks whether those heads have structurally distinctive spectra that explain *why* they were selected.
