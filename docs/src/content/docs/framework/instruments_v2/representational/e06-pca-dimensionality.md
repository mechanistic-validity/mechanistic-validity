---
title: "E06 — PCA Dimensionality"
description: "Measures the effective dimensionality of circuit subspaces via the spectrum of their activation covariance."
---

# E06 — PCA Dimensionality

This framework asks: **How many independent directions does a circuit subspace actually use in practice?**

PCA dimensionality measures the effective rank of a representation by analyzing the eigenvalue spectrum of its covariance matrix. A representation that concentrates variance in few principal components is low-dimensional (even if embedded in a high-dimensional space), suggesting compact, interpretable structure. A flat spectrum indicates the representation uses all available dimensions.

For circuit analysis, PCA dimensionality reveals whether a circuit head's computations are intrinsically low-rank — concentrated in a small subspace — or genuinely high-dimensional. Low effective dimensionality suggests the head performs a simple, potentially interpretable operation.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Ansuini et al., "Intrinsic dimension of data representations in deep neural networks"](https://arxiv.org/abs/1905.12784) | 2019 | Measured intrinsic dimension across layers |
| [Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"](https://arxiv.org/abs/2012.13255) | 2020 | Connected low intrinsic dimension to fine-tuning |
| [Papyan et al., "Prevalence of Neural Collapse"](https://arxiv.org/abs/2008.08186) | 2020 | Spectral collapse in final layers |
| [Hu et al., "LoRA: Low-Rank Adaptation"](https://arxiv.org/abs/2106.09685) | 2021 | Exploited low intrinsic dimensionality for efficient adaptation |

## Core concept

Given activations \( H \in \mathbb{R}^{n \times d} \) (centered), the covariance eigenvalues \( \lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \) define several dimensionality measures:

\[
d_{\text{eff}}^{(90\%)} = \min\left\{ k : \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} \geq 0.9 \right\}
\]

The participation ratio offers a softer measure:

\[
d_{\text{PR}} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}
\]

Low \( d_{\text{eff}} \) relative to ambient dimension \( d \) indicates a low-rank representation. The spectrum shape (exponential decay vs. power-law vs. plateau) reveals the underlying structure of the computation.

## Instruments under E06

### Spectral Analysis (`spectral_dimensionality.py`)

Computes eigenvalue spectrum of per-head activation covariance, reporting effective dimension at 90% and 95% variance thresholds.

**What it establishes:** How many principal directions capture most of the activation variance per circuit component.
**What it does not establish:** Whether the top directions are task-relevant (combine with E01/E02 for that).

**Usage:**
```
uv run python spectral_dimensionality.py --tasks ioi sva --threshold 0.9
```

### Layer-wise Dimensionality Profile

Tracks effective dimensionality across layers, revealing whether representations compress or expand.

**What it establishes:** The dimensionality trajectory through the network — compression points suggest information bottlenecks.
**What it does not establish:** What information is discarded during compression.

**Usage:**
```
uv run python spectral_dimensionality.py --tasks ioi sva --layer-profile
```

## Reading the scores

| Pattern | What it means |
|---|---|
| \( d_{\text{eff}} \ll d \) | Low-rank representation; head performs a simple projection |
| \( d_{\text{eff}} \approx d \) | Full-rank; head uses all available dimensions |
| Dimensionality drops at layer \( \ell \) | Information bottleneck — representations compress |
| Exponential spectral decay | Single dominant direction with rapid falloff |
| Power-law spectrum | Multi-scale structure without clear cutoff |

