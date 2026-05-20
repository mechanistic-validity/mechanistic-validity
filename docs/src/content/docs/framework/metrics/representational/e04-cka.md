---
title: "E04 — Centered Kernel Alignment (CKA)"
description: "Quantifies similarity between representations using kernel alignment, enabling cross-layer and cross-model comparison."
---

# E04 — Centered Kernel Alignment

This framework asks: **How similar are two representations in terms of their learned feature structure, independent of rotation or scale?**

CKA measures the alignment between two sets of representations by comparing their Gram matrices (or linear kernels). Unlike simple correlation of flattened activations, CKA is invariant to orthogonal transformations and isotropic scaling — making it ideal for comparing representations across different layers, training runs, or model architectures.

For circuit analysis, CKA reveals whether two circuit heads learn similar representational structure even when their specific weight matrices differ. It also tracks how representations evolve across layers, identifying computational phases in the network.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Kornblith et al., "Similarity of Neural Network Representations Revisited"](https://arxiv.org/abs/1905.00414) | 2019 | Introduced CKA; showed superiority over CCA and PWCCA |
| [Cortes et al., "Algorithms for Learning Kernels Based on Centered Alignment"](https://doi.org/10.1145/2390524.2390552) | 2012 | Original centered alignment formulation |
| [Nguyen et al., "Do Wide Neural Networks Really Need to be Wide?"](https://arxiv.org/abs/2006.12156) | 2020 | Used CKA to analyze width vs. representation similarity |
| [Raghu et al., "Do Vision Transformers See Like CNNs?"](https://arxiv.org/abs/2108.08810) | 2021 | CKA for cross-architecture comparison |

## Core concept

Given representations \( X \in \mathbb{R}^{n \times p} \) and \( Y \in \mathbb{R}^{n \times q} \) for \( n \) inputs, linear CKA computes:

\[
\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}
\]

after centering both \( X \) and \( Y \). This is equivalent to the HSIC (Hilbert-Schmidt Independence Criterion) normalized by the individual kernel norms. CKA = 1 when representations encode identical structure up to linear transformation; CKA = 0 when they are independent.

The linear kernel suffices for most analyses. For nonlinear structure, RBF-kernel CKA replaces the Gram matrix \( XX^T \) with \( K_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2) \).

## Metrics under E04

### Cross-Layer CKA (`cka_analysis.py`)

Computes pairwise CKA between all layers, producing a similarity heatmap that reveals representational phases.

**What it establishes:** Which layers share representational structure and where phase transitions occur.
**What it does not establish:** What information content changes at each transition.

**Usage:**
```
uv run python cka_analysis.py --tasks ioi sva --kernel linear
```

### Cross-Model CKA

Compares representations between the full model and ablated circuits to quantify how much representational structure a circuit accounts for.

**What it establishes:** Whether a subset of circuit heads captures the full model's representational geometry.
**What it does not establish:** Whether the captured structure is task-relevant (combine with RSA for that).

**Usage:**
```
uv run python cka_analysis.py --tasks ioi sva --compare-ablated
```

## Reading the scores

| Pattern | What it means |
|---|---|
| CKA ~ 1.0 across adjacent layers | Representational continuity; gradual refinement |
| CKA block structure (high within, low across) | Distinct computational phases in the network |
| CKA(full, ablated) > 0.9 | Circuit subset preserves nearly all representational structure |
| CKA(full, ablated) < 0.5 | Ablation destroys significant representational geometry |
| Early-late CKA near zero | Deep transformation; early and late layers share little structure |

