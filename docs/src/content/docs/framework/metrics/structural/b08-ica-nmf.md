---
title: "B08 — ICA / NMF"
description: "Independent component analysis and non-negative matrix factorization for decomposing weight matrices into interpretable parts."
---

# B08 — ICA / NMF

This framework asks: **can weight matrices be decomposed into statistically independent or non-negative components that correspond to interpretable circuit elements?**

While SVD provides the optimal rank-k approximation, its components are constrained to be orthogonal — a mathematical convenience that may not match the actual structure of learned computations. Independent Component Analysis (ICA) relaxes orthogonality to find statistically independent sources, while Non-negative Matrix Factorization (NMF) constrains components to be non-negative, producing parts-based decompositions. Both methods can reveal interpretable structure that SVD misses, particularly when the true computational primitives are non-orthogonal or sparse.

These methods provide an alternative structural vocabulary for circuit description. If ICA or NMF components align with causally identified circuit elements, this provides evidence that the circuit decomposition reflects genuine statistical structure in the weights rather than being an artifact of the analysis method chosen.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Hyvarinen & Oja, "Independent Component Analysis"](https://doi.org/10.1016/S0893-6080(00)00026-5) | 2000 | ICA foundations — maximizing non-Gaussianity for source separation |
| [Lee & Seung, "Learning the parts of objects by NMF"](https://doi.org/10.1038/44565) | 1999 | NMF produces parts-based, interpretable decompositions |
| [Elhage et al., "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) | 2022 | Non-orthogonal feature packing motivates beyond-SVD methods |
| [Sharkey et al., arXiv 2312.09528](https://arxiv.org/abs/2312.09528) | 2023 | Sparse probing and non-orthogonal directions in transformers |
| [Bricken et al., "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 2023 | Learned dictionaries (SAEs) as an alternative to ICA/NMF |

## Core concept

Given a weight matrix \( W \in \mathbb{R}^{m \times n} \), ICA models it as a mixing of independent sources:

\[
W = A S
\]

where \( S \in \mathbb{R}^{k \times n} \) contains \( k \) statistically independent source components and \( A \in \mathbb{R}^{m \times k} \) is the mixing matrix. Independence is measured via non-Gaussianity (kurtosis or negentropy). Each row of \( S \) is a candidate "computational primitive."

NMF instead requires non-negativity:

\[
W \approx W_{\text{basis}} H, \quad W_{\text{basis}} \geq 0, \; H \geq 0
\]

This constraint produces parts-based decompositions — each component represents an additive contribution rather than a cancellation pattern. For weight matrices with non-negative structure (e.g., after ReLU in MLPs), NMF components often correspond to interpretable feature detectors.

Both methods trade optimality (SVD is the best L2 approximation) for interpretability (components may better match the true generative structure of the learned computation).

## Metrics under B08

### Theoretical Framework

ICA and NMF applied to transformer weight matrices remain a theoretical metric in this framework. No dedicated script implements these decompositions end-to-end, but the mathematical framework motivates comparing:

1. **ICA of W_OV** — do independent components correspond to distinct semantic operations (copy, suppress, transform)?
2. **NMF of absolute W_OV** — do non-negative parts correspond to feature-level circuit primitives?
3. **Comparison to SAE features** — do ICA/NMF components recover the same structure as trained sparse autoencoders?

**What it establishes:** Whether the weight matrix has non-orthogonal interpretable structure beyond what SVD reveals.

**What it does not establish:** Causal relevance — interpretable decomposition does not imply causal importance.

## Reading the scores

| Pattern | What it means |
|---|---|
| ICA components align with SAE features | Weight structure has genuinely independent computational primitives |
| NMF produces sparse, interpretable parts | Additive parts-based computation — each component has a clear role |
| ICA/NMF fail to improve over SVD | Orthogonal decomposition is sufficient — computation is low-rank rather than superposed |
| Many ICA components needed | High-dimensional independent structure — possible superposition |
| Few NMF components with high reconstruction | Weight matrix has simple non-negative factorization — clean circuit |

## Connection to other frameworks

ICA/NMF sit between B01 (SVD, which they generalize) and B07 (polysemanticity, which they attempt to resolve). If B07 identifies polysemantic components, ICA/NMF can potentially decompose them into monosemantic sources. The learned dictionary approach (SAEs) from the broader MI literature can be viewed as a nonlinear generalization of NMF with sparsity constraints. B09 (weight classifier) provides an alternative path: rather than decomposing weights, it directly classifies weight patterns as belonging to known circuit motifs.
