---
title: "B03 — OV / QK Decomposition"
description: "Decomposing attention heads into their OV (what to write) and QK (where to attend) circuits."
---

# B03 — OV / QK Decomposition

This framework asks: **what does each attention head copy (OV) and where does it look (QK), and do these decomposed circuits align with the claimed mechanism?**

Every attention head performs two logically distinct operations: the QK circuit determines the attention pattern (which positions attend to which), and the OV circuit determines what information is moved once attention is allocated. The composed matrices \( W_E^T W_Q^T W_K W_E \) (full QK) and \( W_U W_O W_V W_E \) (full OV) can be analyzed independently to characterize a head's structural role. A complete circuit explanation must account for both halves.

This decomposition is the foundation of the "mathematical framework for transformer circuits" and remains the most productive structural lens in mechanistic interpretability. It converts the question "what does this head do?" into two simpler questions with independently testable answers.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | OV/QK decomposition as the fundamental unit of circuit analysis |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | Applied OV/QK to IOI circuit (name movers, induction, S-inhibition) |
| [Olsson et al., arXiv 2209.11895](https://arxiv.org/abs/2209.11895) | 2022 | Induction heads: QK implements "previous token attended to by current" |
| [Nanda et al., arXiv 2301.05217](https://arxiv.org/abs/2301.05217) | 2023 | OV/QK analysis of indirect object identification in GPT-2 |

## Core concept

For a single attention head with parameters \( W_Q, W_K \in \mathbb{R}^{d_m \times d_h} \) and \( W_V, W_O \in \mathbb{R}^{d_m \times d_h} \), the two composed circuits are:

\[
W_{QK} = W_Q W_K^T \in \mathbb{R}^{d_m \times d_m}
\]

\[
W_{OV} = W_O W_V \in \mathbb{R}^{d_m \times d_m}
\]

The QK matrix determines attention scores: position \( i \) attends to position \( j \) proportionally to \( x_i^T W_{QK} x_j \). The OV matrix determines what is written to the residual stream: if position \( j \) is attended to, the contribution is \( W_{OV} x_j \). By composing with the embedding and unembedding:

\[
W_{QK}^{\text{full}} = W_E^T W_{QK} W_E, \quad W_{OV}^{\text{full}} = W_U W_{OV} W_E
\]

we can read off token-to-token attention preferences and token-to-logit contributions directly.

## Instruments under B03

### W_QK Spectral Analysis (`18_weight_extended.py`)

Computes the eigendecomposition of \( W_{QK} \) for circuit heads and reports: (1) the top eigenvalues and their associated input/output directions, (2) whether the QK circuit is symmetric (positional) or asymmetric (token-content), and (3) alignment between top QK directions and the task's causal variables.

**What it establishes:** The structural attention pattern preference encoded in the head's weights, independent of any specific input.

**What it does not establish:** Whether this pattern is actually realized on task-relevant inputs (requires activation-level verification from Pillar A).

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| OV top singular vector aligns with task-relevant token directions | Head structurally encodes the copy/suppress operation the circuit claims |
| QK has low rank with interpretable eigenvectors | Head implements a simple, characterizable attention rule |
| OV eigenspectrum is flat | Head performs distributed transformation — harder to assign single role |
| QK and OV both align with circuit narrative | Strong structural corroboration of the mechanistic claim |

