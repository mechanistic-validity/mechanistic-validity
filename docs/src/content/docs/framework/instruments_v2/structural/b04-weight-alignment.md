---
title: "B04 — Weight Alignment"
description: "Cosine similarity between principal weight directions across circuit and non-circuit heads."
---

# B04 — Weight Alignment

This framework asks: **do circuit heads share privileged directions in weight space, and are those directions distinct from non-circuit heads?**

Weight alignment measures the cosine similarity between the top SVD directions of different attention heads' weight matrices. If multiple heads in a claimed circuit share aligned principal directions, this suggests they operate on a common subspace — potentially implementing a compositional pipeline where one head's output lies in another head's input space. Conversely, high alignment between circuit and non-circuit heads would undermine a claim of structural specialization.

This instrument bridges single-head structural analysis (B01-B03) to multi-head circuit topology. It tests whether the weight-level structure supports the compositional claims that circuit narratives make (e.g., "induction head Q aligns with previous-token head OV output").

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Composition via shared subspaces in the residual stream |
| [Merullo et al., arXiv 2305.16130](https://arxiv.org/abs/2305.16130) | 2023 | Linear representations and subspace alignment across components |
| [Bricken et al., "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 2023 | Feature directions and alignment in representation space |
| [Hanna et al., arXiv 2305.00586](https://arxiv.org/abs/2305.00586) | 2023 | Compositional structure in GPT-2 circuits via weight analysis |

## Core concept

For two heads \( h_1 \) and \( h_2 \), let \( u_1^{(1)} \) denote the top left singular vector of \( W_{OV}^{(h_1)} \) and \( v_1^{(2)} \) denote the top right singular vector of \( W_{QK}^{(h_2)} \). The alignment score is:

\[
\text{align}(h_1 \to h_2) = \left| \cos\left( u_1^{(h_1)}, v_1^{(h_2)} \right) \right| = \frac{| u_1^{(h_1)} \cdot v_1^{(h_2)} |}{\| u_1^{(h_1)} \| \| v_1^{(h_2)} \|}
\]

High alignment indicates that head \( h_1 \)'s primary output direction is head \( h_2 \)'s primary input direction — a necessary condition for sequential composition. The metric generalizes to subspace alignment using the top-k singular vectors and principal angles.

Within a circuit, we compute the mean pairwise alignment among circuit heads and compare it to the mean alignment between circuit and non-circuit heads. A significantly higher within-circuit alignment supports the claim that the circuit forms a coherent computational unit.

## Instruments under B04

### Cosine Alignment of Top SVD Directions (`18_weight_extended.py`)

Computes pairwise \( |\cos(\theta)| \) between the top SVD directions of W_OV and W_QK for all head pairs. Reports: (1) within-circuit mean alignment, (2) between-circuit mean alignment, (3) alignment z-score relative to random head subsets.

**What it establishes:** Whether the identified circuit has structurally coherent weight directions that distinguish it from arbitrary head groupings.

**What it does not establish:** Causal dependence — aligned directions may exist but never be activated together on task-relevant inputs.

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High within-circuit alignment, low between-circuit | Circuit heads form a structurally distinct group |
| Alignment between early-layer OV and late-layer QK | Evidence of sequential composition (V-composition) |
| Uniform alignment across all heads | No structural differentiation — circuit boundary may be arbitrary |
| High alignment z-score (> 2.0) | Within-circuit coherence is unlikely under random grouping |

