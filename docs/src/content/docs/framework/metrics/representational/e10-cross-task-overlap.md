---
title: "E10 — Cross-Task Overlap"
description: "Measures how much representational structure is shared between different tasks via IIA transfer and subspace intersection."
---

# E10 — Cross-Task Overlap

This framework asks: **Do different tasks share representational subspaces, and can causal alignments transfer across tasks?**

Cross-task overlap quantifies the degree to which circuits discovered for one task also encode variables relevant to another. By performing IIA with rotations learned on task A and evaluating on task B, we test whether the model reuses representational structure — revealing shared computational primitives versus task-specific encodings.

This is the representational analog of circuit overlap: while circuit overlap counts shared heads, cross-task overlap measures whether those shared heads encode information *in the same subspace*. Two heads might appear in both circuits but use entirely different directions for each task.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Geiger et al., "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"](https://arxiv.org/abs/2303.02536) | 2023 | DAS transfer methodology |
| [Todd et al., "Function Vectors in Large Language Models"](https://arxiv.org/abs/2310.15213) | 2023 | Shared function representations across tasks |
| [Hernandez et al., "Linearity of Relation Representations in Transformer Language Models"](https://arxiv.org/abs/2311.12786) | 2023 | Linear relation directions transfer across contexts |
| [Merullo et al., "Circuit Component Reuse Across Tasks"](https://arxiv.org/abs/2310.08744) | 2023 | Empirical circuit overlap measurement |

## Core concept

Given DAS rotation \( R_A \) learned on task A with subspace \( S_A \), cross-task IIA evaluates:

\[
\text{IIA}_{\text{transfer}}(A \to B) = \frac{1}{N_B} \sum_{i=1}^{N_B} \mathbb{1}\left[ f\left(\text{do}(h^{(i)}, h^{(j)}, R_A, S_A)\right) = y^{(j)}_{V_B} \right]
\]

High transfer IIA means task B's variable is encoded in the same subspace that task A uses. The overlap coefficient between two learned subspaces \( S_A, S_B \subseteq \mathbb{R}^d \) is:

\[
\text{Overlap}(S_A, S_B) = \frac{\dim(S_A \cap S_B)}{\min(\dim S_A, \dim S_B)}
\]

computed via singular values of \( P_A P_B \) where \( P_A, P_B \) are the projection matrices onto each subspace.

## Metrics under E10

### Cross-Task IIA Transfer (`32_cross_task_iia_transfer.py`)

Trains DAS on each task independently, then evaluates the learned rotations on all other tasks.

**What it establishes:** Whether causal encodings generalize — a shared subspace implies a shared computational primitive.
**What it does not establish:** Whether shared representations reflect genuine reuse vs. coincidental geometric overlap.

**Usage:**
```
uv run python 32_cross_task_iia_transfer.py --tasks ioi sva greater_than
```

### Subspace Intersection Analysis

Computes the principal angles between task-specific DAS subspaces, giving a fine-grained overlap profile.

**What it establishes:** The dimensionality and geometry of shared representational structure between tasks.
**What it does not establish:** Whether the overlap is functionally meaningful (could be residual stream "background").

**Usage:**
```
uv run python 32_cross_task_iia_transfer.py --tasks ioi sva --subspace-angles
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Transfer IIA > 0.8 (A to B) | Tasks share a causal encoding — strong representational reuse |
| Transfer IIA asymmetric (A to B high, B to A low) | Task A's subspace contains B's, but not vice versa |
| Overlap ~ 0 | Tasks use entirely different directions — no representational sharing |
| High overlap but low transfer IIA | Subspaces intersect geometrically but encode different information |
| Cluster of high-overlap tasks | Shared computational primitive (e.g., "subject tracking") |

