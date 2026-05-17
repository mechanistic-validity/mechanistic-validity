---
title: "E09 — Persistent Homology"
description: "Applies topological data analysis to detect loops, voids, and higher-order structure in activation manifolds."
---

# E09 — Persistent Homology

This framework asks: **What topological features (connected components, loops, voids) exist in the activation manifold, and how robust are they?**

Persistent homology tracks topological features across multiple scales by gradually expanding balls around data points and recording when features (connected components, loops, cavities) appear and disappear. Features that persist across many scales are considered genuine topological structure of the data manifold rather than noise artifacts.

For circuit analysis, persistent homology reveals qualitative geometric structure that linear methods (PCA, CKA) miss entirely. A representation might organize inputs along a circle (1-cycle) for periodic variables, or form clusters (0-cycles) for categorical distinctions. These topological signatures provide invariants of the computation that are independent of specific basis or metric choices.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Carlsson, "Topology and data"](https://doi.org/10.1090/S0273-0979-09-01249-X) | 2009 | Foundational survey of TDA for data analysis |
| [Edelsbrunner & Harer, "Persistent homology — a survey"](https://doi.org/10.1090/conm/453/08802) | 2008 | Mathematical foundations of persistence |
| [Rieck et al., "Neural Persistence: A Complexity Measure for Deep Neural Networks"](https://arxiv.org/abs/1811.04492) | 2018 | Applied persistent homology to neural network weights |
| [Naitzat et al., "Topology of Deep Neural Networks"](https://arxiv.org/abs/1811.01122) | 2020 | Tracked topological changes through layers |

## Core concept

Given a point cloud \( X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d \), the Vietoris-Rips complex at scale \( \epsilon \) connects points within distance \( \epsilon \). As \( \epsilon \) grows from 0 to \( \infty \), topological features are born (at \( b_i \)) and die (at \( d_i \)). The persistence diagram records pairs \( (b_i, d_i) \):

\[
\text{Persistence}(f) = d_i - b_i
\]

High persistence indicates robust topological structure. The Betti numbers \( \beta_k \) count features at each dimension:
- \( \beta_0 \): connected components (clusters)
- \( \beta_1 \): loops (circular structure)
- \( \beta_2 \): voids (enclosed cavities)

Wasserstein distance between persistence diagrams provides a metric for comparing topological structure across layers or conditions.

## Instruments under E09

### Activation Topology (`persistent_homology.py`)

Computes persistence diagrams for activations at each layer, identifying robust topological features.

**What it establishes:** Whether the activation manifold has non-trivial topology (loops, voids) that reveals geometric computation structure.
**What it does not establish:** The semantic meaning of topological features (requires pairing with task structure).

**Usage:**
```
uv run python persistent_homology.py --tasks ioi sva --max-dim 2
```

### Topological Complexity Profile

Tracks total persistence (sum of lifetimes) and Betti numbers across layers.

**What it establishes:** Where the network creates or destroys topological structure — simplification points and complexity peaks.
**What it does not establish:** Whether topological complexity correlates with task performance.

**Usage:**
```
uv run python persistent_homology.py --tasks ioi sva --profile
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High \( \beta_1 \) persistence | Stable loops — circular/periodic structure in representations |
| \( \beta_0 \) decreases through layers | Representations progressively merge clusters (categorization) |
| Topological simplification at layer \( \ell \) | Layer removes extraneous structure — decision boundary formation |
| Diagram matches task topology | Representation geometry mirrors the task's inherent structure |
| No persistent features | Representations are topologically trivial (convex, contractible) |

