---
title: "E03 — Representational Similarity Analysis (RSA)"
description: "Compares the geometry of neural representations by correlating pairwise distance matrices across conditions or models."
---

# E03 — Representational Similarity Analysis

This framework asks: **Do two representational spaces organize stimuli in the same geometric pattern?**

RSA abstracts away the specific basis of a representation and instead compares the *relational structure* — which inputs are close to which. By computing pairwise dissimilarity matrices (RDMs) and correlating them, RSA can compare representations across layers, models, or even modalities without requiring aligned dimensions.

For circuit analysis, RSA reveals whether a circuit subspace organizes inputs the same way as a task-relevant variable. If the RDM of a circuit head's activations correlates with the RDM predicted by the task structure, that head likely encodes task-relevant geometry.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Kriegeskorte et al., "Representational similarity analysis"](https://doi.org/10.3389/neuro.06.004.2008) | 2008 | Introduced RSA framework for neural data |
| [Kriegeskorte & Kievit, "Representational geometry"](https://doi.org/10.1016/j.tics.2013.06.007) | 2013 | Formalized representational geometry theory |
| [Diedrichsen & Kriegeskorte, "Representational models"](https://doi.org/10.1371/journal.pcbi.1005508) | 2017 | Extended RSA to model-based hypothesis testing |
| [Shahbazi et al., "Using distance on the Riemannian manifold to compare representations"](https://arxiv.org/abs/2112.09391) | 2021 | Geometric corrections for RDM comparison |

## Core concept

Given \( n \) stimuli and representations \( H \in \mathbb{R}^{n \times d} \), the RDM is:

\[
\text{RDM}_{ij} = d(h_i, h_j)
\]

where \( d \) is typically cosine distance or Euclidean distance. RSA then measures the second-order similarity between two RDMs using Spearman or Kendall correlation:

\[
\text{RSA}(H_1, H_2) = \rho_{\text{Spearman}}\left(\text{vec}(\text{RDM}_1),\, \text{vec}(\text{RDM}_2)\right)
\]

High RSA between a circuit subspace and a task-model RDM (constructed from the known task variables) indicates geometric alignment. The method is basis-invariant — rotations of the representation do not affect the RDM.

## Instruments under E03

### RSA Model Comparison (`rsa_analysis.py`)

Computes RDMs at each layer and correlates them against hypothesis RDMs derived from task structure (e.g., "same IO name" for IOI, "singular vs plural" for SVA).

**What it establishes:** Whether representational geometry at a layer matches task-predicted structure.
**What it does not establish:** Whether that geometry is causally necessary (use E01 for that).

**Usage:**
```
uv run python rsa_analysis.py --tasks ioi sva --metric cosine
```

### Cross-Layer RSA

Computes RSA between all pairs of layers, producing a layer-by-layer similarity matrix that reveals representational transitions.

**What it establishes:** Where in the network the representation undergoes qualitative geometric shifts.
**What it does not establish:** What information is gained or lost at each transition.

**Usage:**
```
uv run python rsa_analysis.py --tasks ioi sva --cross-layer
```

## Reading the scores

| Pattern | What it means |
|---|---|
| RSA > 0.7 with task model | Layer geometry strongly reflects task structure |
| RSA drops between layers \( \ell \) and \( \ell+1 \) | Major representational transformation at that boundary |
| RSA high across many layers | Task structure is preserved passively through the residual stream |
| RSA near zero everywhere | Task variables not geometrically organized (may be nonlinear) |

