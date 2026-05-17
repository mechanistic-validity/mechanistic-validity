---
title: "E01 — Distributed Alignment Search (DAS-IIA)"
description: "Measures whether a learned linear subspace causally encodes a target variable via interchange intervention accuracy."
---

# E01 — Distributed Alignment Search & IIA

This framework asks: **Does a specific linear subspace in the model's residual stream causally encode a particular high-level variable?**

Interchange Intervention Accuracy (IIA) tests causal alignment between model representations and abstract causal variables. Rather than passively observing correlations, IIA actively intervenes: swap the representation from one input into another and check whether the model's output changes as predicted by the abstract causal model.

DAS extends this by learning the optimal rotation of the residual stream in which to intervene. Instead of assuming the variable is axis-aligned, DAS finds the direction that maximizes IIA — making it a constrained linear probe with causal validation built in.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Formalized IIA as causal alignment metric |
| [Geiger et al., "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"](https://arxiv.org/abs/2303.02536) | 2023 | Introduced DAS — learned rotation for distributed IIA |
| [Wu et al., "Interpretability at Scale"](https://arxiv.org/abs/2305.08809) | 2023 | Scaled DAS to large language models |
| [Sutter et al., "Nonlinear Causal Abstractions"](https://arxiv.org/abs/2305.18507) | 2023 | Critiqued linear IIA; proposed nonlinear extensions |

## Core concept

Given a high-level causal model \( \mathcal{C} \) with variable \( V \), DAS learns a rotation matrix \( R \in \mathbb{R}^{d \times d} \) and selects dimensions \( S \subseteq \{1, \ldots, d\} \) such that intervening on \( (Rh)_S \) maximizes:

\[
\text{IIA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ f\left(\text{do}(h^{(i)}, h^{(j)}, R, S)\right) = y^{(j)}_V \right]
\]

where \( \text{do}(h^{(i)}, h^{(j)}, R, S) \) replaces the \( S \)-dimensions of the rotated source with those from the base. High IIA means the subspace causally encodes \( V \); low IIA means the variable is either nonlinearly encoded or distributed across layers.

The key distinction from probing: a probe can achieve high accuracy on linearly decodable but causally inert directions. IIA requires that the direction actually *matters* for downstream computation.

## Instruments under E01

### DAS-IIA Core (`01_das_iia.py`)

Learns rotation \( R \) via gradient descent on IIA loss, then reports final IIA at each layer.

**What it establishes:** Whether a target variable has a clean linear causal encoding at a given site.
**What it does not establish:** Whether that encoding is the *only* pathway, or how the encoding is used downstream.

**Usage:**
```
uv run python 01_das_iia.py --tasks ioi sva
```

### IIA Variants (`15_iia_variants.py`)

Tests boundary conditions: multi-token variables, partial interventions, and nonlinear baselines.

**What it establishes:** Robustness of the linear encoding assumption across intervention granularities.
**What it does not establish:** Optimality of the learned subspace relative to all possible encodings.

**Usage:**
```
uv run python 15_iia_variants.py --tasks ioi sva --variants multi_token partial
```

### Multi-Axis IIA (`31_multi_axis_iia.py`)

Extends DAS to simultaneously align multiple causal variables, measuring orthogonality of their encodings.

**What it establishes:** Whether multiple variables occupy orthogonal subspaces or share directions.
**What it does not establish:** Causal interaction effects between variables.

**Usage:**
```
uv run python 31_multi_axis_iia.py --tasks ioi sva --n-variables 3
```

## Reading the scores

| Pattern | What it means |
|---|---|
| IIA > 0.9 at a single layer | Clean linear causal encoding localized to that layer |
| IIA > 0.9 only with multi-axis | Variable requires >1 dimension for faithful encoding |
| IIA ~ 0.5 everywhere | Variable not linearly encoded; try nonlinear extensions |
| High IIA but low probe accuracy | Causal direction diverges from readout direction |

