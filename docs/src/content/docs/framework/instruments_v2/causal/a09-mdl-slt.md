---
title: "A09 — MDL / Singular Learning Theory"
description: "Minimum Description Length and Singular Learning Theory applied to circuits: measuring effective complexity and degeneracy via the Local Learning Coefficient."
---

# A09 — MDL / Singular Learning Theory

This framework asks: **what is the effective complexity of each circuit component — is it geometrically simple (specialized) or degenerate (polyfunctional)?**

Singular Learning Theory (SLT) provides a geometric characterization of neural network parameters at convergence. The Local Learning Coefficient (LLC) — also called the real log canonical threshold (RLCT) — measures how many effective parameters a component uses relative to its nominal parameter count. A component with low LLC is geometrically simple: it sits near a low-dimensional singularity in parameter space, consistent with implementing a single clean function. A component with high LLC is geometrically complex: it occupies a high-dimensional region, consistent with polyfunctionality or redundancy.

The MDL connection: LLC directly controls the model's Bayesian Information Criterion at the component level. Lower LLC means shorter description length — the component can be described with fewer bits. This maps onto the intuition that interpretable circuits should be *compressible*: a head that implements a single algorithmic role (e.g., "copy the previous token") should have lower effective complexity than a head juggling multiple unrelated tasks.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Watanabe, *Algebraic Geometry and Statistical Learning Theory*](https://doi.org/10.1017/CBO9780511800474) | 2009 | SLT: real log canonical threshold governs generalization in singular models |
| [Lau et al., arXiv 2310.19470](https://arxiv.org/abs/2310.19470) | 2023 | devinterp: practical LLC estimation for neural networks |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Circuit components as functional units amenable to complexity analysis |
| [Olsson et al., "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 2022 | Phase transitions in learning; induction heads as discrete mechanistic structures |

## Core concept: the Local Learning Coefficient

For a model with parameter \( w \) near a critical point \( w_0 \), the LLC \( \lambda \) governs the asymptotic free energy:

\[
F_n = nL_n(w_0) + \lambda \log n + O(\log \log n)
\]

where \( L_n \) is the empirical loss. The LLC is estimated via the SGLD (Stochastic Gradient Langevin Dynamics) trace:

\[
\hat{\lambda} = \frac{n}{m} \left( \hat{L}_n^{\text{SGLD}} - L_n(w_0) \right)
\]

For a regular (non-singular) model, \( \lambda = d/2 \) (half the parameter count). For singular models — which neural networks always are — \( \lambda \) can be much smaller, reflecting the low effective dimensionality of the parameter region. Components that implement clean, specialized functions tend to sit near lower-dimensional singularities and thus have lower LLC.

Hyperparam sensitivity (varying learning rate, data subset, initialization) complements LLC by measuring functional stability: a component whose role changes drastically with small hyperparameter changes is likely sitting in a high-dimensional, degenerate region.

## Instruments under A09

### C10 — LLC Estimation (`10_llc.py`)

Estimates the local learning coefficient for each circuit component by running SGLD chains from the trained weights and measuring the gap between the SGLD average loss and the MAP loss:

\[
\hat{\lambda}(c) = \frac{n}{m} \left( \frac{1}{m} \sum_{t=1}^m L_n(w_t) - L_n(w_0) \right)
\]

where \( w_t \) are SGLD samples restricted to component \( c \)'s parameters.

**What it establishes:** Effective complexity/dimensionality of each component's parameter region. Low LLC indicates a specialized, interpretable role; high LLC indicates polyfunctionality or redundancy.

**What it does not establish:** *What* function the component implements (only its geometric complexity). Must be paired with A05 or A01 for functional characterization.

**Usage:**
```
uv run python 10_llc.py --tasks ioi sva --n-prompts 40
```

### C29 — Hyperparameter Sensitivity (`29_hyperparam_sensitivity.py`)

Measures how stable a component's causal importance score is across hyperparameter variations (ablation method, prompt count, corruption type). High sensitivity indicates the component's role is fragile or method-dependent:

\[
\text{Sensitivity}(c) = \text{CV}\left[ AP(c; \theta_1), \ldots, AP(c; \theta_k) \right]
\]

where \( \theta_i \) are different hyperparameter settings and CV is the coefficient of variation.

**What it establishes:** Robustness of causal conclusions to methodological choices.

**What it does not establish:** Which hyperparameter setting is "correct" — only the variance across them.

**Usage:**
```
uv run python 29_hyperparam_sensitivity.py --tasks ioi sva --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Low LLC, low sensitivity | Clean specialized component; likely interpretable |
| High LLC, low sensitivity | Complex but stable; polyfunctional component reliably used |
| Low LLC, high sensitivity | Specialized but fragile; role depends on exact conditions |
| High LLC, high sensitivity | Degenerate and unstable; difficult to interpret reliably |

## Connection to other frameworks

A09 provides a complexity characterization that complements A05's (MDC/Glennan) mechanistic claims: a component hypothesized to implement a simple logic gate should have low LLC. A04 (Woodward) measures robustness to intervention method; A09's hyperparam sensitivity measures robustness to evaluation method — both address reliability of causal claims. A08 (PID) identifies redundancy between components; high redundancy predicts that the redundant components sit in a degenerate (high-LLC) parameter region where permuting them does not change the loss.
