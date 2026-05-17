---
title: "C08 — Observational Causal Sensitivity Estimation"
description: "Estimates causal influence between circuit components using only observational activation data."
---

# C08 — Observational Causal Sensitivity Estimation

This framework asks: **How sensitive is each downstream component to natural variation in upstream components, without any interventions?**

Observational Causal Sensitivity Estimation (OCSE) measures directed influence between circuit components by exploiting natural variation in activations across inputs. Rather than intervening (ablating or patching), OCSE estimates how much the output would change given a perturbation to an upstream component, using only observed activation covariance. This provides a computationally cheap proxy for causal importance that requires no forward passes beyond the initial activation collection.

OCSE bridges the gap between purely correlational measures (MI) and expensive interventional methods (activation patching). It provides directed importance scores that approximate what interventions would reveal, at a fraction of the computational cost.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Schwab & Bhatt, "CXPlain: Causal Explanations for Model Interpretation"](https://arxiv.org/abs/1910.12336) | 2019 | Learning causal importance from observational data |
| [Janzing et al., "Quantifying Causal Influences"](https://doi.org/10.1214/12-AOS1010) | 2013 | Information-geometric causal influence |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2009 | do-calculus and observational identification |
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Connecting observational and interventional circuit analysis |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC uses edge-level observational statistics |

## Core concept

For a source component with activation \( a_s \) and a target (output logit or downstream component) \( y \), OCSE estimates the causal sensitivity:

\[ \mathrm{OCSE}(s \to y) = \mathbb{E}\left[ \left\| \frac{\partial y}{\partial a_s} \right\|^2 \cdot \mathrm{Var}(a_s) \right] \]

This combines the local gradient (how strongly the target responds to source changes) with the natural variance (how much the source actually varies across inputs). The product gives an estimate of the expected change in \( y \) due to natural fluctuations in \( a_s \).

When gradients are unavailable or expensive, a regression-based estimator fits \( y = f(a_s) + \epsilon \) and uses the explained variance \( R^2 \) as the sensitivity score. Both approaches yield directed scores without requiring any ablation runs.

## Instruments under C08

### OCSE Script (`07_ocse.py`)

Directly implements observational causal sensitivity estimation. Collects activations from circuit heads across a corpus, then estimates pairwise directed influence using gradient-variance products or regression-based explained variance.

**What it establishes:** Directed importance scores between components using only observational data — a cheap proxy for interventional effects.
**What it does not establish:** True causal necessity; OCSE can be confounded by unobserved common causes and cannot distinguish direct from indirect effects.

**Usage:**
```
uv run python 07_ocse.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| High OCSE(head A -> output) | Head A's natural variation strongly predicts output changes |
| High OCSE(A -> B) but low OCSE(B -> A) | Directed information flow from A to B |
| OCSE scores match knockout ordering | Observational proxy is faithful to interventional ground truth |
| OCSE high but knockout effect is low | Redundancy — the effect is absorbed by other components |

## Connection to other frameworks

OCSE implements a practical version of [C07 (Granger Causality)](/framework/instruments_v2/information/c07-granger-causality/) with flexibility for nonlinear dependencies. Its directed scores should correlate with [C03 (Transfer Entropy)](/framework/instruments_v2/information/c03-transfer-entropy/) but are cheaper to compute. Discrepancies between OCSE rankings and interventional results from the [causal pillar](/framework/instruments_v2/causal/) reveal redundancy or backup circuits. [C09 (NOTEARS)](/framework/instruments_v2/information/c09-notears/) can use OCSE scores as edge priors for DAG structure learning.
