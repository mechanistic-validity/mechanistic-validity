---
title: "C07 — Granger Causality"
description: "Tests whether one circuit component's past activations improve prediction of another's future state."
---

# C07 — Granger Causality

This framework asks: **Does knowing a component's earlier-layer activation improve our prediction of a downstream component, beyond what the downstream component's own history provides?**

Granger causality (GC) is a statistical notion of predictive causality: X Granger-causes Y if past values of X contain information about future values of Y that is not already contained in past values of Y alone. Applied to transformer circuits, where "time" is layer depth, GC tests whether an earlier component's output carries predictive signal for a later component's state.

GC is attractive because it requires only observational data — no interventions. This makes it computationally cheap and applicable to any recorded activation trace. However, it captures predictive relationships, not true causal mechanisms — a limitation addressed by the interventional metrics in the causal pillar.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Granger, "Investigating Causal Relations by Econometric Models and Cross-spectral Methods"](https://doi.org/10.2307/1912791) | 1969 | Original Granger causality definition |
| [Barnett et al., "Granger Causality and Transfer Entropy Are Equivalent for Gaussian Variables"](https://doi.org/10.1103/PhysRevLett.103.238701) | 2009 | Equivalence to transfer entropy |
| [Tank et al., "Neural Granger Causality"](https://doi.org/10.1109/TPAMI.2021.3065601) | 2022 | Nonlinear GC via neural networks |
| [Schwab & Bhatt, "CXPlain: Causal Explanations for Model Interpretation"](https://arxiv.org/abs/1910.12336) | 2019 | Granger-style importance for neural network components |

## Core concept

Variable \( X \) Granger-causes \( Y \) if:

\[ H(Y_t \mid Y_{t-1}, \ldots, Y_{t-p}) > H(Y_t \mid Y_{t-1}, \ldots, Y_{t-p}, X_{t-1}, \ldots, X_{t-p}) \]

In the linear Gaussian case, this reduces to comparing the residual variance of two autoregressive models — one with and one without the candidate cause. The F-statistic for this comparison provides a significance test.

For transformer circuits with layer index \( \ell \) as the time axis, let \( A_i^\ell \) be the activation of head \( i \) at layer \( \ell \). Head \( i \) Granger-causes head \( j \) if:

\[ \mathrm{Var}(A_j^{\ell+k} \mid A_j^{\ell+k-1}) > \mathrm{Var}(A_j^{\ell+k} \mid A_j^{\ell+k-1}, A_i^\ell) \]

across the corpus, tested via the F-statistic.

## Metrics under C07

### OCSE Script (`07_ocse.py`)

Observational causal sensitivity estimation implements a form of predictive causality by measuring how natural variation in one component's activation predicts variation in downstream components — the nonparametric analogue of Granger causality.

**What it establishes:** Predictive directed relationships between circuit components without intervention.
**What it does not establish:** True causal necessity — Granger causality can be confounded by common inputs from the residual stream.

**Usage:**
```
uv run python 07_ocse.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| Significant GC from head A to head B | A's activation predicts B beyond B's own context |
| Bidirectional GC | Likely driven by a shared upstream cause |
| GC only at specific layer gaps | Characteristic information relay depth |
| No GC despite same-circuit membership | Parallel computation without interaction |

## Connection to other frameworks

Granger causality is the linear-model counterpart of [C03 (Transfer Entropy)](/framework/metrics/information/c03-transfer-entropy/) — they are equivalent for Gaussian processes but TE captures nonlinear dependencies. [C08 (OCSE)](/framework/metrics/information/c08-ocse/) implements a flexible version of predictive causality. When GC identifies a link, [C09 (NOTEARS)](/framework/metrics/information/c09-notears/) can incorporate it as a constraint in DAG recovery, and the [causal pillar](/framework/metrics/causal/) validates whether the link is functionally necessary via intervention.
