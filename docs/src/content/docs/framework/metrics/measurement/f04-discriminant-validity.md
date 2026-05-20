---
title: "F04 — Discriminant Validity"
description: "Tests whether metrics measuring different constructs produce uncorrelated results."
---

# F04 — Discriminant Validity

This framework asks: **Do methods that measure different things actually produce different answers?**

If a circuit importance score for IOI correlates just as highly with a circuit importance score for SVA as it does with another IOI method, then the metric is not discriminating between tasks — it may be capturing a generic property like head norm rather than task-specific circuit membership. Discriminant validity is the "different trait, same method" check.

A good measurement framework produces high convergent validity (F03) *and* high discriminant validity: metrics agree when they should, and disagree when they should. Without discriminant validity, a high score might simply mean "large weights" rather than "important for this specific task."

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Campbell & Fiske, "Convergent and discriminant validation by the MTMM matrix"](https://doi.org/10.1037/h0046016) | 1959 | Defined the MTMM framework including discriminant cells |
| [Messick, "Validity of Psychological Assessment"](https://doi.org/10.1037/0003-066X.50.9.741) | 1995 | Unified construct validity including discrimination |
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Task-specific causal structure in neural networks |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC — method-specific circuits per task |

## Core concept

Given importance scores from the same method applied to two different tasks \( T_1, T_2 \):

\[
r_{\text{disc}} = \text{Spearman}(\mathbf{a}^{T_1}, \mathbf{a}^{T_2})
\]

Discriminant validity requires \( r_{\text{disc}} < r_{\text{conv}} \). We compute the discriminant ratio:

\[
D = 1 - \frac{r_{\text{disc}}}{r_{\text{conv}}}
\]

Values of \( D > 0.5 \) indicate good discrimination: the metric captures task-specific structure rather than generic head properties. When \( D \approx 0 \), the same heads are flagged regardless of task, suggesting the method is insensitive to the construct.

## Metrics under F04

### Discriminant Validity (`17_discriminant_validity.py`)

Computes cross-task correlations for each circuit-identification method and compares them against within-task convergent correlations. Outputs the MTMM matrix with convergent (diagonal) and discriminant (off-diagonal) cells highlighted.

**What it establishes:** That circuit scores are task-specific — the method captures distinct constructs for distinct tasks.
**What it does not establish:** Which task decomposition is correct — only that the method differentiates between the tasks provided.

**Usage:**
```
uv run python 17_discriminant_validity.py --tasks ioi sva greater_than --methods weight activation
```

## Reading the scores

| Pattern | What it means |
|---|---|
| D > 0.6 | Strong discrimination — circuits are task-specific |
| D 0.3–0.6 | Moderate — some shared structure but meaningful differentiation |
| D < 0.3 | Weak — method may be capturing generic properties (norm, layer depth) |
| Cross-task kappa > within-task kappa | Method failure — "different" tasks produce more agreement than "same" task methods |

