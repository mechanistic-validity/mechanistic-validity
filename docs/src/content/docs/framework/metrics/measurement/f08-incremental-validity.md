---
title: "F08 — Incremental Validity"
description: "Tests whether a circuit-discovery method adds predictive value beyond existing baselines."
---

# F08 — Incremental Validity

This framework asks: **Does our method tell us something that simpler baselines cannot?**

A circuit-discovery method may produce high faithfulness scores, but if a trivial baseline (random circuit, top-norm heads, all-heads-in-layer) achieves the same score, the method adds no value. Incremental validity quantifies the unique contribution of a method above and beyond what is already explained by baselines.

This is the "so what?" test. Many methods can identify circuits, but the practical question is whether the additional complexity of a given approach yields better predictions than methods that are cheaper, simpler, or already established. Incremental validity answers this by hierarchical comparison.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Hunsley & Meyer, "The incremental validity of psychological testing"](https://doi.org/10.1037/1040-3590.15.4.446) | 2003 | Framework for incremental validity in assessment |
| [Sechrest, "Incremental validity: A recommendation"](https://doi.org/10.1207/s15327906mbr1804_3) | 1963 | Original formulation of incremental validity |
| [Haynes & Lench, "Incremental validity of new clinical assessment measures"](https://doi.org/10.1037/1040-3590.15.4.456) | 2003 | Criteria for demonstrating added value |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | Baseline comparisons in circuit discovery |
| [Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery"](https://arxiv.org/abs/2310.10348) | 2023 | Hierarchical method comparison in mechanistic interpretability |

## Core concept

Let \( \theta_{\text{method}} \) be the faithfulness of the circuit identified by our method and \( \theta_{\text{baseline}} \) the faithfulness of the best baseline circuit at the same sparsity level. The incremental validity is:

\[
\Delta\theta = \theta_{\text{method}} - \theta_{\text{baseline}}
\]

To test significance, we use a paired comparison across tasks:

\[
t = \frac{\overline{\Delta\theta}}{SE(\Delta\theta)}, \quad SE = \frac{s_{\Delta}}{\sqrt{K}}
\]

where \( K \) is the number of tasks. We also report the proportion of tasks where the method strictly dominates all baselines, and the effect size (Cohen's \( d \)):

\[
d = \frac{\overline{\Delta\theta}}{s_{\text{pooled}}}
\]

## Metrics under F08

### Incremental Validity Analysis (`36_incremental_validity.py`)

Compares the target method's circuit against a hierarchy of baselines (random, top-norm, top-gradient, layer-wise) at matched sparsity. Reports the incremental gain, statistical significance, effect size, and the proportion of tasks where the method dominates.

**What it establishes:** That the method provides unique value — its circuits are better than what trivial heuristics produce.
**What it does not establish:** *Why* the method works better — only that it does.

**Usage:**
```
uv run python 36_incremental_validity.py --tasks ioi sva greater_than --baselines random norm gradient
```

## Reading the scores

| Pattern | What it means |
|---|---|
| d > 0.8, dominates on all tasks | Large practical gain — method clearly adds value |
| d 0.3–0.8, dominates on most tasks | Moderate gain — method is useful but not transformative |
| d < 0.3 | Small or negligible gain — method may not justify its complexity |
| Method loses to baseline on any task | Critical failure — investigate whether the method has a systematic blind spot |

