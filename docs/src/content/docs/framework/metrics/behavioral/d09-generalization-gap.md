---
title: "D09 — Generalization Gap"
description: "Measures how sensitive the circuit discovery result is to hyperparameters, prompt sampling, and methodological choices."
---

# D09 — Generalization Gap

This framework asks: **How fragile is this circuit — would a different researcher, making slightly different choices, find the same thing?**

The generalization gap quantifies the difference between a circuit's performance on its discovery distribution and its performance on held-out evaluation conditions. This includes sensitivity to hyperparameters (threshold, sparsity penalty), prompt sampling (different random seeds for the evaluation set), and methodological variants (different ablation types, different scoring functions). A large gap indicates that the circuit is overfit to the specific discovery conditions.

This is the meta-metric of the behavioral pillar: it does not measure a single property but quantifies the robustness of all other measurements to researcher degrees of freedom.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | Sensitivity of ACDC to threshold hyperparameter |
| [Miller et al., "Transformer Circuit Faithfulness Metrics"](https://arxiv.org/abs/2404.03214) | 2024 | Faithfulness varies with ablation method choice |
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | Manual vs. automated circuit discovery yield different results |
| [Goldowsky-Dill et al., "Localizing Model Behavior"](https://arxiv.org/abs/2202.05262) | 2023 | Robustness of localization to evaluation set |

## Core concept

The generalization gap is defined as:

\[
G = M_{\text{discovery}} - M_{\text{held-out}}
\]

where \( M_{\text{discovery}} \) is the circuit's faithfulness on the prompts/settings used during discovery, and \( M_{\text{held-out}} \) is faithfulness on a fresh evaluation set. This can be decomposed into sources:

\[
G = G_{\text{prompt}} + G_{\text{hyperparam}} + G_{\text{method}}
\]

Hyperparameter sensitivity measures how much the discovered circuit changes as we vary the discovery threshold \( \tau \):

\[
\text{Sensitivity} = \frac{\partial |C(\tau)|}{\partial \tau} \cdot \frac{\tau}{|C(\tau)|}
\]

A high sensitivity elasticity means small threshold changes produce large circuit changes — a sign that the boundary between "in-circuit" and "out-of-circuit" is arbitrary.

## Metrics under D09

### Hyperparameter Sensitivity (`29_hyperparam_sensitivity.py`)

Sweeps the circuit discovery threshold and measures how circuit size, composition, and faithfulness change.

**What it establishes:** Whether the circuit boundary is robust or arbitrary.
**What it does not establish:** Which threshold is "correct" — only whether the result is stable.

**Usage:**
```
uv run python 29_hyperparam_sensitivity.py --tasks ioi sva
```

### Resample Complement (`35_resample_complement.py`)

Re-runs circuit discovery on different random subsets of the prompt distribution and measures agreement between independently discovered circuits.

**What it establishes:** Whether the circuit is a stable property of the model or of the specific evaluation sample.
**What it does not establish:** Robustness to task definition changes (see D06).

**Usage:**
```
uv run python 35_resample_complement.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Gap < 5% | Robust — circuit generalizes beyond discovery conditions |
| Gap 5–15% | Moderate overfitting to discovery settings |
| Gap > 25% | Fragile — results depend heavily on researcher choices |
| High resample agreement (> 80%) | Circuit is stable across prompt samples |
| Threshold sensitivity > 2.0 | Circuit boundary is arbitrary |

