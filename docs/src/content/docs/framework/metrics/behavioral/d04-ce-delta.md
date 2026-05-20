---
title: "D04 — CE Delta"
description: "Measures the change in cross-entropy loss when the circuit is ablated from the model."
---

# D04 — Cross-Entropy Delta

This framework asks: **How much does the model's prediction quality degrade when we remove the circuit?**

Cross-entropy delta measures the causal importance of a circuit by quantifying how much worse the model predicts when circuit components are ablated. Unlike faithfulness metrics that compare circuit-in-isolation to the full model, CE delta measures the damage of circuit removal — a necessity test rather than a sufficiency test.

This metric connects circuit discovery directly to language modeling performance, making results interpretable in the same units (nats per token) used to evaluate model quality. A large CE delta means the circuit is critical; a small delta means other components can compensate.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | Loss increase under knockout as necessity measure |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | CE-based edge scoring in ACDC |
| [Miller et al., "Transformer Circuit Faithfulness Metrics"](https://arxiv.org/abs/2404.03214) | 2024 | Ablation method affects CE delta magnitude |
| [Hanna et al., "How does GPT-2 compute greater-than?"](https://arxiv.org/abs/2305.00586) | 2023 | Per-component CE contribution in arithmetic circuits |

## Core concept

Given a model with parameters \( \theta \), a circuit \( C \subseteq \theta \), and an ablation function \( a \) (zero, mean, or resample), the CE delta is:

\[
\Delta_{\text{CE}} = \mathcal{L}(\theta \setminus C;\, a) - \mathcal{L}(\theta)
\]

where \( \mathcal{L} \) is the cross-entropy loss averaged over the evaluation set. Positive values indicate the circuit contributes to prediction quality. Per-token decomposition reveals where the circuit matters most:

\[
\Delta_{\text{CE}}^{(t)} = -\log p_{\text{ablated}}(x_t \mid x_{<t}) + \log p_{\text{full}}(x_t \mid x_{<t})
\]

The magnitude of CE delta depends on the ablation method. Mean ablation typically produces smaller deltas than zero ablation because the mean preserves first-order statistics. Resampling ablation provides an unbiased estimate but has higher variance.

## Metrics under D04

### Output Variants — CE Mode (`21_output_variants.py`)

Computes CE delta under multiple ablation strategies (zero, mean, resample) for the identified circuit.

**What it establishes:** The causal necessity of the circuit for language modeling performance.
**What it does not establish:** Sufficiency — other circuits may produce similar CE improvement.

**Usage:**
```
uv run python 21_output_variants.py --tasks ioi sva --metric ce_delta
```

## Reading the scores

| Pattern | What it means |
|---|---|
| CE delta > 2.0 nats | Circuit is critical — model breaks without it |
| CE delta 0.5–2.0 nats | Significant contribution, partial redundancy |
| CE delta < 0.1 nats | Circuit is not necessary (other paths compensate) |
| Zero ablation >> mean ablation | First-order statistics carry most signal |
| High variance across prompts | Circuit importance is context-dependent |

