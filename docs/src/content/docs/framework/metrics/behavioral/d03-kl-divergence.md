---
title: "D03 — KL Divergence"
description: "Measures the information-theoretic distance between the circuit's output distribution and the full model's distribution."
---

# D03 — KL Divergence

This framework asks: **How much information is lost when we replace the full model with its circuit approximation?**

KL divergence provides a principled, distribution-level measure of circuit fidelity. Unlike logit diff (D02), which focuses on a single pair of tokens, KL captures discrepancies across the entire vocabulary — including effects on non-target tokens that may reveal incomplete mechanistic understanding.

As an information-theoretic quantity, KL divergence has a natural interpretation: it measures the expected number of additional nats (or bits) needed to encode samples from the full model's distribution using the circuit's distribution as a codebook.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | KL as primary ACDC optimization target |
| [Goldowsky-Dill et al., "Localizing Model Behavior"](https://arxiv.org/abs/2202.05262) | 2023 | KL for sufficiency/necessity decomposition |
| [Geiger et al., "Causal Abstraction for Faithful Model Interpretability"](https://arxiv.org/abs/2301.04709) | 2023 | KL in interchange intervention settings |
| [Miller et al., "Transformer Circuit Faithfulness Metrics"](https://arxiv.org/abs/2404.03214) | 2024 | Comparison of KL vs other divergence measures |

## Core concept

Given the full model's output distribution \( P \) and the circuit's output distribution \( Q \) at position \( t \):

\[
D_{\text{KL}}(P \| Q) = \sum_{v \in \mathcal{V}} P(v) \log \frac{P(v)}{Q(v)}
\]

For circuit evaluation, we average over prompts in the task distribution:

\[
\overline{D}_{\text{KL}} = \frac{1}{N} \sum_{i=1}^{N} D_{\text{KL}}\bigl(p_{\text{full}}(\cdot \mid x_i) \;\|\; p_{\text{circuit}}(\cdot \mid x_i)\bigr)
\]

KL is asymmetric: \( D_{\text{KL}}(P \| Q) \) penalizes the circuit for assigning low probability to tokens the full model considers likely (mode-dropping), but not for assigning high probability to tokens the model ignores. This makes it a conservative faithfulness test — circuits that hallucinate extra probability mass may still score well.

## Metrics under D03

### Causal Scrubbing KL (`04_causal_scrubbing.py`)

Applies the causal scrubbing protocol, measuring KL between the scrubbed model (circuit-only computation preserved) and the full model.

**What it establishes:** Distributional faithfulness under the causal scrubbing intervention.
**What it does not establish:** Whether low KL is due to circuit completeness vs. task simplicity.

**Usage:**
```
uv run python 04_causal_scrubbing.py --tasks ioi sva
```

### Output Metric Variants (`21_output_variants.py`)

Computes multiple divergence measures (KL, reverse KL, Jensen-Shannon, total variation) to characterize where circuit and model disagree.

**What it establishes:** Whether KL results are robust to divergence measure choice.
**What it does not establish:** Causal mechanism — only distributional match.

**Usage:**
```
uv run python 21_output_variants.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| KL < 0.01 nats | Near-perfect distributional match |
| KL 0.01–0.1 nats | Good circuit, minor tail discrepancies |
| KL 0.1–1.0 nats | Meaningful distributional gaps — missing components |
| KL diverges across tasks | Circuit specialization varies by task |
| JS << KL | Mode-dropping dominates over hallucination |

