---
title: "D05 — Top-K Accuracy"
description: "Measures whether the circuit preserves the model's top-K predicted tokens at the task-critical position."
---

# D05 — Top-K Accuracy

This framework asks: **Does the circuit preserve the model's most confident predictions?**

Top-K accuracy provides a discrete, human-interpretable faithfulness measure: does the circuit's output ranking agree with the full model's top predictions? While KL divergence (D03) and CE delta (D04) capture continuous distributional differences, top-K accuracy answers the practical question of whether the circuit gets the "right answer" — where "right" means agreeing with the full model's top choices.

This metric is especially informative for tasks with multiple valid completions. A circuit may have moderate KL divergence but perfect top-5 accuracy, indicating it captures the core mechanism while differing only in probability mass allocation among low-ranked tokens.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | Accuracy-based metrics alongside logit diff |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | Top-1 accuracy as circuit quality threshold |
| [Goldowsky-Dill et al., "Localizing Model Behavior"](https://arxiv.org/abs/2202.05262) | 2023 | Ranking preservation under intervention |
| [Olsson et al., "In-context Learning and Induction Heads"](https://arxiv.org/abs/2209.11895) | 2022 | Token prediction accuracy for induction tasks |

## Core concept

Let \( \text{top}_K^{\text{full}}(x) \) be the set of K highest-probability tokens under the full model, and \( \text{top}_K^{\text{circuit}}(x) \) the same under the circuit. Top-K accuracy is:

\[
\text{Acc}@K = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{top}_K^{\text{full}}(x_i) \cap \text{top}_K^{\text{circuit}}(x_i)|}{K}
\]

For the special case \( K=1 \), this reduces to exact-match accuracy: does the circuit predict the same top token as the full model? Weighted variants incorporate rank correlation (Kendall's \( \tau \)) over the top-K set:

\[
\tau_K = \text{KendallTau}\bigl(\text{rank}_K^{\text{full}},\; \text{rank}_K^{\text{circuit}}\bigr)
\]

## Metrics under D05

### Output Variants — Top-K Mode (`21_output_variants.py`)

Evaluates top-1, top-5, and top-10 agreement between circuit and full model outputs, plus rank correlation within the top-K set.

**What it establishes:** Whether the circuit preserves the model's discrete predictions.
**What it does not establish:** Probability calibration — two distributions can agree on ranking but differ in confidence.

**Usage:**
```
uv run python 21_output_variants.py --tasks ioi sva --metric topk
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Top-1 accuracy > 95% | Circuit reproduces the model's best guess |
| Top-5 accuracy > 90% | Core ranking structure preserved |
| Top-1 high but top-10 low | Circuit captures dominant mode, misses alternatives |
| Rank correlation > 0.9 | Fine-grained ordering within top-K preserved |
| Top-K drops sharply with K | Circuit is narrow — captures one mechanism only |

