---
title: "D01 — Faithfulness (CMD)"
description: "Measures whether an identified circuit faithfully reproduces the full model's behavior on the target task."
---

# D01 — Circuit Faithfulness

This framework asks: **Does the circuit actually do what we claim the full model does?**

Faithfulness is the foundational behavioral criterion: a circuit explanation is only meaningful if the circuit, run in isolation, reproduces the model's behavior on the task it was discovered from. Without faithfulness, any structural or mechanistic claim is unfalsifiable.

Circuit Metric Distance (CMD) quantifies faithfulness as the gap between the full model's output distribution and the circuit's output distribution, measured across a held-out prompt set. A complementary approach — corrupt-and-restore — verifies that restoring only the circuit components into a corrupted model recovers original performance.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | Defined faithfulness via knockout and patching on IOI |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC faithfulness score across circuit sizes |
| [Goldowsky-Dill et al., "Localizing Model Behavior"](https://arxiv.org/abs/2202.05262) | 2023 | Circuit sufficiency and necessity decomposition |
| [Miller et al., "Transformer Circuit Faithfulness Metrics"](https://arxiv.org/abs/2404.03214) | 2024 | Showed faithfulness depends on ablation method |

## Core concept

Faithfulness decomposes into two sub-properties. **Sufficiency**: running only the circuit components produces the correct output. **Necessity**: ablating the circuit degrades performance to chance. CMD combines both into a single scalar:

\[
\text{CMD} = \frac{1}{N} \sum_{i=1}^{N} d\bigl(p_{\text{full}}(y \mid x_i),\; p_{\text{circuit}}(y \mid x_i)\bigr)
\]

where \( d \) is a divergence measure (typically KL or Jensen-Shannon). A CMD near zero means the circuit is a faithful proxy for the full model on the task distribution.

The corrupt-restore variant measures the complementary direction: starting from a corrupted baseline (e.g., mean-ablated model), restoring only the circuit's edges, and measuring how much task performance recovers relative to the full model.

## Instruments under D01

### Circuit Metric Distance (`26_cmd.py`)

Computes CMD by running the identified circuit in isolation and comparing output distributions to the full model across the task prompt set.

**What it establishes:** Whether the circuit is a sufficient explanation of model behavior.
**What it does not establish:** Whether the circuit is minimal or mechanistically interpretable.

**Usage:**
```
uv run python 26_cmd.py --tasks ioi sva
```

### Corrupt-Restore Faithfulness (`20_corrupt_restore.py`)

Ablates all model components, then restores only circuit edges, measuring task metric recovery.

**What it establishes:** Whether the circuit is necessary and sufficient under corruption.
**What it does not establish:** Robustness to distribution shift or paraphrase.

**Usage:**
```
uv run python 20_corrupt_restore.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| CMD < 0.05 | Highly faithful — circuit reproduces model behavior |
| CMD 0.05–0.2 | Partially faithful — missing minor contributions |
| CMD > 0.5 | Poor faithfulness — circuit is incomplete |
| Restore recovery > 90% | Circuit is both necessary and sufficient |
| Restore recovery < 50% | Significant circuit components are missing |

