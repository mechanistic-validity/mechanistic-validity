---
title: "D02 — Logit Diff Recovery"
description: "Measures how much of the model's logit difference between correct and incorrect answers the circuit recovers."
---

# D02 — Logit Diff Recovery

This framework asks: **Does the circuit recover the model's preference for the correct answer over the incorrect one?**

Logit difference is the canonical scalar metric for binary-choice tasks in mechanistic interpretability. Rather than comparing full output distributions, it isolates the model's "confidence margin" — the gap between the logit assigned to the correct token and the logit assigned to the strongest distractor. Recovery of this gap by a subcircuit is the most widely reported faithfulness metric in the literature.

Logit diff recovery is more targeted than full-distribution faithfulness (D01): a circuit might match the model's top-1 prediction without matching the full tail, or vice versa. This instrument captures the task-relevant signal specifically.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | Introduced logit diff as primary IOI metric |
| [Hanna et al., "How does GPT-2 compute greater-than?"](https://arxiv.org/abs/2305.00586) | 2023 | Logit diff for ordinal comparison tasks |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | Logit diff recovery as circuit quality score |
| [Olsson et al., "In-context Learning and Induction Heads"](https://arxiv.org/abs/2209.11895) | 2022 | Task-specific logit attribution in induction |

## Core concept

For a binary task with correct answer \( t^+ \) and incorrect answer \( t^- \), the logit difference is:

\[
\Delta_{\text{logit}} = \text{logit}(t^+) - \text{logit}(t^-)
\]

Recovery is the fraction of the full model's logit diff preserved by the circuit:

\[
R = \frac{\Delta_{\text{logit}}^{\text{circuit}}}{\Delta_{\text{logit}}^{\text{full}}}
\]

A recovery of \( R = 1.0 \) means the circuit perfectly reproduces the model's decision margin. Values above 1.0 indicate the circuit over-attributes (other components partially cancel the signal). Mean-centered logit variants subtract the mean logit across vocabulary to control for bias shifts.

## Instruments under D02

### Activation Patching Recovery (`02_activation_patching.py`)

Patches activations from clean into corrupted runs, measuring logit diff recovery per component. The aggregate patching score IS the logit diff recovery of the identified circuit.

**What it establishes:** Per-component contribution to the model's decision margin.
**What it does not establish:** Whether the mechanism is interpretable or minimal.

**Usage:**
```
uv run python 02_activation_patching.py --tasks ioi sva
```

### Mean-Centered Logit (`22_mean_centered_logit.py`)

Computes logit diff after subtracting the vocabulary-mean logit, removing constant bias from unembedding.

**What it establishes:** That recovery is not inflated by position-independent biases.
**What it does not establish:** Distributional faithfulness beyond the top-1 margin.

**Usage:**
```
uv run python 22_mean_centered_logit.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Recovery 95–105% | Circuit fully explains the decision margin |
| Recovery 70–95% | Circuit captures main mechanism, minor paths missing |
| Recovery > 110% | Over-attribution — negative contributors excluded |
| Recovery < 50% | Circuit misses primary computation pathway |
| Mean-centered differs from raw | Bias component contributes substantially |

