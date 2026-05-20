---
title: "C05 — Information Bottleneck"
description: "Measures how efficiently circuits compress input information while preserving task-relevant signal."
---

# C05 — Information Bottleneck

This framework asks: **How much does a circuit compress its input, and how much task-relevant information survives that compression?**

The Information Bottleneck (IB) principle characterizes the optimal tradeoff between compression and prediction. A circuit that achieves low description length while retaining high MI with the task output has found an efficient representation. This connects directly to effective model complexity: circuits with fewer active components that still achieve high task performance sit on the IB frontier.

In practice, the local learning coefficient (LLC) measures a related quantity — the effective dimensionality of the loss landscape around a solution. Circuits on the IB frontier should have lower LLC, indicating they use fewer effective parameters.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Tishby et al., "The Information Bottleneck Method"](https://arxiv.org/abs/physics/0004057) | 1999 | Original IB formulation |
| [Shwartz-Ziv & Tishby, "Opening the Black Box of Deep Neural Networks"](https://arxiv.org/abs/1703.00810) | 2017 | IB applied to deep network layers |
| [Saxe et al., "On the Information Bottleneck Theory of Deep Learning"](https://arxiv.org/abs/1801.09125) | 2018 | Critical analysis and phase transitions |
| [Lau et al., "Quantifying Local Learning Coefficient"](https://arxiv.org/abs/2308.12108) | 2023 | LLC as effective complexity measure |
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Minimal sufficient representations in circuits |

## Core concept

The IB objective finds a compressed representation \( T \) of input \( X \) that maximizes information about target \( Y \):

\[ \max_{p(t|x)} \; I(T; Y) - \beta \, I(T; X) \]

The Lagrange multiplier \( \beta \) controls the compression-prediction tradeoff. At \( \beta \to 0 \), maximal compression (T is trivial); at \( \beta \to \infty \), T preserves all of X.

For a circuit with \( k \) active components, we can frame the circuit itself as the bottleneck: \( I(\text{circuit}; Y) \) is the task performance, and \( I(\text{circuit}; X) \) relates to the circuit's capacity (proportional to \( k \) and the effective dimensionality). The LLC provides a tractable proxy for this capacity.

## Metrics under C05

### LLC Script (`10_llc.py`)

Estimates the local learning coefficient around the trained circuit, measuring the effective number of parameters the model uses. Lower LLC indicates a more compressed (bottlenecked) solution.

**What it establishes:** Whether the circuit operates efficiently on the IB frontier — high performance with low effective complexity.
**What it does not establish:** The explicit IB curve or whether alternative circuits with better tradeoffs exist.

**Usage:**
```
uv run python 10_llc.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| Low LLC + high task performance | Efficient circuit on the IB frontier |
| High LLC + high performance | Over-parameterized; compression possible |
| Low LLC + low performance | Under-powered; too compressed |
| LLC drops when removing a head | That head was not contributing effective complexity |

## Connection to other frameworks

The IB perspective complements [C01 (MI)](/framework/metrics/information/c01-mutual-information/) by adding the compression axis: it is not enough for a component to carry information — it must do so efficiently. [C04 (PID)](/framework/metrics/information/c04-pid/) redundancy identifies where compression is possible (redundant heads can be removed without information loss). The [structural pillar](/framework/metrics/structural/) measures circuit size directly, while IB provides the information-theoretic justification for minimality.
