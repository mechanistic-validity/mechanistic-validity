---
title: "D06 — Cross-Task Transfer"
description: "Measures whether a circuit discovered on one task transfers faithfully to a different task."
---

# D06 — Cross-Task Transfer

This framework asks: **Is this circuit task-specific, or does it implement a general-purpose mechanism?**

Cross-task transfer tests whether a circuit identified for task A (e.g., IOI) also performs well on task B (e.g., subject-verb agreement). High transfer suggests the circuit implements a reusable algorithmic primitive — like "copy the attended token" — rather than a task-specific shortcut. Low transfer confirms task specialization and validates that the circuit is genuinely capturing the intended computation.

Both outcomes are informative. Transfer reveals shared computational structure across tasks; non-transfer validates specificity of the circuit discovery method.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | IOI circuit components reappear in related tasks |
| [Olsson et al., "In-context Learning and Induction Heads"](https://arxiv.org/abs/2209.11895) | 2022 | Induction heads transfer across sequence types |
| [Geiger et al., "Causal Abstraction for Faithful Model Interpretability"](https://arxiv.org/abs/2301.04709) | 2023 | Interchange intervention generalization across inputs |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | Circuit overlap statistics across ACDC tasks |

## Core concept

Given a circuit \( C_A \) discovered on task A with metric \( M_A \), cross-task transfer is:

\[
T_{A \to B} = \frac{M_B(C_A)}{M_B(C_B)}
\]

where \( M_B(C_A) \) is the performance of circuit \( C_A \) evaluated on task B's metric, and \( M_B(C_B) \) is the performance of the circuit discovered directly on task B. Transfer is asymmetric: \( T_{A \to B} \neq T_{B \to A} \) in general, because tasks may share mechanisms in one direction but not the other.

The transfer matrix across all task pairs reveals clusters of tasks that share computational structure — a signature of modular, reusable circuits in the model.

## Instruments under D06

### Cross-Task IIA Transfer (`32_cross_task_iia_transfer.py`)

Discovers a circuit via interchange intervention on task A, then evaluates its faithfulness (logit diff recovery, KL) on task B without re-optimization.

**What it establishes:** Whether shared circuit structure exists across tasks.
**What it does not establish:** Why transfer occurs — mechanism-level explanation requires further analysis.

**Usage:**
```
uv run python 32_cross_task_iia_transfer.py --tasks ioi sva greater_than
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Transfer > 80% | Strong shared mechanism between tasks |
| Transfer 40–80% | Partial overlap — shared primitives, different composition |
| Transfer < 20% | Tasks use distinct circuits |
| Asymmetric transfer | One task's circuit subsumes the other's |
| Cluster structure in transfer matrix | Modular computational primitives |

