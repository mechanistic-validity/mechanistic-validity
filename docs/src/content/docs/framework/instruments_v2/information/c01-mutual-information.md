---
title: "C01 — Mutual Information"
description: "Measures the total shared information between circuit components and task performance."
---

# C01 — Mutual Information

This framework asks: **How much information do circuit components share with each other and with the task output?**

Mutual information (MI) quantifies the reduction in uncertainty about one variable given knowledge of another. In the context of circuit analysis, MI between a circuit head's activations and the model's task-relevant output tells us how much that component contributes to the computation. High MI indicates the component carries task-relevant signal; low MI suggests it is peripheral.

Unlike correlation, MI captures arbitrary (including nonlinear) statistical dependencies. This makes it particularly suited to transformer circuits where information flow is mediated by nonlinear attention patterns and MLP activations.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Shannon, "A Mathematical Theory of Communication"](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) | 1948 | Foundational definition of entropy and mutual information |
| [Cover & Thomas, *Elements of Information Theory*](https://doi.org/10.1002/047174882X) | 2006 | Comprehensive treatment of MI estimation and properties |
| [Kraskov et al., "Estimating Mutual Information"](https://doi.org/10.1103/PhysRevE.69.066138) | 2004 | KSG estimator for continuous variables |
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Information-theoretic view of circuit interventions |

## Core concept

Mutual information between random variables \( X \) and \( Y \) is defined as:

\[ I(X; Y) = H(X) - H(X \mid Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \]

For continuous activations (as in transformer hidden states), we use the KSG nearest-neighbor estimator or binning approaches. Given a circuit head with activation vector \( \mathbf{a} \) and model logit output \( \mathbf{y} \), we estimate \( I(\mathbf{a}; \mathbf{y}) \) across a corpus of inputs.

The normalized variant \( \mathrm{NMI}(X; Y) = I(X;Y) / \sqrt{H(X) H(Y)} \) allows comparison across components with different activation scales.

## Instruments under C01

### PID Script (`08_pid.py`)

PID decomposes mutual information into unique, redundant, and synergistic components. The total MI \( I(\{X_1, X_2\}; Y) \) is recovered as the sum of all PID atoms.

**What it establishes:** Total information shared between pairs of circuit heads and the task output.
**What it does not establish:** Directionality of information flow or whether MI reflects causal contribution.

**Usage:**
```
uv run python 08_pid.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| High MI between head and output | Component carries strong task-relevant signal |
| Near-zero MI | Component is informationally decoupled from the task |
| MI(head_A; head_B) >> MI(head_A; output) | Head A informs head B but not the output directly |
| Uniform MI across heads | Distributed representation with no specialization |

## Connection to other frameworks

MI provides the foundation that [C04 (PID)](/framework/instruments_v2/information/c04-pid/) decomposes into finer atoms. When MI is high but [C08 (OCSE)](/framework/instruments_v2/information/c08-ocse/) scores are low, the component carries information but removing it does not degrade performance — suggesting redundancy. The [causal](/framework/instruments_v2/causal/) pillar tests whether high-MI components are actually necessary via intervention.
