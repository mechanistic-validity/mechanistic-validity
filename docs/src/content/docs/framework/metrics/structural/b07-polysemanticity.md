---
title: "B07 — Polysemanticity"
description: "Measuring whether circuit components encode multiple unrelated features in superposition."
---

# B07 — Polysemanticity

This framework asks: **does a circuit component serve a single interpretable role, or does it superpose multiple unrelated computations?**

Polysemanticity — a single neuron or direction encoding multiple unrelated features — is the central obstacle to clean circuit explanations. If a head identified as "the name mover" also implements unrelated computations for other tasks, then its role in the circuit is not cleanly separable from its other functions. Measuring polysemanticity quantifies this threat: highly monosemantic components support clean circuit narratives; highly polysemantic components suggest the circuit boundary cuts through superposed representations.

This metric connects structural analysis (what directions exist in a component) to intervention specificity (does intervening on one claimed function affect others). It operationalizes the question: is the circuit description a faithful decomposition, or is it projecting clean narratives onto a messier underlying computation?

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Elhage et al., "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) | 2022 | Mathematical framework for feature superposition |
| [Bricken et al., "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 2023 | SAE decomposition revealing polysemantic neurons |
| [Cunningham et al., arXiv 2309.08600](https://arxiv.org/abs/2309.08600) | 2023 | Sparse autoencoders for monosemantic feature extraction |
| [Templeton et al., "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | 2024 | Feature splitting and polysemanticity at scale |
| [Marks et al., arXiv 2403.19647](https://arxiv.org/abs/2403.19647) | 2024 | Sparse feature circuits and intervention specificity |

## Core concept

A component is polysemantic if it activates for multiple unrelated input classes. Formally, let \( a(x) \) be the activation of component \( c \) on input \( x \), and let \( T_1, T_2 \) be two unrelated task distributions. Component \( c \) is polysemantic if:

\[
\mathbb{E}_{x \sim T_1}[|a(x)|] > \epsilon \quad \text{and} \quad \mathbb{E}_{x \sim T_2}[|a(x)|] > \epsilon
\]

while \( T_1 \) and \( T_2 \) share no semantic overlap. Intervention specificity provides a causal test: intervene on the component for task \( T_1 \) and measure the effect on task \( T_2 \):

\[
\text{specificity}(c, T_1, T_2) = 1 - \frac{|\Delta \text{perf}(T_2)|}{|\Delta \text{perf}(T_1)|}
\]

High specificity (near 1) means the component is monosemantic for \( T_1 \); low specificity means intervention leaks into \( T_2 \).

## Metrics under B07

### Intervention Specificity (`25_intervention_specificity.py`)

For each circuit component, ablates it while measuring performance on both the target task and a set of control tasks. Reports: (1) per-component specificity scores, (2) mean circuit specificity, (3) identification of the most polysemantic components.

**What it establishes:** Whether circuit components are functionally dedicated to the target task or serve multiple tasks simultaneously.

**What it does not establish:** The *identity* of superposed features — that requires SAE decomposition or other feature-finding methods.

**Usage:**
```
uv run python 25_intervention_specificity.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High specificity (> 0.8) for all circuit components | Circuit is cleanly separable — monosemantic for this task |
| Low specificity (< 0.5) for key components | Critical components serve multiple functions — superposition threat |
| Specificity varies across circuit components | Mixed circuit — some components dedicated, others shared |
| Low specificity correlates with high effective rank (B02) | Structural evidence of superposition confirmed by functional test |

## Connection to other frameworks

Polysemanticity connects B02 (effective rank) to causal findings: high effective rank (B02) suggests capacity for multiple features, and B07 tests whether that capacity is actually used for multiple tasks. It also connects to B08 (ICA/NMF): source separation methods attempt to *resolve* polysemanticity by finding independent components within a superposed representation. The causal metrics in A01 provide the intervention; B07 adds the cross-task measurement to assess specificity.
