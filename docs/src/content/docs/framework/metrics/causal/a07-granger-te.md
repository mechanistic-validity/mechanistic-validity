---
title: "A07 — Granger Causality / Transfer Entropy"
description: "Observational causal discovery via conditional mutual information and temporal precedence in transformer activations."
---

# A07 — Granger Causality / Transfer Entropy

This framework asks: **can we discover causal structure from observational data alone — without interventions — by measuring directed information flow between components?**

Granger causality and transfer entropy provide tools for identifying directed relationships from observational data. A variable \( X \) Granger-causes \( Y \) if past values of \( X \) improve prediction of \( Y \) beyond what past values of \( Y \) alone provide. Transfer entropy generalizes this to the information-theoretic setting: \( T_{X \to Y} = I(Y_t; X_{t-1} \mid Y_{t-1}) \). In transformers, "temporal precedence" maps to layer ordering — earlier-layer activations precede later-layer activations in the computational graph, making Granger-style analysis applicable.

The practical value: observational methods scale to the full model without requiring \( O(n^2) \) interventions. They serve as a discovery tool — identifying candidate causal relationships that can then be verified with interventional methods from A01/A02. The oCSE (observational Causal Structure Evaluation) approach combines conditional mutual information with stability selection to find edges that are robust across subsamples.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Granger, "Investigating Causal Relations by Econometric Models"](https://doi.org/10.2307/1912791) | 1969 | Granger causality: predictive improvement as evidence of causation |
| [Schreiber, "Measuring Information Transfer"](https://doi.org/10.1103/PhysRevLett.85.461) | 2000 | Transfer entropy: information-theoretic generalization of Granger causality |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Layer ordering as temporal structure enabling directed analysis |
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC as interventional circuit discovery (contrast to observational) |

## Core concept: observational circuit discovery

Transfer entropy from component \( X \) (layer \( l \)) to component \( Y \) (layer \( l' > l \)) is:

\[
T_{X \to Y} = I(Y; X \mid \text{Pa}(Y) \setminus X)
\]

where \( \text{Pa}(Y) \) is the set of all components at layers \( \leq l' \) that could influence \( Y \). This conditional mutual information measures the unique information that \( X \) provides about \( Y \) beyond what other parents already provide. High transfer entropy implies a directed information-flow relationship.

The oCSE algorithm applies stability selection: estimate transfer entropy on many bootstrap subsamples of the data, and retain only edges that appear consistently. This controls false discovery rate without interventions. The resulting graph is a candidate circuit that can be validated with A01/A02 methods.

The cross-task transfer variant tests whether causal information relationships discovered on one task generalize to another — evidence of a task-general computational structure rather than task-specific correlation.

## Metrics under A07

### C7 — oCSE: Observational Circuit Structure Evaluation (`07_ocse.py`)

Estimates directed information flow between all pairs of components using conditional mutual information with stability selection:

\[
\hat{T}_{X \to Y} = \hat{I}(Y; X \mid Z) \quad \text{where } Z = \text{Pa}(Y) \setminus X
\]

Edges are retained if they appear in more than a threshold fraction of bootstrap samples. The output is a directed graph over components that can be compared to interventionally-discovered circuits via structural Hamming distance.

**What it establishes:** Candidate causal edges from observational data alone. Scales to full-model analysis without per-edge interventions.

**What it does not establish:** True causation (observational methods cannot distinguish causation from confounding by unobserved variables). Must be validated interventionally.

**Usage:**
```
uv run python 07_ocse.py --tasks ioi sva --n-prompts 40
```

### C32 — Cross-Task IIA Transfer (`32_cross_task_iia_transfer.py`)

Tests whether causal relationships (measured via IIA from A02) transfer across tasks. Trains DAS alignments on one task and evaluates IIA on another, measuring how much of the causal structure is task-general:

\[
\text{Transfer}(t_1 \to t_2) = \frac{\text{IIA}_{t_2}(\tau_{t_1})}{\text{IIA}_{t_2}(\tau_{t_2})}
\]

**What it establishes:** Whether causal structure is task-specific or reflects general computational organization.

**What it does not establish:** The mechanism behind transfer (could be shared representations or shared algorithms).

**Usage:**
```
uv run python 32_cross_task_iia_transfer.py --tasks ioi sva --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| oCSE graph matches interventional circuit (low SHD) | Observational discovery recovers true causal structure |
| High transfer entropy but no interventional effect | Confounded relationship (correlation without causation) |
| High cross-task transfer ratio (> 0.8) | Causal structure is task-general |
| Low cross-task transfer (< 0.4) | Task-specific wiring; different algorithms per task |

## Connection to other frameworks

A07 provides a scalable discovery complement to A01's interventional verification. The workflow is: use A07 (observational) to generate candidate circuits cheaply, then validate the most promising candidates with A01 (activation patching) and A02 (IIA). A13 (Causal Discovery / NOTEARS) offers an alternative observational approach using continuous optimization rather than information-theoretic measures. A12 (Transportability) formalizes the conditions under which cross-task transfer results from A07 are valid.
