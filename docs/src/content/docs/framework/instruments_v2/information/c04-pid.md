---
title: "C04 — Partial Information Decomposition"
description: "Decomposes shared information between circuit heads into unique, redundant, and synergistic atoms."
---

# C04 — Partial Information Decomposition

This framework asks: **Is the information that circuit heads carry about the task unique to each, redundantly shared, or synergistically created by their interaction?**

Partial Information Decomposition (PID) goes beyond pairwise MI to characterize how multiple sources jointly inform a target. Given two circuit heads providing information about the task output, PID separates their joint information into four atoms: unique to head A, unique to head B, redundantly shared by both, and synergistically available only when both are observed together.

This decomposition is essential for understanding circuit structure. Redundancy suggests backup mechanisms; synergy indicates that heads perform a joint computation that neither achieves alone — the hallmark of compositional circuits like those in IOI.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Williams & Beer, "Nonnegative Decomposition of Multivariate Information"](https://arxiv.org/abs/1004.2515) | 2010 | Original PID framework and redundancy axioms |
| [Bertschinger et al., "Quantifying Unique Information"](https://doi.org/10.3390/e16042161) | 2014 | Operational definition of unique information |
| [Griffith & Koch, "Quantifying Synergistic Mutual Information"](https://doi.org/10.3390/e16042161) | 2014 | Synergy quantification via exclusion |
| [Tax et al., "Partial Information Decomposition of Neural Connectivity"](https://doi.org/10.3390/e19100494) | 2017 | PID applied to neural circuits |
| [Rosas et al., "Quantifying High-order Interdependencies"](https://doi.org/10.1103/PhysRevResearch.1.033161) | 2019 | Extension to higher-order interactions |

## Core concept

For two sources \( X_1, X_2 \) and target \( Y \), PID decomposes the joint MI:

\[ I(\{X_1, X_2\}; Y) = \mathrm{Uniq}(X_1) + \mathrm{Uniq}(X_2) + \mathrm{Red}(X_1, X_2) + \mathrm{Syn}(X_1, X_2) \]

where:
- \( \mathrm{Uniq}(X_1) \): information only \( X_1 \) provides
- \( \mathrm{Red}(X_1, X_2) \): information either source alone provides
- \( \mathrm{Syn}(X_1, X_2) \): information available only from both together

The redundancy satisfies \( \mathrm{Red}(X_1, X_2) \leq \min(I(X_1; Y), I(X_2; Y)) \). Using the \( I_\mathrm{min} \) measure from Williams & Beer, redundancy equals the minimum MI that any single source provides about each outcome of \( Y \).

## Instruments under C04

### PID Script (`08_pid.py`)

Computes the full PID lattice for pairs of circuit heads with respect to the task logit output. Uses the \( I_\mathrm{min} \) estimator with KSG-based MI estimation for continuous activations.

**What it establishes:** Whether circuit heads carry unique, redundant, or synergistic information about the task.
**What it does not establish:** Causal necessity — high synergy means both are needed informationally, but not that ablating one destroys performance.

**Usage:**
```
uv run python 08_pid.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| High synergy between two heads | Joint computation — both needed for the information to exist |
| High redundancy | Backup mechanism — either head alone provides the information |
| High unique info for one head | Specialized computation not replicated elsewhere |
| Synergy >> Redundancy across circuit | Compositional, fragile circuit structure |

## Connection to other frameworks

PID refines [C01 (MI)](/framework/instruments_v2/information/c01-mutual-information/) and [C02 (CMI)](/framework/instruments_v2/information/c02-conditional-mi/) into actionable atoms. High-synergy pairs identified here should show superadditive ablation effects in the [causal pillar](/framework/instruments_v2/causal/) — knocking out either head alone should be worse than the sum suggests. [C06 (O-information)](/framework/instruments_v2/information/c06-o-information/) extends this analysis to groups larger than pairs.
