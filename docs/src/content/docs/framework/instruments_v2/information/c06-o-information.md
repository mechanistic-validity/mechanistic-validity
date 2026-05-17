---
title: "C06 — O-Information"
description: "Measures whether a group of circuit components interacts redundantly or synergistically as a whole."
---

# C06 — O-Information

This framework asks: **Does a group of circuit heads collectively exhibit redundancy-dominated or synergy-dominated interactions?**

O-information (also called "information modification") is a signed measure that characterizes the dominant mode of statistical interdependence in a multivariate system. Positive O-information indicates redundancy dominates (components share overlapping information), while negative O-information indicates synergy dominates (the group creates information collectively that no subset possesses). This provides a single scalar summary of higher-order interaction structure.

For circuit analysis, O-information reveals whether the circuit as a whole operates through redundant broadcasting (each head independently captures the relevant signal) or synergistic composition (the circuit's function emerges only from the interaction of its parts).

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Rosas et al., "Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information"](https://doi.org/10.1103/PhysRevResearch.1.033161) | 2019 | O-information definition and properties |
| [Tononi et al., "A Measure for Brain Complexity"](https://doi.org/10.1073/pnas.91.11.5033) | 1994 | Integration/complexity measures that O-info generalizes |
| [Rosas et al., "Disentangling High-order Mechanisms and High-order Behaviours"](https://arxiv.org/abs/2202.05079) | 2022 | Gradient of O-info for identifying synergistic subgroups |
| [Williams & Beer, "Nonnegative Decomposition of Multivariate Information"](https://arxiv.org/abs/1004.2515) | 2010 | PID framework that O-info summarizes at group level |

## Core concept

For a set of \( n \) variables \( \mathbf{X} = \{X_1, \ldots, X_n\} \), the O-information is:

\[ \Omega(\mathbf{X}) = (n-2) \, H(\mathbf{X}) + \sum_{i=1}^n \left[ H(X_i) - H(\mathbf{X} \setminus X_i) \right] \]

Equivalently, it can be expressed as the difference between total correlation (TC) and dual total correlation (DTC):

\[ \Omega(\mathbf{X}) = \mathrm{TC}(\mathbf{X}) - \mathrm{DTC}(\mathbf{X}) \]

When \( \Omega > 0 \), redundancy dominates: individual variables carry more information than their collective entails. When \( \Omega < 0 \), synergy dominates: the group creates information not present in any subset.

## Instruments under C06

### PID Script (`08_pid.py`)

While PID operates on pairs, the O-information computed from the same activation data extends the analysis to the full circuit group. The script computes pairwise PID atoms whose aggregation relates to the group-level O-information.

**What it establishes:** Whether the circuit's dominant interaction mode is redundant or synergistic.
**What it does not establish:** Which specific subgroups contribute the synergy or redundancy.

**Usage:**
```
uv run python 08_pid.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| \( \Omega \gg 0 \) | Redundancy-dominated — heads broadcast similar information |
| \( \Omega \ll 0 \) | Synergy-dominated — circuit function requires interaction |
| \( \Omega \approx 0 \) | Balanced; neither mode dominates |
| \( \Omega \) shifts from positive to negative as circuit grows | Core redundant hub with synergistic periphery |

## Connection to other frameworks

O-information provides the group-level summary that [C04 (PID)](/framework/instruments_v2/information/c04-pid/) details pairwise. A synergy-dominated circuit (\( \Omega < 0 \)) should show superadditive knockout effects in the [causal pillar](/framework/instruments_v2/causal/) — removing any single head should degrade performance more than its individual contribution predicts. The [structural pillar](/framework/instruments_v2/structural/) measures circuit connectivity, while O-information reveals whether that connectivity carries redundant or synergistic function.
