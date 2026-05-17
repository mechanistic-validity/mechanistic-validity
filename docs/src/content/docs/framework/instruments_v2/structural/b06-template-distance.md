---
title: "B06 — Template Distance"
description: "Graph-edit and metric distances between circuits discovered for different tasks."
---

# B06 — Template Distance

This framework asks: **how structurally similar are circuits for different tasks, and do related tasks share circuit topology?**

Template distance measures the structural distance between two circuits — the minimum number of edge/node additions, deletions, or substitutions needed to transform one circuit graph into another. When circuits for semantically related tasks (e.g., IOI and colored objects) have low template distance, this suggests shared computational infrastructure. When unrelated tasks have high template distance, it confirms that circuit discovery is identifying genuinely task-specific structure rather than generic attention patterns.

This instrument operates at the graph level rather than the weight level: it compares circuit topologies (which heads, which edges) rather than individual weight matrices. It is the structural complement to behavioral generalization tests — template distance asks whether the *structure* generalizes, not just the *performance*.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC produces circuit graphs amenable to structural comparison |
| [Sanchez-Lengeling et al., arXiv 2010.00321](https://arxiv.org/abs/2010.00321) | 2020 | Graph-edit distance and graph kernels for comparing computational graphs |
| [Meister & Cotterell, arXiv 2305.15054](https://arxiv.org/abs/2305.15054) | 2023 | Circuit universality — shared structure across models and tasks |
| [Hanna et al., arXiv 2305.00586](https://arxiv.org/abs/2305.00586) | 2023 | Cross-task circuit comparison in GPT-2 |

## Core concept

Let circuits \( C_1 = (V_1, E_1) \) and \( C_2 = (V_2, E_2) \) be directed graphs where vertices are model components (heads, MLPs, residual stream positions) and edges represent information flow. The graph-edit distance is:

\[
d_{\text{GED}}(C_1, C_2) = \min_{\text{edit sequence}} \sum_{i} \text{cost}(e_i)
\]

where edit operations include node insertion/deletion and edge insertion/deletion. For circuits with labeled nodes (layer, head index), a natural cost function assigns zero cost to matching nodes and unit cost to mismatches.

A normalized version — circuit metric distance (CMD) — scales by circuit size:

\[
d_{\text{CMD}}(C_1, C_2) = \frac{d_{\text{GED}}(C_1, C_2)}{|V_1| + |V_2| + |E_1| + |E_2|}
\]

Values near 0 indicate nearly identical circuits; values near 0.5 indicate completely disjoint circuits.

## Instruments under B06

### Circuit Metric Distance (`26_cmd.py`)

Computes pairwise CMD between circuits discovered for different tasks. Reports: (1) the full distance matrix, (2) hierarchical clustering of tasks by circuit similarity, (3) identification of shared "backbone" components present in multiple circuits.

**What it establishes:** Quantitative structural similarity between task circuits, enabling claims about shared vs. task-specific computational infrastructure.

**What it does not establish:** Whether shared structure implies shared mechanism (two circuits may share topology but implement different computations via different weights).

**Usage:**
```
uv run python 26_cmd.py --tasks ioi sva greater_than
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Low CMD between related tasks | Shared computational infrastructure — potential circuit universals |
| High CMD between all task pairs | Each task uses genuinely distinct circuitry |
| Cluster of low-CMD tasks | Family of tasks sharing a computational backbone |
| One component in all circuits | Potential "hub" component — structurally universal |
| CMD near 0.5 | Circuits are maximally different — no structural overlap |

