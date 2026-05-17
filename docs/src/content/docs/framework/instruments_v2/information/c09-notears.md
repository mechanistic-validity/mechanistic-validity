---
title: "C09 — NOTEARS DAG Discovery"
description: "Learns the directed acyclic graph structure of circuit component interactions from observational data."
---

# C09 — NOTEARS DAG Discovery

This framework asks: **What is the directed acyclic graph structure connecting circuit components, learned purely from their observed activations?**

NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning) reformulates DAG discovery as a continuous optimization problem. Rather than searching over the combinatorial space of all possible graphs, NOTEARS uses a smooth algebraic acyclicity constraint to learn a weighted adjacency matrix from data. Applied to circuit components, this recovers the information flow graph without requiring any interventions or prior knowledge of the architecture.

This is uniquely powerful for circuit validation: if the learned DAG matches the hypothesized circuit structure (e.g., name movers receive from S-inhibition heads in IOI), we have observational confirmation of the circuit's connectivity.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning"](https://arxiv.org/abs/1803.01422) | 2018 | Original NOTEARS continuous DAG constraint |
| [Zheng et al., "Learning Sparse Nonparametric DAGs"](https://arxiv.org/abs/1909.13189) | 2020 | Nonlinear extension via neural networks |
| [Lachapelle et al., "Gradient-Based Neural DAG Learning"](https://arxiv.org/abs/1906.02226) | 2019 | GraN-DAG: gradient-based alternative |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC recovers circuit graphs via edge patching |
| [Geiger et al., "Causal Abstractions of Neural Networks"](https://arxiv.org/abs/2106.02997) | 2021 | Ground truth circuit structures for validation |

## Core concept

Given \( n \) circuit components with activation matrix \( \mathbf{X} \in \mathbb{R}^{N \times n} \) (N samples), NOTEARS solves:

\[ \min_{\mathbf{W}} \; \frac{1}{2N} \| \mathbf{X} - \mathbf{X} \mathbf{W} \|_F^2 + \lambda \| \mathbf{W} \|_1 \quad \text{subject to} \quad h(\mathbf{W}) = 0 \]

where the acyclicity constraint is:

\[ h(\mathbf{W}) = \mathrm{tr}(e^{\mathbf{W} \circ \mathbf{W}}) - n = 0 \]

This elegant constraint equals zero if and only if \( \mathbf{W} \) encodes a DAG. The \( L_1 \) penalty encourages sparsity (few edges), matching our expectation that circuits are sparse subgraphs of the full model. The learned \( \mathbf{W} \) gives edge weights interpretable as directed influence strengths.

## Instruments under C09

### NOTEARS Script (`09_notears.py`)

Directly applies the NOTEARS algorithm to circuit head activations. Collects activation vectors for all candidate circuit components across a corpus, then learns the sparse DAG structure connecting them.

**What it establishes:** The directed graph structure of information flow between circuit components, learned from observational data alone.
**What it does not establish:** Whether discovered edges represent direct causal connections or merely predictive relationships mediated by the residual stream.

**Usage:**
```
uv run python 09_notears.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| Learned DAG matches hypothesized circuit | Observational confirmation of circuit structure |
| Extra edges not in hypothesis | Potential undiscovered circuit connections |
| Missing expected edges | Those connections may be indirect or non-informational |
| Dense learned graph | Circuit is highly interconnected; sparse hypothesis may be oversimplified |
| Edge weights match OCSE scores | Convergent evidence from two independent methods |

## Connection to other frameworks

NOTEARS provides the structural graph that [C07 (Granger Causality)](/framework/instruments_v2/information/c07-granger-causality/) and [C08 (OCSE)](/framework/instruments_v2/information/c08-ocse/) estimate edge-by-edge. Where those methods test individual edges, NOTEARS jointly optimizes the entire graph with a global acyclicity constraint. The learned DAG can be compared against circuits recovered by the [causal pillar](/framework/instruments_v2/causal/) (intervention-based) and the [structural pillar](/framework/instruments_v2/structural/) (weight-based) — agreement across all three provides the strongest evidence for circuit validity.
