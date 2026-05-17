---
title: "A13 — Causal Discovery (NOTEARS / PC)"
description: "Continuous optimization and constraint-based algorithms for learning DAGs over circuit components from activation data."
---

# A13 — Causal Discovery (NOTEARS / PC)

This framework asks: **can we learn the causal DAG over circuit components directly from data — recovering which heads causally precede which others without manual hypothesis specification?**

Causal discovery algorithms learn directed acyclic graphs (DAGs) from observational or interventional data. NOTEARS (Zheng et al. 2018) reformulates structure learning as continuous optimization with an acyclicity constraint, making it tractable for moderately-sized graphs. The PC algorithm (Peter-Clark) takes a constraint-based approach, using conditional independence tests to orient edges. Applied to transformer circuits, these methods can recover the information-flow DAG over components — which heads write to the residual stream in ways that downstream heads read — without requiring a researcher to specify the circuit hypothesis in advance.

The key advantage over manual circuit discovery: these algorithms explore the full space of possible DAG structures rather than testing a single human-specified hypothesis. The key limitation: they assume causal sufficiency (no unobserved confounders) or faithfulness (all conditional independences reflect causal structure), which may not hold perfectly in neural networks. The discovered DAG should be treated as a candidate circuit for validation by A01/A02 interventional methods.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Zheng et al., arXiv 1803.01422](https://arxiv.org/abs/1803.01422) | 2018 | NOTEARS: continuous optimization for DAG structure learning |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | PC algorithm and constraint-based causal discovery |
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC: automated circuit discovery (interventional approach for comparison) |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Transformer computational graph as a DAG amenable to structure learning |

## Core concept: DAG structure learning

NOTEARS formulates DAG learning as minimizing a least-squares objective with a continuous acyclicity constraint:

\[
\min_{W} \frac{1}{2n} \|X - XW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad \text{tr}(e^{W \circ W}) - d = 0
\]

where \( W \in \mathbb{R}^{d \times d} \) is the weighted adjacency matrix, \( X \) is the data matrix (component activations across inputs), and the trace-exponential constraint enforces acyclicity. The L1 penalty promotes sparsity — recovering only the strongest causal edges.

The PC algorithm takes the complementary approach: start with a complete undirected graph, remove edges where conditional independence holds (using statistical tests), then orient remaining edges using v-structure detection and orientation rules. PC is provably correct under faithfulness and causal sufficiency assumptions.

Both methods output a DAG (or CPDAG for PC) that can be compared to known circuits via structural Hamming distance (SHD): the number of edge additions, deletions, and reversals needed to transform the learned graph into the ground truth.

## Instruments under A13

### C9 — NOTEARS Structure Learning (`09_notears.py`)

Applies the NOTEARS algorithm to component activations, learning a weighted DAG over attention heads and MLP layers:

\[
\hat{W} = \arg\min_{W} \frac{1}{2n} \|A - AW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t. acyclicity}
\]

where \( A \) is the matrix of component activations (heads/MLPs) across inputs. Reports the learned DAG, its SHD to the reference circuit, and edge weights.

**What it establishes:** A data-driven candidate DAG over components. When SHD to the known circuit is low, observational data alone recovers the causal structure.

**What it does not establish:** True causation (assumes causal sufficiency and no confounders). The learned DAG is a hypothesis, not a verified circuit.

**Usage:**
```
uv run python 09_notears.py --tasks ioi sva --n-prompts 40
```

### C42 — PC Algorithm (`42_pc_algorithm.py`)

**Instrument status:** This script is a stub awaiting implementation. The directory listing confirms it does not yet exist.

An implementation would require:
1. **Conditional independence testing:** For each pair of components, test conditional independence given all subsets of other components (using partial correlation or kernel-based tests).
2. **Skeleton learning:** Remove edges where conditional independence holds at any conditioning set size.
3. **Edge orientation:** Apply v-structure rules and Meek's orientation rules to orient undirected edges.
4. **Comparison:** Report the learned CPDAG, its SHD to the reference circuit, and the set of oriented vs. undetermined edges.

The PC algorithm's advantage over NOTEARS is interpretability of the output (each edge removal/orientation has a specific statistical justification), but it is more sensitive to the choice of independence test and significance level.

**Planned usage:**
```
uv run python 42_pc_algorithm.py --tasks ioi --n-prompts 40 --alpha 0.01
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Low SHD (< 5 edits) | Observational discovery recovers the circuit structure |
| High SHD with correct skeleton | Edge directions wrong but component set is correct |
| Sparse learned DAG matching the reference | Algorithm correctly identifies the minimal circuit |
| Dense learned DAG | Algorithm finds many spurious edges; may need stronger regularization or more data |

## Connection to other frameworks

A13 provides automated discovery that complements A07's (Granger/Transfer Entropy) information-theoretic approach. Both are observational, but A13 learns a full DAG structure while A07 estimates pairwise directed information. The discovered DAG from A13 should be validated using A01 (activation patching) and A02 (IIA) interventional methods. A12 (Transportability) can then assess whether the discovered structure generalizes across domains. ACDC (Conmy et al. 2023) provides the interventional alternative: where A13 discovers structure from observations, ACDC discovers it from systematic interventions — comparing both approaches tests whether observational assumptions hold.
