---
title: "A02 — Counterfactual DAS / IIA"
description: "Distributed Alignment Search and Interchange Intervention Accuracy for testing causal abstraction hypotheses."
---

# A02 — Counterfactual DAS / Interchange Intervention Accuracy

This framework asks: **does the circuit implement a specific causal variable, verified by swapping that variable's representation between inputs?**

Interchange Intervention Accuracy (IIA) is the counterfactual (Rung-3) test for causal abstraction. Where activation patching asks "does this component matter?", IIA asks "does this component encode *this specific causal variable*?" — a strictly stronger claim. Distributed Alignment Search (DAS) extends this to subspaces that may not align with individual components, finding rotated directions that carry causal variables even when no single head or neuron does.

The core logic: if a component encodes causal variable \( Z \), then swapping that component's activation between two inputs that differ only in \( Z \) should produce the same output change predicted by the high-level causal model. When this holds for a substantial fraction of input pairs, the component is a *faithful implementation* of that variable. DAS generalizes this by learning a linear subspace (rotation matrix) that maximizes IIA, discovering distributed representations of causal variables that are invisible to per-component patching.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Geiger et al., arXiv 2106.02997](https://arxiv.org/abs/2106.02997) | 2021 | Causal abstraction: mapping high-level causal models onto neural network components |
| [Geiger et al., arXiv 2303.02536](https://arxiv.org/abs/2303.02536) | 2023 | DAS: gradient-based search for subspaces that maximize IIA |
| [Wu et al., arXiv 2402.14843](https://arxiv.org/abs/2402.14843) | 2024 | Boundless DAS: continuous relaxation removing fixed-dimension constraints |
| [Mueller et al., arXiv 2406.14673](https://arxiv.org/abs/2406.14673) | 2024 | MIB benchmark: standardized IIA evaluation across tasks and methods |
| [Goldowsky-Dill et al., arXiv 2304.05969](https://arxiv.org/abs/2304.05969) | 2023 | Path patching: restricting interchange interventions to specific edges |

## Core concept: causal abstraction

A high-level causal model \( \mathcal{M} \) specifies variables \( Z_1, \ldots, Z_k \) and their relationships. A neural network \( \mathcal{N} \) implements \( \mathcal{M} \) if there exists an alignment \( \tau \) mapping each \( Z_i \) to a set of neural components such that interchange interventions on those components produce behavior consistent with interventions on \( Z_i \) in \( \mathcal{M} \). IIA is the fraction of (input, counterfactual-input) pairs for which this consistency holds.

DAS learns the alignment \( \tau \) as a rotation matrix \( R \in \mathbb{R}^{d \times k} \), projecting activations into a \( k \)-dimensional subspace where IIA is maximized. This handles the common case where causal variables are encoded in distributed directions rather than axis-aligned components.

## Metrics under A02

### C1 — DAS / IIA (`01_das_iia.py`)

The primary IIA metric. For each causal variable in the task's high-level model, trains a DAS rotation (or uses pre-specified component alignments) and evaluates:

\[
\text{IIA}(Z_i, \tau) = \frac{1}{N} \sum_{(x, x')} \mathbf{1}\left[ \mathcal{N}[\tau(Z_i) \leftarrow \tau(Z_i)(x')](x) = \mathcal{M}[Z_i \leftarrow Z_i(x')](x) \right]
\]

**What it establishes:** That a specific subspace faithfully implements a named causal variable.

**What it does not establish:** That the alignment is unique or that the variable decomposition is correct.

**Usage:**
```
uv run python 01_das_iia.py --tasks ioi sva --n-prompts 40
```

### C15 — IIA Variants (`15_iia_variants.py`)

Evaluates multiple IIA operationalizations: hard vs. soft matching, per-token vs. sequence-level accuracy, and different counterfactual sampling strategies.

**Usage:**
```
uv run python 15_iia_variants.py --tasks ioi --n-prompts 40
```

### C20 — Corrupt-Restore Protocol (`20_corrupt_restore.py`)

Measures restoration IIA: patches the circuit's components with clean activations starting from a corrupted baseline and checks whether clean output is restored. **Usage:** `uv run python 20_corrupt_restore.py --tasks ioi sva --n-prompts 40`

### C31 — Multi-Axis IIA (`31_multi_axis_iia.py`)

Tests IIA along multiple causal variables simultaneously, verifying joint interventions. **Usage:** `uv run python 31_multi_axis_iia.py --tasks ioi --n-prompts 40`

### C33 — Path Patching (`33_path_patching.py`)

Restricts interchange interventions to specific edges (Goldowsky-Dill et al. 2023), testing whether information flows along the hypothesized path. **Usage:** `uv run python 33_path_patching.py --tasks ioi --n-prompts 40`

### C34 — Counterfactual Consistency (`34_counterfactual_consistency.py`)

Checks whether IIA scores generalize across different counterfactual input pairs rather than overfitting to specific corruptions. **Usage:** `uv run python 34_counterfactual_consistency.py --tasks ioi sva --n-prompts 40`

## Reading the scores

| Pattern | What it means |
|---|---|
| IIA > 0.9 across variable pairs | Circuit faithfully implements the causal variable |
| High IIA on DAS but low on axis-aligned | Variable is distributed (not localized to one head) |
| IIA degrades across tasks | Alignment is task-specific, not a general feature |
| Path-patching IIA < full-node IIA | Information leaks through alternative paths |

## Connection to other frameworks

A02 operationalizes the Rung-3 counterfactual tests that A01 (SCM) formalizes. Where A01 provides the language, A02 provides the measurement. A04 (Woodward) offers philosophical criteria for what makes an intervention "surgical" rather than confounded. A06 (Mediation) decomposes the total causal effect into direct and indirect paths, complementing A02's binary pass/fail with continuous effect decomposition.
