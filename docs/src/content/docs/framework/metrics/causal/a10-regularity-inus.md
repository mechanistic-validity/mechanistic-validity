---
title: "A10 — Regularity / INUS Conditions"
description: "Mackie's INUS conditions applied to circuits: identifying components that are Insufficient but Necessary parts of Unnecessary but Sufficient sets."
---

# A10 — Regularity / INUS Conditions

This framework asks: **is this component an Insufficient but Necessary part of an Unnecessary but Sufficient set for producing the behavior?**

Mackie's INUS conditions (1965) provide a formal account of causation under redundancy: a cause is typically not sufficient on its own (Insufficient), but is a required part (Necessary) of some condition-set that *is* sufficient (Sufficient), though that set may not be the only sufficient set (Unnecessary). In circuit terms: a head is an INUS condition if it is necessary within its particular sub-circuit but the model has alternative sub-circuits that can also produce the behavior. This directly formalizes the "backup circuit" phenomenon observed in many transformer models.

This framework bridges A01's necessity testing with A08's redundancy analysis by asking a structured question: for each component, identify all minimal sufficient sets it belongs to, and test its necessity within each set. The result is a map of the model's causal redundancy structure at a finer grain than binary "necessary / not necessary."

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Mackie, "Causes and Conditions"](https://doi.org/10.2307/2024505) | 1965 | INUS conditions: causes as parts of minimal sufficient sets |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | Structural formalization of regularity accounts |
| [McGrath et al., arXiv 2307.15771](https://arxiv.org/abs/2307.15771) | 2023 | Hydra effect as empirical evidence of INUS structure in transformers |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | Multiple sufficient paths in the IOI circuit |

## Core concept: minimal sufficient sets

A minimal sufficient set (MSS) for behavior \( B \) is a set of components \( S \) such that: (1) \( S \) is sufficient for \( B \) (the circuit produces the behavior when only \( S \) is active), and (2) no proper subset of \( S \) is sufficient. Component \( c \) is an INUS condition if:

\[
c \in S_i \text{ (Necessary within } S_i\text{)} \quad \text{and} \quad \exists S_j \neq S_i \text{ also sufficient (}S_i \text{ is Unnecessary)}
\]

The INUS structure of a circuit can be represented as a disjunction of conjunctions (DNF): \( B \iff (c_1 \wedge c_2 \wedge c_3) \vee (c_1 \wedge c_4 \wedge c_5) \vee \ldots \). Each conjunct is a minimal sufficient set. Components appearing in multiple conjuncts are "harder" INUS conditions (more broadly necessary); components in only one conjunct are more replaceable.

## Why INUS matters for MI

Standard necessity/sufficiency testing treats each component in isolation: "is head A necessary?" "is set S sufficient?" INUS conditions reveal the *combinatorial* structure of causation. In the IOI circuit, the name-mover heads are individually insufficient (neither alone produces correct output) but jointly sufficient — and they form one of potentially several sufficient configurations (backup name movers exist). This cannot be captured by per-component scores alone; it requires exhaustive or heuristic search over component combinations.

The INUS framework also predicts ablation sensitivity: components appearing in many minimal sufficient sets are "harder to kill" because alternative pathways exist. Components in only one MSS are critical failure points.

## Metrics under A10

### C39 — INUS Conditions (`39_inus_conditions.py`)

**Metric status:** This script is a stub awaiting implementation. The directory listing confirms it does not yet exist.

An implementation would require:
1. **Exhaustive or heuristic search** over component subsets to identify all minimal sufficient sets (those achieving > threshold faithfulness when only they are active).
2. **Necessity testing** within each sufficient set (ablating one component from the set and checking whether sufficiency breaks).
3. **INUS classification** for each component: how many MSS it appears in, and whether it is necessary in all of them (making it a full necessary condition) or only some (true INUS).
4. **Output:** A table mapping each component to its INUS status, the number of minimal sufficient sets it belongs to, and the size of those sets.

The computational cost scales combinatorially with circuit size, so practical implementations would use greedy search or build on A01's activation patching scores to prune the search space.

**Planned usage:**
```
uv run python 39_inus_conditions.py --tasks ioi --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Component in all MSS | Fully necessary; not INUS but a genuine necessary cause |
| Component in multiple but not all MSS | True INUS condition; necessary within its sub-circuit but replaceable |
| Component in exactly one MSS | Fragile; no backup circuit for this component's role |
| Many MSS of similar size | Highly redundant circuit; robust to ablation |

## Connection to other frameworks

A10 provides the formal vocabulary for interpreting A08's (PID) redundancy findings: high redundancy between components A and B means they likely appear in alternative minimal sufficient sets (each is INUS). A01 (SCM) measures necessity and sufficiency but does not decompose them into INUS structure. A03 (CATE) identifies heterogeneous effects — components that are INUS (rather than fully necessary) may show high CATE variance because they are only causally relevant when their particular MSS is the active one.
