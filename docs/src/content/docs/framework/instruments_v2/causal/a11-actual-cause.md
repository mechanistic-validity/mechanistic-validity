---
title: "A11 — Actual Causation (Halpern-Pearl)"
description: "Halpern-Pearl actual causation applied to circuits: identifying which components were the actual causes on specific inputs, not just potential causes in general."
---

# A11 — Actual Causation (Halpern-Pearl)

This framework asks: **on this specific input, which components were the *actual* causes of the output — not just potential causes, but the ones that actually made the difference?**

Halpern and Pearl's theory of actual causation (2005) addresses a subtle distinction that aggregate metrics miss: the difference between a component that *could* cause the output (type-level causation) and one that *actually did* cause it on a specific token sequence (token-level causation). Standard activation patching gives type-level evidence — "this head is generally important for IOI." Actual causation asks the token-level question: "on this specific input where the model correctly predicted ' Mary', was head 9.9 the actual cause, or did the model arrive at the answer through a different route this time?"

This is particularly important for understanding when circuits are active versus dormant. A head may be part of the IOI circuit (type-level) but on certain easy inputs the model solves the task via a simpler heuristic that does not engage that head (token-level). Actual causation identifies which components were genuinely operative on each input.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Halpern & Pearl, "Causes and Explanations: A Structural-Model Approach"](https://doi.org/10.1613/jair.1391) | 2005 | Formal definition of actual causation via structural models + contingency |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | Structural models as the foundation for token-level causal claims |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | IOI circuit with variable engagement across prompts |
| [Miller et al., arXiv 2407.08734](https://arxiv.org/abs/2407.08734) | 2024 | Prompt-dependence of causal attributions |

## Core concept: actual cause vs. type-level cause

In Halpern-Pearl's definition, \( X = x \) is an actual cause of \( Y = y \) in context \( u \) if:

1. **AC1 (Factuality):** \( X = x \) and \( Y = y \) both hold in context \( u \).
2. **AC2 (Counterfactual contingency):** There exists a set of variables \( W \) and values \( w' \) such that setting \( X = x' \) and \( W = w' \) would change \( Y \) — i.e., the causal relationship manifests under some contingency.
3. **AC3 (Minimality):** No proper subset of the conjunct \( X = x \) satisfies AC1 and AC2.

In MI terms: AC1 requires the component to be active on this input. AC2 requires that intervening on the component (under some contingency about other components) changes the output. AC3 prevents over-attribution by requiring minimality. The "contingency" clause (AC2) is crucial — it handles cases where a component's effect is masked by downstream compensation.

This per-input attribution can reveal activation patterns invisible to aggregate metrics: heads that are actual causes only on syntactically complex inputs, or heads that are actual causes on early tokens but not late ones.

## Why token-level causation matters

Aggregate metrics hide critical structure. Consider an attention head with activation-patching score 0.5 averaged over 100 inputs. This could mean: (a) the head contributes moderately on all inputs, or (b) the head is the decisive actual cause on 50% of inputs and completely irrelevant on the other 50%. These are profoundly different mechanistic stories. Case (a) suggests a distributed, graded contribution. Case (b) suggests a binary "on/off" mechanism with specific activation conditions — and those conditions are what characterize the circuit's algorithm.

Actual causation also addresses the "sufficiency vs. actual role" gap: a component may be part of a sufficient circuit (passes causal scrubbing in A01) without being the actual cause on any given input, if the model solves the task through an alternative route that happens to also be consistent with the circuit hypothesis.

## Instruments under A11

### C40 — Actual Causation (`40_actual_causation.py`)

**Instrument status:** This script is a stub awaiting implementation. The directory listing confirms it does not yet exist.

An implementation would require:
1. **Per-input causal attribution:** For each input in the evaluation set, determine which components satisfy AC1-AC3.
2. **Contingency search:** For each candidate actual cause, identify the contingency set \( W = w' \) that reveals the counterfactual dependence (this may require iterating over ablation patterns).
3. **Minimality check:** Verify that the attributed cause is minimal — no proper subset suffices.
4. **Aggregation:** Report per-component frequencies of being an actual cause across the input distribution, and cluster inputs by their actual-cause profiles.

The key challenge is the contingency search (AC2), which in the worst case requires exponential search over subsets. Practical implementations can use greedy heuristics: start with the full complement ablated and progressively restore components until the counterfactual dependence appears.

**Planned usage:**
```
uv run python 40_actual_causation.py --tasks ioi --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Component is actual cause on > 80% of inputs | Constitutive role; always operative |
| Component is actual cause on 20-50% of inputs | Context-dependent activation; circuit engages selectively |
| Component is actual cause only with specific contingencies | Masked by compensation; effect only visible when backups are disabled |
| Few components are actual causes per input | Sparse mechanism; few components do the work on any given input |

## Connection to other frameworks

A11 refines A01's (SCM) aggregate causal claims to token-level actual causation. Where A01 says "this head is important on average," A11 says "this head was the actual cause *on this input*." A03 (CATE) identifies input covariates that modulate causal importance — A11 makes this per-input rather than per-stratum. A10 (INUS) provides the type-level redundancy structure; A11 reveals which of the multiple sufficient sets was actually operative on each input. The contingency clause in AC2 connects to A08 (PID): synergistic components may fail AC2 individually but satisfy it jointly.
