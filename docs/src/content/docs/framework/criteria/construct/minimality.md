---
title: "Minimality"
validity_type: "Construct"
criterion_id: "C4"
---

# Criterion C4 — Minimality

| | |
|---|---|
| Validity type | Construct |
| Pass condition | The circuit is the smallest set of components satisfying sufficiency; removing any member degrades performance |
| Evidence family | Causal (per-component ablation) |
| Minimum reporting | Per-component ablation result for each member; confirmation no member can be removed without behavioral degradation |
| Common failure mode | Circuit grown by correlation or attribution score without a per-component pruning pass |

## What this criterion requires

Minimality requires no redundant members. Every component must be individually necessary given the other components. If removing c_i while leaving all other members intact has no effect on performance, c_i is redundant and should be removed.

Formally: for circuit C = {c_1, ..., c_n}, minimality requires that for every c_i in C, ablating c_i while leaving all others intact produces a performance decrease.

## The per-component pruning pass

1. Take proposed circuit C.
2. Ablate only c_i (zero or resample).
3. Measure performance on target task.
4. If not degraded relative to full-circuit ablation: remove c_i from C.
5. Repeat until no member can be removed.

Report the set of members pruned and ablation results for each.

## Minimality and redundancy

Multiple minimal circuits can exist (backup name-mover heads in IOI). This is a finding about architectural redundancy, not a failure of minimality. Report both minimal circuits and describe their relationship.

## Minimum reporting rule

- List every component in the final circuit.
- Per-component ablation result (method, metric, delta) for each member.
- Set of components pruned during minimality pass, if any.
- If no minimality pass was run, flag as open criterion.

## Minimality type declaration

Three kinds of minimality can come apart completely. A claim must declare which it asserts:

| Type | Definition | How established |
|---|---|---|
| **Statistical** | Smallest faithful subgraph under the discovery procedure | Greedy edge-pruning, ACDC threshold sweep |
| **Causal** | Fewest components that are INUS conditions for the behavior | Per-component necessity testing (I1) with confound control (I5) |
| **Mechanistic** | Decomposition carved at the model's natural joints | Independent structural evidence (weight-space signatures, convergent validity across methods) |

A circuit found by greedy edge-pruning is statistically minimal by construction. Calling it causally minimal requires per-component necessity evidence. Calling it mechanistically natural requires convergent structural evidence from independent methods — the decomposition must correspond to structure the model actually uses, not just the smallest set the discovery algorithm converged to.

**Pass condition for type declaration:** The type of minimality is explicitly stated and the evidence matches that declaration. Mismatch (e.g., statistical minimality claimed as mechanistic naturalness without structural evidence) is a criterion failure.

## Common failure

Discovery methods (ACDC, EAP, attribution patching) return components ranked by attribution score. Taking top-k without per-component ablation check risks including redundant members.
