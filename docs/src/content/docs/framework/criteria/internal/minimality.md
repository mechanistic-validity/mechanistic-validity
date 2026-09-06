---
title: "Minimality"
validity_type: "Internal"
criterion_id: "I3"
---

# Criterion I3 — Minimality

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Every component in the proposed circuit earns its place — removing any one degrades the behavior |
| Evidence family | Causal |
| Minimum reporting | Per-component ablation results, threshold used to define "earns its place" |
| Common failure mode | Reporting circuit membership without testing whether each member is individually necessary |

## What this criterion requires

Minimality asks whether the circuit is as small as it needs to be. A circuit where half the components can be removed without degrading behavior is not minimal — it contains passengers.

Satisfied when:

1. **Each component is individually ablated.** Every head, MLP, or feature in the proposed circuit is removed one at a time.
2. **Removing any component degrades behavior.** The target metric drops for each removal.
3. **The threshold is stated.** What counts as "degrades" — a fixed delta, a percentage, a statistical test.

## MI example

The induction heads (token copying) case study is capped partly by I3. The proposed circuit includes multiple heads labeled as induction heads, but not every head has been individually tested for necessity within the circuit. Some may be redundant — present because they were included in the initial identification pass, not because each is individually required.

## Relation to I1 and I5

I1 (necessity) tests whether the circuit as a whole is required. I3 tests whether each *component* of the circuit is required. I5 (rival mechanism exclusion) tests whether a different set of components could do the same job. A circuit can pass I1 but fail I3 if it contains unnecessary components.

## Connection to the chain

Required for Validated tier. Without minimality, the circuit boundary is not justified — it may include components that are correlated with the behavior but not causally required.
