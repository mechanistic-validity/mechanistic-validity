---
title: "Knockout Ordering"
validity_type: "Internal"
criterion_id: "I8"
---

# Criterion I8 — Knockout Ordering

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Ordered ablation produces monotonic degradation with Spearman rho at least 0.7 |
| Evidence family | Causal |
| Minimum reporting | Predicted ordering and basis, observed degradation curve, Spearman rho, comparison to random ordering, number of components |
| Common failure mode | Using only one ordering criterion (e.g., only activation magnitude) without comparing to alternative orderings |
| Lens | Genetics |

## What this criterion requires

Knockout ordering tests whether the proposed circuit has a well-defined importance hierarchy: sequentially ablating components in predicted importance order produces monotonically increasing degradation. This validates that the circuit discovery method's importance ranking reflects genuine functional structure.

Satisfied when:

1. **Ordered ablation is monotonic.** Ablating components one at a time in predicted importance order (most important first) produces a degradation curve with Spearman rho at least 0.7 between ablation step and degradation magnitude.
2. **The ordering outperforms random.** The predicted ordering produces a steeper initial degradation curve than random orderings (measured by area under the degradation curve for the first 50% of components).
3. **Multiple ordering criteria are compared.** At least two bases for ordering (e.g., activation magnitude, causal effect size, weight norm) are tested to confirm robustness.

Knockout ordering does not establish that the ordering reflects information flow direction, nor does it establish sufficiency. It establishes that the circuit's components have a real, rankable importance structure rather than a flat contribution profile.

## Distinction from I1 — Necessity

I1 tests whether individual components are necessary. I8 tests whether the relative ordering of component importance is meaningful. A circuit can have all-necessary components (I1 pass) yet have a flat importance profile where ordering is arbitrary (I8 fail).

## Minimum reporting rule

- Predicted importance ordering and the basis used to derive it.
- Observed degradation curve (metric value at each ablation step).
- Spearman rho between step number and degradation.
- Area under degradation curve compared to random ordering baseline.
- Number of components in the circuit.
- Whether multiple ordering criteria were compared, and results for each.
