---
title: "Construct Coverage"
validity_type: "Measurement"
criterion_id: "M6"
---

# Criterion M6 — Construct Coverage

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | The instrument measures its nominal target, not a correlated proxy |
| Evidence family | Measurement, Representational |
| Minimum reporting | Explicit statement of the gap between nominal and actual measurement target; constrained vs. unconstrained IIA comparison |
| Common failure mode | Treating the instrument's nominal claim as its actual claim |

## What this criterion requires

Every instrument has a gap between what it nominally measures and what it actually measures. Construct coverage requires making this gap explicit.

**IIA:** Nominally measures "does this subspace contain the causal variable?" Actually measures "can a linear map from this subspace predict the interchange outcome?" A linear map has enough degrees of freedom to fit patterns correlated with the variable but not isomorphic to it. Construct coverage for IIA: (a) use constrained linear maps (not unconstrained or nonlinear); (b) compare constrained result to unconstrained — if unconstrained substantially outperforms constrained, constrained IIA may underestimate true causal alignment.

**Nonlinear IIA:** Construct coverage *fails* by definition. Unconstrained nonlinear alignment maps can achieve IIA ≈ 1.0 on random models, meaning nonlinear IIA measures alignment map flexibility, not causal variable representation. Any result using nonlinear IIA must be accompanied by a comparison to the unconstrained baseline.

**Weight classifier:** Nominally measures "does this component have the structural signature of a circuit member?" Actually measures "does this component's weight-space representation exceed a learned threshold for circuit membership trained on specific published circuits." Components implementing the same computation via different parametric structure may be missed. Construct coverage: report out-of-distribution performance (on circuit types not in the training set).

**Faithfulness:** Nominally measures "does the circuit implement the same input-output mapping as the full model?" Actually measures "does the circuit produce the same logit difference on the training prompt distribution?" A circuit can have high faithfulness on training prompts and low faithfulness on held-out templates. Construct coverage: test on held-out templates (overlaps with E5 robustness).

## The constrained vs. unconstrained comparison

1. Run DAS-IIA with a constrained linear alignment (orthogonal rotation or rank-r linear map with fixed r).
2. Run same setup with unconstrained linear map (full d_model × d_model matrix).
3. Compare the two IIA scores.

If constrained = 0.48 and unconstrained = 0.51: constraint has not substantially reduced performance — constrained result provides genuine information. If constrained = 0.48 and unconstrained = 0.89: the unconstrained map is fitting much richer structure — the constraint is doing real work and the constrained result measures something more specific.

## Minimum reporting rule

- Explicit statement of what the instrument nominally measures and what it actually measures.
- For IIA: constrained vs. unconstrained IIA comparison.
- For weight classifier: whether tested on circuit types outside its training distribution.
- For faithfulness: whether held-out template testing was included.
