---
title: "Stability"
validity_type: "Measurement"
criterion_id: "M3"
---

# Criterion M3 — Stability

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | The classification or score is robust to perturbation of analysis choices |
| Evidence family | Measurement-theoretic |
| Minimum reporting | Which analysis choices were varied, how much the score moved, whether the conclusion changed |
| Common failure mode | Reporting one configuration as the result without testing alternatives |

## What this criterion requires

Stability asks whether the measurement outcome changes when the analyst makes different defensible choices — a different threshold, a different number of components, a different ablation method. A result that holds under one configuration but flips under another is not stable.

Satisfied when:

1. **At least two defensible analysis configurations are tested.** Threshold, component count, method variant, or hyperparameter is varied.
2. **The qualitative conclusion is unchanged.** The classification (e.g., "necessary") holds across configurations; quantitative values may shift but the sign and order are preserved.
3. **The sensitivity is reported.** If the result is sensitive to a particular choice, that choice is named.

## MI example

IOI faithfulness scores span below 0% to over 100% across six methodological choices (Miller et al.). The qualitative conclusion — "this is a faithful circuit" — is not stable across methods. The measurement is highly sensitive to ablation method, baseline, and metric choice. This is a stability failure, distinct from reliability (M1), which asks about repeated runs under the same configuration.

## Connection to the chain

Stability is a measurement property. An unstable measurement cannot support an internal-validity claim — the causal conclusion depends on which configuration the analyst chose. Required for full measurement validity at Validated tier.
