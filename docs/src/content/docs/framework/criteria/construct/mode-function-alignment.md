---
title: "Mode-Function Alignment"
validity_type: "Construct"
criterion_id: "C7"
---

# Criterion C7 — Mode-Function Alignment

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Top-k Koopman modes predict task-relevant feature with R^2 >= 0.5 |
| Evidence family | Representational |
| Minimum reporting | Mode interpretations, R^2 with bootstrap CI, comparison to prediction from raw activations and from random modes |
| Common failure mode | Probing modes without reporting random-mode baseline R^2 |
| Lens | Dynamical Systems |

## What this criterion requires

Mode-function alignment asks whether the dominant dynamical modes of the circuit track the claimed computational feature. A circuit may have clean low-rank dynamics (C6) but those dynamics may have nothing to do with the task the circuit is supposed to perform. This criterion bridges the gap between dynamical structure and functional interpretation.

The criterion is satisfied when:

1. **The top-k Koopman modes (from C6) predict the task-relevant feature with R^2 >= 0.5.** For IOI, this means the modes predict IO token identity; for SVA, number agreement. The prediction is via linear regression from mode coefficients to the task label.
2. **The R^2 exceeds what random modes achieve.** A random rotation of the same dimensionality applied to the same data should produce substantially lower R^2. Report bootstrap confidence intervals for both.
3. **The R^2 is compared to prediction from raw activations.** If raw activations predict just as well without mode decomposition, the dynamical decomposition adds no interpretive value — the modes are not identifying structure beyond what is trivially available.

This criterion does not establish that the modes play a causal role in the computation. A mode may track a feature because both are downstream of a common cause. It also does not establish uniqueness — multiple mode decompositions may achieve similar alignment.

## Minimum reporting rule

- Interpretation of each of the top-k modes (what feature each tracks, qualitatively).
- R^2 of linear prediction from top-k mode coefficients to task label, with bootstrap 95% CI.
- R^2 from random modes of same dimensionality, with bootstrap 95% CI.
- R^2 from raw activations (same layer, same dimensionality) for comparison.
- Number of modes used and justification for k.
