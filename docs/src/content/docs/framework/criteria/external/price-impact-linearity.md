---
title: "Price Impact Linearity"
validity_type: "External"
criterion_id: "E9"
---

# Criterion E9 — Price Impact Linearity

| | |
|---|---|
| Validity type | External |
| Pass condition | R-squared ≥ 0.8 for linear fit of effect vs intervention magnitude across ≥ 5 strengths; max residual/predicted < 0.2 |
| Evidence family | Causal |
| Minimum reporting | Intervention magnitudes, effects at each, R-squared, residuals, comparison to monotonic-only (E2), identification of any threshold or saturation regime |
| Common failure mode | Fitting linear model without reporting residuals; masking threshold effects |
| Lens | Economics |

## What this criterion requires

Price impact linearity tests whether an intervention's effect scales linearly — not merely monotonically — with its magnitude. In market microstructure, small trades move prices linearly (Kyle's lambda); deviations from linearity reveal market depth constraints and information asymmetry. For circuits, linearity of the intervention-effect relationship is a stronger structural claim than monotonicity alone: it implies the circuit's contribution is additive and does not interact nonlinearly with the rest of the model.

The test fits a linear model to (intervention magnitude, measured effect) pairs across ≥ 5 intervention strengths. The criterion is satisfied when R-squared ≥ 0.8 and no single residual exceeds 20% of the predicted value at that point. The residual check matters because a high R-squared can mask localized nonlinearities (threshold effects, saturation) that are mechanistically informative.

This criterion is strictly stronger than E2 (Graded Response). E2 requires monotonicity — the effect goes up as intervention goes up. E9 requires the relationship to be approximately linear. A circuit that shows strong monotonic degradation but with a sharp threshold at one intervention strength will pass E2 but fail E9. This distinction matters: linearity suggests a proportional contribution, while threshold behavior suggests a switch-like mechanism. Both are valid findings, but they support different mechanistic interpretations.

## Minimum reporting rule

- ≥ 5 intervention magnitudes tested, evenly spaced across the operating range.
- Effect value at each magnitude.
- Linear fit R-squared.
- Per-point residuals (absolute and as fraction of predicted value).
- Comparison to monotonic-only fit (E2) — does linearity add explanatory power?
- Identification of any threshold or saturation regime where linearity breaks down, with the corresponding intervention magnitude range.
- If R-squared < 0.8 or max residual/predicted ≥ 0.2: criterion is unsatisfied; report the actual values and characterize the nonlinearity.
