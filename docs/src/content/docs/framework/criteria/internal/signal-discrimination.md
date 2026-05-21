---
title: "Signal Discrimination"
validity_type: "Internal"
criterion_id: "I15"
---

# Criterion I15 — Signal Discrimination

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Effect ratio (task-relevant / random perturbation) at least 2.0; p < 0.01 |
| Evidence family | Causal |
| Minimum reporting | Task-relevant perturbation effect, random perturbation effect, ratio, p-value, perturbation magnitude (must be matched), number of samples per condition (at least 50) |
| Common failure mode | Using different magnitudes for task-relevant vs random perturbations |
| Lens | Economics |

## What this criterion requires

Signal discrimination tests whether the circuit responds preferentially to task-relevant perturbations over random perturbations of equal magnitude. A circuit that reacts equally to signal and noise is not discriminating -- it is merely sensitive. A circuit that reacts substantially more to task-relevant input changes demonstrates functional tuning for the target computation.

Satisfied when:

1. **Task-relevant perturbations produce larger effects.** The circuit's output change (measured on the target metric) is at least 2x larger for task-relevant perturbations than for random perturbations.
2. **Perturbation magnitudes are matched.** Task-relevant and random perturbations have the same L2 norm (or other magnitude measure). Unmatched magnitudes invalidate the comparison entirely.
3. **Statistical significance is established.** The difference in effect sizes is significant at p < 0.01, tested over at least 50 samples per condition.

Signal discrimination does not establish what the circuit does with the signal, nor does it establish sufficiency. It establishes that the circuit's response profile is tuned to task-relevant structure rather than responding generically to any input perturbation.

## Distinction from I3 — Specificity

I3 tests whether the circuit's causal effect is selective for one task over another (cross-task comparison). I15 tests whether the circuit discriminates signal from noise within a single task (within-task comparison). A circuit can be task-specific (I3 pass) yet fail to discriminate signal from noise within that task (I15 fail) if it responds to any perturbation, task-relevant or not.

## Minimum reporting rule

- Task-relevant perturbation effect size (mean and variance).
- Random perturbation effect size (mean and variance).
- Effect ratio (task-relevant / random).
- p-value for the difference.
- Perturbation magnitude and confirmation that magnitudes are matched.
- Number of samples per condition (minimum 50).
- Definition of "task-relevant" perturbation and justification.
