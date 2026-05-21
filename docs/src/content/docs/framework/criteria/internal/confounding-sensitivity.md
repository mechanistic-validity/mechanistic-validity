---
title: "Confounding Sensitivity"
validity_type: "Internal"
criterion_id: "I10"
---

# Criterion I10 — Confounding Sensitivity

| | |
|---|---|
| Validity type | Internal |
| Pass condition | E-value at least 2.0 |
| Evidence family | Causal |
| Minimum reporting | Point estimate of causal effect, 95% CI, E-value for point estimate, E-value for CI bound, interpretation |
| Common failure mode | Computing E-value without reporting the point estimate and CI it is based on |
| Lens | Genetics |

## What this criterion requires

Confounding sensitivity quantifies how strong an unmeasured confounder would need to be to explain away the observed causal effect. The E-value (introduced by VanderWeele and Ding, 2017) provides this bound: it is the minimum strength of association that an unmeasured confounder would need to have with both the treatment (circuit activation) and the outcome (model behavior) to fully account for the observed effect.

Satisfied when:

1. **E-value for the point estimate is at least 2.0.** A confounder would need to at least double both its association with the circuit and its association with the outcome to explain the result.
2. **E-value for the confidence interval bound is reported.** The E-value for the lower bound of the 95% CI indicates robustness: if this is also above 2.0, even the weakest plausible version of the effect is robust to moderate confounding.
3. **Interpretation is stated in plain language.** The report includes a sentence of the form: "A confounder would need to [X] the effect to explain it away."

Confounding sensitivity does not establish that no confound exists. It establishes a lower bound on how strong any confound would need to be, allowing readers to judge plausibility.

## Distinction from I5 — Confound Control

I5 asks "did you control for confounds?" by comparing ablation methods and checking for mean-field disruption. I10 asks "how robust is your claim to confounds you did not control for?" I5 is a procedural check; I10 is a quantitative sensitivity analysis. They are complementary: I5 addresses known confounds, I10 bounds the impact of unknown ones.

## Minimum reporting rule

- Point estimate of the causal effect (e.g., ablation delta, patching recovery fraction).
- 95% confidence interval for the effect.
- E-value for the point estimate.
- E-value for the CI bound closer to null.
- Plain-language interpretation of the E-value.
