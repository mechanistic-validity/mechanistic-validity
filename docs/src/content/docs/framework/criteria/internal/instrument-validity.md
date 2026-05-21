---
title: "Instrument Validity"
validity_type: "Internal"
criterion_id: "I9"
---

# Criterion I9 — Instrument Validity

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Instrument relevance F > 10 (Staiger-Stock); Sargan/Hansen overidentification p > 0.05 |
| Evidence family | Causal |
| Minimum reporting | Instrument definition, relevance F-statistic, overidentification test statistic and p-value, first-stage R-squared, IV estimate vs OLS estimate |
| Common failure mode | Weak instruments (F < 10); not testing overidentification when multiple instruments available |
| Lens | Genetics |

## What this criterion requires

Instrument validity uses instrumental variable (IV) analysis to test whether an upstream variable affects the outcome only through the proposed circuit (the exclusion restriction). This addresses unmeasured confounding that standard ablation cannot rule out: if an instrument is correlated with the circuit's activation but affects the output only through the circuit, the IV estimate isolates the circuit's causal contribution.

Satisfied when:

1. **Instrument relevance.** The instrument has a first-stage F-statistic exceeding the Staiger-Stock threshold of 10, confirming it is strongly correlated with the circuit's activation.
2. **Exclusion restriction is tested.** When multiple instruments are available, the Sargan/Hansen overidentification test has p > 0.05, consistent with all instruments satisfying the exclusion restriction.
3. **IV estimate is compared to naive estimate.** The IV estimate of the circuit's causal effect is compared to the OLS (ordinary intervention) estimate. Large divergence suggests confounding in the naive estimate.

Instrument validity does not establish the direct causal effect of the circuit on its own, nor does it establish sufficiency. It establishes that the causal pathway from input to output runs through the circuit, addressing confounds that ablation-based methods miss.

## Distinction from I5 — Confound Control

I5 controls for known confounds by comparing ablation methods and checking for mean-field disruption. I9 uses instruments to address unknown confounds: variables that the experimenter has not identified or cannot directly measure. I5 asks "did you control for confounds you know about?" I9 asks "can you rule out confounds you don't know about?"

## Minimum reporting rule

- Definition of the instrument(s) and why exclusion restriction is plausible.
- First-stage F-statistic and R-squared.
- Overidentification test statistic and p-value (when multiple instruments are used).
- IV estimate of the causal effect alongside OLS estimate.
- If F < 10: flag as weak-instrument concern.
