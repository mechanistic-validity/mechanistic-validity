---
title: "Invariance"
validity_type: "Measurement"
criterion_id: "M2"
---

# Criterion M2 — Invariance

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | The instrument gives comparable results across model sizes and model families — the same construct is being measured |
| Evidence family | Structural, Behavioral |
| Minimum reporting | Cross-scale transfer result (F1 or equivalent) with null expectation; template invariance test |
| Common failure mode | Treating cross-scale agreement as assumed; never testing whether the instrument measures the same thing in different models |

## What this criterion requires

Invariance asks: when the same instrument is applied to a different model size or family, is it measuring the same construct?

**Cross-scale invariance:** Weight classifier trained on GPT-2 Small achieves non-zero F1 when applied to Pythia-160M or GPT-2 Medium, above the null expectation for random cross-model comparison. Operationalized via `c13invariance.py`. Cross-scale F1 substantially above null indicates invariance; cross-scale F1 equal to null indicates the classifier is measuring something specific to GPT-2 Small's initialization.

**Template invariance:** The metric does not vary significantly across prompt templates (e.g., different sentence structures for the same task). Welch ANOVA across template groups; p > 0.05 is a pass. Significant variation across template groups may indicate the instrument measures something that varies with template structure — which may be expected or problematic.

## Invariance gates cross-architecture claims

Invariance testing at the measurement level is the prerequisite for making cross-architecture generalization claims (E6). You cannot claim the *mechanism* generalizes across models if you have not established that the *instrument* is measuring the same thing across models.

## Minimum reporting rule

- Cross-scale transfer result with null expectation.
- Template invariance test (Welch ANOVA or equivalent).
- If invariance was not tested: cross-architecture claims (E6) cannot be made — flag as open criterion.
