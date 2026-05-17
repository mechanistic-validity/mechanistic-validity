---
title: "Graded Response"
validity_type: "External"
criterion_id: "E2"
---

# Criterion E2 — Graded Response

| | |
|---|---|
| Validity type | External |
| Pass condition | The effect scales monotonically with intervention strength; a threshold and plateau are visible |
| Evidence family | Causal |
| Minimum reporting | ≥7 intervention-strength values; task metric at each; identification of threshold and plateau |
| Common failure mode | Testing only 1–2 intervention strengths; never characterizing the dose-response curve |

## What this criterion requires

A mechanistically real intervention produces a response that:

1. **Scales monotonically** with strength (more intervention → more effect, up to plateau).
2. **Has a threshold**: below some minimal strength, the effect is negligible. A real mechanism has a threshold; random noise does not.
3. **Has a plateau**: at some maximal strength, the effect saturates. Beyond the plateau, off-task metrics should degrade before on-task metrics plateau.

Operationalized via steering multiplier sweep using `steering multiplier sweep` in `factor_analysis.py`: sweep the `strength` parameter from 0 to 20 across ≥7 values; measure the target task metric at each; identify threshold and plateau.

## Why the plateau matters

A result showing only the increasing part — without the plateau — is consistent with the hypothesis that the intervention is generically disrupting the model rather than specifically saturating a mechanism. A genuine mechanistic intervention will plateau when the specific computation is saturated; generic disruption keeps degrading all metrics without a meaningful plateau.

## Minimum reporting rule

- Full dose-response curve with ≥7 strength values.
- Identify threshold and plateau (or note that neither was found, with explanation).
- Compare on-task and off-task metrics at each strength level.
- If only 1–2 strength values tested: graded response is unsatisfied — flag it.
