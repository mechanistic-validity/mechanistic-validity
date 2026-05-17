---
title: "Level–Evidence Match"
validity_type: "Interpretive"
criterion_id: "V2"
---

# Criterion V2 — Level–Evidence Match

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | The evidence collected is sufficient to license the declared mode tag |
| Evidence family | N/A (criterion is about evidence–claim logic) |
| Minimum reporting | List of criteria met for each validity type required by the declared tag; explicit statement of any criteria not met |
| Common failure mode | Declaring `[causal-mechanistic]` based on necessity alone; not listing which criteria are satisfied |

## What this criterion requires

Level–evidence match checks that the declared mode tag (V1) is licensed by the evidence in hand. It is the criterion that prevents level inflation — the most common error in MI.

For each mode tag, the required evidence:

| Tag | Required criteria (minimum) |
|---|---|
| `[functional]` | Behavioral evidence only — no validity criteria required |
| `[representational]` | M3 (baseline separation); M5 (calibration); M1 (reliability) |
| `[causal-mechanistic]` | I1 (necessity); I2 (sufficiency); I3 (specificity); M3 (baseline separation) |
| `[structural-mechanistic]` | C2 (structural plausibility); I1 (necessity); M3 (baseline separation) |
| `[transportable]` | E5 (robustness) minimum; E6 (cross-architecture) for strongest form |

To satisfy level–evidence match: list every required criterion for the declared tag; report whether each criterion was satisfied; if any required criterion is unsatisfied, the declared tag must be downgraded.

## Example: SVA circuit at L8.MLP

Declared tag: `[causal-mechanistic]`  
Required criteria: I1, I2, I3, M3

| Criterion | Status | Evidence |
|---|---|---|
| I1 Necessity | **Partial** — one ablation method only | Zero ablation only; resample not run |
| I2 Sufficiency | **Open** — circuit-only forward pass not run | — |
| I3 Specificity | **Open** — control-axis IIA not computed | — |
| M3 Baseline separation | **Open** — random-vector baseline not computed | — |

**Verdict:** Evidence does not yet license `[causal-mechanistic]`. Current justified tag: `[representational]` for L8.MLP as an SVA-associated subspace. Upgrade path: run I2 (complement ablation), I3 (control-axis IIA), M3 (random-vector baseline).

## Minimum reporting rule

For every published claim: list all required criteria for the declared tag, with status (satisfied/partial/open) and the specific evidence for each. If any criterion is open or partial, the verdict must be downgraded to the highest tag whose required criteria are all satisfied.
