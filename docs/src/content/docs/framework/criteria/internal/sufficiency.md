---
title: "Sufficiency"
validity_type: "Internal"
criterion_id: "I2"
---

# Criterion I2 — Sufficiency

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Isolating or restoring the proposed component(s) reproduces the target behavior |
| Evidence family | Causal |
| Minimum reporting | Circuit-only forward pass result OR patching-in result; metric and recovery fraction; comparison to full-model baseline |
| Common failure mode | Showing necessity only; never running a circuit-only forward pass or restoration test |

## What this criterion requires

Sufficiency asks: does keeping *only* this component (ablating everything else) reproduce the behavior?

Two operationalizations:

**Circuit-only forward pass (complement ablation):** Ablate all components *not* in the proposed circuit. Measure whether the circuit alone produces the target behavior at ≥ 70% of the full-model level (threshold should be pre-stated per C1).

**Patching-in:** Start with a corrupted run. Patch the circuit's activations from a clean run into the corrupted run. Measure recovery. ≥ 70% of the clean-corrupted difference is a standard threshold.

## The faithfulness metric

```
faithfulness = (logit_diff(patched circuit) - logit_diff(corrupted))
               / (logit_diff(clean) - logit_diff(corrupted))
```

Reference points (GPT-2 Small):
- IOI circuit: 87% (Wang et al. 2022)
- Greater-Than circuit: 89.5% (Hanna et al. 2023)
- SVA circuit: 93% (Lazo et al. 2025)

## What sufficiency does not establish

Sufficiency does not establish specificity. A circuit that recovers 90% of the logit difference might also recover 85% on a completely unrelated task — a general-purpose structure. Task specificity (C3) must be tested separately.

## Why sufficiency is required for mechanistic claims

- A component might be necessary because it is upstream of the actual mechanism — ablating breaks everything downstream, but restoring alone doesn't reproduce the behavior.
- A component might appear necessary due to mean-field confounds.

Sufficiency rules out both: if restoring alone reproduces the behavior, the component is doing the relevant computation directly.

## Minimum reporting rule

- Which operationalization of sufficiency was used.
- Recovery fraction and comparison to full-model baseline.
- Pre-stated threshold and whether it was met.
- If sufficiency was not tested: flag as open criterion.
