# ADR-013: Epistemic Framing Claim Spec

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Add a MechanisticClaimSpec for the epistemic framing circuit (4 heads, 3 roles). This is the first claim spec for a non-published, experimentally-discovered circuit.

## Circuit Structure

3 roles forming a linear pipeline with a skip connection:

```
detector (L6H5) ──→ integrator (L9H2, L9H5) ──→ executor (L10H5)
       └──────────────────────────────────────────→ (skip)
```

- **Detector**: Identifies epistemic verb tokens (think/believe/know)
- **Integrator**: Combines epistemic signal with factual content
- **Executor**: Adjusts output logit distribution

## Predictions

4 positive predictions + 2 negative controls:

| Prediction | Type | Threshold |
|-----------|------|-----------|
| Ablate executor → output decreases | positive | 0.15 |
| Ablate integrator → output decreases | positive | 0.10 |
| Ablate detector → output decreases | positive | 0.05 |
| Ablate detector → integrator decreases | positive | 0.05 |
| Ablate executor → detector unchanged | negative | invariant |
| Ablate integrator → detector unchanged | negative | invariant |

Thresholds are conservative (lower than other specs) because:
1. This is a 4-head circuit — individual heads contribute less than in IOI's 26-head circuit
2. The epistemic effect may be subtle compared to factual completion
3. No prior measurements to calibrate against — thresholds will need recalibration after first run

## Why This Matters

- First Track 3 spec for our own experimental discovery (not published literature)
- Tests whether the spec framework works for novel, smaller circuits
- The epistemic circuit is interesting because it's truth-insensitive and subject-invariant
- 4 variants exist (core, expanded, tight, EAP) — enables mechanism adjudication (V4)

## Impact

- `epistemic_framing/claim_spec.py`: EPISTEMIC_SPEC with 3 steps, 3 edges, 4 predictions, 2 negative controls
- `_builtins.py`: import + `get_claim_spec()` on EpistemicFramingTask
- Total claim specs: 5 → 6
