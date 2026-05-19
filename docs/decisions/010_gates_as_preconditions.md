# ADR-010: Gates as Preconditions, Not Scores

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Gates (G0-G3) are boolean precondition checks, not scored metrics. They return pass/fail with details, not numeric values.

## Justification

Gates check "is it meaningful to run this Track/View?" not "how well does this circuit perform?"

- **G0 Construct Operationalization**: Does the task have prompts and a circuit? If not, nothing can be measured.
- **G1 Measurement Calibration**: Do bootstrap confidence intervals and seed variance indicate stable measurements? If not, Track scores are noise.
- **G2 Identifiability**: Can the causal effects in this spec be estimated with available interventions? If not, Track 3 results are uninterpretable.
- **G3 Superposition Risk**: Are ablations confounded by polysemantic collateral damage? If polysemanticity is high, ablation effects may be misleading.

G0-G1 use existing infrastructure (prompt generators, calibration metrics). G2-G3 read metadata from the MechanisticClaimSpec.

## Alternatives Considered

1. **Gates as scored metrics** — Returns 0-1 instead of pass/fail. Overcomplicates interpretation.
2. **No gates** — Users run Track 3 on specs with high superposition risk without warning.
3. **Automatic gating** — verify() refuses to run if gates fail. Too restrictive — sometimes you want to see results even with caveats.

## Impact

- `gates.py`: `GateResult` model + `check_gate()` dispatcher
- `spec.py`: `IdentifiabilityGate`, `SuperpositionGate` on MechanisticClaimSpec
- `__init__.py`: `mv.check_gate()` exposed
