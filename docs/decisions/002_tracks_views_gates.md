# ADR-002: 3 Tracks + 4 Views + 4 Gates Architecture

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Organize the benchmark into 3 competitive Tracks, 4 scoring Views, and 4 precondition Gates.

### Tracks (methods compete, leaderboards)
- **Track 1 — Circuit Localization**: "What's the circuit?" Input: model + task. Output: edge set. Score: CMD, faithfulness. (MIB-compatible)
- **Track 2 — Causal Variable Localization**: "What are the variables?" Input: model + task + variable. Output: aligned subspace. Score: IIA. (MIB-compatible)
- **Track 3 — Causal Model Testing**: "Is this mechanism real?" Input: model + task + MechanisticClaimSpec. Output: verdict profile. Score: confirmation rate + claim ceiling. (Novel — ours)

### Views (aggregations over existing metrics)
- V1 Effect Estimation, V2 Transportability, V3 Counterfactual, V4 Adjudication

### Gates (preconditions)
- G0 Construct Operationalization, G1 Measurement Calibration, G2 Identifiability, G3 Superposition Risk

## Alternatives Considered

1. **Flat metric list** (status quo) — 82 metrics with no organizing principle
2. **MIB-only tracks** — Would miss Track 3 entirely, which is our novel contribution
3. **Single scoring dimension** — Would collapse distinct validity types into one number

## Justification

- Maps directly to causal inference literature: Tracks 1-3 correspond to causal discovery, causal abstraction, and causal model testing
- Cross-field alignment: each Track/View maps to neuroscience, psychometrics, pharmacology, and philosophy of science equivalents (documented in plan)
- Views are NOT new measurement infrastructure — they aggregate existing metrics. This means 0 new measurements needed for Views.
- Gates enforce preconditions that make Track scores meaningful (can't interpret Track 3 if the construct isn't operationalized)
- MIB compatibility on Tracks 1-2 ensures our work plugs into the existing benchmark ecosystem

## Impact

- `views.py`: 4 view definitions with metric lists + `run_view()` dispatcher
- `gates.py`: 4 gate checks + `check_gate()` dispatcher
- `__init__.py`: `mv.run_view()`, `mv.check_gate()` exposed at top level
- Track 3 is the largest piece — see ADR-003
