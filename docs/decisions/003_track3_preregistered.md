# ADR-003: Track 3 as Pre-Registered Hypothesis Testing

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Track 3 works by pre-registering mechanistic hypotheses as `MechanisticClaimSpec` objects (computational DAG + testable predictions + negative controls), then executing those predictions with real model interventions.

The spec declares:
- **Steps**: Named computational components (nodes in a mechanism DAG)
- **Edges**: Directed information flow between steps
- **Predictions**: "If I ablate step X, step Y's output should decrease by at least Z"
- **Negative Controls**: "If I ablate step X, upstream step W should be unaffected"

`mv.verify(spec)` runs all predictions, returns per-prediction verdicts, confirmation rate, and a claim ceiling (highest description mode where all predictions pass).

## Alternatives Considered

1. **Post-hoc analysis only** — Run all metrics, find patterns. Problem: p-hacking, no falsifiability
2. **Fixed intervention protocol** — Same intervention on every circuit. Problem: different circuits have different causal structures
3. **Manual test scripts** — Write custom scripts per circuit. Problem: no standardization, no comparison

## Justification

- Pre-registration is the standard for credible scientific claims (clinical trials, replication crisis response)
- The DAG structure makes causal claims explicit and testable
- Negative controls are critical: if ablating a downstream component affects an upstream one, your causal model is wrong (the effect is going the wrong direction, or there's an unmapped pathway)
- Confirmation rate is the primary Track 3 score: what fraction of your predictions are confirmed by real interventions?
- Claim ceiling maps to the Marr/description-mode hierarchy: you can only claim your mechanism is at a given level of abstraction if all predictions at that level pass

## Key Design Choices

1. **Predictions reference step names, not head indices** — This decouples the hypothesis from the implementation. "Ablating the duplicate token detector should reduce s-inhibition" is testable regardless of which specific heads implement duplicate token detection.

2. **Thresholds are per-prediction** — Different predictions have different expected effect sizes. "Ablating the most critical role should reduce output by 80%" vs "ablating a supporting role should reduce output by at least 20%."

3. **`_extract_value()` handles both dicts and EvalResult objects** — The metric layer returns EvalResult dataclasses, but the framework layer works with Pydantic models. The bridge function handles both.

## Results

IOI: 5/5 positive predictions pass, 3/3 negative controls pass. 100% confirmation rate.
Greater-than: 1/3 attention-only predictions pass (MLP support needed for the other 2).
Induction: 0/3 pass (thresholds too aggressive for a 2-head circuit).
SVA: 1/4 pass (multi-layer distributed circuit, attention-only ablation insufficient).
RTI: 1/4 pass (similar to SVA — distributed circuit).

## Impact

- `spec.py`: MechanisticClaimSpec, CausalPrediction, SpecVerificationResult
- `__init__.py`: `mv.verify()` top-level function
- `role_ablation.py`: the metric that executes predictions
- 5 claim specs: IOI, greater-than, induction, SVA, RTI
