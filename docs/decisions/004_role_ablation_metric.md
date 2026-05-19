# ADR-004: role_ablation as the Core Track 3 Metric

**Date:** 2025-05-19
**Status:** Implemented

## Decision

A single metric, `role_ablation`, serves as the execution engine for Track 3 predictions. It takes an `intervention_target` (the step to ablate) and a `measurement_target` (the step or "output" to measure), performs the ablation, and returns the normalized effect size.

Effect formula: `(ablated_mean - clean_mean) / |clean_mean|`

## Alternatives Considered

1. **One metric per prediction type** — `ablation_effect_on_output`, `ablation_effect_on_role`, etc. Problem: combinatorial explosion, duplicate logic
2. **Generic `intervention` metric** — Takes intervention type as parameter (ablate, patch, clamp, resample). Better but over-engineered for now.
3. **Direct logit diff only** — Only measure output logit diff. Problem: can't test role-to-role predictions ("ablating DTH reduces S-inhibition")

## Justification

- A single metric with `intervention_target` and `measurement_target` parameters covers all Track 3 prediction types
- When `measurement_target == "output"`, it measures logit diff change (the standard)
- When `measurement_target` is another step, it measures activation norm change at the measurement target's components
- The normalized effect formula gives interpretable values: -1.0 = complete destruction, -0.5 = 50% reduction, +1.0 = 100% increase

## Two measurement modes

### Output measurement
Ablate intervention target, run forward, compare logit diff (correct vs incorrect token) clean vs ablated. This is the standard activation patching / ablation metric.

### Role-to-role measurement
Ablate intervention target, capture activations at measurement target's components, compare activation norms clean vs ablated. Uses `run_with_hooks` with capture hooks to get post-ablation activations without needing two separate forward passes for the ablation and the capture.

Key implementation detail: `run_with_cache()` doesn't accept `fwd_hooks`, so we can't ablate and cache in one call. Instead, we create capture hooks that write to a shared dict during `run_with_hooks`, then measure from that dict.

## Calibration

Each spec's predictions include `expected_threshold` values calibrated from real measurements. This was essential for IOI — initial thresholds were too aggressive (e.g., expecting name mover ablation to reduce output by 50%, when backup name movers compensate). After running measurements, thresholds were set to match observed effects with margin.

## Impact

- `role_ablation.py`: ~380 lines, the largest single metric
- `__init__.py`: registered in `_METRIC_REGISTRY`
- `verify()`: routes CausalPredictions to role_ablation when they have intervention/measurement targets
