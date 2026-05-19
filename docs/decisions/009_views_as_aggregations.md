# ADR-009: Views as Aggregations, Not New Metrics

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Views (V1-V4) are scoring aggregations over existing metrics, not new measurement infrastructure. `run_view("effect_estimation")` runs the 7 metrics in V1 and returns their scores together with a composite aggregate.

## Justification

- The 82+ existing metrics already cover all the measurement we need
- Views answer "how does this circuit score on causal effect estimation?" by collecting the relevant subset of metrics
- No new TransformerLens hooks, no new model interventions
- Adding a View is adding a list of metric names + an aggregation function

### View definitions

| View | Causal inference grounding | Metrics |
|------|---------------------------|---------|
| V1 Effect Estimation | Pearl/Rubin effect estimation | mediation, cate, dose_response, effect_size, pse, intervention_specificity |
| V2 Transportability | Pearl/Bareinboim transportability | cross_task_generalization, cross_model_invariance, generalization_gap |
| V3 Counterfactual | Pearl rung-3 counterfactuals | das_iia, iia_variants, counterfactual_consistency, corrupt_restore |
| V4 Adjudication | SEM equivalent-models testing | discriminant_validity |

## Alternatives Considered

1. **Views as new metric types** — Unnecessary complexity, would duplicate existing measurement logic
2. **No Views** — Users would have to manually select and aggregate metrics
3. **Single composite score** — Would hide which sub-metrics contributed

## Impact

- `views.py`: metric lists + `run_view()` function
- `__init__.py`: `mv.run_view()` exposed
