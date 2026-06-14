---
title: "Confound Control"
validity_type: "Internal"
criterion_id: "I5"
---

# Criterion I5 — Confound Control

| | |
|---|---|
| Validity type | Internal |
| Pass condition | The ablation effect is not explained by collateral disruption to non-circuit components |
| Evidence family | Causal |
| Minimum reporting | Component-specific ablation result; comparison to full-circuit ablation; mean vs. zero vs. resample comparison |
| Common failure mode | Running only mean ablation; not checking whether the effect is due to mean-field signal disruption |

## What this criterion requires

Confound control asks: is the observed effect due to the loss of the nominated component's specific computation, or due to collateral disruption — unintended side effects affecting downstream components?

Satisfied when:

1. **Component-specific ablation** (ablating only the target, not adjacent components) produces the same degradation as the full-circuit ablation.
2. **Mean ablation is compared to zero and resample ablation.** Mean ablation changes the mean-field signal to all downstream heads — can cause cascade failures unrelated to the target component. If zero and resample produce comparable degradation, the mean-field confound is ruled out.
3. **The effect is not due to hub disruption.** Component-specific ablation paired with specificity test (I3) together rule out the hub confound.

## The mean-field confound

When a component's activation is replaced with the dataset mean, all downstream components receive the mean vector. For components with linear operations, the mean vector carries average training-distribution content, which may be task-relevant. This can suppress or amplify downstream effects unrelated to the ablated component's specific computation.

**Fix:** Always run zero and resample alongside mean. If all three show comparable degradation, mean-field confound is ruled out. If mean ablation shows large degradation but zero shows small degradation — the effect may be mean-field disruption, not loss of specific computation.

## Causal scrubbing as confound control

Causal scrubbing (`c04causalscrubbing.py`, A01 SCMPearl) replaces activations at each component with activations from a run where the causal variable is absent, while leaving all other activations unchanged. This is a component-specific counterfactual that controls for all pathway confounds simultaneously. A successful causal scrubbing result is the strongest single piece of confound-control evidence.

## Confound control vs. rival mechanisms

I5 asks whether the causal effect is due to the nominated component's *specific computation* or to *collateral disruption*. It does not ask whether a *different* set of components could produce the same effect without collateral disruption — that is the domain of [I6 Rival Mechanism Exclusion](/framework/criteria/internal/rival-mechanism-exclusion/). A claim can pass I5 (the effect is genuinely due to this component, not collateral damage) while failing I6 (other components could produce the same effect).

## Minimum reporting rule

- Whether component-specific ablation was compared to full-circuit ablation.
- Whether ≥2 ablation methods were compared to check mean-field confound.
- Whether causal scrubbing was run (if so, report the result).
- If only one ablation method used: confound control is partial pass — flag explicitly.
