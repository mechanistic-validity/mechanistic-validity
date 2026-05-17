---
title: "Intervention Reach"
validity_type: "External"
criterion_id: "E1"
---

# Criterion E1 — Intervention Reach

| | |
|---|---|
| Validity type | External |
| Pass condition | A measurement confirms the target component's activations changed in the predicted direction and magnitude |
| Evidence family | Causal |
| Minimum reporting | Activation delta at hook point (before vs. after intervention); direction confirmation |
| Common failure mode | Assuming the intervention reached the target because the code ran; never measuring the activation change |

## What this criterion requires

Intervention reach verifies that the intervention *actually changed the target activations in the predicted way*. Failure modes:

1. **Wrong hook point:** The hook name did not match the intended component.
2. **Effect absorbed upstream:** Skip connections or normalization layers partially mask the intervention.
3. **Magnitude near zero:** The component was near-inactive on test prompts; ablation delta is near zero regardless.
4. **Wrong direction:** A steering intervention added a vector not in the expected direction.

Satisfied when: activation value at the hook point is measured before and after intervention; delta is in the predicted direction; magnitude is non-trivial (not near zero).

## Minimum reporting rule

- Measure and report activation value at the target hook point before and after intervention (sample of test prompts).
- Report mean absolute delta and confirm direction.
- If delta < 0.01 in normalized units: flag — the intervention may not have reached the target.

## Why this is an external validity criterion

Intervention reach is a prerequisite for any external generalization claim. If the intervention did not reach the target reliably in the original setting, there is no basis for cross-model or cross-task generalization. It also functions as a measurement validity check on the experimental procedure.
