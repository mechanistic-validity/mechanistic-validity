---
title: "Specificity"
validity_type: "Internal"
criterion_id: "I3"
---

# Criterion I3 — Specificity

| | |
|---|---|
| Validity type | Internal |
| Pass condition | The effect is selective; control-axis IIA ≈ 0 while causal-axis IIA is high |
| Evidence family | Representational (multi-axis IIA), Causal (cross-task ablation) |
| Minimum reporting | Control-axis IIA value; comparison to causal-axis IIA; ideally both representational and behavioral specificity tests |
| Common failure mode | Showing necessity without any control task or control axis test |

## What this criterion requires

Specificity asks: is the effect of the intervention *specific* to the claimed computation, or does it reflect generic disruption?

Two operationalizations:

**Representational specificity (multi-axis IIA):** Define a causal axis (e.g., subject number) and a control axis (e.g., object number or verb number). Run IIA on each axis using the same subspace. Specificity requires: causal-axis IIA high (above baseline), control-axis IIA near zero.

**Behavioral specificity (cross-task ablation):** Run the same ablation on an unrelated control task at equal intervention strength. Specificity requires: ablation degrades the target task substantially more than the control.

## The failure mode specificity catches

A component causally necessary for the target behavior *because* it is necessary for a large number of behaviors. A key hub in the residual stream degrades almost anything when ablated — not because it specifically implements the claimed computation, but because many computations flow through it.

Without specificity, necessity is consistent with the hub hypothesis. With control-axis IIA ≈ 0, the hub interpretation is ruled out.

## Worked example

For subject-verb agreement: causal axis = subject number (singular vs. plural). Natural control axis = object number — same sentence, similar position, but not the claimed causal variable. If IIA is high on subject-number and near zero on object-number, specificity is satisfied.

## Minimum reporting rule

- Control axis or control task used.
- Control-axis or control-task score alongside causal-axis score.
- Ratio.
- If no control included: specificity is unsatisfied — report explicitly.
