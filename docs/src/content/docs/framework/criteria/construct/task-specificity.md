---
title: "Task Specificity"
validity_type: "Construct"
criterion_id: "C3"
---

# Criterion C3 — Task Specificity

| | |
|---|---|
| Validity type | Construct |
| Pass condition | The proposed circuit does not score highly on unrelated tasks under the same metric and intervention strength |
| Evidence family | Representational (cross-task IIA), Behavioral (cross-task faithfulness) |
| Minimum reporting | IIA or faithfulness on ≥1 unrelated control task; ratio of on-task to off-task score |
| Common failure mode | Reporting only the target-task score; never testing whether the same circuit scores well on completely unrelated behaviors |

## What this criterion requires

Task specificity is the discriminant side of construct validity. It asks whether the circuit is specific to the claimed computation, or is a general-purpose structure that would score well on any task.

Satisfied when:

1. **The circuit is tested on ≥1 unrelated control task.** Different in computational structure, not just a variant. For SVA: Greater-Than is a good control. For IOI: Greater-Than or gendered pronoun.
2. **The circuit's score on the control task is substantially lower.** On-task:off-task ratio ≥ 2:1, or off-task score not significantly above chance using the same metric and strength.
3. **Cross-task IIA contamination is near zero.** Using the same subspace alignment, the IIA on the control task should be near baseline.

## The specificity ratio

```
R = IIA(target task) / IIA(control task)
```

R = 1.0 means the circuit scores equally on both — task specificity fails. R ≥ 2.0 with off-task score not significantly above baseline is a reasonable pass threshold.

## Worked example

An SVA circuit claim requires: IIA on a control task (e.g., Greater-Than or IOI) using the same DAS-IIA setup. If the component shows high IIA on SVA and low IIA on Greater-Than (R ≥ 2.0), task specificity is satisfied. If IIA is comparable on both tasks (R ≈ 1.0), the construct is "general information-processing locus," not SVA-specific — a weaker and different claim.

## Minimum reporting rule

- Control task, intervention strength used for control.
- Off-task score with same precision as on-task score.
- Specificity ratio.
- If off-task score is unexpectedly high, discuss — may mean the construct is under-specified.
