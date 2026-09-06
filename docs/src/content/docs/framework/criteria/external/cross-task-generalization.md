---
title: "Cross-Task Generalization"
validity_type: "External"
criterion_id: "E3"
---

# Criterion E3 — Cross-Task Generalization

| | |
|---|---|
| Validity type | External |
| Pass condition | The mechanism transfers to related tasks, not only the task it was discovered on |
| Evidence family | Behavioral, Causal |
| Minimum reporting | Related tasks tested, effect sizes on each, similarity metric between tasks |
| Common failure mode | Claiming generality from a single task |

## What this criterion requires

Cross-task generalization asks whether the mechanism operates beyond the specific task used to discover it. An "induction head" found on copying tasks should also operate on other in-context learning tasks if the claim is that it implements general copying.

Satisfied when:

1. **Related tasks are identified and tested.** Tasks that share the hypothesized function but differ in surface form.
2. **The mechanism operates on those tasks.** Ablation or patching shows the mechanism is causally involved in the related tasks.
3. **The scope is bounded.** The claim states which tasks the mechanism covers and which it does not.

## MI example

The greater-than circuit was discovered on "The war lasted from 1723 to 17__" prompts. Cross-task generalization asks whether the same circuit handles other ordinal comparisons — months, alphabet position, numbered lists. If it only works on year comparisons, the mechanism is narrower than "greater-than" implies.

## Connection to the chain

Required for Validated tier. Without cross-task generalization, a mechanism labeled with a general function word ("greater-than," "copying," "refusal") may be specific to one task format. The label imports scope the evidence has not established (see V4, unlicensed labeling).
