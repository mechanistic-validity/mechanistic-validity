---
title: "Cross-Model Generalization"
validity_type: "External"
criterion_id: "E4"
---

> The framework paper's appendix scorecards call this criterion *cross-model recurrence*. It is the same criterion; this page keeps the name used in the criterion table, and the URL keeps the earlier slug so existing links resolve.

# Criterion E4 — Cross-Model Generalization

| | |
|---|---|
| Validity type | External |
| Pass condition | The corresponding causal organization recurs across independently trained models under a declared correspondence criterion |
| Evidence family | Causal, Structural |
| Minimum reporting | Models tested, correspondence criterion, whether the causal organization (not just the behavior) recurs |
| Common failure mode | Observing the same behavior in a second model and calling it the same mechanism |

## What this criterion requires

Cross-model generalization asks whether the mechanism appears in other models — not just the behavior, but the causal organization behind it. Different models can produce the same input-output relation through different internal mechanisms, so behavioral agreement alone does not establish generalization.

Satisfied when:

1. **A correspondence criterion is declared.** What counts as "the same mechanism" across models — same heads, same subspace alignment, same causal graph structure.
2. **The causal organization is tested in a second model.** Not just behavioral output, but the internal structure — which components, which connections, which information flow.
3. **The correspondence holds or fails transparently.** If the mechanism recurs under one criterion but not another, both are reported.

## MI example

Knowledge neurons claims "can be easily generalized" to other models, but the study examines only one model. Cross-model generalization is untested. This is a scope-creep failure (V5) compounded by the absence of E4 evidence.

## Connection to the chain

Required for Triangulated where the claim asserts reach beyond the systems tested; where it does not, prompt generalization (E2) carries the gate instead. Without cross-model generalization, the mechanism may be an idiosyncrasy of one training run rather than a general computational strategy. Steel (2008): recurrence alone supports limited induction; stronger extrapolation requires evidence that the causally relevant process is preserved.
