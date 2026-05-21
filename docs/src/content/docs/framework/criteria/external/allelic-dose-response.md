---
title: "Allelic Dose-Response"
validity_type: "External"
criterion_id: "E7"
---

# Criterion E7 — Allelic Dose-Response

| | |
|---|---|
| Validity type | External |
| Pass condition | Monotonic degradation across ≥ 3 qualitatively different intervention types; rank correlation ≥ 0.8 |
| Evidence family | Causal |
| Minimum reporting | Intervention types in severity order, performance at each, rank correlation, comparison to E2 dose-response at fixed intervention type |
| Common failure mode | Using only 2 intervention types; not establishing a severity ordering a priori |
| Lens | Genetics |

## What this criterion requires

E7 tests whether different *kinds* of intervention — not just different *amounts* of one intervention — produce degradation that tracks with intervention severity. In genetics, allelic series provide a graded loss-of-function spectrum: null, hypomorph, neomorph. The analogous test for circuits is to apply qualitatively distinct ablation methods (zero ablation, mean ablation, resample ablation, rank reduction) and verify that the resulting performance degradation follows the a priori severity ordering.

The severity ordering must be declared before running the experiment. A natural ordering for ablation methods is: rank reduction (mild, preserves dominant subspace) < resample ablation (moderate, preserves marginal statistics) < mean ablation (strong, collapses to a single point) < zero ablation (severe, removes all signal). The criterion is satisfied when task performance degrades monotonically along this ordering and the Spearman rank correlation between severity rank and performance degradation is ≥ 0.8.

This criterion is strictly stronger than E2 (Graded Response). E2 varies how much of a single intervention is applied — a scalar dose. E7 varies what kind of intervention is applied — a categorical dose across qualitatively different perturbation types. A circuit that passes E2 with resample ablation but shows identical degradation under mean and zero ablation has a weaker mechanistic claim than one that shows graded degradation across all three.

## Minimum reporting rule

- Intervention types used, listed in their a priori severity order with justification for the ordering.
- Task performance at each intervention type.
- Spearman rank correlation between severity rank and performance degradation.
- Comparison to E2 dose-response within a single intervention type (to distinguish categorical from scalar sensitivity).
- If fewer than 3 intervention types were tested: criterion is unsatisfied.
