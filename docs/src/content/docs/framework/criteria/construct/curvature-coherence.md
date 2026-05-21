---
title: "Curvature Coherence"
validity_type: "Construct"
criterion_id: "C10"
---

# Criterion C10 — Curvature Coherence

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Top-3 principal curvature directions predict task features with R^2 >= 0.5; anisotropy ratio >= 5 |
| Evidence family | Representational |
| Minimum reporting | Eigenvalue spectrum of Fisher matrix, R^2 from top-3 eigenvectors to task labels, anisotropy ratio, random component comparison |
| Common failure mode | Reporting anisotropy without checking alignment with task features (high anisotropy from architectural structure, not task structure) |
| Lens | Geometry |

## What this criterion requires

Curvature coherence asks whether the Fisher information metric of the circuit has structure that matches the task. The Fisher matrix captures how sensitively the model's output distribution changes with respect to perturbations in different directions of parameter or activation space. If the circuit is performing a specific computation, perturbations along task-relevant directions should produce large changes (high curvature), while perturbations along task-irrelevant directions should produce small changes (low curvature).

The criterion is satisfied when:

1. **The top-3 principal curvature directions predict task features with R^2 >= 0.5.** Linear regression from the top-3 eigenvectors of the Fisher matrix to the task label (e.g., correct token, agreement class) must explain at least half the variance. This ensures the high-curvature directions are aligned with the task, not with incidental architectural features.
2. **The anisotropy ratio is >= 5.** The ratio of the largest to the smallest eigenvalue of the Fisher matrix must be at least 5. This ensures the curvature structure is non-trivial — an isotropic Fisher matrix has no preferred directions and provides no interpretive information.
3. **The curvature structure is specific to the circuit.** A random subset of components at the same layers should show lower anisotropy or worse alignment with task features. Without this comparison, high anisotropy may reflect architectural regularities (e.g., LayerNorm creating preferred directions) rather than task-specific computation.

This criterion does not establish that the high-curvature directions play a causal role in computation. It also does not establish cross-model generalization — the curvature structure may be model-specific.

## Minimum reporting rule

- Full eigenvalue spectrum of the Fisher matrix (or at minimum the top-10 and bottom-10 eigenvalues).
- R^2 from linear regression of top-3 eigenvectors to task labels, with bootstrap 95% CI.
- Anisotropy ratio (max eigenvalue / min eigenvalue).
- Same analysis on >= 1 random component set of matched size, with R^2 and anisotropy for comparison.
- If anisotropy is high but R^2 is low, report this as a negative result — the curvature is not task-aligned.
