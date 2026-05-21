---
title: "Angular Separability"
validity_type: "Construct"
criterion_id: "C11"
---

# Criterion C11 — Angular Separability

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Mean pairwise angle between task subspaces >= 30 degrees; separation >= 3x within-subspace spread |
| Evidence family | Representational |
| Minimum reporting | Pairwise angles, within-subspace spread, separation-to-spread ratio, random Grassmannian baseline |
| Common failure mode | Reporting angles without within-subspace spread (high angles can be trivial in high dimensions) |
| Lens | Geometry |

## What this criterion requires

Angular separability asks whether the circuit's task-relevant subspaces are geometrically separated in a way that enables downstream linear readout. If two task-relevant representations (e.g., "singular subject" and "plural subject" in SVA) occupy overlapping subspaces, a linear downstream computation cannot reliably distinguish them. Angular separation is a necessary condition for the circuit to support the claimed computational structure through simple (linear or near-linear) mechanisms.

The criterion is satisfied when:

1. **The mean pairwise angle between task subspaces is >= 30 degrees.** For each pair of task-relevant subspaces (e.g., subspaces for different task labels or different computational roles), the principal angle must be at least 30 degrees. This threshold is chosen because in high-dimensional spaces, random subspaces are nearly orthogonal — the comparison to random baselines (below) is what gives the angle meaning.
2. **The separation-to-spread ratio is >= 3.** The mean pairwise angle between subspaces must be at least 3 times the mean within-subspace angular spread. This ensures the subspaces are tight clusters, not diffuse regions that happen to have separated centroids.
3. **The separation exceeds what random subspaces achieve on a Grassmannian baseline.** In high-dimensional spaces, random subspaces have predictable angular distributions. The observed separation must exceed the random baseline by a meaningful margin, otherwise high angles are an artifact of dimensionality, not task structure.

This criterion does not establish that the model actually uses the angular separation for its computation. The subspaces may be separated but read out through a nonlinear mechanism that ignores the angular structure. It also does not establish uniqueness — different basis choices may produce different separability results.

## Minimum reporting rule

- Pairwise principal angles between all pairs of task subspaces.
- Within-subspace angular spread for each subspace.
- Separation-to-spread ratio.
- Random Grassmannian baseline: expected pairwise angle for random subspaces of the same dimensionality in the same ambient space.
- If separation-to-spread ratio is < 3 despite high absolute angles, report this as a failure — the subspaces are not well-separated relative to their internal variability.
