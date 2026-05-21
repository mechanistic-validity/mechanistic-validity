---
title: "Metric Space Comparison"
validity_type: "Measurement"
criterion_id: "M12"
---

# Criterion M12 — Metric Space Comparison

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | Gromov-Hausdorff distance ≤ 0.3 x diameter of either metric space |
| Evidence family | Representational |
| Minimum reporting | GH distance, diameter of each space, normalized distance, optimal correspondence, comparison to GH distance between circuits and random components |
| Common failure mode | Comparing circuits of different sizes without normalizing by diameter |
| Lens | Geometry |

## What this criterion requires

Metric space comparison tests whether two circuits have comparable geometric structure without requiring them to live in the same ambient space. The Gromov-Hausdorff (GH) distance measures how far two metric spaces are from being isometric — it finds the best possible correspondence between points in the two spaces and reports the worst-case distortion under that correspondence. Unlike Euclidean distance or cosine similarity, GH distance requires no shared coordinate system, making it appropriate for comparing circuits across different models or different training runs.

The criterion is satisfied when the GH distance between the two circuits' activation metric spaces is ≤ 30% of the diameter of either space (using the smaller diameter as the reference). Normalization by diameter is essential: a GH distance of 5.0 means nothing without knowing whether the spaces have diameter 10 (50% distortion, failing) or diameter 100 (5% distortion, passing). The failure mode of comparing circuits of different sizes without normalization produces misleading results because larger circuits have larger diameters and therefore larger absolute GH distances even when their structures are proportionally similar.

This criterion does not establish functional equivalence or causal similarity between the compared circuits. Two circuits can have nearly isometric activation geometries while implementing entirely different computations — geometric similarity is a necessary but not sufficient condition for mechanistic equivalence. The GH distance between the circuit pair should be compared to the GH distance between each circuit and a size-matched random component set to establish that the measured similarity exceeds the baseline.

## Minimum reporting rule

- Gromov-Hausdorff distance between the two circuit metric spaces.
- Diameter of each metric space.
- Normalized GH distance (GH distance / min diameter).
- Optimal correspondence (or an approximation, with the algorithm used).
- GH distance between each circuit and size-matched random components (baseline).
- If normalized distance exceeds 0.3 or baseline comparison is not provided: criterion is unsatisfied.
