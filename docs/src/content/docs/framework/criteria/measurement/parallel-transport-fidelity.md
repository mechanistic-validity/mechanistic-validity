---
title: "Parallel Transport Fidelity"
validity_type: "Measurement"
criterion_id: "M10"
---

# Criterion M10 — Parallel Transport Fidelity

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | Mean sheaf consistency ≥ 0.8 across circuit edges; no edge below 0.5 |
| Evidence family | Representational |
| Minimum reporting | Mean consistency, per-edge scores, minimum edge, shuffled-graph baseline, identification of low-consistency edges |
| Common failure mode | Computing consistency without shuffled-graph baseline |
| Lens | Geometry |

## What this criterion requires

Parallel transport fidelity tests whether a representation maintains its meaning as it moves through circuit components. In differential geometry, parallel transport moves a vector along a curve while preserving its relationship to the local geometry. For circuits, this means: if a feature is represented at one component's output, does it arrive at the next component's input with the same meaning, or does the intervening transformation distort it?

Sheaf consistency formalizes this. Model the circuit as a graph where nodes are components and edges are the connections between them. Each node has a local representation space. Each edge has a transport map (the linear transformation between adjacent components' representation spaces). Sheaf consistency on an edge measures whether the transported representation at one end matches the actual representation at the other end — a score of 1.0 means perfect preservation, 0.0 means complete distortion. The criterion requires mean consistency ≥ 0.8 across all circuit edges and no individual edge below 0.5.

The shuffled-graph baseline is essential. Compute the same consistency scores on a graph with the same nodes but randomly permuted edges. This baseline measures the consistency you would expect if components were connected arbitrarily rather than by the circuit's actual wiring. Without this baseline, high consistency could reflect properties of the representation spaces (e.g., all components using similar bases) rather than meaningful transport through the circuit. This criterion does not establish that the transported representation is causally used by downstream components or that the interpretation of the representation is correct.

## Minimum reporting rule

- Mean sheaf consistency across all circuit edges.
- Per-edge consistency scores (or at minimum: max, min, quartiles).
- Minimum-consistency edge identified by name.
- Shuffled-graph baseline: mean consistency on ≥ 50 random edge permutations (mean, 95% CI).
- Separation between real-graph and shuffled-graph consistency.
- If any edge falls below 0.5 or shuffled baseline is not computed: criterion is unsatisfied; identify the failing edges.
