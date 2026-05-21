---
title: "Free Energy Efficiency"
validity_type: "Measurement"
criterion_id: "M8"
---

# Criterion M8 — Free Energy Efficiency

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | ≥ 50% of circuit heads Pareto-efficient; mean Pareto efficiency ≥ 0.6 |
| Evidence family | Information-theoretic |
| Minimum reporting | Accuracy metric, complexity metric, Pareto frontier plot, fraction of circuit heads on frontier, mean Pareto efficiency, comparison to non-circuit heads |
| Common failure mode | Defining accuracy and complexity on the same metric (circular) |
| Lens | Information Theory |

## What this criterion requires

Free energy efficiency tests whether circuit components achieve a good tradeoff between accuracy (how much they contribute to the task) and complexity (how much representational capacity they consume). In statistical physics, free energy = energy - temperature x entropy; systems at equilibrium minimize free energy, balancing accuracy against model complexity. For circuits, components that lie on the Pareto frontier of accuracy vs. complexity are using their capacity efficiently.

The accuracy metric measures each component's contribution to task performance (e.g., logit difference contribution, faithfulness when ablated). The complexity metric measures representational cost (e.g., effective rank of weight matrices, number of active features, description length). These two metrics must be independently defined — using the same quantity for both (e.g., weight norm as both "how much it contributes" and "how complex it is") makes the criterion circular and vacuous.

Pareto efficiency for a component is computed as 1 minus the normalized distance from the component to the Pareto frontier. A component on the frontier has efficiency 1.0; a component far from the frontier has efficiency near 0. The criterion is satisfied when at least half the circuit heads lie on or near the Pareto frontier and the mean efficiency is ≥ 0.6. This criterion does not establish causal necessity — a component can be Pareto-efficient but redundant. It tests whether the measurement procedure is identifying components that use their capacity well, not components that happen to be large or active.

## Minimum reporting rule

- Accuracy metric definition and values per component.
- Complexity metric definition and values per component (must be independently defined from accuracy metric).
- Pareto frontier plot with circuit heads and non-circuit heads distinguished.
- Fraction of circuit heads on the Pareto frontier.
- Mean Pareto efficiency for circuit heads vs. non-circuit heads.
- If accuracy and complexity metrics are not independently defined: criterion is automatically unsatisfied.
