---
title: "Observability"
validity_type: "Measurement"
criterion_id: "M9"
---

# Criterion M9 — Observability

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | Observability Gramian rank ≥ 0.8 x dim(circuit state); linear probe R-squared ≥ 0.7 |
| Evidence family | Representational |
| Minimum reporting | Gramian eigenvalue spectrum, rank, probe R-squared per component, comparison to random components |
| Common failure mode | Probing from all model outputs rather than circuit outputs only |
| Lens | Control Theory |

## What this criterion requires

Observability tests whether a circuit's internal state is fully decodable from its outputs alone. In control theory, a system is observable if its full internal state can be reconstructed from its output trajectory. For circuits, this means that the information the circuit computes internally is accessible in the activations it passes downstream — nothing is hidden.

The observability Gramian is constructed from the circuit's output map: if the circuit has state dimension d and output dimension m, the Gramian is the m x d matrix of linear relationships between internal states and outputs. Its rank indicates how many dimensions of the internal state are recoverable. The criterion requires this rank to be at least 80% of the circuit state dimension. As a complementary check, a linear probe trained to predict internal circuit state from circuit outputs (not from the full model's residual stream) must achieve R-squared ≥ 0.7.

The critical methodological constraint is that probing must use circuit outputs only — the activations at the circuit's output positions, not the full model residual stream. Probing from the full residual stream tests whether the model as a whole encodes the information, not whether the circuit's outputs carry it. A circuit that computes the correct answer internally but passes it downstream through a non-circuit pathway fails observability: the measurement procedure cannot see the circuit's contribution through the circuit's own outputs.

## Minimum reporting rule

- Gramian eigenvalue spectrum (all eigenvalues, not just the top few).
- Effective rank of the Gramian (number of eigenvalues above a noise threshold).
- Rank as a fraction of circuit state dimension.
- Linear probe R-squared per circuit component, trained on circuit outputs only.
- Comparison: same probe trained on random (non-circuit) component outputs.
- If probing was done from the full residual stream rather than circuit outputs: criterion is unsatisfied regardless of probe accuracy.
