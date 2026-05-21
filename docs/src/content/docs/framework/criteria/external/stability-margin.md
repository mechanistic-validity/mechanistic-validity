---
title: "Stability Margin"
validity_type: "External"
criterion_id: "E8"
---

# Criterion E8 — Stability Margin

| | |
|---|---|
| Validity type | External |
| Pass condition | Gain margin ≥ 2.0 (can double perturbation before target metric drops below 50%) |
| Evidence family | Causal |
| Minimum reporting | Perturbation magnitudes tested, target metric at each, gain margin, perturbation-response curve plot, off-target metric at each magnitude |
| Common failure mode | Testing only at 1x and 2x; need ≥ 5 perturbation magnitudes for smooth curve |
| Lens | Control Theory |

## What this criterion requires

Stability margin quantifies how much perturbation a circuit absorbs before its behavior qualitatively changes. In control theory, gain margin is the factor by which loop gain can be multiplied before a stable system becomes unstable. For circuits, the gain margin is the factor by which intervention magnitude can be scaled (relative to the standard intervention) before the target metric drops below 50% of its unperturbed value.

The test sweeps perturbation magnitude across ≥ 5 values (e.g., 0.5x, 1x, 1.5x, 2x, 3x, 5x of the standard intervention strength) and records both the target metric and at least one off-target metric at each level. The gain margin is read off the perturbation-response curve as the x-intercept of the 50% threshold. A gain margin of 2.0 means the perturbation can be doubled before qualitative failure — the circuit has a comfortable operating envelope rather than a knife-edge equilibrium.

This criterion is distinct from E5 (Robustness). E5 asks a binary question: does the claim survive variation? E8 gives a continuous answer: how much variation does it survive, and what does the degradation curve look like? A circuit can pass E5 (the claim holds under paraphrase) while having a gain margin of 1.1 (any slight increase in perturbation magnitude causes failure). E8 exposes this fragility.

## Minimum reporting rule

- ≥ 5 perturbation magnitudes tested, spanning at least a 4x range.
- Target metric value at each magnitude.
- Off-target metric value at each magnitude.
- Gain margin (the perturbation multiplier at which target metric crosses 50%).
- Perturbation-response curve plot showing both target and off-target metrics.
- If gain margin < 2.0: criterion is unsatisfied; report the actual margin.
