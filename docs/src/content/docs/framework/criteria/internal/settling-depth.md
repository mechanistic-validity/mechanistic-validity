---
title: "Settling Depth"
validity_type: "Internal"
criterion_id: "I14"
---

# Criterion I14 — Settling Depth

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Residual stream distance returns to within 20% of pre-perturbation within 3 layers; if never settles (>50% through remaining layers), component lacks compensation |
| Evidence family | Causal |
| Minimum reporting | Perturbation method and magnitude, recovery curve (distance at each subsequent layer), settling depth, steady-state error, comparison across perturbation magnitudes |
| Common failure mode | Perturbation magnitude not controlled; large perturbations can exceed the linear regime |
| Lens | Control Theory |

## What this criterion requires

Settling depth measures the temporal dynamics of recovery after a perturbation: how many layers does it take for the residual stream to return to its pre-perturbation trajectory? This characterizes whether the model compensates for circuit disruption (fast settling) or whether the perturbation propagates indefinitely (no settling), revealing the circuit's role in the model's error-correction landscape.

Satisfied when:

1. **Recovery curve is measured.** After perturbing a component at layer L, the L2 distance between the perturbed and clean residual streams is measured at each subsequent layer.
2. **Settling depth is bounded.** The distance returns to within 20% of the pre-perturbation baseline within 3 layers, indicating the model compensates quickly.
3. **Non-settling is flagged.** If the perturbation persists through more than 50% of the remaining layers, the component lacks downstream compensation, suggesting it occupies a critical, uncompensated role.

Settling depth does not establish why recovery occurs. Fast settling could reflect backup mechanisms, redundancy, or simple noise absorption. Slow settling could reflect genuine computational criticality or an overpowered perturbation. The perturbation magnitude must be controlled to stay within a regime where the model's response is informative.

## Distinction from I1 — Necessity

I1 measures the final behavioral effect of removing a component. I14 measures the layer-by-layer dynamics of how the model responds to a perturbation. A component can be necessary (I1 pass, large final effect) yet settle quickly (I14, fast recovery) if the model compensates internally but imperfectly.

## Minimum reporting rule

- Perturbation method (zero, noise, resample) and magnitude.
- Recovery curve: residual stream distance at each layer after perturbation.
- Settling depth (number of layers to reach 20% threshold).
- Steady-state error (asymptotic distance if it does not return to baseline).
- Comparison across at least two perturbation magnitudes to check linearity.
