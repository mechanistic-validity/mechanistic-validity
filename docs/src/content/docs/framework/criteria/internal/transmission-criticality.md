---
title: "Transmission Criticality"
validity_type: "Internal"
criterion_id: "I12"
---

# Criterion I12 — Transmission Criticality

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Circuit R0 > 1.0; removing hub components drops R0 below 1.0 |
| Evidence family | Information-theoretic |
| Minimum reporting | Circuit R0, R0 after hub removal, hub component identities, comparison to R0 of random component sets |
| Common failure mode | Computing R0 without removing hubs to test criticality |
| Lens | Information Theory |

## What this criterion requires

Transmission criticality adapts the basic reproduction number (R0) from SIR epidemiology to information propagation in circuits. R0 measures whether a perturbation injected at one component amplifies through the circuit (R0 > 1, supercritical) or decays (R0 < 1, subcritical). A circuit with R0 > 1 self-sustains information propagation; hub components whose removal drops R0 below 1 are transmission-critical.

Satisfied when:

1. **Circuit R0 exceeds 1.0.** The circuit as a whole is supercritical: information injected at any component propagates to downstream components with amplification.
2. **Hub removal drops R0 below 1.0.** Removing the identified hub components (those with highest transmission rates) collapses the circuit to subcritical. This confirms the hubs are not merely high-activation but structurally critical for information propagation.
3. **Comparison to random sets.** R0 computed over random component sets of equal size serves as a baseline. The circuit's R0 should exceed the random baseline.

Transmission criticality does not establish what information is being propagated or whether the circuit is sufficient for the behavior. It establishes which components are structurally critical for maintaining information flow through the circuit.

## Distinction from I1 — Necessity

I1 measures behavioral degradation when a component is removed. I12 measures whether information propagation through the circuit self-sustains, and identifies components whose removal collapses that propagation. A component can be necessary (I1 pass) without being transmission-critical (I12 fail) if its contribution is local rather than propagating.

## Minimum reporting rule

- Circuit R0 value.
- R0 after removal of each identified hub component.
- Identity of hub components and basis for hub identification.
- R0 of random component sets of equal size (baseline).
- Method used to estimate transmission rates between components.
