---
title: "Spectral Identifiability"
validity_type: "Construct"
criterion_id: "C6"
---

# Criterion C6 — Spectral Identifiability

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Koopman reconstruction error ≤ 0.15 with k ≤ 5 modes |
| Evidence family | Structural |
| Minimum reporting | Singular value spectrum, reconstruction error vs number of modes, eigenvalue magnitudes, comparison to random baseline |
| Common failure mode | Reporting reconstruction error without comparing to random component sets |
| Lens | Dynamical Systems |

## What this criterion requires

Spectral identifiability asks whether the circuit's layer-to-layer dynamics are low-rank — whether a small number of dynamical modes capture most of the variance in how information propagates through the circuit. A circuit that requires many modes to reconstruct is not a coherent dynamical unit; it is a collection of loosely related components whose joint behavior resists compact description.

The criterion is satisfied when:

1. **The Koopman operator for the circuit's layer-to-layer map admits a low-rank approximation.** Specifically, k ≤ 5 modes must achieve reconstruction error ≤ 0.15 (normalized Frobenius norm of the residual).
2. **The reconstruction error is substantially lower than what random component sets achieve.** A random subset of the same size from the same layers should require more modes to reach the same error, or should not reach it at all with k ≤ 5.
3. **The singular value spectrum shows a clear gap.** A gradual decay without a gap indicates the circuit's dynamics are not well-separated from noise.

This criterion does not establish that the identified modes correspond to the claimed computation (that is C7), nor that the circuit components are causally necessary (I1). It establishes only that the circuit has identifiable low-dimensional dynamics — a prerequisite for any dynamical-systems-level interpretation.

## Minimum reporting rule

- Singular value spectrum of the Koopman operator (all values, not just top-k).
- Reconstruction error as a function of number of modes retained (1 through at least 10).
- Magnitudes of the top-k eigenvalues.
- Same analysis on ≥1 random component set of matched size and layer distribution.
- If no clear spectral gap exists, report this explicitly.
