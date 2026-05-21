---
title: "Controllability"
validity_type: "Internal"
criterion_id: "I13"
---

# Criterion I13 — Controllability

| | |
|---|---|
| Validity type | Internal |
| Pass condition | Controllability Gramian rank at least 0.8 x dim(circuit state); condition number < 100 |
| Evidence family | Structural |
| Minimum reporting | Gramian eigenvalue spectrum, rank, condition number, comparison to random component set |
| Common failure mode | Computing Gramian rank without reporting condition number (near-singular Gramians have nominal high rank but poor practical controllability) |
| Lens | Control Theory |

## What this criterion requires

Controllability tests whether the circuit's state can be steered to an arbitrary target by intervening on its inputs. Borrowing from linear control theory, the controllability Gramian characterizes the reachable state space: a full-rank Gramian means any target activation pattern is achievable through input perturbations. This validates that the circuit is a controllable subsystem, not a fixed-function pipeline.

Satisfied when:

1. **Gramian rank is high.** The controllability Gramian has rank at least 0.8 times the dimensionality of the circuit's state space, confirming that most target states are reachable.
2. **Condition number is bounded.** The Gramian's condition number is below 100, confirming that reachable states are practically achievable (not requiring astronomically large inputs to reach some directions).
3. **Comparison to random baseline.** The circuit's Gramian rank and condition number are compared to those of a random component set of equal size, confirming the circuit has better controllability than arbitrary components.

Controllability does not establish that the circuit is necessary for the behavior, nor that steering is used during normal model operation. It establishes that the circuit is a well-conditioned subsystem amenable to targeted intervention.

## Distinction from I2 — Sufficiency

I2 tests whether the circuit alone reproduces the target behavior. I13 tests whether the circuit can be commanded to produce arbitrary target states, a stronger structural property. A circuit can be sufficient (I2 pass) for one behavior yet poorly controllable (I13 fail) if it is locked into a narrow range of states.

## Minimum reporting rule

- Gramian eigenvalue spectrum (top-k eigenvalues and decay profile).
- Gramian rank and fraction of circuit state dimensionality.
- Condition number.
- Comparison to random component set of equal size.
- Method used to linearize the circuit (if applicable).
