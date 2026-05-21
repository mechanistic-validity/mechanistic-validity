---
title: "Symmetry Equivariance"
validity_type: "Construct"
criterion_id: "C13"
---

# Criterion C13 — Symmetry Equivariance

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Equivariance error \|\|W rho(g) - rho(g) W\|\|_F / \|\|W\|\|_F < 0.2 for all generators of the symmetry group |
| Evidence family | Structural |
| Minimum reporting | Symmetry group and generators, equivariance error per generator per weight matrix, comparison to random weight matrices |
| Common failure mode | Not identifying the correct symmetry group from the computational claim |
| Lens | Geometry |

## What this criterion requires

Symmetry equivariance asks whether the circuit's weight matrices respect the symmetries implied by the computational claim. If a circuit is claimed to perform modular arithmetic, its weights should approximately commute with the cyclic group's representation. If it is claimed to perform a permutation-invariant computation, its weights should be approximately equivariant under the symmetric group. A circuit whose weights violate the symmetries of its claimed computation is either performing a different computation or performing the claimed computation through a less structured mechanism than implied.

The criterion is satisfied when:

1. **The symmetry group is correctly identified from the computational claim.** This is a non-trivial step. The claim "this circuit performs IOI" implies invariance under name permutation (swapping IO and S names should produce correspondingly swapped outputs). The claim "this circuit detects even/odd" implies Z/2Z equivariance. The group and its generators must be explicitly stated.
2. **The equivariance error is small for all generators.** For each generator g of the symmetry group and each weight matrix W in the circuit, the normalized Frobenius error ||W rho(g) - rho(g) W||_F / ||W||_F must be < 0.2. It suffices to check generators because equivariance under generators implies equivariance under the full group.
3. **The error is lower than for random weight matrices.** Random matrices of the same shape and spectral norm should show higher equivariance error. Without this comparison, low error may reflect properties of the matrix distribution (e.g., near-identity matrices are approximately equivariant under any group).

This criterion does not establish that the model uses the symmetry structure in its computation. A weight matrix can be equivariant by accident or as a side effect of training dynamics. It also does not establish functional correctness — equivariant weights are necessary but not sufficient for the claimed computation.

## Minimum reporting rule

- The symmetry group claimed, with explicit generators and their matrix representations rho(g).
- Equivariance error per generator per weight matrix in the circuit.
- Same computation on random weight matrices (matched shape and spectral norm) as baseline.
- If equivariance error exceeds 0.2 for any generator, identify which weight matrix and which symmetry is violated.
- Justification for why this symmetry group is the correct one for the computational claim.
