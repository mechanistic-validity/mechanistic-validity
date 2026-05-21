---
title: "Scale Invariance"
validity_type: "Construct"
criterion_id: "C8"
---

# Criterion C8 — Scale Invariance

| | |
|---|---|
| Validity type | Construct |
| Pass condition | RG beta function |beta| < 0.1 for >= 2 directions; fraction relevant directions <= 0.3 |
| Evidence family | Structural |
| Minimum reporting | Beta function values per direction, fraction relevant/irrelevant, comparison across >= 2 scales |
| Common failure mode | Testing only one coarse-graining scale |
| Lens | Dynamical Systems |

## What this criterion requires

Scale invariance asks whether the circuit's structure is preserved under coarse-graining — whether the same computational story holds when you describe it at the level of individual neurons, attention heads, layers, or multi-layer blocks. A circuit claim that depends critically on a particular granularity is fragile: it may reflect an artifact of the resolution at which analysis was performed rather than a genuine structural feature.

The criterion is satisfied when:

1. **The renormalization group (RG) beta function has small magnitude in most directions.** Specifically, |beta| < 0.1 for at least 2 coupling directions, meaning these aspects of circuit structure do not change under coarse-graining. Directions with |beta| >= 0.1 are "relevant" — they change with scale and represent scale-dependent features.
2. **The fraction of relevant directions is small.** At most 30% of coupling directions should be relevant. A circuit where most directions are scale-dependent does not have stable structure across scales.
3. **The analysis is performed at >= 2 coarse-graining scales.** Testing a single transition (e.g., neurons to heads) is insufficient. The beta function should be consistent across at least two transitions (e.g., neurons to heads, and heads to layers).

This criterion does not establish that the finest-grained description of the circuit is correct — it only establishes that structure is preserved across scales. It also does not establish cross-model generalization; a circuit may be scale-invariant within one model but not present in another.

## Minimum reporting rule

- Beta function value for each coupling direction at each scale transition.
- Classification of each direction as relevant (|beta| >= 0.1) or irrelevant.
- Fraction of relevant directions at each scale.
- At least 2 coarse-graining scales tested, with the scale transitions named explicitly (e.g., neuron to head, head to layer).
- If the fraction of relevant directions exceeds 0.3, discuss which directions are scale-dependent and why.
