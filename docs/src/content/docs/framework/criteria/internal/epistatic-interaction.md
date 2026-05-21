---
title: "Epistatic Interaction"
validity_type: "Internal"
criterion_id: "I6"
---

# Criterion I6 — Epistatic Interaction

| | |
|---|---|
| Validity type | Internal |
| Pass condition | At least one component pair with |synergy| > 2x individual effects; Shapley interaction p < 0.01 |
| Evidence family | Causal |
| Minimum reporting | Synergy scores per pair, Shapley interaction indices, p-values, comparison to additive model, number of pairs tested |
| Common failure mode | Testing only pairs without higher-order interactions |
| Lens | Genetics |

## What this criterion requires

Epistatic interaction tests whether components in the proposed circuit functionally couple: their joint effect differs from the sum of their individual effects. This is the mechanistic-interpretability analogue of epistasis in genetics, where the phenotypic effect of one gene depends on the state of another.

Satisfied when:

1. **Pairwise synergy exceeds additivity.** For at least one component pair (A, B), the joint ablation effect exceeds 2x the sum of individual ablation effects: |effect(A,B) - effect(A) - effect(B)| > 2 * max(|effect(A)|, |effect(B)|).
2. **Shapley interaction index is significant.** The Shapley interaction index for that pair has p < 0.01 against a null distribution from random component pairs.
3. **Higher-order interactions are probed.** At minimum, triplet interactions are tested for the top synergistic pairs.

Epistatic interaction does not establish the direction of interaction (whether A modulates B or B modulates A), nor whether the interaction is necessary for the computation. It establishes that the circuit's components are functionally coupled rather than independently contributing.

## Distinction from I3 — Specificity

I3 tests whether a component's effect is selective for a particular task. I6 tests whether two components interact with each other. A component can be highly task-specific (I3 pass) yet act independently of all other circuit members (I6 fail). Conversely, two components can interact strongly (I6 pass) on a non-specific behavior (I3 fail).

## Minimum reporting rule

- Synergy score for each tested pair, with sign (superadditive vs. subadditive).
- Shapley interaction indices and associated p-values.
- Comparison to additive baseline model predictions.
- Number of pairs tested and multiple-comparison correction method.
- Whether higher-order (triplet+) interactions were tested, and results if so.
