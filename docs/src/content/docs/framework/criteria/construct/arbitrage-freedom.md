---
title: "Arbitrage Freedom"
validity_type: "Construct"
criterion_id: "C12"
---

# Criterion C12 — Arbitrage Freedom

| | |
|---|---|
| Validity type | Construct |
| Pass condition | No alternative subset of equal/smaller size achieves >= 90% of circuit performance; top-5 substitutes each < 50% |
| Evidence family | Structural |
| Minimum reporting | Best substitute set performance, top-5 substitute performances, search method (greedy/random/exhaustive), number of candidates tested |
| Common failure mode | Testing only random substitutes, not greedy/best-case substitutes |
| Lens | Economics |

## What this criterion requires

Arbitrage freedom asks whether there exists a cheaper way to achieve the same computation by rerouting through non-circuit components. If an alternative subset of components — not overlapping with the proposed circuit — can replicate the circuit's performance at equal or smaller size, the circuit claim is weakened: the claimed components are not uniquely important, and the computational structure may be distributed more broadly than the circuit claim implies.

The criterion is satisfied when:

1. **No alternative subset of equal or smaller size achieves >= 90% of the circuit's performance.** The "performance" here is measured using the same metric used to evaluate the circuit (e.g., faithfulness, IIA). The search over alternative subsets must be non-trivial — not just random sampling.
2. **The top-5 individual substitute components each achieve < 50% of circuit performance.** No single non-circuit component should be a near-substitute for the full circuit. If one component outside the circuit achieves 80% of the circuit's performance, the circuit claim overstates the importance of its members.
3. **The search method is adequate.** Random sampling alone is insufficient because combinatorial spaces are large and random subsets are unlikely to find good substitutes. At minimum, a greedy search (iteratively adding the best remaining component) should be performed. Exhaustive search is ideal for small circuits but computationally prohibitive for large ones.

This criterion does not establish that the circuit is minimal in the deletion sense (C4) — minimality asks whether each member is individually necessary, while arbitrage freedom asks whether the entire set is collectively irreplaceable. A circuit can be minimal (every member is needed) but not arbitrage-free (a completely different set of components works equally well).

## Minimum reporting rule

- Performance of the best alternative subset found, and the subset's composition.
- Performance of the top-5 individual substitute components (non-circuit components ranked by how well they approximate circuit function).
- Search method used: random, greedy, beam search, or exhaustive.
- Number of candidate subsets evaluated.
- Size of the proposed circuit and the best substitute set.
- If the best substitute achieves >= 90%, report this as a failure and describe the substitute.
