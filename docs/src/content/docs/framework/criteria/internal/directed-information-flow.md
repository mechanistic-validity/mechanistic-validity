---
title: "Directed Information Flow"
validity_type: "Internal"
criterion_id: "I11"
---

# Criterion I11 — Directed Information Flow

| | |
|---|---|
| Validity type | Internal |
| Pass condition | At least 50% of claimed edges significant after Bonferroni (p < 0.05); edge Jaccard with ground truth at least 0.3 |
| Evidence family | Information-theoretic |
| Minimum reporting | Number of raw edges, number after Bonferroni, edge Jaccard with claimed topology, top hub heads, F-statistics for significant edges |
| Common failure mode | Not correcting for multiple comparisons; reporting raw edges without Bonferroni |
| Lens | Information Theory |

## What this criterion requires

Directed information flow tests whether information actually transfers along the edges claimed by the circuit topology. Using Granger causality or transfer entropy between component activations across layers, this criterion checks that the proposed wiring diagram reflects real directional signal propagation rather than mere co-activation or structural adjacency.

Satisfied when:

1. **Edge significance survives correction.** At least 50% of the claimed circuit edges show significant directed information transfer (Granger F-test or transfer entropy permutation test) after Bonferroni correction at p < 0.05.
2. **Topology overlap is non-trivial.** The Jaccard similarity between the significant-edge set and the claimed circuit topology is at least 0.3. This ensures the information flow map aligns with the proposed circuit, not an unrelated subgraph.
3. **Hub structure is reported.** Components with disproportionately many significant incoming or outgoing edges are identified as hubs, enabling comparison with the circuit's claimed architecture.

Directed information flow does not establish that the flow is causally necessary for the behavior, nor does it identify what information is being transmitted. It establishes that the circuit's wiring diagram reflects real signal propagation.

## Distinction from I1 — Necessity

I1 tests whether removing a component hurts behavior. I11 tests whether information flows along the claimed path between components. A component can be necessary (I1 pass) without information flowing along the specific edges claimed (I11 fail) if the component contributes through an alternative pathway.

## Minimum reporting rule

- Total number of edges tested (raw).
- Number of edges significant after Bonferroni correction.
- Edge Jaccard with the claimed circuit topology.
- Hub components (top nodes by in-degree and out-degree in the significant-edge graph).
- F-statistics or transfer entropy values for significant edges.
- Multiple comparison correction method used.
