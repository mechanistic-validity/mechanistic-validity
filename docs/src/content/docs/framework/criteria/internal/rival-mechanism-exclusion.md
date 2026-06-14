---
title: "Rival Mechanism Exclusion"
validity_type: "Internal"
criterion_id: "I6"
---

# Criterion I6 — Rival Mechanism Exclusion

| | |
|---|---|
| Validity type | Internal |
| Pass condition | No alternative component set achieves comparable faithfulness under the same intervention regime, or rivals are explicitly reported and the claim scoped accordingly |
| Evidence family | Causal |
| Minimum reporting | At least one rival circuit tested; faithfulness gap between target and best rival; claim language reflecting uniqueness or non-uniqueness |
| Common failure mode | Reporting a single discovered circuit as "the circuit" without testing whether alternative component sets achieve comparable faithfulness |

## What this criterion requires

Causal intervention establishes that a component set is *a* sufficient mechanism — it does not establish that the component set is *the* mechanism. Rival mechanism exclusion requires testing whether alternative decompositions achieve comparable sufficiency, and scoping the claim accordingly.

Satisfied when either:

1. **No rival achieves comparable faithfulness.** At least one alternative component set (of similar size) has been tested and achieves less than 80% of the target circuit's faithfulness under the same ablation regime. The gap is reported explicitly.
2. **Rivals exist and are declared.** Alternative component sets achieve comparable faithfulness, and the claim is explicitly scoped to "a sufficient mechanism" rather than "the mechanism." The rival circuits are named and their faithfulness reported.

## Why this is separate from V4 (Alternative Exclusion)

[V4 Alternative Exclusion](/framework/criteria/interpretive/alternative-exclusion/) addresses competing *interpretations* of the same evidence — different narrative accounts of what the circuit does. Rival mechanism exclusion addresses competing *circuits* — different component sets that produce the same behavior with comparable faithfulness. The first is an interpretive question (what does the evidence mean?). The second is an empirical question (is this the only component set that works?).

A claim can pass V4 (no better interpretation of the evidence exists) while failing I6 (a different set of components produces the same behavior equally well). This is the "a circuit vs. the circuit" distinction.

## The Méloux et al. problem

[Méloux et al. (2024)](https://arxiv.org/abs/2407.07498) demonstrated that multiple distinct circuits can achieve comparable faithfulness on the same task in the same model. The IOI circuit identified by Wang et al. is *a* sufficient mechanism for IOI, but alternative head sets also achieve high faithfulness. This finding does not invalidate the original circuit — it invalidates the implicit uniqueness claim.

Without I6, a researcher who finds a faithful circuit has no obligation to test alternatives. The circuit is reported as "the IOI circuit" (definite article implying uniqueness) when the evidence only supports "an IOI circuit" (indefinite article implying sufficiency without uniqueness).

## How to test for rivals

1. **Permutation test:** Randomly sample component sets of the same size as the proposed circuit. Measure faithfulness for each. If the proposed circuit is in the top 1% of random samples, it is meaningfully better than chance — but other high-performing sets may exist.
2. **Greedy re-discovery:** Run the circuit discovery procedure (ACDC, EAP, attribution patching) with different random seeds or hyperparameters. If different runs return substantially different component sets with comparable faithfulness, the decomposition is non-unique.
3. **Complementary discovery:** Run a *different* discovery method on the same task. If method A and method B return different component sets (low Jaccard overlap) but both achieve high faithfulness, neither is "the" circuit.

## Relation to other criteria

- **[C4 Minimality](/framework/criteria/construct/minimality/):** A minimal circuit can still be non-unique — multiple minimal circuits may exist (backup mechanisms, distributed computation). Minimality ensures no redundant members within a circuit; I6 ensures no rival circuits across decompositions.
- **[I5 Confound Control](/framework/criteria/internal/confound-control/):** Confound control asks whether the effect is due to the nominated component or collateral disruption. I6 asks whether *different* components could produce the same effect without collateral disruption.
- **[I4 Consistency](/framework/criteria/internal/consistency/):** Consistency ensures the finding replicates across seeds and methods. I6 goes further: even if the finding replicates perfectly, is the discovered circuit the only one that would?

## Minimum reporting rule

- Number of rival circuits tested and how they were generated.
- Faithfulness of target circuit and best rival (same metric, same ablation method).
- Whether the claim uses "the" or "a" — definite vs. indefinite framing.
- If no rivals were tested: flag I6 as open and restrict claim to "a sufficient mechanism."
