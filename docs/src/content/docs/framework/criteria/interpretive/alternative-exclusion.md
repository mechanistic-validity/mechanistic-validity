---
title: "Alternative Exclusion"
validity_type: "Interpretive"
criterion_id: "V4"
---

# Criterion V4 — Alternative Exclusion

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | Competing mechanism descriptions have been considered and addressed |
| Evidence family | Depends on which alternative is being excluded (usually causal or representational) |
| Minimum reporting | List of considered alternatives; evidence or argument addressing each; remaining ambiguities noted |
| Common failure mode | Reporting the favored interpretation without mentioning the alternatives it competes with |

## What this criterion requires

Alternative exclusion requires that the researcher has identified the main competing interpretations of the evidence and has either (a) provided evidence against them or (b) acknowledged that the ambiguity remains unresolved.

For any MI circuit claim, the standard alternatives are:

1. **The hub alternative:** The nominated component is a general-purpose hub, not specifically involved in the claimed computation. Addressed by specificity (I3) and task specificity (C3).

2. **The correlation alternative:** The component correlates with the behavior but does not cause it. Addressed by sufficiency (I2) and confound control (I5).

3. **The architectural prior alternative:** The finding reflects the model's architecture rather than its learned representations. Addressed by baseline separation (M3: untrained-model baseline).

4. **The prompt-artifact alternative:** The finding reflects regularities in the specific prompt distribution used for discovery, not a general mechanism. Addressed by robustness (E5).

5. **The rival-mechanism alternative:** A different set of components achieves comparable faithfulness on the same task. The evidence establishes *a* sufficient mechanism, not *the* mechanism. Addressed by rival mechanism exclusion ([I6](/framework/criteria/internal/rival-mechanism-exclusion/)). This alternative is distinct from the others because it does not challenge the evidence — it challenges the uniqueness of the conclusion.

6. **The method-disagreement alternative:** Two discovery methods nominate different component sets for the same task. The alternatives are: (a) method A is wrong; (b) method B is wrong; (c) both measure different real properties (complementary evidence). Each must be considered, with evidence or argument — typically by testing faithfulness of each component set independently.

## Minimum reporting rule

- List the competing interpretations considered.
- For each: either cite the criterion and result that addresses it, or note that it remains unresolved.
- If an interpretation remains unresolved, state this explicitly in the verdict.
