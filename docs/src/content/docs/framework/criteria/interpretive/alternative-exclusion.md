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

5. **The Jaccard-disagreement alternative (project-specific):** The weight-circuit and EAP-circuit nominate different components (Jaccard ≈ 0). The alternatives are: (a) weight classifier is wrong; (b) EAP is wrong; (c) both are measuring different real properties. Each must be considered, with evidence or argument.

## The Jaccard ≈ 0 case

For this project, alternative exclusion for the Jaccard ≈ 0 finding requires:

- **If weight classifier is wrong:** What methodological flaw would explain it finding circuit-like structural properties in components that EAP identifies as causally inactive? (Possible: the classifier was trained on templates that happen to share structural properties with non-circuit heads in this architecture.)
- **If EAP is wrong:** What methodological flaw would explain EAP missing the structurally plausible components? (Possible: EAP overweights high-activation layers; if weight-circuit components are in lower-activation layers, EAP may undercount them.)
- **If both are partially right:** This is the "complementary evidence" interpretation — weight classifier finds structurally consistent components that are not the primary causal route, while EAP finds the primary causal route. This interpretation requires its own empirical test (DAS-IIA on both sets of components to determine which has higher causal alignment).

## Minimum reporting rule

- List the competing interpretations considered.
- For each: either cite the criterion and result that addresses it, or note that it remains unresolved.
- If an interpretation remains unresolved, state this explicitly in the verdict.
