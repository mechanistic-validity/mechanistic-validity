---
title: "Intervention Reach"
validity_type: "External"
criterion_id: "E1"
---

# Criterion E1 — Intervention Reach

| | |
|---|---|
| Validity type | External |
| Pass condition | The result has been reproduced under at least two genuinely distinct intervention families (distinct do-operators, not ablation variants), and they agree |
| Evidence family | Causal |
| Minimum reporting | Each intervention family used and the operator behind it; the effect estimate under each; whether the families agree in direction and magnitude |
| Common failure mode | Running several ablation variants that share one underlying do-operator and reporting the agreement as convergence; reporting a single ablation number at origin |

## What this criterion requires

Ablation and activation patching are formally different do-operators. Ablation performs do(h := 0) or do(h := E[h]) — it removes a component. Patching performs do(h := h') where h' comes from a counterfactual input — it inserts a specific value. These correspond to different interventional distributions and can yield different causal conclusions, which is why mean ablation and resample ablation produce different faithfulness numbers for the same circuit.

Satisfied when:

1. **At least two intervention families are applied.** The families must differ in kind — removal versus insertion, activation edit versus weight edit — not merely in ablation value.
2. **The families agree.** They return the same sign and a comparable magnitude for the effect.
3. **Disagreement is reported, not averaged.** Where families disagree, the disagreement is the result and the criterion is Inconclusive.

## Minimum reporting rule

- Name every intervention family used and the do-operator each implements.
- Report the effect estimate under each family, on the same metric and prompt set.
- State whether the families agree; if they disagree, report the spread rather than a single number.
- If only one intervention family was used: E1 is unsatisfied — every agreement is the method with itself.

## Why this is an external validity criterion

Intervention reach is required for Mechanistically Supported, alongside sufficiency (I2) and specificity (I4). A result that holds under one intervention family has not been separated from a property of that family: the same IOI faithfulness quantity spans below 0% to over 100% across methodological choices, so a single number carries no information about which of them the mechanism survives.
