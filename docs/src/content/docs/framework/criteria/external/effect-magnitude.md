---
title: "Effect Magnitude"
validity_type: "External"
criterion_id: "E4"
---

# Criterion E4 — Effect Magnitude

| | |
|---|---|
| Validity type | External |
| Pass condition | The absolute effect is large enough to support the computational story being told |
| Evidence family | Behavioral |
| Minimum reporting | Absolute recovery fraction (not just statistical significance); comparison to published baselines |
| Common failure mode | Reporting statistical significance without absolute magnitude |

## What this criterion requires

Effect magnitude distinguishes between a component that is *causally necessary* (absence degrades behavior by any nonzero amount) and one that *implements the computation* (absence degrades behavior by a large fraction of the total effect).

A head accounting for 3% of the logit difference satisfies necessity in the strict sense but is not doing most of the work. Calling it "a core circuit member" when it contributes 3% while another contributes 60% is effect-magnitude overclaiming.

**Recovery fraction:**

```
recovery_fraction = |ablation_delta| / |full_model_logit_diff|
```

Threshold: ≥ 0.10 (≥10% contribution) for inclusion in a primary mechanism claim. Components with < 0.05 should be classified as minor contributors.

## Effect magnitude vs. faithfulness

- **Faithfulness (I2):** How well the *circuit as a whole* recovers the behavior.
- **Effect magnitude (E4):** How much *individual components* contribute to the total.

A circuit with high faithfulness (87%) may contain components with low individual effect magnitude (3%) where the high faithfulness comes from many small contributors. Effect magnitude analysis identifies which components do most of the work.

## Published calibration points

| Task | Published faithfulness | Source |
|---|---|---|
| IOI | 87% logit-diff recovery | Wang et al. 2022 |
| Greater-Than | 89.5% prob-diff recovery | Hanna et al. 2023 |
| SVA | 93% logit-diff recovery | Lazo et al. 2025 |

A circuit at 40% recovery is not yet competitive with these baselines.

## Minimum reporting rule

- Absolute recovery fraction for each circuit member.
- Statistical significance *in addition to*, not instead of, absolute magnitude.
- Comparison to published baselines using the table above.
