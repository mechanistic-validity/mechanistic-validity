---
title: "F07 — Measurement Invariance"
description: "Tests whether an evaluation metric means the same thing across different groups or conditions."
---

# F07 — Measurement Invariance

This framework asks: **Does our faithfulness metric carry the same meaning when applied to different tasks, model sizes, or prompt distributions?**

A score of 0.85 faithfulness on IOI and 0.85 on SVA should reflect comparable circuit quality. But if the metric's relationship to the underlying construct shifts between contexts — for example, if IOI prompts have less variance so 0.85 is easier to achieve — then cross-context comparisons are invalid. Measurement invariance tests whether the metric functions equivalently across groups.

Without invariance, claims like "the IOI circuit is more faithful than the SVA circuit" are meaningless because the ruler changes between measurements. This is the measurement-theoretic concept of factorial invariance applied to circuit evaluation.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Meredith, "Measurement invariance, factor analysis and factorial invariance"](https://doi.org/10.1007/BF02294825) | 1993 | Formal hierarchy of invariance levels |
| [Vandenberg & Lance, "A review of measurement invariance testing practices"](https://doi.org/10.1177/1094428106296543) | 2000 | Practical guidelines for testing invariance |
| [Chen, "Sensitivity of goodness of fit indexes to lack of measurement invariance"](https://doi.org/10.1080/10705510701301834) | 2007 | Delta-CFI criteria for invariance testing |
| [Putnick & Bornstein, "Measurement invariance conventions and reporting"](https://doi.org/10.1016/j.dr.2016.06.004) | 2016 | Modern reporting standards |

## Core concept

Measurement invariance is tested at progressive levels:

1. **Configural invariance**: The same items load on the same construct across groups (same circuit structure applies).
2. **Metric invariance**: The loadings are equal — a one-unit increase in the construct produces the same score change across groups.
3. **Scalar invariance**: Intercepts are equal — the same construct level produces the same observed score.

Formally, for groups \( g \in \{1, 2\} \), the measurement model is:

\[
x_g = \boldsymbol{\tau}_g + \boldsymbol{\Lambda}_g \eta_g + \boldsymbol{\epsilon}_g
\]

Metric invariance requires \( \boldsymbol{\Lambda}_1 = \boldsymbol{\Lambda}_2 \); scalar invariance additionally requires \( \boldsymbol{\tau}_1 = \boldsymbol{\tau}_2 \). In practice, we test whether the metric's rank-ordering of circuits is preserved across conditions.

## Metrics under F07

### Measurement Invariance Test (`13_measurement_invariance.py`)

Evaluates all circuits on multiple tasks and prompt distributions, then tests whether the metric's ranking of circuits is preserved (rank invariance) and whether absolute scores are comparable (scalar invariance) via DIF (Differential Item Functioning) analysis.

**What it establishes:** That cross-task and cross-distribution comparisons of faithfulness scores are valid.
**What it does not establish:** That the metric is measuring faithfulness specifically — only that it measures the *same thing* across contexts.

**Usage:**
```
uv run python 13_measurement_invariance.py --tasks ioi sva greater_than --distributions pile ioi_prompts
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Rank correlation > 0.9 across groups | Strong invariance — cross-group comparisons are valid |
| Rank correlation 0.7–0.9 | Partial invariance — ordinal comparisons are valid but absolute differences are not |
| Rank correlation < 0.7 | Invariance violated — the metric means different things in different contexts |
| DIF flagged on > 20% of items | Substantial non-invariance — investigate which prompts function differently |

