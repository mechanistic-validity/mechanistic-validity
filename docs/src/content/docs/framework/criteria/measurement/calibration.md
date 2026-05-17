---
title: "Calibration"
validity_type: "Measurement"
criterion_id: "M5"
---

# Criterion M5 — Calibration

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | Raw scores are interpretable relative to known reference points; a number without context is not a result |
| Evidence family | Measurement |
| Minimum reporting | Comparison to ≥1 published baseline on the same task and model; the score in the context of the published range |
| Common failure mode | Reporting scores without calibration context; treating an absolute number as self-interpreting |

## What this criterion requires

Calibration transforms a raw score into an interpretable claim. Every reported score must be placed in the context of at least one of:

1. **A published baseline on the same task and model.**
2. **The SOTA range for the metric.**
3. **A within-project comparison.**

## The calibration table for this project

| Task | Metric | Published baseline | Source |
|---|---|---|---|
| IOI | Logit diff faithfulness | 87% recovery | Wang et al. 2022 |
| IOI | Circuit CMD (lower=better) | UGS: 0.035; EAP(CF): 0.214; random: ~0.75 | MIB benchmark |
| Greater-Than | Prob diff recovery | 89.5% | Hanna et al. 2023 |
| SVA | Logit diff faithfulness | 93% | Lazo et al. 2025 |
| SVA | DAS-IIA (transcoder/CLT) | 0.40–0.60 | Mueller et al. MIB; transcoder papers |
| Gendered pronoun | Logit diff faithfulness | ≥ full model | Mathwin 2023 |
| BLiMP SVA | Behavioral accuracy | 95–97% | Warstadt et al. 2020 |
| BLiMP anaphor_gender | Behavioral accuracy | 99% | Warstadt et al. 2020 |

Every result in the project should include a calibration sentence: *"This score of X is [above/within/below] the published range of Y–Z for [task] in [model] ([source])."*

If no published baseline exists for the task/model combination, state this explicitly and propose the relevant comparison.

## The SVA IIA = 0.48 finding contextualized

> "The DAS-IIA score of 0.48 at L8.MLP for SVA in GPT-2 Small is within the published transcoder baseline range of 0.40–0.60 (Mueller et al. MIB; Lazo et al. 2025), making it competitive with SOTA for this task. Subject to baseline separation confirmation (M3), this constitutes a calibrated, competitive result."

This is what a calibrated result statement looks like.

## Minimum reporting rule

Every IIA, faithfulness, or classification score must include a calibration sentence referencing a specific published baseline. If no baseline exists, state this explicitly.
