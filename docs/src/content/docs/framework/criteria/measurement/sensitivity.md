---
title: "Sensitivity"
validity_type: "Measurement"
criterion_id: "M4"
---

# Criterion M4 — Sensitivity

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | Instrument detects real circuits at acceptable hit rates (AUROC ≥ 0.85) without excessive false positives (AUPRC above random baseline) |
| Evidence family | Measurement |
| Minimum reporting | AUROC for circuit head membership classification; AUPRC with random baseline comparison |
| Common failure mode | Reporting only positive results; never characterizing the false positive rate |

## What this criterion requires

Sensitivity is the detection-theory characterization of the measurement instrument.

Satisfied when:

1. **AUROC ≥ 0.85** for circuit head membership classification. AUROC = 0.5 means chance performance; 1.0 is perfect.
2. **AUPRC is above the random baseline.** For a circuit of k heads out of n total, random baseline AUPRC = k/n. For GPT-2 Small IOI (26 heads out of 144 total): random baseline = 0.18. The instrument's AUPRC should substantially exceed this.

## The Test 16 AP = 1.0 result

The project's circuit scan results report Test 16 AP = 1.0 — the weight classifier achieves perfect average precision on the held-out test set. This is the strongest possible sensitivity result and is a primary finding that should be highlighted in any write-up. This means the classifier can perfectly separate circuit members from non-members on the held-out set (at this circuit size / task combination).

## d-prime

A more complete characterization uses d-prime from signal detection theory:

```
d' = Z(hit_rate) - Z(false_alarm_rate)
```

where Z is the inverse normal CDF. d' ≥ 2.0 indicates good discrimination. Report d' alongside AUROC when circuit membership is known from published circuits.

## Sensitivity for small circuits

AUPRC (not AUROC) is preferred for small circuits because AUROC is insensitive to false positive rates when the positive class is rare. For a circuit of 5 heads (random baseline AUPRC = 0.035), AUPRC = 0.40 represents excellent sensitivity. Report both metrics.

## Minimum reporting rule

- AUROC and AUPRC with random baseline.
- d-prime if circuit membership is known.
- The Test 16 AP result if computed.
- If sensitivity was not characterized: flag as partially satisfied.
