---
title: "Reliability"
validity_type: "Measurement"
criterion_id: "M1"
---

# Criterion M1 — Reliability

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | IIA and faithfulness scores are stable across prompt splits, random seeds, and model checkpoints |
| Evidence family | Measurement |
| Minimum reporting | Bootstrap CI on faithfulness (≥100 subsamples); test-retest Pearson r across ≥3 prompt splits; seed variance |
| Common failure mode | Reporting a single point estimate with no confidence interval or variance |

## What this criterion requires

Reliability is the test-retest stability of the measurement instrument. A reliable instrument produces the same result when re-applied under the same conditions.

Three dimensions:

**Prompt-sample reliability:** Bootstrap 100 random subsamples (80% of full size each); compute target metric on each; report the 95% CI. A 95% CI width ≤ 0.05 (on a 0–1 metric) is a reasonable threshold.

**Seed reliability:** Any step involving random initialization (DAS alignment search, bootstrap sampling, prompt shuffling) produces consistent results across ≥3 seeds. SD ≤ 0.02 is a reasonable threshold.

**Checkpoint reliability:** The claim should hold across multiple training checkpoints (not just the final checkpoint). A result that holds at only one checkpoint may be a training-stage artifact. Particularly important for LLC-based analyses.

The `c11bootstrap.py` script implements bootstrap stability for circuit faithfulness. The `reconstruction_metrics` function in `factor_analysis.py` provides stability for reconstruction quality metrics.

## Reliability vs. consistency (I4)

- **Reliability (M1):** Does the *instrument* give the same answer when re-applied?
- **Consistency (I4):** Does the *causal effect* hold across experimental variation?

High reliability, low consistency: instrument is stable but the effect it measures varies across prompt families. Low reliability, high apparent consistency: instrument is noisy but averages out to a consistent-looking result. Report both.

## Minimum reporting rule

- Bootstrap CI on all primary metrics (faithfulness, IIA, AUROC).
- Seed variance for any random process.
- Whether checkpoint stability was tested; if not, flag it.
- A result without any variance estimate fails this criterion.
