---
title: "Consistency"
validity_type: "Internal"
criterion_id: "I4"
---

# Criterion I4 — Consistency

| | |
|---|---|
| Validity type | Internal |
| Pass condition | The finding holds across prompt samples, ablation methods, and random seeds |
| Evidence family | Causal, Measurement |
| Minimum reporting | Variance estimate across ≥3 prompt splits or seeds; sigma-ablation across ≥3 ablation methods |
| Common failure mode | Reporting a single point estimate with no variance; sigma-ablation not run |

## What this criterion requires

Consistency is the replication criterion for internal validity. Three forms:

**Prompt-sample consistency:** Stable across different random samples from the same distribution. Minimum: 3 independent prompt splits; sigma-ablation ≤ 0.05 across splits; test-retest Pearson r ≥ 0.9.

**Method consistency (sigma-ablation):** Direction and approximate magnitude hold across multiple ablation methods. The sigma-ablation framework tests this by running 8 ablation variants and measuring variance across methods. sigma-ablation ≤ 0.05 is a reasonable threshold.

**Seed consistency:** Results hold across ≥3 random seeds for any analysis involving random initialization.

## Consistency vs. reliability (M1)

- **Consistency (I4):** Stability of the *finding* — does the causal effect hold across experimental variation?
- **Reliability (M1):** Stability of the *metric* — does the measurement tool give the same answer when re-applied?

These can diverge: a reliable metric can measure an inconsistent effect (the effect changes sign across prompt families). Report both.

## Sigma-ablation

Implemented in `c03sigmaablation.py` (A04 Woodward). Runs the same component ablation across multiple methods, computes standard deviation of metric values. Low sigma (≤ 0.05 on 0–1 normalized metric) means the finding is method-invariant. High sigma means effect size depends on ablation method choice — a consistency failure requiring investigation.

The script reports: mean ablation effect, sigma across methods, min, max, and a flag for methods where effect direction reverses.

## Minimum reporting rule

- Number of prompt splits and variance/sigma across splits.
- Sigma-ablation result (methods used, sigma value) if run.
- Seed count if any random process was involved.
- If variance was not estimated: flag consistency as partial pass.
