---
title: "Baseline Separation"
validity_type: "Measurement"
criterion_id: "M3"
---

# Criterion M3 — Baseline Separation

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | The IIA score exceeds both the random-vector baseline AND the untrained-model baseline by a practically meaningful margin (≥0.10 above the higher baseline) |
| Evidence family | Measurement |
| Minimum reporting | Random-vector baseline value; untrained-model baseline value; separation magnitude; published SOTA baseline |
| Common failure mode | Reporting only the absolute IIA score; never computing the baselines |

## What this criterion requires

**Baseline separation is the single most important measurement validity criterion and the most commonly violated one in the current literature.**

An IIA score of 0.48 means nothing without knowing:

1. **What IIA do you get by swapping random vectors instead of factor activations?** (Random-vector baseline.) In high-dimensional spaces (d_model = 768 for GPT-2 Small), a random unit vector has non-trivial alignment with the causal variable's subspace simply due to dimensionality. If random vectors produce IIA = 0.44 and your circuit produces IIA = 0.48, the 0.04 gap is not a finding.

2. **What IIA does an untrained (randomly initialized) model produce?** (Untrained-model baseline.) Measures the architectural prior: how much IIA does the procedure produce before any learning has occurred?

The criterion is satisfied when the score exceeds *both* baselines by ≥0.10 (practically meaningful) AND is stable across prompt splits.

## The SVA competitive finding

For GPT-2 Small SVA, published transcoder IIA scores are in the 0.40–0.60 range. The project's IIA = 0.48 at L8.MLP is right in that range — **competitive with published baselines and potentially publishable**. But *only if baseline separation is reported explicitly.* Without the random-vector and untrained-model baselines, 0.48 is just a number; with baselines showing, e.g., random-vector = 0.31 and untrained-model = 0.22, it becomes a finding (separation of +0.17 and +0.26 respectively).

This is the core of the original question: "IIA scores are meaningless without baselines."

## The three required baselines

| Baseline | How to compute | What it measures |
|---|---|---|
| Random-vector | Run DAS-IIA with random unit vectors instead of factor decoder directions | Dimensionality-driven false positive rate |
| Untrained-model | Run DAS-IIA on same architecture with random (untrained) weights | Architectural prior |
| Published SOTA | IIA from published transcoders/CLT on same task | Competitive calibration |

Expected values for GPT-2 Small SVA:
- Published transcoder baseline: 0.40–0.60
- Random-vector baseline: ~0.25–0.35 (expected; to be computed)
- Untrained-model baseline: ~0.15–0.25 (expected; to be computed)

## Minimum reporting rule

Every IIA result must report:
1. The IIA score for the claimed subspace.
2. The random-vector baseline.
3. The untrained-model baseline.
4. The published SOTA baseline (for calibration, M5).

A paper reporting only the absolute IIA score fails this criterion unconditionally.
