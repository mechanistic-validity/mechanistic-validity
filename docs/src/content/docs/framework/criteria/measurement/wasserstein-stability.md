---
title: "Wasserstein Stability"
validity_type: "Measurement"
criterion_id: "M11"
---

# Criterion M11 — Wasserstein Stability

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | W1 between split-half activation distributions ≤ 0.2 x W1 between circuit and random components |
| Evidence family | Representational |
| Minimum reporting | W1 between bootstrap resamples (mean, 95% CI), W1 circuit-vs-random, ratio, ≥ 50 resamples |
| Common failure mode | Reporting absolute W1 without circuit-vs-random reference scale |
| Lens | Geometry |

## What this criterion requires

Wasserstein stability tests whether a circuit's activation geometry is stable across prompt samples. The 1-Wasserstein distance (earth mover's distance) between two distributions measures the minimum cost of transforming one into the other — it respects the metric structure of the activation space rather than just comparing marginal statistics.

The test splits the prompt corpus into two halves, computes the activation distribution of the circuit on each half, and measures the W1 distance between them. This is repeated across ≥ 50 bootstrap resamples to obtain a distribution of within-circuit W1 distances. The reference scale is the W1 distance between the circuit's activation distribution and the activation distribution of a size-matched set of random (non-circuit) components on the same prompts. The criterion is satisfied when the within-circuit W1 (measuring sampling variability) is ≤ 20% of the circuit-vs-random W1 (measuring the structural difference between circuit and non-circuit components).

The ratio matters more than the absolute W1 value. A circuit with large absolute W1 between splits may still pass if the circuit-vs-random distance is much larger — this means the circuit has a distinctive geometric signature that dwarfs the sampling noise. Conversely, a circuit with small absolute W1 may fail if the circuit-vs-random distance is similarly small — the circuit's geometry is not distinguishable from random components even when measured stably. This criterion does not establish that the measured geometry is correct or causally relevant — only that it is reproducible.

## Minimum reporting rule

- W1 between split-half activation distributions: mean and 95% CI across ≥ 50 bootstrap resamples.
- W1 between circuit and size-matched random components: mean and 95% CI.
- Ratio of within-circuit W1 to circuit-vs-random W1.
- Number of bootstrap resamples and split-half size.
- If ratio exceeds 0.2 or fewer than 50 resamples were used: criterion is unsatisfied.
