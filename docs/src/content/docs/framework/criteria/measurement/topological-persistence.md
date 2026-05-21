---
title: "Topological Persistence"
validity_type: "Measurement"
criterion_id: "M7"
---

# Criterion M7 — Topological Persistence

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | ≥ 2 features with persistence ratio ≥ 3x median; bootstrap bottleneck distance ≤ 0.2 x max persistence |
| Evidence family | Representational |
| Minimum reporting | Persistence diagram, persistence ratios for top features, bottleneck distances across ≥ 50 bootstrap samples, shuffled baseline comparison |
| Common failure mode | Reporting persistence diagram without bootstrap stability test |
| Lens | Dynamical Systems |

## What this criterion requires

Topological persistence tests whether a circuit's activation geometry contains stable topological features — connected components, loops, voids — that survive across prompt variations. Persistent homology computes a persistence diagram: each topological feature is born at some filtration scale and dies at another, and features with high persistence (large birth-to-death ratio) represent robust geometric structure rather than noise.

The criterion requires two things. First, at least 2 features must have a persistence ratio (death/birth) of at least 3x the median persistence ratio, indicating that they stand out from the noise floor. Second, these features must be stable: the bottleneck distance between persistence diagrams computed on different bootstrap samples of the prompt distribution must be ≤ 0.2 times the maximum persistence. This ensures the topological structure reflects the circuit's geometry rather than the particular prompts sampled.

This criterion does not establish causal relevance or interpretability. A circuit can have highly persistent topological features that play no functional role. Topological persistence is a measurement validity criterion: it tests whether the geometric structure you are measuring is real and stable, not whether it matters. Pair with causal criteria (E1-E9) to establish that the measured structure is mechanistically relevant.

## Minimum reporting rule

- Persistence diagram for H0 (connected components) and H1 (loops) at minimum.
- Persistence ratios for the top features, with the median persistence ratio as reference.
- Bottleneck distances across ≥ 50 bootstrap resamples of the prompt distribution (mean, 95% CI).
- Shuffled baseline: persistence diagram computed on the same activations with token positions shuffled; bottleneck distance between real and shuffled diagrams.
- If no features exceed 3x median persistence or bootstrap bottleneck distance exceeds threshold: criterion is unsatisfied.
