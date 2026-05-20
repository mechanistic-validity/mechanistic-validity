---
title: "Convergent Validity"
validity_type: "Construct"
criterion_id: "C5"
---

# Criterion C5 — Convergent Validity

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Multiple independent metrics from *different* evidence families nominate the same components |
| Evidence family | Measurement (cross-metric agreement) |
| Minimum reporting | Jaccard similarity between metric outputs; both metrics' component lists; interpretation of any disagreement |
| Common failure mode | Running multiple metrics but not checking whether they agree; treating agreement as assumed |

## What this criterion requires

Convergent validity is the core of this entire project. Running one metric and claiming a circuit is not enough.

Satisfied when:

1. **≥2 metrics from different evidence families** have been run. Two ablation variants (both causal) do not satisfy — they measure the same thing differently. Metrics must come from different epistemic traditions: e.g., causal + structural, or representational + behavioral.
2. **The metrics agree on component membership.** Jaccard ≥ 0.5 is a reasonable pass threshold for circuits of ≤ 20 components.
3. **Disagreement is treated as a finding.** Jaccard ≈ 0 must be reported as a primary result and investigated — not silently resolved by choosing the preferred metric.

## The Jaccard ≈ 0 finding in this project

The weight-circuit (structural analysis) and EAP-circuit (attribution patching) have Jaccard ≈ 0. This is one of the project's most informative findings. It means one of:

- One metric has a methodological flaw explaining its output;
- The two metrics are sensitive to different real properties (structurally similar but causally inactive components vs. causally active but structurally atypical ones);
- The construct "SVA circuit" is underspecified and metrics track different things under the same label.

All three interpretations are scientifically important. Convergent validity is failed; the disagreement is the finding.

## Metric coverage for convergent validity

| Metric | Evidence family |
|---|---|
| Weight classifier (B03) | Structural |
| EAP attribution patching (A01) | Causal |
| DAS-IIA (A02) | Representational |
| Activation patching (A01) | Causal |
| LLC per component (B04) | Structural |
| Bootstrap stability (F01) | Measurement |

Any two metrics from different rows constitute a valid test.

## Minimum reporting rule

- List every metric used and its evidence family.
- For each pair from different families: Jaccard similarity.
- If Jaccard ≥ 0.5: note which components appear in all metrics.
- If Jaccard < 0.5: report disagreement explicitly, list components unique to each metric, state which interpretation is most consistent with the data.
