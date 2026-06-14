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

Running one metric and claiming a circuit is not enough. Convergent validity requires triangulation across independent measurement traditions.

Satisfied when:

1. **≥2 metrics from different evidence families** have been run. Two ablation variants (both causal) do not satisfy — they measure the same thing differently. Metrics must come from different epistemic traditions: e.g., causal + structural, or representational + behavioral.
2. **The metrics agree on component membership.** Jaccard ≥ 0.5 is a reasonable pass threshold for circuits of ≤ 20 components.
3. **Disagreement is treated as a finding.** Low Jaccard overlap must be reported as a primary result and investigated — not silently resolved by choosing the preferred metric.

## When methods disagree

When two discovery methods from different evidence families nominate substantially different component sets (Jaccard < 0.5), the disagreement typically means one of:

- One metric has a methodological flaw explaining its output;
- The two metrics are sensitive to different real properties (e.g., structurally consistent but causally inactive components vs. causally active but structurally atypical ones);
- The construct (e.g., "the SVA circuit") is underspecified and metrics track different things under the same label.

All three interpretations are scientifically important. Convergent validity is failed; the disagreement is the finding.

## Metric coverage for convergent validity

| Metric type | Evidence family |
|---|---|
| Attribution patching | Causal |
| DAS-IIA | Representational |
| Weight-space analysis (SVD, composition scores) | Structural |
| Activation statistics (LLC, probe accuracy) | Structural / Representational |
| Bootstrap stability | Measurement |

Any two metrics from different evidence families constitute a valid test.

## Minimum reporting rule

- List every metric used and its evidence family.
- For each pair from different families: Jaccard similarity.
- If Jaccard ≥ 0.5: note which components appear in all metrics.
- If Jaccard < 0.5: report disagreement explicitly, list components unique to each metric, state which interpretation is most consistent with the data.
