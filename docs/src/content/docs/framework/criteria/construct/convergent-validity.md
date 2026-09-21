---
title: "Convergent Validity"
validity_type: "Construct"
criterion_id: "C3"
---

# Criterion C3 — Convergent Validity

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

1. **≥3 metrics from different evidence families** have been run. Two [ablation](/mechanistic-validity/glossary/#ablation) variants (both causal) do not satisfy — they measure the same thing differently. Families count as independent when the failure of one's core assumption would not automatically invalidate the other: e.g., causal + structural + representational.
2. **The metrics agree on component membership.** Overlap is reported, not thresholded: what counts as agreement depends on the size of the circuit and on the spread each metric shows under resampling.
3. **Disagreement is treated as a finding.** Low Jaccard overlap must be reported as a primary result and investigated — not silently resolved by choosing the preferred metric.

## When methods disagree

When two discovery methods from different evidence families nominate substantially different component sets, the disagreement typically means one of:

- One metric has a methodological flaw explaining its output;
- The two metrics are sensitive to different real properties (e.g., structurally consistent but causally inactive components vs. causally active but structurally atypical ones);
- The construct (e.g., "the SVA circuit") is underspecified and metrics track different things under the same label.

All three interpretations are scientifically important. Convergent validity is failed, and the disagreement is reported as a result.

## Metric coverage for convergent validity

| Metric type | Evidence family |
|---|---|
| [Attribution patching](/mechanistic-validity/glossary/#attribution-patching) | Causal |
| [DAS-IIA](/mechanistic-validity/glossary/#das) | Representational |
| Weight-space analysis (SVD, [composition scores](/mechanistic-validity/glossary/#composition-score)) | Structural |
| Activation statistics (LLC, probe accuracy) | Structural / Representational |
| Bootstrap stability | Measurement |

Three metrics from different evidence families constitute the test at Triangulated; two establish a partial result.

## Minimum reporting rule

- List every metric used and its evidence family.
- For each pair from different families: Jaccard similarity.
- Note which components appear in every metric, and which are unique to one.
- Where the metrics disagree, report the disagreement explicitly and state which of the three interpretations above is most consistent with the data.
