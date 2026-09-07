---
title: "Selection Correction"
validity_type: "Measurement"
criterion_id: "M7"
---

# Criterion M7 — Selection Correction

| | |
|---|---|
| Validity type | Measurement |
| Pass condition | When k findings are selected from N candidates, N is reported and multiplicity is controlled |
| Evidence family | Measurement-theoretic |
| Minimum reporting | Total number of candidates tested (N), number reported (k), correction method if any |
| Common failure mode | Reporting selected results without naming the search space |

## What this criterion requires

Selection correction addresses the multiple-comparisons problem: when many candidates are tested and only the best are reported, the reported results are biased upward. The criterion requires transparency about the search and, where appropriate, statistical correction.

Satisfied when:

1. **N is reported.** The total number of candidates tested (heads, features, circuits, [SAE](/mechanistic-validity/glossary/#sae) latents) is stated.
2. **The selection rule is stated.** How the reported k were chosen from N — top-k by effect size, threshold, manual inspection.
3. **Multiplicity is controlled or acknowledged.** Bonferroni, FDR, or permutation correction is applied, or the selection is explicitly labeled as exploratory.

## MI example

SAE feature analysis often examines thousands of latents, identifies the ones with interpretable activation patterns, and reports those as "features the model uses." The denominator — how many latents were examined and discarded — is rarely stated. Without it, the reported features are a selected sample and their apparent interpretability may reflect selection rather than structure.

## Connection to the chain

Selection correction is a measurement property. A selected result without correction overstates the evidence, inflating construct validity (C3 convergent validity) and internal validity (I4 specificity). Required for full measurement validity at Validated tier.
