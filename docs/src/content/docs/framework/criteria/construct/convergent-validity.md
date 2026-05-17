---
title: "Convergent Validity"
validity_type: "Construct"
criterion_id: "C5"
---

# Criterion C5 — Convergent Validity

| | |
|---|---|
| Validity type | Construct |
| Pass condition | Multiple independent instruments from *different* evidence families nominate the same components |
| Evidence family | Measurement (cross-instrument agreement) |
| Minimum reporting | Jaccard similarity between instrument outputs; both instruments' component lists; interpretation of any disagreement |
| Common failure mode | Running multiple instruments but not checking whether they agree; treating agreement as assumed |

## What this criterion requires

Convergent validity is the core of this entire project. Running one instrument and claiming a circuit is not enough.

Satisfied when:

1. **≥2 instruments from different evidence families** have been run. Two ablation variants (both causal) do not satisfy — they measure the same thing differently. Instruments must come from different epistemic traditions: e.g., causal + structural, or representational + behavioral.
2. **The instruments agree on component membership.** Jaccard ≥ 0.5 is a reasonable pass threshold for circuits of ≤ 20 components.
3. **Disagreement is treated as a finding.** Jaccard ≈ 0 must be reported as a primary result and investigated — not silently resolved by choosing the preferred instrument.

## The Jaccard ≈ 0 finding in this project

The weight-circuit (structural analysis) and EAP-circuit (attribution patching) have Jaccard ≈ 0. This is one of the project's most informative findings. It means one of:

- One instrument has a methodological flaw explaining its output;
- The two instruments are sensitive to different real properties (structurally similar but causally inactive components vs. causally active but structurally atypical ones);
- The construct "SVA circuit" is underspecified and instruments track different things under the same label.

All three interpretations are scientifically important. Convergent validity is failed; the disagreement is the finding.

## Instrument coverage for convergent validity

| Instrument | Evidence family |
|---|---|
| Weight classifier (B03) | Structural |
| EAP attribution patching (A01) | Causal |
| DAS-IIA (A02) | Representational |
| Activation patching (A01) | Causal |
| LLC per component (B04) | Structural |
| Bootstrap stability (F01) | Measurement |

Any two instruments from different rows constitute a valid test.

## Minimum reporting rule

- List every instrument used and its evidence family.
- For each pair from different families: Jaccard similarity.
- If Jaccard ≥ 0.5: note which components appear in all instruments.
- If Jaccard < 0.5: report disagreement explicitly, list components unique to each instrument, state which interpretation is most consistent with the data.
