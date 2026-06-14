---
title: "Falsifiability"
validity_type: "Construct"
criterion_id: "C1"
---

# Criterion C1 — Falsifiability

| | |
|---|---|
| Validity type | Construct |
| Pass condition | A named, specific result is stated *before data collection* that would disconfirm the circuit claim |
| Evidence family | Any (criterion is about claim structure, not the metric) |
| Minimum reporting | The disconfirmation condition verbatim, with a threshold value, stated prospectively |
| Common failure mode | "The circuit is real unless ablation fails" — too inclusive to be falsifiable |

## What this criterion requires

Falsifiability is the minimum condition for a claim to have determinate content. A circuit claim that cannot be disconfirmed by any specifiable result is not a scientific claim; it is a label applied to whatever the metrics found.

The criterion requires:

1. **A stated disconfirmation condition** — a specific result that, if obtained, would lead the researcher to conclude the circuit claim is wrong. Must be stated *before* the disconfirmatory test is run.
2. **A threshold** — quantitative or at minimum ordinal. "IIA < 0.10 above the random-vector baseline across three prompt splits" is acceptable; "IIA is not significantly different from baseline" is not.
3. **Prior commitment** — the condition must be committed to before collecting the relevant data. A disconfirmation condition invented after a negative result is rationalization, not falsifiability.

## Why this is a construct criterion

Falsifiability lives under construct validity because the problem is about the *content* of the claim, not whether an experiment succeeded. A claim defined only by what the metrics found — "the SVA circuit is whatever components show above-chance IIA on SVA" — is unfalsifiable by construction: adding components in response to any negative result always restores the claim.

## Worked example

> **Falsification condition:** If DAS-IIA on a held-out prompt set (n ≥ 200 prompts, 3 random seeds) does not exceed the random-vector baseline by ≥ 0.10 in at least 2 of 3 seeds, the claim that the nominated component is a primary causal locus for the target task is disconfirmed.

## Common failures

**Too inclusive:** "The circuit is real unless ablation fails." A sufficiently large circuit almost never fails ablation — you can always add the ablated component back.

**Post-hoc:** The threshold was set after seeing the data.

**Vague:** "If the circuit doesn't generalize, we would revisit the claim." Names no metric, no threshold, no specific result.

## Relation to other criteria

Falsifiability interacts with **minimality (C4)**: a non-minimal circuit is harder to falsify because many components provide fallback explanations. It also interacts with **convergent validity (C5)**: when metrics from different evidence families disagree, the falsifiability condition must specify which metric's result counts as disconfirmation.
