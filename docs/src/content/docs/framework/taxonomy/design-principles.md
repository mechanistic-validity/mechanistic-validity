---
title: "Design Principles"
description: "The five foundational commitments that explain why the framework is structured the way it is."
---

# Design Principles

The framework's structure reflects five explicit design commitments, each one a response to a recurring failure mode in published MI work.

## Principle 1: No scalar summary score

The framework does not produce a single number summarizing circuit quality. It produces a structured verdict: a pattern of which validity dimensions have evidence and which do not.

Two circuits with identical scalar scores can have radically different validity profiles. Circuit A with high internal validity but zero external validity and Circuit B with moderate internal validity but two tested model families are not equivalent claims, and no aggregation preserves the distinction. This principle also means the framework cannot be gamed: a claim cannot be upgraded to *Validated* by accumulating more evidence of one type if the other types remain unaddressed.

## Principle 2: Partial-pass language is the default

Most MI findings should not receive a *Validated* verdict. The appropriate default is a partial-pass verdict naming exactly which validity types have been addressed. *Mechanistically supported* is a real finding. *Causally suggestive* is a real result that warrants follow-up. Neither needs to be inflated to *Validated* to be publishable, and inflating either creates a false impression about what the evidence shows.

Practically: the default verdicts for most circuit-discovery work on a single model and task should be *Causally suggestive* or *Mechanistically supported*. *Triangulated* requires cross-task or cross-scale evidence. *Validated* requires a full five-type audit.

## Principle 3: Description level is a property of the verdict, not the evidence

Instruments do not have description levels. Activation patching is not an algorithmic-level test. DAS-IIA is not a representational-level test. Description level is a property of the *claim* the researcher makes using those instruments.

The failure mode this principle is designed to catch is **level inflation**: reporting an implementational result as an algorithmic claim. Level inflation is the most common misuse of MI evidence. It makes circuits sound more understood than they are, and it makes failed replications look like contradictions when they are actually tests of a different claim.

## Principle 4: Instrument disagreement is a finding, not a problem

When weight-classifier analysis and EAP attribution patching nominate different sets of components as the circuit (Jaccard ≈ 0, as observed in this project), the correct response is to treat the disagreement as a primary result, not to choose one instrument and discard the other.

Instrument disagreement reveals one of three things: (1) one instrument has a methodological flaw, (2) the two instruments are sensitive to different real properties of the circuit, or (3) the construct is underspecified. All three are informative. The practical rule: every multi-instrument study should report Jaccard similarity between instrument outputs as a first-class result.

## Principle 5: Evidence tells us what we measured; validity tells us what we can conclude

This is the single distinction that organizes the whole framework. An IIA score of 0.48 at L8.MLP is a measurement. Whether that measurement constitutes evidence that L8.MLP implements the SVA mechanism is a validity question — and it cannot be answered from the score alone. It requires knowing the random-vector baseline, the untrained-model baseline, the cross-task control, and the replication rate across seeds.

Every field that studies invisible mechanisms — neuroscience, pharmacology, measurement theory, epidemiology — has learned this distinction the hard way. Mechanistic interpretability is learning it now. There is no such thing as "too much evidence": evidence from a different evidence family still matters because it tests convergent validity, even if an earlier instrument already covered the same criterion.
