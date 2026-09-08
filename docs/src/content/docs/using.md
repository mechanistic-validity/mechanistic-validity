---
title: Using the Framework
description: "How to apply the Mechanistic Validity pipeline to a mechanistic claim."
---

# Using the Framework

This page walks through how to apply the Mechanistic Validity pipeline to evaluate a mechanistic claim. The pipeline has six layers; the first two scope the claim, the third produces evidence, and the last three score it.

## 1. Declare a description mode

State what level of explanation the claim is making. A claim that names components (topographic) requires different evidence than one that attributes a function to them (functional) or describes an algorithm (algorithmic). The declaration constrains what counts as evidence and what counts as a gap.

See [Description Modes](/mechanistic-validity/framework/description-modes/) for the seven modes.

## 2. Identify evidence families

Classify the available evidence by source (weights, activations, behavior, training) and access (observational vs. interventional). This produces an 8-cell matrix. Most published MI evidence concentrates in a single cell — interventional activations. Triangulation means covering multiple cells with independent failure modes.

See [Evidence Families](/mechanistic-validity/framework/evidence-families/) for the full matrix.

## 3. Run metrics and calibrations

Produce the evidence. This is the iterative phase — run metrics, check which criteria are weak, gather more evidence where needed. Calibrations (bootstrap stability, baseline separation, seed variance) cut across all evidence families and should be run alongside the primary metrics.

See [Metrics](/mechanistic-validity/framework/metrics/) for the metric catalog.

## 4. Score criteria

Score the 36 criteria against the evidence. Each criterion receives one of six statuses: Confirmed, Partially confirmed, Inconclusive, Untested, Disconfirmed, or Not applicable. The scoring is deterministic given the evidence — the same record always produces the same statuses.

See [Criteria](/mechanistic-validity/framework/criteria/) for all 36 criteria across five validity types.

## 5. Assess validity types

Aggregate the criterion scores by validity type. The five types form a dependency chain: construct → measurement → internal → external → interpretive. A debt in an upstream type is not repayable by evidence in a downstream one.

See [Validity Types](/mechanistic-validity/framework/validity-types/) for the five types and their dependency structure.

## 6. Issue a verdict

The verdict reports three things: the strongest reading of the claim the record warrants, the tier that reading reaches (Proposed through Validated), and the criterion capping it — which names the experiment that would lift it.

See [Verdicts](/mechanistic-validity/framework/verdicts/) for the five tiers and three diagnostic labels.

## Worked examples

The [Case Studies](/mechanistic-validity/framework/examples/) apply this pipeline to sixteen published claims. The [IOI circuit walkthrough](/mechanistic-validity/framework/#running-example-activation-patching-on-the-ioi-circuit) on the Framework Overview page shows the pipeline applied step by step to a single claim.
