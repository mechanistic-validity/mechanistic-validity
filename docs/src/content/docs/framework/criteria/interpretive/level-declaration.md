---
title: "Level Declaration"
validity_type: "Interpretive"
criterion_id: "V1"
---

# Criterion V1 — Level Declaration

| | |
|---|---|
| Validity type | Interpretive |
| Pass condition | A specific description-mode tag is stated explicitly in the verdict |
| Evidence family | N/A (criterion is about claim structure) |
| Minimum reporting | The mode tag, stated verbatim, in the verdict section of any published claim |
| Common failure mode | Publishing a result without a declared mode tag; leaving the claim's scope implicit |

## What this criterion requires

Level declaration is the simplest interpretive criterion: before any claim can be evaluated for interpretive validity, it must declare what level of description it is making. Without a declared level, there is no standard against which to measure evidence–claim fit.

The five description mode tags (from [../taxonomy/](../taxonomy/)):

| Tag | Meaning | Minimum validity requirements |
|---|---|---|
| `[functional]` | Describes input-output behavior without mechanism | None beyond behavioral evidence |
| `[representational]` | Claims a variable is encoded at a component | Baseline-separated IIA or equivalent |
| `[causal-mechanistic]` | Claims a component causally implements a computation | Necessity + sufficiency established |
| `[structural-mechanistic]` | Claims a component's weights implement a computation | Structural plausibility + causal support |
| `[transportable]` | Claims the mechanism generalizes across contexts | At least one robustness result |

A verdict without one of these tags is not a verdict — it is a measurement with a story attached.

## Why this is required

The most common interpretive failure in MI is implicit level inflation: a paper establishes `[representational]` evidence (high IIA) and implicitly claims `[causal-mechanistic]` status without the additional evidence that requires. Level declaration forces the implicit claim to be explicit, where it can be evaluated.

## Minimum reporting rule

Every verdict must contain one of the five tags verbatim, followed immediately by the scope restriction:

> "**Verdict:** `[causal-mechanistic]` for L8.MLP as a primary SVA locus in GPT-2 Small on the Linzen et al. prompt distribution. Not yet `[transportable]` — cross-architecture generalization has not been established."

## Relation to other interpretive criteria

Level declaration (V1) is the prerequisite for level–evidence match (V2). You cannot check whether the evidence licenses the claimed level until the claimed level is stated.
