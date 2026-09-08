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

The seven [description modes](/mechanistic-validity/framework/description-modes/):

| Mode | Meaning | Minimum validity requirements |
|---|---|---|
| [`[computational]`](/mechanistic-validity/framework/modes/computational) | States what problem the system solves, without mechanism | Behavioral evidence on a stated distribution |
| [`[algorithmic]`](/mechanistic-validity/framework/modes/algorithmic) | Names the steps and their order | Causal evidence for each step and for the ordering |
| [`[representational]`](/mechanistic-validity/framework/modes/representational) | Claims a variable is encoded at a component | Baseline-separated [IIA](/mechanistic-validity/glossary/#iia) or equivalent |
| [`[implementational-topographic]`](/mechanistic-validity/framework/modes/implementational-topographic) | Locates the computation in named components | Necessity, with a matched-set control |
| [`[implementational-connectomic]`](/mechanistic-validity/framework/modes/implementational-connectomic) | Claims a specific edge structure between components | Edge-level causal evidence, not node-level |
| [`[implementational-functional]`](/mechanistic-validity/framework/modes/implementational-functional) | Claims a component's weights implement the computation | Structural plausibility plus causal support |
| [`[implementational-activation]`](/mechanistic-validity/framework/modes/implementational-activation) | Claims a specific activation geometry carries the computation | Geometry measured and separated from a baseline |

A verdict without a declared mode is not a verdict — it is a measurement with a story attached.

## Why this is required

The most common interpretive failure in MI is implicit level inflation: a paper establishes `[representational]` evidence (high IIA) and implicitly claims an `[algorithmic]` account without the additional evidence that requires. Level declaration forces the implicit claim to be explicit, where it can be evaluated.

## Minimum reporting rule

Every verdict must name one of the seven modes, followed immediately by the scope restriction:

> "**Verdict:** `[implementational-topographic]` for L8.MLP as a primary SVA locus in GPT-2 Small on the Linzen et al. prompt distribution. No `[algorithmic]` claim is made — the operation L8.MLP performs is not established."

## Relation to other interpretive criteria

Level declaration (V1) is the prerequisite for level–evidence match (V2). You cannot check whether the evidence licenses the claimed level until the claimed level is stated.
