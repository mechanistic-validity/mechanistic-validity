---
title: "Insufficient"
description: "The construct is not defined precisely enough to score — a diagnostic label indicating the claim cannot yet be evaluated."
---

# Verdict: Insufficient

| | |
|---|---|
| Label | Diagnostic (outside progression) |
| What it means | The construct is not defined precisely enough to be evaluated against the criteria |
| When to assign | The claim lacks a falsifiable statement, an operationalized target, or both — scoring would be premature |
| Resolution path | Sharpen the construct definition until C1 (Falsifiability) and C2 (Structural plausibility) can be assessed |
| Relationship to progressive tiers | Insufficient replaces the tier -- the claim holds no position in the hierarchy, rather than a position beneath Proposed |

## What this verdict establishes

Insufficient is not a negative result. It is a statement that the claim has not yet been formulated precisely enough for the framework to evaluate it. The criteria require a well-defined target: a component, circuit, or mechanism whose existence would be falsifiable and whose structure is at least plausible given the architecture. Without that, scoring the remaining criteria produces meaningless numbers.

Construct validity can fail to be established in two distinct ways, and either is sufficient to assign this label. The target concept can be incoherent — "safety module," "world model," "reasoning" — with no operational definition that a measurement could confirm or refute. Or the concept can be coherent while the available measurement instruments cannot separate it from a null baseline: if no metric distinguishes the claimed component from a random component of the same size and type (M2 baseline separation fails for every candidate instrument), the claim cannot be scored regardless of how precisely it is stated.

This verdict separates two failure modes that are otherwise conflated: "we tested the claim and it failed" (Disconfirmed) vs. "we cannot test the claim because it is not stated precisely enough" (Insufficient). The distinction matters because the remedies differ — Disconfirmed claims need new evidence or a different hypothesis, while Insufficient claims need a sharper formulation.

## Example verdict statement

> **Verdict:** Insufficient
> **Claim:** "The model has a safety module."
> **Why insufficient:** "Safety module" is not operationalized — it does not specify which components, what behavior they produce, or what would count as evidence against the claim. The term could refer to refusal heads, a linear direction, a distributed circuit, or an emergent property of fine-tuning. Until the claim specifies what is meant, no criterion can be scored.
> **Resolution:** Reformulate as a testable claim, e.g., "Refusal behavior in Llama-2-Chat is mediated by a linear direction in residual stream space at layers 15–20."

## Minimum reporting for this label

- The claim as stated
- Which construct-level criteria (C1–C6) cannot be assessed and why
- What additional specification would move the claim to Proposed

## When Insufficient applies

| Situation | Verdict |
|---|---|
| The claim names a behavior but not a mechanism | Insufficient |
| The claim names a mechanism but provides no falsifiable prediction | Insufficient |
| The claim uses a label ("world model", "planning module") without operationalizing it | Insufficient |
| The claim is well-defined but no available metric separates the target from a random baseline | Insufficient |
| The claim is well-defined but no evidence has been collected | Proposed |
| The claim is well-defined but the evidence is negative | Disconfirmed |

## Characteristic examples

- **"LLMs have emergent reasoning capabilities"** — "emergent" and "reasoning" are each undefined. The claim cannot be tested until both are operationalized.
- **"The model understands X"** — "understands" carries implications (generalization, compositionality, robustness) that are not stated as testable predictions. Different operationalizations would lead to different verdicts.
- **Unnamed circuits in scaling-law papers** — claims about internal structure that reference scale but not specific components cannot be scored.

## Relationship to other diagnostic labels

Insufficient, Underdetermined, and Disconfirmed are all diagnostic labels that sit outside the five progressive tiers (Proposed through Validated). They serve different functions:

- **Insufficient**: the claim is not yet evaluable
- **Underdetermined**: the claim is evaluable but the evidence does not distinguish between competing accounts
- **Disconfirmed**: the claim is evaluable and the evidence is decisively against it
