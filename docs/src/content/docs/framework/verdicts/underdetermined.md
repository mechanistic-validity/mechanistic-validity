---
title: "Underdetermined"
description: "The evidence is consistent with multiple mechanistic accounts and does not distinguish between them — a lateral verdict."
---

# Verdict: Underdetermined

| | |
|---|---|
| Tier | Lateral (outside progression) |
| What it means | Multiple mechanistic accounts are consistent with all available evidence — the data does not distinguish between them |
| When to assign | Two or more hypotheses have comparable evidential support and no available experiment has been performed to separate them |
| Resolution path | Identify the discriminating experiment and perform it — the claim then moves to the appropriate progressive tier |
| Relationship to progressive tiers | A claim at any progressive tier can additionally be flagged as Underdetermined if competing accounts exist at that tier |

## What this verdict establishes

Underdetermined is not a failure. It is a precise characterization of the epistemic state: the evidence is real, the measurements are sound, but the data is equally consistent with multiple distinct mechanistic accounts. The informative response is to name the competing accounts and identify what experiment would distinguish them.

This verdict prevents premature commitment to one mechanistic story when the evidence does not support that commitment. It also provides a clear research direction: the discriminating experiment.

Formally, let $H_1, H_2, \ldots, H_n$ be competing mechanistic hypotheses for behavior $B$. Underdetermination holds when the available evidence $\mathcal{E}$ is approximately equally likely under all competing hypotheses, and the posterior ratio is determined primarily by priors rather than evidence.

## Example verdict statement

> **Verdict:** Underdetermined — `[implementational-topographic]`
> **Claim:** The Docstring Circuit implements variable binding.
> **Competing accounts:** (1) Variable binding — tracking which variable name maps to which argument position. (2) Positional copying — copying from a fixed offset regardless of variable identity.
> **Evidence status:** Activation patching results are consistent with both accounts. Neither predicts distinct behavior on the tested prompts.
> **Discriminating experiment:** Test on prompts where variable names are reordered relative to argument positions — the two accounts predict different outputs.
> **Scope:** GPT-2 Small, Python docstring completion, Heimersheim & Janiak prompt distribution

## Minimum reporting for this tier

- All competing hypotheses stated explicitly with their predictions
- Evidence that supports each hypothesis listed
- Explanation of why the available evidence does not discriminate
- At least one discriminating experiment identified (what would the competing accounts predict differently?)
- Current progressive tier of the evidence (Underdetermined is overlaid on a progressive tier)

## Resolution paths

| Resolution | Outcome |
|---|---|
| Discriminating experiment favors $H_i$ | Claim moves to appropriate progressive tier under $H_i$; other hypotheses become Disconfirmed or deprioritized |
| All hypotheses shown to be equivalent | The accounts are notational variants — dissolve into a single claim at its progressive tier |
| New hypothesis $H_{n+1}$ subsumes all others | The more general account replaces the competing specific ones |

## Characteristic occupants

- **Docstring Circuit** ([Heimersheim & Janiak, 2023](https://arxiv.org/abs/2307.13057)) — variable binding vs. positional copying accounts both consistent with observed patching results
- **IOI S-inhibition heads** — inhibition vs. attention-sink accounts both predict the observed attention patterns; targeted intervention on sink tokens would discriminate
- **Superposition vs. polysemanticity** in real (non-toy) models — whether observed feature interference is superposition (geometric packing) or polysemanticity (intrinsic multi-functionality) remains underdetermined in many cases

## Why this is lateral, not lower

Underdetermined does not mean "bad evidence." A claim can have strong causal evidence (Tier 2 or 3 level) while simultaneously being underdetermined between competing accounts. The progressive tier captures evidential strength; the Underdetermined flag captures interpretive ambiguity. Both dimensions matter.

## Key references

- Heimersheim & Janiak (2023). *A Circuit for Python Docstrings in a 4-Layer Attention-Only Transformer.* [arXiv:2307.13057](https://arxiv.org/abs/2307.13057)
- Elhage et al. (2022). *Toy Models of Superposition.* [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes.* [doi:10.1017/CBO9780511621123](https://doi.org/10.1017/CBO9780511621123)
