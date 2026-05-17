---
title: "A06 — Do-Calculus Mediation (NDE/NIE)"
description: "Natural Direct and Indirect Effects decomposition for transformer circuits: quantifying how much of a causal effect flows through each path."
---

# A06 — Do-Calculus Mediation (NDE / NIE)

This framework asks: **how much of the total causal effect flows *through* a specific intermediate component versus around it?**

Causal mediation analysis decomposes the total effect of an input change on the output into a Natural Direct Effect (NDE) — the portion that bypasses the mediator — and a Natural Indirect Effect (NIE) — the portion that flows through the mediator. In a transformer, this is the question of path-specific causation: when we change the input from clean to corrupted, how much of the logit-difference change is mediated by a specific head's activation, versus flowing through the residual stream around it?

This goes beyond activation patching (which measures total effect of ablating a component) to decompose *where* the effect comes from. A head might have low activation-patching score because its contribution is split across multiple paths — mediation analysis reveals the indirect effects that aggregate patching misses.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Pearl, "Direct and Indirect Effects"](https://ftp.cs.ucla.edu/pub/stat_ser/R273-U.pdf) | 2001 | NDE/NIE definitions via nested counterfactuals |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | Do-calculus foundations for mediation |
| [Vig et al., arXiv 2004.14944](https://arxiv.org/abs/2004.14944) | 2020 | First application of causal mediation to transformers (gender bias) |
| [Finlayson et al., arXiv 2106.06087](https://arxiv.org/abs/2106.06087) | 2021 | Causal mediation for factual knowledge in language models |
| [Goldowsky-Dill et al., arXiv 2304.05969](https://arxiv.org/abs/2304.05969) | 2023 | Path patching as edge-level mediation analysis |

## Core concept: path-specific effects

The total effect (TE) of changing input from \( x \) to \( x' \) decomposes as:

\[
TE = NDE + NIE
\]

where the Natural Indirect Effect through mediator \( M \) is:

\[
NIE_M = \mathbb{E}[Y(x, M(x')) - Y(x, M(x))]
\]

and the Natural Direct Effect is:

\[
NDE_M = \mathbb{E}[Y(x', M(x)) - Y(x, M(x))]
\]

The NIE measures how much the output changes when we let the mediator respond to the input change while holding everything else fixed. The NDE measures how much the output changes through all paths *except* the mediator. A component with high NIE/TE ratio is a strong mediator — most of the causal effect flows through it.

The Path-Specific Effect (PSE) extends this to arbitrary paths in the computational graph, allowing decomposition of effects along specific edges (e.g., "the effect of the input change on the output that flows through head 9.9's value vector specifically").

## Instruments under A06

### C5 — Mediation Analysis (`05_mediation.py`, `05_mediation_v2.py`)

Computes NDE and NIE for each component and decomposes the total causal effect into path-specific contributions:

\[
\frac{NIE(c)}{TE} = \text{fraction of total effect mediated by component } c
\]

The v2 variant uses improved counterfactual sampling (resampling from the empirical distribution rather than a single corrupted input) for more robust estimates.

**What it establishes:** The fraction of total causal effect flowing through each component. Components with high NIE/TE are strong mediators.

**What it does not establish:** Whether the mediation is necessary (the effect might flow through alternative paths if this one is blocked) or whether the mediator's contribution is linear.

**Usage:**
```
uv run python 05_mediation_v2.py --tasks ioi sva --n-prompts 40
```

### C24 — Path-Specific Effects (`24_pse.py`)

Extends mediation to specific edges in the computational graph. For each directed path \( \pi \) from input to output, estimates the effect flowing along that path:

\[
PSE(\pi) = \mathbb{E}[Y_{\pi(x'), \bar{\pi}(x)} - Y(x)]
\]

where \( \pi(x') \) denotes setting variables along path \( \pi \) to their values under \( x' \) while holding all other paths at their values under \( x \).

**What it establishes:** Which specific information-flow paths carry the causal effect. Distinguishes between a head's contribution via Q, K, V, or residual stream.

**What it does not establish:** Whether the path decomposition is unique (interactions between paths can make PSEs non-additive).

**Usage:**
```
uv run python 24_pse.py --tasks ioi --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| NIE/TE > 0.7 for a component | Strong mediator — most effect flows through it |
| NDE dominates (NIE/TE < 0.3) | Effect mostly bypasses this component |
| PSE concentrated on one edge | Clean information-flow path identified |
| PSEs non-additive (sum != TE) | Path interactions present; non-linear mechanism |

## Connection to other frameworks

A06 decomposes the aggregate causal effects measured by A01 (activation patching) into path-specific contributions. Where A01 gives a single importance score, A06 reveals the routing structure. A02 (DAS/IIA) tests whether a *variable* is implemented in a component; A06 tests whether the *effect* of that variable flows through a specific path. A05 (MDC/Glennan) predicts which paths should carry effects based on weight-space organization — A06 verifies this prediction with activation-level measurements.
