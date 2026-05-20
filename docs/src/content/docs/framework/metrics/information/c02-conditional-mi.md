---
title: "C02 — Conditional Mutual Information"
description: "Measures information shared between components after conditioning on other parts of the circuit."
---

# C02 — Conditional Mutual Information

This framework asks: **How much unique information does a component carry about the task, beyond what other components already provide?**

Conditional mutual information (CMI) extends MI by asking: given that we already observe component B, how much additional information does component A provide about the output? This distinguishes genuinely unique contributions from redundant ones. A circuit head with high MI but low CMI (conditioned on the rest of the circuit) is informationally redundant.

CMI is central to understanding circuit minimality. If every component has positive CMI given the others, the circuit is informationally non-redundant — each part contributes something the others cannot.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Cover & Thomas, *Elements of Information Theory*](https://doi.org/10.1002/047174882X) | 2006 | Chain rule and CMI properties |
| [Williams & Beer, "Nonnegative Decomposition of Multivariate Information"](https://arxiv.org/abs/1004.2515) | 2010 | CMI as building block for PID |
| [Wyner, "The Common Information of Two Random Variables"](https://doi.org/10.1109/TIT.1975.1055346) | 1975 | Common information and conditional independence |
| [Runge et al., "Detecting and quantifying causal associations"](https://doi.org/10.1126/sciadv.aau4996) | 2019 | CMI for causal discovery in time series |

## Core concept

The conditional mutual information of \( X \) and \( Y \) given \( Z \) is:

\[ I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z) \]

Equivalently, via the chain rule: \( I(X; Y \mid Z) = I(X; Y, Z) - I(X; Z) \). This quantity is zero if and only if \( X \) and \( Y \) are conditionally independent given \( Z \).

For circuit analysis, let \( A_i \) be the activation of head \( i \), \( Y \) the task output, and \( A_{-i} \) all other circuit heads. Then:

\[ I(A_i; Y \mid A_{-i}) \]

measures the unique information head \( i \) carries. If this is near zero, the head is informationally redundant — its contribution is already captured by the other components.

## Metrics under C02

### PID Script (`08_pid.py`)

The PID decomposition uses CMI as its foundation. The unique information of source \( X_1 \) about target \( Y \) is bounded by \( I(X_1; Y \mid X_2) \), with equality under certain PID definitions.

**What it establishes:** Whether a component's information is unique or shared with other components.
**What it does not establish:** Whether the unique information is causally necessary (it could be carried but unused).

**Usage:**
```
uv run python 08_pid.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| \( I(A_i; Y \mid A_{-i}) \approx 0 \) | Head is informationally redundant |
| \( I(A_i; Y \mid A_{-i}) \approx I(A_i; Y) \) | Head carries fully unique information |
| CMI drops when conditioning on one specific head | Those two heads share redundant info |
| CMI increases with conditioning (interaction info) | Synergistic computation requiring both heads |

## Connection to other frameworks

CMI directly feeds into [C04 (PID)](/framework/metrics/information/c04-pid/) which formalizes the redundancy/synergy decomposition. Low CMI combined with high causal effect under [C08 (OCSE)](/framework/metrics/information/c08-ocse/) would be paradoxical — it would mean a component is redundant informationally but not functionally, suggesting backup circuits. See also the [causal pillar](/framework/metrics/causal/) for testing whether CMI tracks with intervention effects.
