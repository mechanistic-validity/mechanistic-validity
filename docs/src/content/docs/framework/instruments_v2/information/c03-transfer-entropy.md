---
title: "C03 — Transfer Entropy"
description: "Measures directed information flow between circuit components across layers."
---

# C03 — Transfer Entropy

This framework asks: **Does information flow directionally from one circuit component to another across the computation?**

Transfer entropy (TE) quantifies the directed, time-asymmetric information flow from a source process to a target process. In transformer circuits, "time" corresponds to layer depth: information flows from earlier layers to later layers through the residual stream. TE measures how much knowing the activation of an earlier component reduces uncertainty about a later component, beyond what the later component's own history provides.

This is crucial for circuit discovery because it distinguishes genuine information transmission from mere correlation. Two heads may be correlated because they both read from the same residual stream position, but TE identifies which one actually informs the other.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Schreiber, "Measuring Information Transfer"](https://doi.org/10.1103/PhysRevLett.85.461) | 2000 | Original transfer entropy definition |
| [Cover & Thomas, *Elements of Information Theory*](https://doi.org/10.1002/047174882X) | 2006 | Directed information and causal conditioning |
| [Barnett et al., "Granger Causality and Transfer Entropy Are Equivalent for Gaussian Variables"](https://doi.org/10.1103/PhysRevLett.103.238701) | 2009 | Equivalence to Granger causality under Gaussianity |
| [Bossomaier et al., *An Introduction to Transfer Entropy*](https://doi.org/10.1007/978-3-319-43222-9) | 2016 | Comprehensive treatment with estimation methods |

## Core concept

Transfer entropy from process \( X \) to process \( Y \) is defined as:

\[ T_{X \to Y} = I(Y_t; X_{t-1} \mid Y_{t-1}) \]

In the circuit context, let \( X_\ell \) be a component's output at layer \( \ell \) and \( Y_{\ell+k} \) a downstream component at layer \( \ell+k \). The transfer entropy becomes:

\[ T_{X \to Y} = H(Y_{\ell+k} \mid Y_{\ell+k-1}) - H(Y_{\ell+k} \mid Y_{\ell+k-1}, X_\ell) \]

This is positive only if knowing \( X_\ell \) reduces uncertainty about \( Y_{\ell+k} \) beyond what the target's own earlier state provides. Asymmetry (\( T_{X \to Y} \neq T_{Y \to X} \)) reveals directed information flow.

## Instruments under C03

### OCSE Script (`07_ocse.py`)

Observational causal sensitivity estimation captures directed influence by measuring how perturbations to one component's activation propagate to downstream components — a finite-difference analogue of transfer entropy.

**What it establishes:** Directed information flow between circuit components across layers.
**What it does not establish:** Whether the transferred information is task-relevant (high TE could reflect noise propagation).

**Usage:**
```
uv run python 07_ocse.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|---|
| High \( T_{X \to Y} \), low \( T_{Y \to X} \) | Genuine directed flow from X to Y |
| High TE in both directions | Shared input or confounded relationship |
| TE peaks at specific layer gaps | Information is transmitted with characteristic depth |
| Zero TE despite high MI | Components share info via a common cause, not direct transmission |

## Connection to other frameworks

Transfer entropy is the information-theoretic analogue of [C07 (Granger Causality)](/framework/instruments_v2/information/c07-granger-causality/) — they are equivalent for Gaussian processes. Where TE finds directed flow, [C08 (OCSE)](/framework/instruments_v2/information/c08-ocse/) validates it via observational perturbation, and [C09 (NOTEARS)](/framework/instruments_v2/information/c09-notears/) attempts to recover the full DAG structure. The [causal pillar](/framework/instruments_v2/causal/) then tests whether these directed flows are necessary via intervention.
