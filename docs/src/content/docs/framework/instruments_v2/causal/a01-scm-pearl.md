---
title: "A01 — SCM / Pearl Causal Hierarchy"
description: "Structural causal models and the do-calculus as the formal language for circuit claims."
---

# A01 — Structural Causal Models (Pearl Causal Hierarchy)

This framework asks: **can the circuit's behavior be written as a causal model that survives intervention?**

Activation patching and causal scrubbing both implement Pearl's do-calculus — they differ in which nodes of the causal graph they intervene on and how they evaluate the circuit hypothesis against the result. The SCM framework is not an instrument itself; it is the formal language that makes activation patching and causal scrubbing *mean* something rather than being ad hoc measurement procedures.

A structural causal model consists of endogenous variables \( X_1, \ldots, X_n \), exogenous noise \( U_1, \ldots, U_n \), and structural equations \( X_i = f_i(\text{pa}(X_i), U_i) \). An intervention \( do(X_i = v) \) removes the structural equation for \( X_i \) and replaces it with constant \( v \), breaking all incoming edges. This distinguishes intervention from conditioning: conditioning asks what correlates with outcome \( Y \); intervention asks what \( Y \) would be if we *forced* \( X_i = v \).

In a transformer, component activations at each layer and position are endogenous variables. The structural equations are attention and MLP computations. An ablation — zero, mean, or resample — is a \( do \) intervention. A circuit hypothesis is a claim about the topology of the causal graph: that a specific subset of components mediates the model's task behavior.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | Structural causal models, do-calculus, three-rung hierarchy |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Residual stream as causal graph; circuits as sub-graphs |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | First systematic application of activation patching to discover a circuit |
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC: automated circuit discovery via causal graph pruning |

## The causal hierarchy in MI

Pearl's three rungs map directly onto MI methodology. **Rung 1 (Association):** probing, linear regression, feature visualization — they identify co-variation but cannot establish that intervention changes output. **Rung 2 (Intervention):** activation patching, mean ablation, resample ablation — they manipulate structural equations and measure consequences. **Rung 3 (Counterfactual):** causal scrubbing and interchange interventions — they require specifying a complete counterfactual input and verifying the circuit hypothesis predicts the outcome.

Most MI work operates on Rung 2. Causal scrubbing targets Rung 3 and is a strictly stronger test: it asks not just whether intervention changes output, but whether it changes output *in the way the circuit predicts*.

## Instruments under A01

### C2 — Activation Patching (`02_activation_patching.py`)

For each component \( c \), measures the fraction of the clean-to-corrupted logit-difference gap restored by patching that component's activation:

\[
AP(c) = \frac{LD_{\text{patched at } c} - LD_{\text{corrupted}}}{LD_{\text{clean}} - LD_{\text{corrupted}}}
\]

This is the standard faithfulness numerator from Wang et al. (2022). The intervention is a Rung-2 do-intervention: \( do(X_c = x_c^{\text{clean}}) \) while the rest of the model runs on corrupted input.

**What it establishes:** Necessity-oriented attribution — components with high \( AP(c) \) are causally important. The score is ablation-method-dependent (Miller et al. 2024) and prompt-distribution-dependent.

**What it does not establish:** Sufficiency, specificity without a random baseline, or uniqueness of the component subset.

**Usage:**
```
uv run python 02_activation_patching.py --tasks ioi sva --n-prompts 40
```

### C4 — Causal Scrubbing (`04_causal_scrubbing.py`)

Tests a full circuit hypothesis \( H \) under the strict scrubbing criterion. For each node, resamples activations from a *compatible* input and measures KL divergence from the clean output:

\[
CS(H) = \mathbb{E}\left[ D_{\text{KL}}\left( P_{\text{model}}(\cdot \mid x) \;\|\; P_{\text{scrubbed}}(\cdot \mid x, H) \right) \right]
\]

A circuit passing causal scrubbing with low KL is *explanatorily complete* — every activation that matters is accounted for by the hypothesis.

**What it establishes:** Sufficiency of the circuit hypothesis at the causal-variable level.

**What it does not establish:** Uniqueness of the hypothesis or correctness of the named causal variables.

**Usage:**
```
uv run python 04_causal_scrubbing.py --tasks ioi --n-prompts 20
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High AP for circuit, low for non-circuit | Components are causally load-bearing (necessary) |
| Low KL under causal scrubbing | Circuit hypothesis is explanatorily sufficient |
| High AP, high KL | Components matter but hypothesis mis-specifies *how* |
| Low AP, low KL | Distributed mechanism — hypothesis passes but individual components aren't decisive |

