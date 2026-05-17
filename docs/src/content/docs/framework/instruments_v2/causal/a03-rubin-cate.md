---
title: "A03 — Rubin Potential Outcomes / CATE"
description: "Conditional Average Treatment Effects applied to circuit components: estimating heterogeneous causal effects across input subpopulations."
---

# A03 — Rubin Potential Outcomes / CATE

This framework asks: **how much does including a component in the circuit *cause* the output to change, and does this effect vary across input subpopulations?**

The Rubin causal model defines causation through potential outcomes: for each unit (here, each input prompt), there is a potential outcome \( Y(1) \) under treatment (component active) and \( Y(0) \) under control (component ablated). The individual treatment effect \( Y(1) - Y(0) \) is never directly observed for both conditions simultaneously, but the Conditional Average Treatment Effect (CATE) can be estimated across subgroups. This brings the statistical machinery of heterogeneous treatment effects to MI — asking not just "does this component matter on average?" but "for which inputs does it matter most?"

Where activation patching (A01) gives a single average score per component, CATE estimation reveals that a head might be critical for long-distance subject-verb agreement but irrelevant for adjacent cases. This heterogeneity is invisible to aggregate faithfulness metrics and critical for understanding when circuits activate versus remain dormant.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Rubin, "Estimating Causal Effects of Treatments"](https://doi.org/10.2307/2529685) | 1974 | Potential outcomes framework; fundamental problem of causal inference |
| [Wager & Athey, "Estimation and Inference of Heterogeneous Treatment Effects"](https://doi.org/10.1080/01621459.2017.1319839) | 2018 | Causal forests for CATE estimation with valid confidence intervals |
| [Wang et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | Activation patching as treatment/control design |
| [Miller et al., arXiv 2407.08734](https://arxiv.org/abs/2407.08734) | 2024 | Ablation-method dependence of causal effect estimates in circuits |

## Core concept: heterogeneous treatment effects

Define treatment \( T = 1 \) as "component \( c \) runs normally" and \( T = 0 \) as "component \( c \) is ablated." For input \( x \) with covariates \( W(x) \) (e.g., token distance, syntax type, prompt length), the CATE is:

\[
\tau(w) = \mathbb{E}[Y(1) - Y(0) \mid W = w]
\]

When \( \tau(w) \) varies substantially across covariate strata, the component's causal role is context-dependent. A causal forest partitions the input space into regions of homogeneous treatment effect, revealing which structural features of the input determine whether the component is load-bearing.

This directly addresses a limitation of standard activation patching: a head with average \( AP = 0.3 \) might have \( \tau = 0.9 \) on syntactically complex inputs and \( \tau = 0.0 \) on simple ones. The aggregate hides the mechanism's activation conditions.

## Instruments under A03

### C6 — CATE Estimation (`06_cate.py`)

Estimates conditional average treatment effects for each circuit component using ablation as treatment assignment. Partitions inputs by structural covariates and reports per-stratum effects:

\[
\hat{\tau}(w) = \frac{1}{|S_w|} \sum_{x \in S_w} \left[ LD(x, T=1) - LD(x, T=0) \right]
\]

where \( S_w \) is the set of inputs in covariate stratum \( w \). Also fits a causal forest for continuous covariate spaces.

**What it establishes:** Which input features modulate a component's causal importance. Identifies the activation conditions for circuit engagement.

**What it does not establish:** Why the effect is heterogeneous (mechanism), or whether the covariates chosen are the right partitioning variables.

**Usage:**
```
uv run python 06_cate.py --tasks ioi sva --n-prompts 40
```

### C25 — Intervention Specificity (`25_intervention_specificity.py`)

Tests whether a component's causal effect is specific to the target task or whether ablation causes collateral damage to unrelated behaviors. Measures the ratio of on-task effect to off-task disruption:

\[
\text{Specificity}(c) = \frac{\tau_{\text{on-task}}(c)}{\tau_{\text{on-task}}(c) + \tau_{\text{off-task}}(c)}
\]

**What it establishes:** Whether the component is specialized (high specificity) or polyfunctional (low specificity).

**What it does not establish:** The full set of tasks the component participates in.

**Usage:**
```
uv run python 25_intervention_specificity.py --tasks ioi sva --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Uniform CATE across strata | Component is always active (constitutive role) |
| High CATE in one stratum, near-zero elsewhere | Context-dependent activation; circuit engages conditionally |
| High specificity (> 0.8) | Component is task-specialized |
| Low specificity (< 0.5) | Polyfunctional component; ablation disrupts multiple behaviors |

