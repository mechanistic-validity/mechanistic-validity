---
title: "B05 — Norm Trajectory"
description: "Spectral norm ratios tracking signal amplification through circuit components."
---

# B05 — Norm Trajectory

This framework asks: **how does signal magnitude evolve through the circuit, and do circuit components amplify their inputs more than non-circuit components?**

The spectral norm of a weight matrix bounds the maximum amplification it can apply to any input direction. By tracking spectral norms layer-by-layer through a circuit, we obtain a "norm trajectory" that reveals where the circuit amplifies signal and where it attenuates noise. A well-structured circuit should show systematic norm differences between circuit and non-circuit components — the circuit amplifies task-relevant directions while non-circuit components are neutral or attenuating.

This metric connects weight-level structure to information flow: high spectral norm in a circuit component is a necessary (though not sufficient) condition for that component to have large causal effect. It provides a structural prior for which components *could* be important, complementing the causal measurements that determine which components *are* important.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Vershynin, *High-Dimensional Probability*](https://doi.org/10.1017/9781108231596) | 2018 | Spectral norm as operator norm; concentration inequalities |
| [Sankararaman et al., arXiv 2301.12971](https://arxiv.org/abs/2301.12971) | 2023 | Norm growth and signal propagation in transformers |
| [He et al., arXiv 2305.19778](https://arxiv.org/abs/2305.19778) | 2023 | Residual stream norm dynamics during training |
| [Noci et al., arXiv 2310.17813](https://arxiv.org/abs/2310.17813) | 2023 | Signal propagation and effective depth via spectral analysis |

## Core concept

The spectral norm of a matrix \( W \) is its largest singular value:

\[
\| W \|_2 = \sigma_1(W) = \max_{\|x\|=1} \|Wx\|
\]

For an attention head's OV circuit, the spectral norm bounds how much the head can amplify any residual stream direction. The norm ratio between circuit and non-circuit heads provides a structural signal-to-noise measure:

\[
R_{\text{norm}} = \frac{\text{mean}(\| W_{OV}^{\text{circuit}} \|_2)}{\text{mean}(\| W_{OV}^{\text{non-circuit}} \|_2)}
\]

A ratio significantly above 1 indicates that circuit components have greater amplification capacity. Tracking this ratio across layers reveals where in the network the circuit concentrates its signal power.

The norm trajectory \( [\| W^{(l)} \|_2]_{l=0}^{L} \) through successive circuit components also reveals potential instabilities: if norms grow exponentially, small input perturbations get amplified, making the circuit sensitive to intervention (which connects to causal findings).

## Metrics under B05

### Spectral Norm Ratio (`18_weight_extended.py`)

Computes \( \| W_{OV} \|_2 \) and \( \| W_{QK} \|_2 \) for every attention head. Reports: (1) per-head spectral norms, (2) circuit vs. non-circuit ratio \( R_{\text{norm}} \), (3) layer-by-layer trajectory for circuit heads.

**What it establishes:** Whether circuit components have greater signal amplification capacity in their weight matrices.

**What it does not establish:** Whether this capacity is utilized on task inputs — a high-norm head may amplify irrelevant directions.

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| \( R_{\text{norm}} > 1.5 \) | Circuit heads have substantially higher amplification capacity |
| Norm peaks at specific layers | Circuit concentrates signal power at identifiable processing stages |
| Flat norm trajectory | No structural differentiation in amplification — circuit boundary may lack weight support |
| High norm + high activation patching score | Structural capacity aligns with causal importance |

