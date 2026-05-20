---
title: "E08 — Participation Ratio"
description: "Quantifies how many dimensions are effectively active in a representation via the normalized squared eigenvalue sum."
---

# E08 — Participation Ratio

This framework asks: **How many dimensions carry meaningful variance — is the representation concentrated or distributed?**

The participation ratio (PR) is a single scalar that summarizes spectral concentration. Unlike threshold-based measures (e.g., "dimensions for 90% variance"), PR provides a smooth, threshold-free estimate of effective dimensionality. Originally from condensed matter physics (localization of wavefunctions), PR elegantly captures whether a representation is dominated by a few directions (low PR) or spreads evenly across many (high PR).

For circuit analysis, PR distinguishes heads that perform rank-1-like operations (low PR, interpretable as single-feature detectors) from heads that implement genuinely high-dimensional transformations (high PR, likely performing complex combinatorial operations).

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Bell & Dean, "Atomic vibrations in vitreous silica"](https://doi.org/10.1080/14786437008238472) | 1970 | Original participation ratio in physics |
| [Gao et al., "A theory of multineuron dimensionality, dynamics and measurement"](https://doi.org/10.1101/214262) | 2017 | PR for neural population dimensionality |
| [Litwin-Kumar et al., "Optimal Degrees of Synaptic Connectivity"](https://doi.org/10.1016/j.neuron.2017.01.030) | 2017 | PR in biological neural circuits |
| [Recanatesi et al., "Dimensionality compression and expansion in deep neural networks"](https://arxiv.org/abs/1906.00443) | 2019 | PR dynamics through network layers |

## Core concept

Given the eigenvalue spectrum \( \lambda_1, \ldots, \lambda_d \) of the activation covariance matrix, the participation ratio is:

\[
\text{PR} = \frac{\left(\sum_{i=1}^d \lambda_i\right)^2}{\sum_{i=1}^d \lambda_i^2}
\]

PR ranges from 1 (all variance in one dimension) to \( d \) (uniform across all dimensions). It can be normalized:

\[
\text{PR}_{\text{norm}} = \frac{\text{PR}}{d} \in [1/d, \; 1]
\]

The inverse participation ratio (IPR = 1/PR) measures *localization* — how concentrated the variance is. For a rank-\( k \) representation with equal eigenvalues, PR = \( k \) exactly. For exponentially decaying spectra, PR is dominated by the few largest eigenvalues.

## Metrics under E08

### Per-Head Participation Ratio (`participation_ratio.py`)

Computes PR for each circuit head's output activations across a corpus, reporting both raw PR and normalized PR.

**What it establishes:** The effective dimensionality of each head's operation — low PR means rank-deficient, high PR means full-rank.
**What it does not establish:** What the active dimensions encode (combine with E01/E02 for interpretation).

**Usage:**
```
uv run python participation_ratio.py --tasks ioi sva
```

### PR Dynamics

Tracks how PR evolves across layers, revealing dimensionality compression/expansion dynamics.

**What it establishes:** Where the network compresses (decreasing PR) or expands (increasing PR) its representational capacity.
**What it does not establish:** Whether compression is lossy or lossless.

**Usage:**
```
uv run python participation_ratio.py --tasks ioi sva --dynamics
```

## Reading the scores

| Pattern | What it means |
|---|---|
| PR ~ 1-3 | Near rank-1; head implements a simple projection or feature detector |
| PR ~ 10-30 (for d=64) | Moderate dimensionality; structured but non-trivial operation |
| PR ~ d | Full-rank; head uses all available dimensions |
| PR decreases through layers | Progressive compression toward a decision subspace |
| PR varies across heads at same layer | Functional specialization — some heads simple, others complex |

