---
title: "E02 — Linear Probe"
description: "Measures whether a target variable is linearly decodable from intermediate representations at each layer."
---

# E02 — Linear Probe

This framework asks: **Is the information about a high-level variable linearly accessible in the model's representations?**

Linear probing trains a simple linear classifier or regressor on frozen intermediate activations to predict a target variable. High probe accuracy indicates that the representation geometrically separates the variable's values in a linearly decodable way. This is a necessary (but not sufficient) condition for the model to *use* that information downstream.

Probing is the representational complement to causal intervention: it measures what information is *present*, while IIA (E01) measures what information is *used*. The gap between probe accuracy and IIA reveals "latent" information — decodable but causally inert features stored in the residual stream.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Alain & Bengio, "Understanding intermediate layers using linear classifier probes"](https://arxiv.org/abs/1610.01644) | 2016 | Introduced linear probes for representation analysis |
| [Hewitt & Manning, "A Structural Probe for Finding Syntax in Word Representations"](https://arxiv.org/abs/1906.02715) | 2019 | Structural probes for tree distance |
| [Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances"](https://arxiv.org/abs/2102.12452) | 2021 | Survey of probing methodology and pitfalls |
| [Voita & Titov, "Information-Theoretic Probing with MDL"](https://arxiv.org/abs/2003.12298) | 2020 | MDL-based probe complexity as representation quality |

## Core concept

A linear probe learns \( W \in \mathbb{R}^{c \times d} \) and \( b \in \mathbb{R}^c \) minimizing:

\[
\mathcal{L}_{\text{probe}} = -\sum_{i} \log \text{softmax}(W h_\ell^{(i)} + b)_{y_i}
\]

where \( h_\ell^{(i)} \) is the residual stream at layer \( \ell \) for input \( i \), and \( y_i \) is the target label. Probe accuracy reflects linear decodability; MDL-based approaches additionally penalize probe complexity to avoid overfitting to memorization.

The critical distinction: probe accuracy is an *upper bound* on what the model could extract linearly, not evidence that it does. A direction may be decodable simply because the embedding preserves input features that the model never attends to.

## Metrics under E02

### Linear Probe (`01_das_iia.py` — probe mode)

DAS with identity rotation reduces to a constrained linear probe. Running without learned rotation gives probe-only baselines.

**What it establishes:** Whether target information is linearly separable at each layer.
**What it does not establish:** Whether the model causally relies on that linear encoding.

**Usage:**
```
uv run python 01_das_iia.py --tasks ioi sva --probe-only
```

### Selectivity Baseline

Compares probe accuracy on the target variable against a control task (random labels or unrelated variable). Selectivity = target accuracy minus control accuracy.

**What it establishes:** Whether high accuracy reflects genuine structure vs. probe memorization capacity.
**What it does not establish:** Causal role of the decoded information.

**Usage:**
```
uv run python 01_das_iia.py --tasks ioi sva --probe-only --selectivity
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Probe accuracy high, IIA high | Variable linearly encoded and causally used |
| Probe accuracy high, IIA low | Information present but not causally active at that site |
| Probe accuracy low everywhere | Variable not linearly decodable; may require nonlinear read-out |
| Accuracy jumps at layer \( \ell \) | Layer \( \ell \) computes or consolidates the variable |
| High selectivity (> 0.3) | Genuine structure, not probe capacity artifact |

