---
title: "B09 — Weight Classifier"
description: "Training classifiers on weight matrices to predict circuit membership and comparing to activation-based circuit discovery."
---

# B09 — Weight Classifier

This framework asks: **can circuit membership be predicted from weight structure alone, and does the weight-derived circuit agree with activation-based methods?**

A weight classifier takes raw weight matrices (or derived features like SVD spectra, norms, alignment scores) as input and predicts whether a component belongs to a task circuit. If such a classifier achieves high accuracy, it demonstrates that circuit membership is encoded in static weight structure — the model's architecture contains enough information to identify circuits without running any forward passes. Comparing weight-derived circuits to activation-based circuits (e.g., from EAP or activation patching) quantifies the agreement between structural and functional perspectives.

This instrument represents the ultimate test of structural circuit analysis: if weights alone suffice to identify circuits, then the weight-level instruments (B01-B08) are not merely descriptive but genuinely predictive. If weight classifiers fail, it suggests that circuit identity emerges from activation dynamics rather than static structure.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Syed et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | ACDC automated circuit discovery — provides ground-truth circuits for training classifiers |
| [Nanda et al., arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | 2022 | Manual circuit identification providing labeled training data |
| [Lieberum et al., arXiv 2304.01421](https://arxiv.org/abs/2304.01421) | 2023 | Edge attribution patching (EAP) as efficient alternative to full patching |
| [Conmy et al., arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | 2023 | Benchmarking circuit discovery methods — framework for comparison |
| [Marks et al., arXiv 2403.19647](https://arxiv.org/abs/2403.19647) | 2024 | Sparse feature circuits — weight structure predicts feature membership |

## Core concept

The weight classifier pipeline has three stages:

1. **Feature extraction:** For each attention head, compute a feature vector from its weight matrices:
\[
\phi(h) = [\text{erank}(W_{QK}^h), \text{erank}(W_{OV}^h), \|W_{OV}^h\|_2, \sigma_1/\sigma_2, \text{align}(h, h_{\text{ref}}), \ldots]
\]

2. **Training:** Using circuits identified by activation-based methods (EAP, ACDC, manual annotation) as ground truth, train a classifier (logistic regression, decision tree, or MLP) on \( \phi(h) \to \{0, 1\} \).

3. **Evaluation:** Measure agreement between weight-predicted circuits and activation-derived circuits via Jaccard similarity:
\[
J(C_{\text{weight}}, C_{\text{EAP}}) = \frac{|C_{\text{weight}} \cap C_{\text{EAP}}|}{|C_{\text{weight}} \cup C_{\text{EAP}}|}
\]

High Jaccard indicates that weight structure is predictive of functional role; low Jaccard suggests the two perspectives capture different aspects of circuit organization.

## Instruments under B09

### Weight-EAP Jaccard (`28_weight_eap_jaccard.py`)

Computes Jaccard similarity between circuits derived from weight-level features and circuits derived from Edge Attribution Patching. Reports: (1) per-task Jaccard scores, (2) which weight features are most predictive, (3) confusion matrix showing where the methods disagree.

**What it establishes:** Quantitative agreement between weight-based and activation-based circuit identification.

**What it does not establish:** Which method is "correct" when they disagree — both may identify valid but different aspects of the circuit.

**Usage:**
```
uv run python 28_weight_eap_jaccard.py --tasks ioi sva
```

### Incremental Validity (`36_incremental_validity.py`)

Tests whether weight-derived circuit features add predictive value beyond activation-based baselines. Fits a model predicting task performance from activation features alone, then adds weight features and measures improvement.

**What it establishes:** Whether weight structure provides *additional* information beyond what activations already reveal.

**What it does not establish:** Sufficiency of either method in isolation.

**Usage:**
```
uv run python 36_incremental_validity.py --tasks ioi sva
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Jaccard > 0.7 | Strong agreement — weight structure predicts activation-based circuits |
| Jaccard < 0.3 | Methods capture different structure — complementary rather than redundant |
| Incremental validity significant | Weight features add information beyond activation baselines |
| Weight classifier accuracy > 0.85 | Circuit membership is largely determined by static weight structure |
| Specific weight features dominate | Identifies which structural properties (norm, rank, alignment) matter most |

