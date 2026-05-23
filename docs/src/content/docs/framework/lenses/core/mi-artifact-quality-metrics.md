---
title: "MI Artifact Quality Metrics"
description: "Evaluation metrics for learned decompositions: SAE feature quality, transcoder composability, and crosscoder validation."
---

# MI Artifact Quality Metrics

This page documents metrics that evaluate the quality, interpretability, and validity of learned decomposition artifacts -- SAE features, transcoder circuits, and crosscoder representations. These metrics test whether the artifacts produced by a decomposition method are faithful, interpretable, and free from known pathologies.

---

## SAE and Feature-Level Quality

These metrics evaluate whether individual features from sparse autoencoders or similar decompositions are interpretable, stable, and semantically valid.

---

### AQ01 -- Automated Interpretability

**ID.** `AQ01.autointerp` | **File.** `EX3_autointerp.py`

**What it computes.** Tests whether artifact features have human-interpretable descriptions that predict activation patterns, via detection (can a judge identify activating examples?), fuzzing (do synthetic inputs activate the feature?), and intervention (does ablation produce predicted changes). Falls back to activation-statistics proxies (monosemanticity, sparsity) without an LLM API.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** Proxy-based; varies by sub-metric.

**Reference.** Bills et al. (2023); Templeton et al. (2024).

---

### AQ02 -- Feature Absorption

**ID.** `AQ02.feature_absorption` | **File.** `EX4_feature_absorption.py`

**What it computes.** Detects the absorption pathology where a parent feature fails to fire because a more specific child feature absorbs its activation (e.g., an "A" feature fails on "Attention" because an "Attention"-concept feature captures the signal instead).

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `absorption_rate < 0.15` (lower is better)

**Reference.** Klindt, Bloom et al. (2025), NeurIPS 2025 Oral.

---

### AQ03 -- NLA Reconstruction Fidelity

**ID.** `AQ03.nla_reconstruction` | **File.** `EX5_nla_reconstruction.py`

**What it computes.** Tests whether a feature's activation pattern is coherent enough to survive a compress-decompress cycle by identifying top-k activating positions, reconstructing a concept direction via PCA, and comparing against the original feature direction.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `mean_roundtrip_fidelity > 0.3`

**Reference.** Anthropic (2026), transformer-circuits.pub/2026/nla/.

---

### AQ04 -- TopK SAE Scaling

**ID.** `AQ04.topk_scaling` | **File.** `EX7_topk_scaling.py`

**What it computes.** Three complementary SAE quality sub-metrics: (a) planted feature recovery (whether SAE directions recover planted random unit directions), (b) downstream effect sparsity (L0 of per-feature logit effects through W_U), and (c) activation explainability (kurtosis and sparsity of activation patterns).

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `recovery > 0.3; sparsity > 0.8; explainability > 0.5`

**Reference.** Gao, Dupre la Tour et al. (2025), ICLR 2025 Oral.

---

### AQ05 -- RoPE Massive Value Filter

**ID.** `AQ05.rope_massive_value` | **File.** `EX8_rope_massive_value.py`

**What it computes.** Detects SAE features contaminated by RoPE positional encoding artifacts (massive values in Q/K projections) rather than representing genuine semantic features, via alignment with RoPE frequency bases, position-dependence, and magnitude checks.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `clean_fraction > 0.9`

**Reference.** Jin et al. (2025), ICML 2025.

---

### AQ06 -- Latent Self-Consistency

**ID.** `AQ06.latent_self_consistency` | **File.** `EX11_self_consistency.py`

**What it computes.** Tests SAE encoder-decoder stability via iterated encode-decode roundtrips, measuring whether latent representations converge (stable fixed point) or drift (saddle point instability) across multiple cycles.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `single_roundtrip_consistency > 0.85`

**Reference.** Buchanan et al. (2026), ICLR 2026.

---

### AQ07 -- GradSAE Causal Influence Score

**ID.** `AQ07.gradsae_causal_influence` | **File.** `113_gradsae_causal.py`

**What it computes.** For each SAE latent, computes activation times gradient (of output probability w.r.t. latent activation) to produce per-feature causal influence scores. The causal dissociation rate measures how many top-by-activation features are NOT in the top-by-causal-influence ranking.

**Evidence family.** Internal (I1 Necessity, I2 Sufficiency)

**Pass threshold.** `causal_dissociation_rate < 0.3`

**Reference.** Shu et al. (2025), EMNLP 2025.

---

### AQ08 -- Output-Centric Description Score

**ID.** `AQ08.output_centric_description` | **File.** `116_output_centric.py`

**What it computes.** Compares input-centric descriptions (top activating tokens) against output-centric descriptions (decoder direction projected through the unembedding matrix W_U) to detect correlational features that lack causal execution validity. Classifies features into a 2x2 encoding/execution grid (HH, HL, LH, LL).

**Evidence family.** Internal (I2 Sufficiency), Construct (C5 Convergent Validity)

**Pass threshold.** `encoding_execution_agreement > 0.2`

**Reference.** Gur-Arieh et al. (2025), ACL 2025.

---

### AQ09 -- Neuronpedia Agreement

**ID.** `AQ09.neuronpedia_agreement` | **File.** `126_neuronpedia_agreement.py`

**What it computes.** Cross-references weight-based top-k token analysis (decoder direction projected through W_U) against Neuronpedia's automated feature descriptions and top activating tokens, measuring external convergent validity via Jaccard overlap.

**Evidence family.** Construct (C5 Convergent Validity)

**Pass threshold.** `neuronpedia_weight_agreement > 0.2`

**Reference.** Neuronpedia community infrastructure (neuronpedia.org).

---

## Transcoder Quality

These metrics evaluate validity properties specific to transcoders -- models that decompose MLP computation into input-output feature circuits rather than internal representations.

---

### AQ10 -- Transcoder Circuit Composability

**ID.** `AQ10.transcoder_composability` | **File.** `139_transcoder_composability.py`

**What it computes.** Tests whether transcoder decoder directions compose correctly with downstream weight matrices by comparing weight-predicted effects (decoder direction projected through next layer's input weights) against patching-measured effects (observed change when zeroing a feature's contribution).

**Evidence family.** Internal (C2 Structural Plausibility, I2 Sufficiency)

**Pass threshold.** `composability_score > 0.5`

**Reference.** Dunefsky, Chanin, Nanda (2024), arXiv:2406.11944.

---

### AQ11 -- Transcoder vs. SAE Feature Agreement

**ID.** `AQ11.transcoder_sae_agreement` | **File.** `140_transcoder_sae_agreement.py`

**What it computes.** Compares transcoder features (MLP input-to-output decomposition) against SAE features (MLP output state decomposition) at the same layer, measuring direction similarity and top-activating-token overlap of matched feature pairs.

**Evidence family.** Construct (C5 Convergent Validity)

**Pass threshold.** `transcoder_sae_agreement > 0.3`

**Reference.** Dunefsky, Chanin, Nanda (2024), arXiv:2406.11944.

---

### AQ12 -- Transcoder Decomposition Fraction

**ID.** `AQ12.transcoder_decomposition` | **File.** `141_transcoder_decomposition.py`

**What it computes.** Measures what fraction of MLP output variance is captured by the input-dependent sparse features versus the input-invariant bias/mean term, testing whether sparse features do real computational work or the MLP is bias-dominated.

**Evidence family.** Construct (M6 Construct Coverage), Measurement (C2 Structural Plausibility)

**Pass threshold.** `input_dependent_fraction > 0.6`

**Reference.** Dunefsky, Chanin, Nanda (2024), arXiv:2406.11944.

---

## Crosscoder Quality

These metrics evaluate validity properties specific to crosscoders -- models that jointly decompose representations across multiple layers or multiple models.

---

### AQ13 -- Crosscoder Cross-Layer Persistence

**ID.** `AQ13.crosscoder_persistence` | **File.** `142_crosscoder_persistence.py`

**What it computes.** Tests whether crosscoder features claimed to span multiple layers actually activate coherently at all those layers by measuring Pearson correlation of per-position projection magnitudes between adjacent layers.

**Evidence family.** Construct (C2 Structural Plausibility), Measurement (M2 Invariance)

**Pass threshold.** `cross_layer_coherence > 0.5`

**Reference.** Lindsey et al. (2024), transformer-circuits.pub.

---

### AQ14 -- Crosscoder Model Diffing Validity

**ID.** `AQ14.crosscoder_model_diff` | **File.** `143_crosscoder_model_diff.py`

**What it computes.** Validates the shared/exclusive feature classification by checking activation behavior: exclusive features should have near-zero activation in the excluded model, shared features should activate similarly in both models.

**Evidence family.** Internal (C3 Task Specificity), Measurement (M3 Baseline Separation)

**Pass threshold.** `diff_classification_accuracy > 0.7`

**Reference.** Lindsey et al. (2024), transformer-circuits.pub.

---

### AQ15 -- Crosscoder L1 Artifact Detection

**ID.** `AQ15.crosscoder_artifact_detection` | **File.** `144_crosscoder_artifact_detection.py`

**What it computes.** Detects the Muto et al. sparsity artifact where L1-penalized crosscoder training inflates apparent feature exclusivity, reclassifying genuinely shared features as exclusive. Compares decoder-norm classification against activation-based classification.

**Evidence family.** Measurement (M6 Construct Coverage), Internal (I5 Confound Control)

**Pass threshold.** `artifact_rate < 0.15`

**Reference.** Muto et al. (2025), NeurIPS 2025.

---

## Summary Table

| ID | Name | File | Evidence Family | Threshold |
|---|---|---|---|---|
| AQ01 | Automated Interpretability | `EX3_autointerp.py` | Measurement | proxy-based |
| AQ02 | Feature Absorption | `EX4_feature_absorption.py` | Measurement | `< 0.15` |
| AQ03 | NLA Reconstruction Fidelity | `EX5_nla_reconstruction.py` | Measurement | `> 0.3` |
| AQ04 | TopK SAE Scaling | `EX7_topk_scaling.py` | Measurement | varies |
| AQ05 | RoPE Massive Value Filter | `EX8_rope_massive_value.py` | Measurement | `> 0.9` |
| AQ06 | Latent Self-Consistency | `EX11_self_consistency.py` | Measurement | `> 0.85` |
| AQ07 | GradSAE Causal Influence | `113_gradsae_causal.py` | Internal | `< 0.3` |
| AQ08 | Output-Centric Description | `116_output_centric.py` | Internal / Construct | `> 0.2` |
| AQ09 | Neuronpedia Agreement | `126_neuronpedia_agreement.py` | Construct | `> 0.2` |
| AQ10 | Transcoder Composability | `139_transcoder_composability.py` | Internal | `> 0.5` |
| AQ11 | Transcoder-SAE Agreement | `140_transcoder_sae_agreement.py` | Construct | `> 0.3` |
| AQ12 | Transcoder Decomposition | `141_transcoder_decomposition.py` | Construct | `> 0.6` |
| AQ13 | Crosscoder Persistence | `142_crosscoder_persistence.py` | Construct | `> 0.5` |
| AQ14 | Crosscoder Model Diff | `143_crosscoder_model_diff.py` | Internal / Measurement | `> 0.7` |
| AQ15 | Crosscoder Artifact Detection | `144_crosscoder_artifact_detection.py` | Measurement | `< 0.15` |

---

## Relationship to Other Pages

For circuit faithfulness metrics (CLT attribution, cross-prompt consistency, minimality), see [MI Faithfulness Metrics](/framework/lenses/core/mi-faithfulness-metrics). For safety-relevant construct validation, see [MI Safety Metrics](/framework/lenses/core/mi-safety-metrics). For the overall MI lens framework, see [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability).

---

<!-- REDISTRIBUTION NOTE: The following metrics from mi-evaluation-metrics.md are cross-model or miscellaneous
     and belong on their natural evidence-family pages. To be redistributed in a follow-up:

     - EX12 (107 MAS) -> mi-representational-metrics.md (cross-model alignment)
     - EX14 (109 activation reasoning) -> mi-causal-metrics.md
     - EX16 (112 layer navigator) -> mi-representational-metrics.md
     - EX19 (121 functional localizer) -> mi-causal-metrics.md
     - EX21 (123 latent reasoning) -> mi-causal-metrics.md
     - EX22 (124 semantic hub) -> mi-representational-metrics.md
     - EX31 (130 NLA semantic validity) -> mi-behavioral-metrics.md
     - EX32 (131 weight-sparse circuit) -> mi-structural-metrics.md
     - EX26 (133 atlas alignment) -> mi-representational-metrics.md
     - EX27 (134 MOT alignment) -> mi-representational-metrics.md
     - EX6 (rule descriptions) -> mi-structural-metrics.md
-->
