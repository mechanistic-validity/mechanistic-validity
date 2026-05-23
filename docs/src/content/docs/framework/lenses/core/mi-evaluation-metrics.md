---
title: "MI Evaluation Metrics"
description: "Evaluation metrics from the mechanistic interpretability lens: circuit faithfulness, feature quality, safety analysis, and method comparison."
---

# MI Evaluation Metrics

This page documents the 43 evaluation metrics implemented in `mechval_v2.core.mechanistic_interpretability.evaluation`. Each metric tests a specific validity property of a mechanistic interpretability claim -- circuit faithfulness, feature quality, safety-relevant construct stability, method agreement, or decomposition completeness. The metrics are grouped by the type of artifact or claim they evaluate.

---

## Circuit and Model-Level Evaluation

These metrics evaluate whether proposed circuits or model-level mechanisms are faithful, necessary, sufficient, and robust.

---

### EX12 -- Model Alignment Search (MAS)

**ID.** `EX12.mas_alignment` | **File.** `107_mas.py`

**What it computes.** Learns an invertible linear transformation between two models' activation spaces and measures bidirectional interchange intervention accuracy (IIA) -- whether causal variables can be freely exchanged between models through the learned alignment.

**Evidence family.** Construct (C5 Cross-Model Agreement)

**Pass threshold.** `alignment_score > 0.5`

**Reference.** Grant (2026), ICLR Re-Align Workshop.

---

### FH07 -- ModCirc Cross-Task Modularity

**ID.** `FH07.modcirc_modularity` | **File.** `122_modcirc.py`

**What it computes.** For each pair of tasks with defined circuits, evaluates whether a circuit discovered on task A maintains faithfulness when evaluated on task B's prompts, plus Jaccard overlap of circuit head sets.

**Evidence family.** Construct (C5 Convergent Validity), Internal (E2 Causal Sufficiency)

**Pass threshold.** `mean_cross_task_faithfulness > 0.4`

**Reference.** He et al. (2025), ICML 2025.

---

### S02 -- Adversarial Ablation Verification

**ID.** `S02.adversarial_ablation_gap` | **File.** `127_adversarial_ablation.py`

**What it computes.** Tests whether circuit heads that appear necessary under standard (mean) ablation remain necessary under adversarial ablation, where non-circuit heads are replaced with maximally disruptive values to detect false necessity.

**Evidence family.** Internal (I2 Sufficiency)

**Pass threshold.** `mean adversarial_gap < 0.3`

**Reference.** Sharkey et al. (2026), Goodfire / Apollo Research.

---

### FH08 -- Actionability Score

**ID.** `FH08.actionability` | **File.** `128_actionability.py`

**What it computes.** Measures whether circuit-level insights translate into actionable steering interventions, combining concreteness (norm ratio of circuit-derived vs. full-model steering vectors) with validation (fraction of prompts where circuit-derived steering shifts output toward the correct answer).

**Evidence family.** External (E1 Downstream Utility)

**Pass threshold.** `actionability > 0.1`

**Reference.** Orgad, Barez et al. (2026), ICML 2026.

---

### FH09 -- Surprise Reduction

**ID.** `FH09.surprise_reduction` | **File.** `129_surprise_reduction.py`

**What it computes.** Measures how much knowing the circuit reduces uncertainty about model outputs by comparing output entropy with and without the circuit's contribution (entropy increase upon circuit ablation divided by ablated entropy).

**Evidence family.** Measurement (M4 Construct Coverage)

**Pass threshold.** `surprise_reduction > 0.05`

**Reference.** ARC (2026); Hilton et al. (2026), AlignmentForum.

---

### FH10 -- Behavior vs. Capability Gap

**ID.** `FH10.behavior_capability_gap` | **File.** `130_behavior_capability_gap.py`

**What it computes.** Compares circuit faithfulness on prompts where the model gets the correct answer (capability) against prompts where the model errs (behavior), detecting circuits that only explain success but not failure.

**Evidence family.** External (E3 Generalizability)

**Pass threshold.** `behavior_gap < 0.3`

**Reference.** Steinhardt (2026), AlignmentForum.

---

### EX32a -- Weight-Sparse Circuit Completeness

**ID.** `EX32a.weight_sparse_circuit` | **File.** `131_weight_sparse_circuit.py`

**What it computes.** Iteratively prunes smallest-magnitude weights until task performance drops, then tests whether the surviving circuit is both sufficient (circuit alone recovers task performance) and necessary (removing any circuit head degrades performance).

**Evidence family.** Internal (E2 Causal Sufficiency, I1 Component Necessity)

**Pass threshold.** `sufficiency > 0.8; necessity > 0.2; circuit_size < 0.1`

**Reference.** Gao et al. (2025), OpenAI.

---

## SAE and Feature-Level Evaluation

These metrics evaluate the quality, interpretability, and validity of individual features from sparse autoencoders or similar decompositions.

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

### EX6 -- Rule-Based Feature Descriptions

**ID.** `EX6.rule_descriptions` | **File.** `EX6_rule_descriptions.py`

**What it computes.** Tests whether features can be described by formal rules (skip-gram patterns, absence rules, counting rules) rather than just exemplars. Absence rules (features firing when a token is NOT present) appear in over 25% of attention features.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `mean_rule_coverage > 0.3`

**Reference.** Friedman, Bhaskar et al. (2025), Princeton NLP.

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

### EX31 -- NLA Semantic Validity Gap

**ID.** `EX31.nla_semantic_validity` | **File.** `130_nla_semantic_validity.py`

**What it computes.** Tests whether NLA-style round-trip reconstruction accuracy implies semantic validity by comparing reconstruction fidelity (PCA recovery of feature direction from top-k activating positions) against semantic prediction accuracy (whether the reconstructed direction predicts held-out activations). Large gap reveals informational but not semantic validity.

**Evidence family.** Construct (E1 Content Validity, C5 Convergent Validity)

**Pass threshold.** `mean_semantic_validity_gap < 0.3`

**Reference.** Anthropic (2026), transformer-circuits.pub/2026/nla/.

---

## Representation and Steering Evaluation

These metrics evaluate the quality and causal relevance of representational directions used for model steering or interpretive claims.

---

### FH06 -- CoT Faithfulness

**ID.** `FH06.cot_faithfulness` | **File.** `108_cot_faithfulness.py`

**What it computes.** Detects unfaithful Chain-of-Thought reasoning by presenting paired contradictory comparison questions (A > B? and B > A?) and measuring the rate of logical contradictions, revealing post-hoc rationalization.

**Evidence family.** Internal (I2 Compositional Sufficiency)

**Pass threshold.** `contradiction_rate < 0.05`

**Reference.** Arcuschin et al. (2025), Google DeepMind, ICLR 2025.

---

### EX14 -- Activation Reasoning

**ID.** `EX14.activation_reasoning` | **File.** `109_activation_reasoning.py`

**What it computes.** Maps SAE features to logical propositions and applies symbolic composition rules (AND, OR, NOT) to test whether feature descriptions enable correct downstream reasoning about task behavior.

**Evidence family.** External (E5 Feature-Description Downstream Reasoning)

**Pass threshold.** `downstream_accuracy > 0.6`

**Reference.** Helff et al. (2026), ICLR 2026.

---

### EX16 -- LayerNavigator

**ID.** `EX16.layer_navigator` | **File.** `112_layer_navigator.py`

**What it computes.** Scores each layer's suitability for activation steering via discriminability (AUROC of mean-difference direction) times consistency (stability of per-example difference directions), identifying the optimal layer for a concept.

**Evidence family.** Measurement (M4 Sensitivity)

**Pass threshold.** `best_layer_score > 0.3`

**Reference.** Sun et al. (2025), NeurIPS 2025.

---

### EX21 -- Latent Reasoning Validity

**ID.** `EX21.latent_reasoning_validity` | **File.** `123_latent_reasoning.py`

**What it computes.** Tests whether intermediate representations claimed to encode reasoning states actually do so, via steering sensitivity (output change under targeted PCA-direction perturbation) and shortcut exploitation (accuracy on corrupted/reversed prompts).

**Evidence family.** Internal (E1 Content Validity, I2 Sufficiency), Construct (C5 Convergent Validity)

**Pass threshold.** `steering_sensitivity > 0.3; shortcut_exploitation_rate < 0.2`

**Reference.** Zhang et al. (2025), arXiv:2512.21711.

---

### EX22 -- Semantic Hub Convergence

**ID.** `EX22.semantic_hub_convergence` | **File.** `124_semantic_hub.py`

**What it computes.** Measures whether semantically equivalent inputs in different surface forms (digits vs. words, code vs. description, formal vs. informal) converge to similar representations at intermediate layers, identifying the hub layer of peak cross-form similarity.

**Evidence family.** Construct (C5 Convergent Validity)

**Pass threshold.** `hub_convergence_score > 0.5`

**Reference.** Wu et al. (2025), ICLR 2025.

---

### S01 -- Dual Mechanism Discriminant Validity

**ID.** `S01.dual_mechanism` | **File.** `125_dual_mechanism.py`

**What it computes.** Decomposes representation directions for a behavioral construct into intrinsic (baseline) and prompted (instruction-following) mechanisms, testing whether they are genuinely distinct and whether each steers independently after removing the shared component.

**Evidence family.** Construct (C4 Discriminant Validity)

**Pass threshold.** `discriminant_separation > 0.2; independent_steering_effect > 0.1`

**Reference.** Han et al. (2025), NeurIPS 2025 / ICML 2026.

---

### EX19 -- Functional Localizer

**ID.** `EX19.functional_localizer` | **File.** `121_functional_localizer.py`

**What it computes.** Applies the neuroscience functional localizer paradigm to LLMs: identifies domain-selective units via contrast conditions (linguistic vs. non-linguistic stimuli), then confirms causal necessity by comparing selective ablation deficit against random ablation baseline.

**Evidence family.** Internal (I1 Necessity, I3 Specificity), Construct (C5 Convergent)

**Pass threshold.** `causal_deficit > 0.2; selectivity_dprime > 1.0`

**Reference.** AlKhamissi et al. (2025), EMNLP 2025.

---

## Safety and Alignment Evaluation

These metrics evaluate validity properties specific to safety-relevant mechanistic claims -- reliability, construct stability, and the relationship between alignment and interpretability.

---

### S03 -- Safety Claim Reliability

**ID.** `S03.safety_claim_reliability` | **File.** `131_safety_claim_reliability.py`

**What it computes.** Tests whether circuit-based behavioral claims are consistent across different prompt templates and ablation calibration seeds, measuring reliability via coefficient of variation.

**Evidence family.** Measurement (M1 Reliability)

**Pass threshold.** `safety_reliability > 0.7`

**Reference.** Nanda (2025), AlignmentForum.

---

### S04 -- Assistant Axis Causal Stability

**ID.** `S04.assistant_axis` | **File.** `132_assistant_axis.py`

**What it computes.** Tests three validity properties of a dominant representational direction extracted via persona contrasts: causal sufficiency (suppression degrades behavior), reliability (direction stable across extraction runs), and discriminant validity (direction specific to target construct).

**Evidence family.** Internal (E2 Causal Sufficiency), Measurement (M1 Reliability), Construct (C4 Discriminant)

**Pass threshold.** `causal_deficit > 0.3; direction_stability > 0.8; discriminant_ratio > 2.0`

**Reference.** MATS + Anthropic Fellows (2026).

---

### S05 -- Safety Subspace Causal Validation

**ID.** `S05.safety_subspace` | **File.** `135_safety_subspace.py`

**What it computes.** Identifies safety-relevant directions by contrasting activations on safe vs. unsafe prompts, constructs a low-rank subspace via PCA, then tests causal sufficiency (linear classifier accuracy on projected activations) and necessity (refusal behavior change upon subspace ablation).

**Evidence family.** Internal (E2 Causal Sufficiency, I1 Necessity)

**Pass threshold.** `sufficiency > 0.6; ablation_deficit > 0.3`

**Reference.** NCSU (2025), arXiv:2512.23260.

---

### S06 -- Single-Shot Safety Recovery

**ID.** `S06.safety_one_shot` | **File.** `137_safety_one_shot.py`

**What it computes.** Tests whether the safety construct is compact enough that a single safety example's gradient aligns with the safety subspace, validating that safety alignment occupies a low-rank gradient structure.

**Evidence family.** Internal (E2 Causal Sufficiency)

**Pass threshold.** `gradient_alignment > 0.5; recovery_rate > 0.3`

**Reference.** Anonymous (2026), arXiv:2601.01887.

---

### S07 -- Alignment-Interpretability Trade-off

**ID.** `S07.alignment_interpretability` | **File.** `138_alignment_interpretability.py`

**What it computes.** Quantifies the trade-off between interpretability (activation consistency of per-unit responses) and representational richness (effective rank of activation matrices), testing whether they are discriminant constructs and how alignment affects the balance.

**Evidence family.** Construct (C4 Discriminant Validity)

**Pass threshold.** Diagnostic (no binary pass/fail). Reports whether alignment improves interpretability at the cost of richness.

**Reference.** Colin, Oliver, Serre (2026), ICLR 2026 Re-Align Workshop.

---

## Cross-Model Alignment Evaluation

These metrics evaluate whether representational structure transfers across models, architectures, or training runs.

---

### EX26 -- Atlas-Alignment Cross-Model Convergence

**ID.** `EX26.atlas_alignment` | **File.** `133_atlas_alignment.py`

**What it computes.** Computes layer-wise CKA (Centered Kernel Alignment) between two model instances on shared inputs to measure how well internal representations align, enabling concept label transfer from a reference model to a target.

**Evidence family.** Construct (C5 Convergent Validity, cross-model)

**Pass threshold.** `atlas_alignment_score > 0.3; feature_transfer_rate > 0.5`

**Reference.** Puri et al. (2025), ICLR 2026 Re-Align Workshop.

---

### EX27 -- MOT Global Alignment Score

**ID.** `EX27.mot_alignment` | **File.** `134_mot_alignment.py`

**What it computes.** Computes a global alignment score between two sets of representations using optimal transport (Sinkhorn) over pairwise CKA similarity, producing soft layer-to-layer couplings that handle depth mismatches.

**Evidence family.** Construct (C5 Convergent Validity, cross-architecture)

**Pass threshold.** `mot_global_score > 0.3`

**Reference.** Shah, Khosla (2025), ICLR 2026.

---

## Transcoder Evaluation

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

## Crosscoder Evaluation

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

## CLT (Circuit Tracing) Evaluation

These metrics evaluate the validity of attribution graphs produced by Anthropic's Circuit Tracing / CLT framework, testing faithfulness, reliability, completeness, and sensitivity to pruning decisions.

---

### FH01 -- CLT Attribution Graph Faithfulness

**ID.** `FH01.clt_graph_faithfulness` | **File.** `EX29_clt_graph_faithfulness.py`

**What it computes.** Measures how well a pruned attribution graph (keeping only the top-k highest-attribution heads) preserves the model's behavior, computed as the ratio of pruned-model logit difference to full-model logit difference.

**Evidence family.** Internal (I2 Compositional Sufficiency, C2 Structural Plausibility)

**Pass threshold.** `graph_faithfulness > 0.8`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH02 -- CLT Cross-Prompt Consistency

**ID.** `FH02.clt_cross_prompt_consistency` | **File.** `EX30_clt_cross_prompt_consistency.py`

**What it computes.** Tests M1 reliability of circuit identification by computing pairwise Jaccard similarity of top-k causally important head sets across semantically equivalent prompts (paraphrases targeting the same answer).

**Evidence family.** Measurement (M1 Reliability, M2 Invariance)

**Pass threshold.** `cross_prompt_consistency > 0.4`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH03 -- CLT Error Node Fraction

**ID.** `FH03.clt_error_fraction` | **File.** `EX31_clt_error_fraction.py`

**What it computes.** Quantifies the replacement model gap by measuring what fraction of the model's behavior is NOT captured by individually attributable head contributions, approximating the "error node" concept from the CLT framework.

**Evidence family.** Measurement (M6 Construct Coverage), Internal (I5 Confound Control)

**Pass threshold.** `error_fraction < 0.2`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH04 -- CLT Missing Attention Quantification

**ID.** `FH04.clt_missing_attention` | **File.** `EX32_clt_missing_attention.py`

**What it computes.** Quantifies the systematic explanatory gap from CLT attribution graphs' exclusion of attention mechanisms (QK circuits) by measuring the fraction of a task's causal effect attributable to attention heads versus MLP layers.

**Evidence family.** Internal (I5 Confound Control, M6 Construct Coverage)

**Pass threshold.** `attention_gap_fraction < 0.3`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

### FH05 -- CLT Graph Minimality Sensitivity

**ID.** `FH05.clt_minimality_sensitivity` | **File.** `EX33_clt_minimality_sensitivity.py`

**What it computes.** Tests whether the pruned attribution graph's size is stable across pruning thresholds, detecting fragile minimality where small threshold changes cause large jumps in graph size.

**Evidence family.** Construct (C4 Minimality), Measurement (M4 Sensitivity)

**Pass threshold.** `minimality_stability > 0.5`

**Reference.** Ameisen, Lindsey et al. (2025), Anthropic.

---

## Summary Table

| ID | Name | File | Evidence Family | Threshold |
|---|---|---|---|---|
| AQ01 | Automated Interpretability | `EX3_autointerp.py` | Measurement | proxy-based |
| AQ02 | Feature Absorption | `EX4_feature_absorption.py` | Measurement | `< 0.15` |
| AQ03 | NLA Reconstruction Fidelity | `EX5_nla_reconstruction.py` | Measurement | `> 0.3` |
| EX6 | Rule-Based Descriptions | `EX6_rule_descriptions.py` | Measurement | `> 0.3` |
| AQ04 | TopK SAE Scaling | `EX7_topk_scaling.py` | Measurement | varies |
| AQ05 | RoPE Massive Value Filter | `EX8_rope_massive_value.py` | Measurement | `> 0.9` |
| AQ06 | Latent Self-Consistency | `EX11_self_consistency.py` | Measurement | `> 0.85` |
| EX12 | Model Alignment Search | `107_mas.py` | Construct | `> 0.5` |
| FH06 | CoT Faithfulness | `108_cot_faithfulness.py` | Internal | `< 0.05` |
| EX14 | Activation Reasoning | `109_activation_reasoning.py` | External | `> 0.6` |
| EX16 | LayerNavigator | `112_layer_navigator.py` | Measurement | `> 0.3` |
| AQ07 | GradSAE Causal Influence | `113_gradsae_causal.py` | Internal | `< 0.3` |
| AQ08 | Output-Centric Description | `116_output_centric.py` | Internal / Construct | `> 0.2` |
| EX19 | Functional Localizer | `121_functional_localizer.py` | Internal / Construct | `> 0.2` deficit |
| FH07 | ModCirc Modularity | `122_modcirc.py` | Construct | `> 0.4` |
| EX21 | Latent Reasoning Validity | `123_latent_reasoning.py` | Internal / Construct | `> 0.3` steering |
| EX22 | Semantic Hub Convergence | `124_semantic_hub.py` | Construct | `> 0.5` |
| S01 | Dual Mechanism | `125_dual_mechanism.py` | Construct | `> 0.2` separation |
| AQ09 | Neuronpedia Agreement | `126_neuronpedia_agreement.py` | Construct | `> 0.2` |
| EX26 | Atlas-Alignment | `133_atlas_alignment.py` | Construct | `> 0.3` |
| EX27 | MOT Alignment | `134_mot_alignment.py` | Construct | `> 0.3` |
| S05 | Safety Subspace | `135_safety_subspace.py` | Internal | `> 0.6` suff. |
| FH01 | CLT Graph Faithfulness | `EX29_clt_graph_faithfulness.py` | Internal | `> 0.8` |
| FH02 | CLT Cross-Prompt Consistency | `EX30_clt_cross_prompt_consistency.py` | Measurement | `> 0.4` |
| FH03 | CLT Error Fraction | `EX31_clt_error_fraction.py` | Measurement | `< 0.2` |
| EX31 | NLA Semantic Validity Gap | `130_nla_semantic_validity.py` | Construct | `< 0.3` |
| FH04 | CLT Missing Attention | `EX32_clt_missing_attention.py` | Internal | `< 0.3` |
| S02 | Adversarial Ablation | `127_adversarial_ablation.py` | Internal | `< 0.3` |
| EX32a | Weight-Sparse Circuit | `131_weight_sparse_circuit.py` | Internal | `> 0.8` suff. |
| AQ13 | Crosscoder Persistence | `142_crosscoder_persistence.py` | Construct | `> 0.5` |
| S06 | Safety One-Shot | `137_safety_one_shot.py` | Internal | `> 0.5` align. |
| FH05 | CLT Minimality Sensitivity | `EX33_clt_minimality_sensitivity.py` | Construct | `> 0.5` |
| FH08 | Actionability | `128_actionability.py` | External | `> 0.1` |
| S04 | Assistant Axis | `132_assistant_axis.py` | Internal / Measurement | `> 0.3` deficit |
| S07 | Alignment-Interpretability | `138_alignment_interpretability.py` | Construct | diagnostic |
| AQ11 | Transcoder-SAE Agreement | `140_transcoder_sae_agreement.py` | Construct | `> 0.3` |
| FH09 | Surprise Reduction | `129_surprise_reduction.py` | Measurement | `> 0.05` |
| AQ12 | Transcoder Decomposition | `141_transcoder_decomposition.py` | Construct | `> 0.6` |
| AQ15 | Crosscoder Artifact Detection | `144_crosscoder_artifact_detection.py` | Measurement | `< 0.15` |
| FH10 | Behavior vs. Capability Gap | `130_behavior_capability_gap.py` | External | `< 0.3` |
| S03 | Safety Claim Reliability | `131_safety_claim_reliability.py` | Measurement | `> 0.7` |
| AQ10 | Transcoder Composability | `139_transcoder_composability.py` | Internal | `> 0.5` |
| AQ14 | Crosscoder Model Diff | `143_crosscoder_model_diff.py` | Internal / Measurement | `> 0.7` |

---

## Relationship to Other Pages

For the **benchmark** metrics (AxBench, SAEBench, CE-Bench, MIB, and others), see [MI Benchmarks](/framework/lenses/core/mi-benchmarks-metrics). For the genetics lens metrics (GN1--GN5) and protocols, see [Genetics Metrics](/framework/lenses/supporting/genetics-metrics). For measurement theory metrics (reliability, validity, sensitivity), see [Measurement Theory Metrics](/framework/lenses/core/measurement-theory-metrics). For the overall MI lens framework, see [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability).
