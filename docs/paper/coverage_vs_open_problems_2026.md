# Coverage Comparison: mechval vs "Open Problems in Mechanistic Interpretability" (Sharkey et al., Jan 2026)

Source: Lee Sharkey et al. (30 authors, 2026). "Open Problems in Mechanistic Interpretability."
https://arxiv.org/abs/2501.16496 / https://openreview.net/forum?id=91H76m9Z94

## Section 2.1: Reverse Engineering

### 2.1.2 Decomposition

- [COVERED] Networks don't naturally decompose (polysemanticity) -> `prism_polysemanticity`, `superposition_regime`, `feature_absorption`
- [PARTIAL] Dimensionality reduction limitations -> related: `spectral_svd`, `superposition_regime`; gap: no explicit test of whether SDL can find more directions than activation dimensions
- [PARTIAL] SDL reconstruction errors too high -> related: `ce_bench`, `saebench`; gap: no metric isolating reconstruction error impact on downstream task performance
- [GAP] SDL computational expense and scaling -- no metric for computational cost analysis or scaling properties of SDL
- [PARTIAL] Linear representation assumption may not hold -> related: `superposition_regime`; gap: no explicit nonlinear representation detection test
- [COVERED] Sparsity as proxy: splitting, absorption, composition -> `feature_absorption`, `topk_scaling`, `adaptive_sparsity`
- [GAP] Feature geometry unexplained: SDL ignores semantic relationships between features -> no inter-feature semantic relationship metric
- [PARTIAL] Multi-architecture SDL limitations -> related: `cka_cross_arch`, `cross_model_invariance`; gap: no metric for SDL applicability across attention heads and distributed representations
- [PARTIAL] Mechanism vs activation gap: SDL describes activations, not weight mechanisms -> related: `circuitlens_weight_circuits`, `weight_extended`, `weightlens_convergence`; gap: no formal metric bridging activation-SDL to weight-space mechanisms
- [GAP] Task-dependent latents: SDL latents may miss concepts needed for downstream use -> no task-conditioned latent completeness metric
- [GAP] Lack of theoretical foundations: no formal definition of "features" -> no formal feature definition validation metric
- [PARTIAL] Superposition hypothesis validity -> related: `superposition_regime`; gap: no definitive validity test (only regime detection)
- [GAP] Intrinsic decomposability trade-off: training for both performance and interpretability -> no performance-interpretability Pareto frontier metric

### 2.1.3 Describing Component Roles

- [PARTIAL] Highly activating examples -- projection bias -> related: `autointerp`; gap: no alien-concept detection (human projection bias measurement)
- [COVERED] Interpretability illusions from dataset bias -> `crosscoder_artifact_detection`, `saebench_audit`
- [PARTIAL] Plausible explanations for random directions -> related: `autointerp`, `rule_based_descriptions`; gap: no random-direction control baseline for explanation quality
- [PARTIAL] Attribution methods -- theoretical gaps (first-order approximation) -> related: `activation_patching`, `eap`, `path_patching`; gap: no higher-order attribution method
- [COVERED] Some attribution methods model-independent -> `relevance_patching`, `relp` (LRP is model-dependent by design)
- [COVERED] Perturbation methods -- distribution shift -> `corrupt_restore`, `corrupt_restore_behavioral`, `resample_complement`
- [GAP] Feature synthesis limitations -- no metric for evaluating synthesized vs natural feature examples
- [GAP] Logit lens and activation steering lack formal causal grounding -> no formal causal grounding evaluation for these methods

### 2.1.4 Validation of Descriptions

- [PARTIAL] No standardized evaluation framework -> related: `reproducibility_check`, `normative_account`; gap: no universal validation standard
- [COVERED] Distinguishing genuine insights from spurious correlations -> `causal_scrubbing`, `das_iia`, `counterfactual_consistency`
- [PARTIAL] Ablation may trigger reconfiguration rather than revealing function -> related: `adversarial_ablation_verification`; gap: no explicit reconfiguration detection metric
- [GAP] Single-component interventions insufficient for multi-component interactions -> no multi-component interaction intervention metric
- [GAP] Circular reasoning risk in validation -> no circularity detection metric for mechanistic validation
- [GAP] Generalization of mechanistic descriptions to new domains -> no cross-domain mechanistic transfer metric

### 2.2 Concept-Based Interpretability

- [COVERED] Concept probes detect correlations not causal variables -> `probe_decodability`, `das_iia`, `causal_representation`
- [GAP] Probe data quality and concept definition challenges -> no concept boundary quality metric
- [COVERED] Causality vs correlation in probes -> `das_iia`, `causal_representation`, `iia_variants`
- [GAP] Concept-based intrinsic interpretability: interpretability-constrained training -> no interpretability-constrained training evaluation

### 2.3 Proceduralizing Circuit Discovery

- [COVERED] Standardizing circuit discovery pipelines -> `automatic_circuit_discovery`, `eap`, `path_identification`, `sparse_feature_circuits`
- [GAP] Lack of systematic procedures across domains -> no cross-domain circuit discovery transferability metric
- [GAP] Manual steps prevent scaling -> no automation completeness metric for circuit discovery

### 2.4 Automating Interpretability

- [COVERED] Automating interpretation steps -> `autointerp`, `natural_language_autoencoder`
- [GAP] AI-generated interpretations risk reproducing biases at scale -> no bias amplification detection in automated interpretations
- [GAP] Evaluating quality of auto-generated interpretations -> no ground-truth interpretation quality benchmark (autointerp measures correlation, not correctness)

## Section 3: Applications

### 3.1 Axes of Progress

- [GAP] Methods on small models don't transfer to large models -> no scale-transfer evaluation metric
- [GAP] Decomposition costs scale poorly -> no computational scaling analysis
- [GAP] Most work focuses on transformer LLMs; vision, RNNs, hybrids under-studied -> no cross-architecture MI coverage metric

### 3.2 Safety Monitoring and Auditing

- [PARTIAL] Detecting unsafe cognition -> related: `safety_sve`, `safety_subspace`, `safety_one_shot`, `misalignment`; gap: no deception or goal-misalignment detection in latent space
- [COVERED] Control of AI behavior via activation steering and parameter editing -> `contrastive_activation_addition`, `steering_reliability`, `concept_erasure`
- [GAP] Understanding unintended consequences of mechanistic interventions -> no side-effect measurement for interpretability-based interventions
- [GAP] Scaling targeted behavioral modifications -> no scaling evaluation for steering/editing

### 3.3 Better Predictions

- [PARTIAL] Predicting behavior in novel situations -> related: `failure_prediction`, `held_out_prediction`, `generalization_gap`; gap: no mechanistic-knowledge-driven OOD prediction
- [PARTIAL] Predicting capability emergence during training -> related: `behavior_capability_gap`; gap: no training-dynamics capability emergence prediction

### 3.4 Improving Inference, Training, and Mechanisms

- [GAP] Using MI to reduce model size or inference cost -> no MI-guided pruning evaluation
- [GAP] Identifying and removing redundant mechanisms -> no redundancy-guided compression metric
- [GAP] Transferring mechanisms between models -> no mechanism transfer metric
- [GAP] MI insights for curriculum learning -> no curriculum design from MI metric

### 3.5 Microscope AI

- [GAP] Extracting latent scientific knowledge from models -> no scientific knowledge extraction evaluation
- [GAP] Distinguishing understanding from pattern matching -> no understanding-vs-memorization metric
- [GAP] Validating whether model mechanisms correspond to real-world processes -> no mechanism-to-reality correspondence metric

### 3.6 Broader Model Coverage

- [PARTIAL] MI for vision transformers, CNNs, RNNs -> related: `cka_cross_arch`; gap: no architecture-specific MI evaluation suite
- [GAP] Interpreting multimodal models -> no multimodal MI evaluation
- [GAP] Understanding sparse/MoE architectures -> no MoE-specific MI evaluation
- [GAP] MI for reward models, RL agents, planning systems -> no RL/reward model MI evaluation
- [GAP] Interpreting fine-tuned models and adaptation mechanisms -> no fine-tuning mechanism evaluation

### 3.7 Human-Computer Interaction

- [GAP] Designing effective interfaces for MI exploration -> HCI, not a metric
- [GAP] Real-time visualization during inference -> tooling, not a metric
- [GAP] Enabling non-expert interaction with internals -> accessibility, not a metric
- [GAP] Human-AI collaboration for mechanism discovery -> workflow, not a metric

## Section 4: Socio-Technical Problems

### 4.1 Policy and Governance

- [PARTIAL] Translating MI to regulatory requirements -> related: `safety_claim_reliability`; gap: no regulatory compliance metric
- [GAP] Standards for interpretability audits in governance -> governance standard, not a metric
- [GAP] International alignment on MI standards -> policy, not a metric

### 4.2 Social and Philosophical Problems

- [GAP] Whether mechanistic explanations constitute genuine understanding -> philosophical, not a metric
- [GAP] Interpretability misuse / dual-use concerns -> sociotechnical, not a metric
- [GAP] Equity in interpretability accessibility -> social, not a metric

## Summary

| Status  | Count |
|---------|-------|
| COVERED | 12    |
| PARTIAL | 16    |
| GAP     | 39    |
| **Total** | **67** |

**Coverage rate:** 18% fully covered, 24% partially covered, 58% genuine gaps.

Note: Many GAP items in Sections 3.7 and 4 are HCI, policy, or philosophical problems that are inherently outside the scope of a computational metrics framework. Filtering to only computational/empirical problems:

| Status  | Count (computational only) |
|---------|---------------------------|
| COVERED | 12                        |
| PARTIAL | 16                        |
| GAP     | 28                        |
| **Total** | **56**                  |

**Adjusted coverage rate:** 21% fully covered, 29% partially covered, 50% genuine gaps.

### Key gap themes

1. **SDL foundations and scaling** (reconstruction error impact, computational cost scaling, formal feature definitions, feature geometry): Theoretical and practical SDL limitations are not evaluated.
2. **Cross-architecture and cross-scale** (non-transformer models, multimodal, MoE, RL, frontier scaling): Framework is GPT-2-centric; no evaluation infrastructure for other architectures or scales.
3. **Validation methodology** (circularity detection, multi-component interactions, cross-domain transfer, reconfiguration detection): Validation of MI methods themselves is underserved.
4. **Practical MI applications** (MI-guided pruning, mechanism transfer, curriculum design, scientific knowledge extraction): No metrics measuring the practical utility of MI findings.
5. **Safety-specific gaps** (deception detection, intervention side effects, capability emergence prediction, robust-to-training): Safety monitoring goes beyond what current metrics cover.
6. **Automation quality** (bias in auto-generated interpretations, ground-truth interpretation benchmarks): Autointerp exists but no evaluation of its failure modes.
