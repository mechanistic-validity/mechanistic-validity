# Coverage Comparison: mechval vs Sharkey "Sparsify: A MI Research Agenda" (2024)

Source: Lee Sharkey (2024). "Sparsify: A Mechanistic Interpretability Research Agenda."
https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm

## Objective 1: Improved SAEs

- [COVERED] Benchmarking SAEs: "principled metrics and standardized comparison" -> `saebench`, `saebench_audit`, `axbench`, `ce_bench`
- [GAP] End-to-End SAE Training: train SAEs using reconstruction loss on later layers rather than single-layer -- no end-to-end SAE training quality metric
- [GAP] Better Sparsity Penalties: replace L1 with L0<p<1 penalties to reduce feature count while maintaining performance -- no sparsity penalty comparison metric
- [PARTIAL] Feature Suppression: address systematic undershooting in SAE encoder outputs -> related: `feature_absorption`; gap: no explicit suppression magnitude measurement
- [PARTIAL] Attention Head Superposition: gated attention blocks or sparse transcoders to decompose attention head interactions -> related: `transcoder_decomposition`, `attention_entropy`; gap: no destructive interference quantification between attention heads
- [GAP] Applying SAEs to Attention: resolve remaining challenges in attention block interpretation -> no attention-specific SAE evaluation (attention SAE faithfulness, attention job decomposition)
- [GAP] Better Hyperparameter Selection Methods: understand hyperparameter interactions to avoid expensive sweeps -> no hyperparameter sensitivity metric for SAEs specifically
- [GAP] Computationally Efficient Sparse Coding: explore initialization, preprocessing, and more efficient sparse coding methods -> no sparse coding efficiency benchmark

## Objective 2: Decompiled Networks

- [PARTIAL] Network Decompilation: transform networks to perform inference in the sparse feature basis -> related: `transcoder_decomposition`, `transcoder_composability`; gap: no full-network decompilation fidelity metric (inference entirely in feature basis)
- [PARTIAL] Transcoder Architecture Research: investigate linear vs nonlinear transcoders -> related: `transcoder_sae_agreement`; gap: no explicit transcoder architecture comparison framework
- [PARTIAL] Interaction Feature Analysis: study interaction features and interaction strengths between layers -> related: `pairwise_synergy`, `epistasis`, `k_composition`; gap: no transcoder-derived interaction feature causal influence metric
- [GAP] Policy Goals and Standards: produce decompiled versions alongside base models for auditing -> policy/governance goal, not a metric; but no decompilation completeness standard

## Objective 3: Abstraction Above Raw Decompilations

- [COVERED] Circuit Identification: identify multi-layer modules of causally interacting features -> `automatic_circuit_discovery`, `sparse_feature_circuits`, `path_identification`
- [GAP] Meta-SAEs for Abstraction: sparse coding on transcoder features to identify co-activating feature groups -> no meta-SAE evaluation metric
- [GAP] Hierarchical Abstraction Levels: further sparse coding iterations to climb abstraction hierarchies -> no hierarchical abstraction quality metric
- [PARTIAL] Description Length Optimization: minimize mathematical or semantic description length -> related: `mdl_compression`, `kolmogorov_complexity`; gap: no comparison between mathematical and semantic description length

## Objective 4: Deep Description

- [PARTIAL] Automated Feature Labeling Beyond Shallow Descriptions -> related: `autointerp`, `natural_language_autoencoder`, `nla_semantic_validity`; gap: no multi-layer causal chain description quality metric
- [GAP] Iterative-Forward-Backwards Procedure: explain features via earlier layers (forward) and later layer effects (backward) iteratively -> no iterative bidirectional description refinement metric
- [PARTIAL] Integration of Multiple Description Types: combine visualization, activation atlases, logit lens, causal interventions -> related: `neuronpedia_agreement`, `nla_sae_convergence`; gap: no unified multi-source description integration quality metric
- [COVERED] Hypothesis Testing for Feature Descriptions: gradient-based attribution, causal scrubbing, activation predictions -> `causal_scrubbing`, `activation_patching`, `das_iia`, `autointerp`

## Objective 5: MI Applications

- [PARTIAL] MI-Based Model Red-Teaming: find inputs producing concerning feature combinations -> related: `safety_one_shot`, `safety_subspace`, `safety_sve`; gap: no proactive feature-combination-driven input search
- [GAP] Backward Tracing from Dangerous Features: work backward from undesirable features to identify triggering inputs -> no backward tracing metric
- [PARTIAL] MI-Based Benchmarking: standardized tests evaluating internal activations -> related: `saebench`, `axbench`, `ce_bench`; gap: these benchmark SAEs not internal activations directly for safety
- [PARTIAL] Alignment Method Evaluation: use mechanistic evals to compare alignment approaches -> related: `alignment_interpretability`, `misalignment`; gap: no systematic alignment method comparison using mechanistic evidence
- [COVERED] Targeted Model Interventions: ablating knowledge, whitelisting capabilities, steering vectors -> `contrastive_activation_addition`, `steering_reliability`, `concept_erasure`, `role_ablation`
- [PARTIAL] Capability Prediction: predict whether models possess representations for dangerous capabilities -> related: `behavior_capability_gap`, `failure_prediction`; gap: no latent capability prediction from mechanistic structure
- [GAP] During-Training Interpretability: efficient methods for continuous interpretation during training -> no training-time interpretability metric
- [GAP] Robust-to-Training Interpretability: test whether methods remain valid when models are trained against them -> no adversarial robustness-to-training metric for interpretability

## Underlying Open Questions

- [GAP] Formal justification for sparsity yielding short semantic descriptions -> no formal sparsity-to-semantics connection metric
- [PARTIAL] Is the superposition/polysemanticity abstraction correct? -> related: `superposition_regime`, `prism_polysemanticity`; gap: no definitive test of the superposition hypothesis itself
- [GAP] Can >75% of model computation be recovered through interpretable explanations? -> no computation recovery percentage metric
- [COVERED] Do SAEs actually work sufficiently well? -> `saebench`, `saebench_audit`, `ce_bench`, `axbench`
- [PARTIAL] What is the correct notion of internal validity? -> related: `normative_account`, `reproducibility_check`; gap: no formal internal validity definition evaluation
- [GAP] How do SAE results scale to frontier-scale models? -> no scaling analysis metric
- [PARTIAL] Formal definition of "explanation of (network, dataset)" -> related: `mdl_compression`, `kolmogorov_complexity`; gap: no Kolmogorov or proof-theoretic formalization of explanations
- [GAP] What constitute truly "fundamental objects" from the network's perspective? -> no network-native feature boundary detection
- [PARTIAL] What downstream tasks demonstrate SAE building blocks work? -> related: `saebench`, `axbench`; gap: these validate SAEs, not specifically downstream safety utility
- [PARTIAL] How to measure monosemanticity rigorously? -> related: `prism_polysemanticity`, `autointerp`; gap: no ground-truth monosemanticity metric
- [GAP] Can MI actually improve AI safety in practice? -> no empirical safety-improvement-via-MI metric

## Summary

| Status  | Count |
|---------|-------|
| COVERED | 5     |
| PARTIAL | 17    |
| GAP     | 16    |
| **Total** | **38** |

**Coverage rate:** 13% fully covered, 45% partially covered, 42% genuine gaps.

### Key gap themes

1. **SAE training methodology** (e2e training, sparsity penalties, hyperparameter selection, efficient sparse coding): No metrics for evaluating or comparing SAE training procedures.
2. **Hierarchical abstraction** (meta-SAEs, abstraction hierarchies): No metrics for evaluating abstraction quality above raw feature/circuit level.
3. **Deep description** (iterative forward-backward, multi-source integration): No metrics for evaluating description depth beyond single-layer labeling.
4. **Safety-specific MI** (backward tracing, during-training, robust-to-training): Missing proactive safety-oriented MI evaluation methods.
5. **Scaling and formal foundations** (frontier scaling, sparsity-semantics connection, computation recovery): No metrics connecting MI findings to formal guarantees or frontier-scale applicability.
6. **Network decompilation completeness**: No metric for measuring what fraction of a model's computation has been successfully decompiled into interpretable form.
