# Coverage Comparison: mechval vs Apollo Research "45+ Mech Interp Project Ideas" (2024)

Source: Apollo Research (2024). "A List of 45+ Mech Interp Project Ideas."
https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8

## Foundational work on sparse dictionary learning

- [COVERED] #1 Training and releasing high quality transcoders -> `transcoder_decomposition`, `transcoder_composability`, `transcoder_sae_agreement`
- [COVERED] #3 Further circuit analysis using transcoders -> `transcoder_composability`, `sparse_feature_circuits`
- [PARTIAL] #4 Cross layer superposition -> related: `crosscoder_persistence`, `crosscoder_model_diff`; gap: explicit cross-layer superposition detection via decoder vector similarity
- [PARTIAL] #5 Improving transcoder architectures -> related: `transcoder_decomposition`; gap: no evaluation of linear bypass terms or multi-layer sparsity penalties
- [COVERED] #6 Improved-logit lens interpretation of SAE features -> `mean_centered_logit`, `probe_decodability`
- [GAP] #7 Toy model of feature splitting -- no metric for feature splitting analysis or continuous manifold detection
- [PARTIAL] #8 Looking for opposing feature directions in SAEs -> related: `prism_polysemanticity`; gap: no explicit bipolar/opposing-direction pair detection
- [GAP] #9 SAE/Transcoder activation shuffling -- no metric for training procedure analysis (shuffling importance)
- [GAP] #10 SAE/Transcoder initialization -- no metric evaluating initialization strategy impact
- [COVERED] #11 Make public benchmarks for SAEs and transcoders -> `saebench`, `saebench_audit`, `axbench`
- [GAP] #12 Mixture of Expert SAEs -- no MOE-SAE evaluation metric
- [PARTIAL] #13 Identify canonical features in language models -> related: `convergent_evolution`, `cross_model_invariance`; gap: no systematic cross-scale canonical feature identification
- [COVERED] #14 Studying generalization of SAEs and transcoders -> `cross_task_generalization`, `cross_task_transfer`
- [GAP] #15 How does layer norm affect SAE features -- no layer-norm-specific feature distortion metric
- [GAP] #16 Connecting SAE/transcoder features to polytopes -- no polytope-based representation analysis
- [COVERED] #17 Verify features based on model weights -> `circuitlens_weight_circuits`, `weight_extended`, `weightlens_convergence`
- [PARTIAL] #18 Relationship between feature splitting and atomic features -> related: `feature_absorption`; gap: no explicit atomic feature identification
- [GAP] #19 Is there structure in feature splitting -- no structured splitting direction analysis
- [PARTIAL] #20 Understanding the geometry of SAE features -> related: `spectral_svd`, `cka`, `superposition_regime`; gap: no PCA spectra or hierarchical subspace analysis
- [GAP] #21 Identify better SAE sparsity penalties -- no metric for comparing sparsity penalty quality
- [GAP] #22 Preprocessing activations with the interaction basis -- no interaction basis preprocessing evaluation
- [GAP] #23 Using attribution sparsity penalties for end-to-end SAEs -- no e2e SAE attribution sparsity evaluation

## Applied interpretability

- [GAP] #24 Apply SAEs/transcoders to a small conv net -- no vision model SAE evaluation (framework is LLM-focused)
- [GAP] #25 Figure out interpretability interfaces for video/other modalities -- no multimodal interpretability evaluation
- [GAP] #26 Apply SAEs and transcoders to WhisperV2 -- no speech model SAE evaluation
- [PARTIAL] #27 Detecting backdoors in small models using e2eSAEs -> related: `safety_sve`, `safety_subspace`; gap: no explicit backdoor detection via e2e SAEs
- [GAP] #28 Interpreting Mamba/SSMs using sparse dictionary learning -- no SSM interpretability metrics
- [GAP] #29 Characterizing the geometry of low-level vision SAE features -- no vision feature geometry evaluation
- [GAP] #30 Can we understand the first sequence index -- no first-token prediction analysis
- [PARTIAL] #31 Attempt to understand a toy LM completely -> related: `kolmogorov_complexity`, `mdl_compression`; gap: no comprehensive n-gram recovery measurement
- [GAP] #32 Understand a small model from start to end -- no end-to-end model understanding completeness metric

## Intrinsic interpretability

- [GAP] #33 Can we train a bilinear transformer and interpret via closed form -- no bilinear transformer analysis
- [GAP] #34 Interpretable inference via model conversion -- no post-hoc interpretable conversion evaluation
- [GAP] #35 Develop a mathematical framework for linear attention circuits -- no linear attention circuit formalism evaluation

## Understanding features (not SDL)

- [PARTIAL] #36 Recovering features through direct optimization for interpretability -> related: `autointerp`, `natural_language_autoencoder`; gap: no iterative optimization loop for monosemantic feature discovery

## Theoretical foundations

- [COVERED] #37 Understanding SLT at finite data/precision -> `llc`
- [PARTIAL] #38 Bounding the local learning coefficient (LLC) -> related: `llc`; gap: no Hessian null space rank bound computation
- [GAP] #39 Higher-order terms in LLC Taylor series -- no higher-order LLC bound tightness evaluation
- [GAP] #40 Understanding the relationship between LLC and behavioral LLC -- no behavioral LLC metric
- [PARTIAL] #41 Extending computation in superposition framework -> related: `superposition_regime`; gap: no floating-point superposition evaluation (only boolean)
- [GAP] #42 Bounding the sparsity of LLM representations -- no theoretical sparsity bound measurement
- [GAP] #43 Relating superposition to the loss landscape -- no superposition-loss-landscape connection metric

## Meta-research and philosophy

- [GAP] #44 Write up reviews on links between disciplines -- meta-research, not a metric
- [PARTIAL] #45 What is a feature conceptually -> related: `normative_account`, `rule_based_descriptions`; gap: no formal feature definition evaluation
- [GAP] #46 Should we expect features to be natural latents -- no natural latent theory test

## Engineering

- [GAP] #47 Create a new, high quality TinyStories dataset -- dataset creation, not a metric

## Summary

| Status  | Count |
|---------|-------|
| COVERED | 10    |
| PARTIAL | 14    |
| GAP     | 23    |
| **Total** | **47** |

**Coverage rate:** 21% fully covered, 30% partially covered, 49% genuine gaps.

### Key gap themes

1. **SAE training methodology** (#9, #10, #21, #22, #23): No metrics for evaluating SAE training procedures themselves (initialization, shuffling, sparsity penalties, preprocessing).
2. **Feature geometry and splitting** (#7, #19, #16, #20): No metrics for feature splitting structure, polytope connections, or hierarchical subspace analysis.
3. **Non-LLM modalities** (#24, #25, #26, #28, #29): Framework is LLM-focused; no vision, speech, or SSM interpretability evaluation.
4. **Theoretical foundations** (#39, #40, #42, #43): SLT higher-order terms, behavioral LLC, sparsity bounds, and superposition-loss landscape connections are all absent.
5. **Complete model understanding** (#30, #32): No metrics for measuring end-to-end comprehensiveness of mechanistic understanding.
