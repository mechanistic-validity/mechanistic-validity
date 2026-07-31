# Coverage Analysis: mechval vs. ICML 2026 "Interpretability Can Be Actionable"

Source: Orgad, Barez et al. "Interpretability Can Be Actionable: Evaluation
Criteria for Interpretability in Practice." arXiv:2605.11161, ICML 2026.

Core thesis: evaluation criteria -- not methods -- are the bottleneck.
Two dimensions: concreteness (enables specific intervention?) and validation
(empirically tested?). Five domains where MI provides unique leverage.

---

## 1. Two Evaluation Dimensions

### 1.1 Concreteness (enables a specific intervention with measurable outcome?)

- [COVERED] "Does the finding identify a specific model component?" -> `functional_localizer`, `activation_patching`, `path_patching`
- [COVERED] "Does the finding specify a specific intervention type?" -> `contrastive_activation_addition`, `concept_erasure`, `steering_reliability`
- [COVERED] "Does the finding specify a measurable outcome?" -> `logit_diff`, `ce_delta`, `dose_response`
- [COVERED] "Actionability score (concreteness x validation)" -> `actionability_score`

### 1.2 Validation (intervention empirically tested on held-out data?)

- [COVERED] "Has the intervention been tested on held-out data?" -> `cross_task_generalization`, `cross_task_transfer`, `held_out_prediction`
- [COVERED] "Does the finding replicate across seeds?" -> `reproducibility_check`, `hyperparam_sensitivity`
- [COVERED] "Does the finding transfer across architectures?" -> `cross_model_transfer`, `cross_model_invariance`, `cka_cross_arch`

---

## 2. Five Domains of Unique Leverage

### 2.1 Problems Scaling Doesn't Solve

- [COVERED] "Identify mechanistic origins of hallucinations" -> `cot_faithfulness`, `latent_reasoning_validity`, `latent_self_consistency`
- [COVERED] "Identify mechanistic origins of adversarial brittleness" -> `adversarial_ablation_verification`, `adversarial_parameter_decomposition`, `boundary_sweep`
- [PARTIAL] "Identify mechanistic origins of catastrophic forgetting" -> related: `cross_task_generalization`, gap: no forgetting-specific metric that tracks which circuits degrade during continued training
- [PARTIAL] "Identify mechanistic origins of bias" -> related: `concept_erasure`, `safety_subspace`, gap: no bias-specific circuit identification metric; concept erasure tests removal but not discovery of bias circuits

### 2.2 Alignment

- [COVERED] "Audit for deceptive capabilities" -> `misalignment`, `eval_awareness_format_control`, `safety_claim_reliability`
- [COVERED] "Detect backdoors via internal representations" -> `safety_subspace`, `safety_sve`, `crosscoder_artifact_detection`
- [COVERED] "Verify intended behavior mechanistically" -> `causal_scrubbing`, `activation_patching`, `das_iia`
- [GAP] "Goal-representation auditing: verify model represents intended goals vs instrumentally deceptive goals" -- no metric tests whether internal goal representations match stated objectives

### 2.3 Surgical Interventions

- [COVERED] "Model editing: insert/remove/correct behaviors" -> `concept_erasure`, `knock_in`, `contrastive_activation_addition`
- [COVERED] "Activation steering at inference" -> `steering_reliability`, `contrastive_activation_addition`, `dose_response`
- [COVERED] "Targeted pruning preserving other functionality" -> `role_ablation`, `graph_minimality`, `edge_necessity`
- [COVERED] "Intervention specificity: affects target, preserves non-target" -> `intervention_specificity`, `path_specificity`
- [PARTIAL] "Representation fine-tuning (ReFT) as interpretability-grounded LoRA alternative" -> related: `steering_reliability`, gap: no metric specifically evaluating ReFT vs LoRA efficiency with mechanistic grounding

### 2.4 Architectural Design

- [PARTIAL] "Link design choices to behavioral effects" -> related: `architecture_duality`, `cka_cross_arch`, gap: no metric that maps architectural decisions to specific behavioral changes
- [GAP] "Narrow the space of plausible architectural modifications using MI" -- no metric evaluates whether interpretability findings reduce the architecture search space
- [GAP] "Principled architecture innovation informed by mechanistic insights" -- no metric measures whether MI-derived architectural changes outperform uninformed changes

### 2.5 Concept Translation

- [COVERED] "Translate internal signals into domain-appropriate concepts" -> `autointerp`, `natural_language_autoencoder`, `nla_semantic_validity`
- [COVERED] "Human understandability of explanations" -> `rule_based_descriptions`, `output_centric_description`, `procedure_specification`
- [PARTIAL] "Domain-specific concept translation (clinical, legal, etc.)" -> related: `autointerp`, `natural_language_autoencoder`, gap: no domain-specific concept translation evaluation beyond general NL descriptions

---

## 3. Evaluation Criteria by Action Type

### 3.1 Output Modification Evaluation

- [COVERED] "Comparative utility: performance against non-MI baselines (prompting, fine-tuning)" -> `ce_delta`, `logit_diff`, `saebench`
- [COVERED] "Mechanistic faithfulness: interventions produce predicted changes" -> `activation_patching`, `das_iia`, `causal_scrubbing`
- [COVERED] "Generalization across model sizes, seeds, perturbations" -> `cross_model_transfer`, `cross_model_invariance`, `reproducibility_check`
- [COVERED] "Specificity: component explains target better than alternatives" -> `intervention_specificity`, `path_specificity`, `edge_necessity`
- [GAP] "Comparative advantage over non-interpretability baselines: marginal benefit of MI-based intervention vs standard fine-tuning/prompting on same task" -- no metric directly computes the MI advantage delta

### 3.2 Deployment Evaluation

- [GAP] "Human-subject studies: do explanations improve human decision-making?" -- no metric includes human-in-the-loop evaluation of explanation quality
- [PARTIAL] "Understandability: explanations comprehensible to practitioners" -> related: `autointerp`, `rule_based_descriptions`, gap: no user-study-based comprehensibility metric
- [COVERED] "Reliability: stable explanations across seeds and perturbations" -> `reproducibility_check`, `hyperparam_sensitivity`, `core_stability`

### 3.3 Future Practice Evaluation

- [GAP] "Feasibility for regulators using MI for governance" -- no metric evaluates regulatory applicability
- [GAP] "Support for policy tools: risk audits, model cards, licensing" -- no metric maps MI outputs to policy-ready formats
- [PARTIAL] "Legibility to non-experts" -> related: `rule_based_descriptions`, `output_centric_description`, gap: no non-expert comprehension evaluation

---

## 4. Specific Barriers to Address

- [COVERED] "Oversimplified setups (single next-token vs multi-token)" -> `per_token_nll`, `output_variants`, `output_variants_kl`
- [COVERED] "Insufficient comparative analysis" -> `saebench_audit`, `weight_eap_jaccard`
- [PARTIAL] "Open-weights assumption restricts applicability to proprietary models" -> related: `cross_model_invariance`, gap: no metric specifically tests whether findings transfer to black-box/API-only models
- [GAP] "21.8% of ICML 2025 MI submissions flagged as insufficiently actionable" -- no meta-metric evaluates the actionability of a research contribution

---

## 5. Specific Interventions

- [COVERED] "Model editing (MLP key-value stores)" -> `knock_in`, `concept_erasure`
- [COVERED] "Activation steering" -> `steering_reliability`, `contrastive_activation_addition`
- [COVERED] "Concept erasure (nullspace projection)" -> `concept_erasure`
- [COVERED] "Sparse autoencoder features for steering/safety" -> `sparse_feature_circuits`, `saebench`
- [PARTIAL] "Latent adversarial training" -> related: `adversarial_ablation_verification`, gap: no metric evaluates latent adversarial training as a defense mechanism

---

## Summary

| Status  | Count | Percentage |
|---------|-------|------------|
| COVERED | 27    | 56%        |
| PARTIAL | 10    | 21%        |
| GAP     | 11    | 23%        |
| **Total** | **48** | **100%** |

### Key gaps (genuine unsolved problems for the field):
1. Goal-representation auditing for alignment verification
2. Human-subject evaluation of explanation quality
3. Regulatory/policy feasibility of MI outputs
4. Comparative advantage quantification (MI vs non-MI baselines)
5. Architecture search space reduction via MI
6. Catastrophic forgetting circuit identification
7. Domain-specific concept translation evaluation
