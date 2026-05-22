"""Metric and calibration registries.

Maps metric/calibration names to (module_path, function_name) tuples
for lazy import dispatch. External metrics registered via
``mechval.registry.register_metric()`` are merged automatically.

Organized by lens (disciplinary framework) at the top level.
Each metric is tagged with validity_type and originating lens via
METRIC_METADATA.
"""
import importlib
import inspect

from mechval.registry import get_external_metrics


METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    # ── Mechanistic Interpretability: Structural ─────────────────────────
    "k_composition": ("mechval.metrics.mechanistic_interpretability.core.structural.k_composition.B10_k_composition", "run"),
    "copying_score": ("mechval.metrics.mechanistic_interpretability.core.structural.copying_score.B11_copying_score", "run"),
    "qk_norms": ("mechval.metrics.mechanistic_interpretability.core.structural.qk_norms.B12_qk_norms", "run"),
    "cmd": ("mechval.metrics.mechanistic_interpretability.core.structural.template_distance.26_cmd", "run_cmd"),
    "edge_jaccard": ("mechval.metrics.mechanistic_interpretability.core.structural.template_distance.27_edge_jaccard", "run_edge_jaccard"),
    "weight_eap_jaccard": ("mechval.metrics.mechanistic_interpretability.core.structural.weight_alignment.28_weight_eap_jaccard", "run_weight_eap_jaccard"),
    "network_motifs": ("mechval.metrics.mechanistic_interpretability.core.structural.network_motifs.97_network_motifs", "run_network_motifs"),
    "motif_enrichment": ("mechval.metrics.mechanistic_interpretability.core.structural.network_motifs.G7_motif_enrichment", "run_motif_enrichment"),
    "capacity_utilization": ("mechval.metrics.mechanistic_interpretability.core.structural.effective_rank.B13_capacity_utilization", "run"),
    "k_alignment": ("mechval.metrics.mechanistic_interpretability.core.structural.effective_rank.B14_k_alignment", "run"),
    "weight_extended": ("mechval.metrics.mechanistic_interpretability.core.structural.effective_rank.18_weight_extended", "run_weight_extended"),
    "spectral_svd": ("mechval.metrics.mechanistic_interpretability.core.structural.spectral_svd.18_weight_extended", "run_weight_extended"),
    "path_identification": ("mechval.metrics.mechanistic_interpretability.core.structural.edge_analysis.82_path_identification", "run_path_identification"),
    "edge_necessity": ("mechval.metrics.mechanistic_interpretability.core.structural.edge_analysis.83_edge_necessity", "run_edge_necessity"),
    "path_specificity": ("mechval.metrics.mechanistic_interpretability.core.structural.edge_analysis.84_path_specificity", "run_path_specificity"),
    "compositional_sufficiency": ("mechval.metrics.mechanistic_interpretability.core.structural.edge_analysis.85_compositional_sufficiency", "run_compositional_sufficiency"),
    "graph_minimality": ("mechval.metrics.mechanistic_interpretability.core.structural.edge_analysis.86_graph_minimality", "run_graph_minimality"),
    "attention_clustering": ("mechval.metrics.mechanistic_interpretability.core.structural.attention_clustering.96_attention_clustering", "run_attention_clustering"),
    "llc": ("mechval.metrics.mechanistic_interpretability.core.structural.llc_rllc.10_llc", "run_llc"),

    # ── Mechanistic Interpretability: Causal ─────────────────────────────
    "activation_patching": ("mechval.metrics.mechanistic_interpretability.core.causal.scm_pearl.02_activation_patching", "run_activation_patching"),
    "logit_diff": ("mechval.metrics.mechanistic_interpretability.core.causal.scm_pearl.logit_diff", "run_logit_diff"),
    "role_ablation": ("mechval.metrics.mechanistic_interpretability.core.causal.scm_pearl.role_ablation", "run_role_ablation"),
    "causal_scrubbing": ("mechval.metrics.mechanistic_interpretability.core.causal.scm_pearl.04_causal_scrubbing", "run_causal_scrubbing"),
    "das_iia": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.01_das_iia", "run_das_iia"),
    "iia_variants": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.15_iia_variants", "run_iia_variants"),
    "path_patching": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.33_path_patching", "run_path_patching"),
    "counterfactual_consistency": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.34_counterfactual_consistency", "run_counterfactual_consistency"),
    "multi_axis_iia": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.31_multi_axis_iia", "run_multi_axis_iia"),
    "corrupt_restore": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.20_corrupt_restore", "run_corrupt_restore"),
    "intermediate_state_prediction": ("mechval.metrics.mechanistic_interpretability.core.causal.counterfactual_das.79_intermediate_state_prediction", "run_intermediate_state_prediction"),
    "eap": ("mechval.metrics.mechanistic_interpretability.core.causal.eap.91_eap", "run_eap"),
    "cross_task_transfer": ("mechval.metrics.mechanistic_interpretability.core.causal.granger_te.32_cross_task_iia_transfer", "run_cross_task_transfer"),
    "cross_model_invariance": ("mechval.metrics.mechanistic_interpretability.core.causal.transportability.38_cross_model_invariance", "run_cross_model_invariance"),
    "hyperparam_sensitivity": ("mechval.metrics.mechanistic_interpretability.core.causal.mdl_slt.29_hyperparam_sensitivity", "run_hyperparam_sensitivity"),
    "pid": ("mechval.metrics.mechanistic_interpretability.core.causal.pid.08_pid", "run_pid"),

    # ── Mechanistic Interpretability: Information-theoretic ───────────────
    "ocse": ("mechval.metrics.mechanistic_interpretability.core.information.osce.07_ocse", "run_ocse"),
    "notears": ("mechval.metrics.mechanistic_interpretability.core.information.notears_dag.09_notears", "run_notears"),
    "mutual_information": ("mechval.metrics.mechanistic_interpretability.core.information.mutual_info.54_mutual_information", "run_mutual_information"),
    "conditional_mi": ("mechval.metrics.mechanistic_interpretability.core.information.conditional_mi.55_conditional_mi", "run_conditional_mi"),
    "granger_causality": ("mechval.metrics.mechanistic_interpretability.core.information.granger.56_granger_causality", "run_granger"),
    "info_bottleneck": ("mechval.metrics.mechanistic_interpretability.core.information.info_bottleneck.57_info_bottleneck", "run_info_bottleneck"),
    "o_information": ("mechval.metrics.mechanistic_interpretability.core.information.synergistic_info.58_o_information", "run_o_information"),
    "transfer_entropy": ("mechval.metrics.mechanistic_interpretability.core.information.transfer_entropy.53_transfer_entropy", "run_transfer_entropy"),

    # ── Mechanistic Interpretability: Representational ───────────────────
    "attention_entropy": ("mechval.metrics.mechanistic_interpretability.core.representational.attention_entropy.E11_attention_entropy", "run"),
    "cka": ("mechval.metrics.mechanistic_interpretability.core.representational.cka.92_cka", "run_cka"),
    "cka_cross_arch": ("mechval.metrics.mechanistic_interpretability.core.representational.cka.E6b_cka_cross_arch", "run_cka_analysis"),
    "probe_decodability": ("mechval.metrics.mechanistic_interpretability.core.representational.linear_probe.75_probe_decodability", "run_probe_decodability"),
    "causal_representation": ("mechval.metrics.mechanistic_interpretability.core.representational.linear_probe.76_causal_representation", "run_causal_representation"),

    # ── Mechanistic Interpretability: Behavioral ─────────────────────────
    "ce_delta": ("mechval.metrics.mechanistic_interpretability.core.behavioral.ce_delta.43_ce_delta", "run_ce_delta"),
    "per_token_nll": ("mechval.metrics.mechanistic_interpretability.core.behavioral.per_token_nll.44_per_token_nll", "run_per_token_nll"),
    "calibration": ("mechval.metrics.mechanistic_interpretability.core.behavioral.calibration.45_calibration", "run_calibration"),
    "generalization_gap": ("mechval.metrics.mechanistic_interpretability.core.behavioral.generalization_gap.46_generalization_gap", "run_generalization_gap"),
    "mdl_compression": ("mechval.metrics.mechanistic_interpretability.core.behavioral.mdl_compression.47_mdl_compression", "run_mdl_compression"),
    "subnetwork_probe": ("mechval.metrics.mechanistic_interpretability.core.behavioral.subnetwork_probe.48_subnetwork_probe", "run_subnetwork_probe"),
    "output_variants": ("mechval.metrics.mechanistic_interpretability.core.behavioral.logit_diff_recovery.21_output_variants", "run_output_variants"),
    "output_variants_kl": ("mechval.metrics.mechanistic_interpretability.core.behavioral.kl_divergence.21_output_variants", "run_output_variants"),
    "output_variants_topk": ("mechval.metrics.mechanistic_interpretability.core.behavioral.topk_accuracy.21_output_variants", "run_output_variants"),
    "corrupt_restore_behavioral": ("mechval.metrics.mechanistic_interpretability.core.behavioral.logit_diff_recovery.20_corrupt_restore", "run_corrupt_restore"),
    "mean_centered_logit": ("mechval.metrics.mechanistic_interpretability.core.behavioral.logit_diff_recovery.22_mean_centered_logit", "run_mean_centered_logit"),
    "normative_account": ("mechval.metrics.mechanistic_interpretability.core.behavioral.generalization_gap.80_normative_account", "run_normative_account"),
    "error_boundary": ("mechval.metrics.mechanistic_interpretability.core.behavioral.generalization_gap.81_error_boundary_analysis", "run_error_boundary_analysis"),
    "boundary_sweep": ("mechval.metrics.mechanistic_interpretability.core.behavioral.construct_boundary.RI3b_boundary_sweep", "run_boundary_sweep"),
    "epistemic_gradient": ("mechval.metrics.mechanistic_interpretability.core.behavioral.minimal_pairs.A5_epistemic_gradient", "run_epistemic_gradient"),
    "cross_task_generalization": ("mechval.metrics.mechanistic_interpretability.core.behavioral.cross_task_transfer.E6b_cross_task_generalization", "run_cross_task_generalization"),

    # ── Mechanistic Interpretability: Methods / Discovery ────────────────
    "sparse_feature_circuits": ("mechval.metrics.mechanistic_interpretability.methods.discovery.sparse_feature_circuits.92_sfc", "run_sfc"),
    "automatic_circuit_discovery": ("mechval.metrics.mechanistic_interpretability.methods.discovery.automatic_circuit_discovery.94_acdc", "run_acdc"),
    "relevance_patching": ("mechval.metrics.mechanistic_interpretability.methods.discovery.relevance_patching.95_relp", "run_relevance_patching"),
    "contextual_decomposition": ("mechval.metrics.mechanistic_interpretability.methods.discovery.contextual_decomposition.96_contextual_decomposition", "run_contextual_decomposition"),
    "information_bottleneck_circuit": ("mechval.metrics.mechanistic_interpretability.methods.discovery.information_bottleneck.97_information_bottleneck", "run_information_bottleneck"),
    "position_aware_eap": ("mechval.metrics.mechanistic_interpretability.methods.discovery.position_aware_eap.98_position_aware_eap", "run_position_aware_eap"),
    "relp": ("mechval.metrics.mechanistic_interpretability.methods.discovery.relevance_patching_lrp.106_relp", "run_relp"),
    "adversarial_parameter_decomposition": ("mechval.metrics.mechanistic_interpretability.methods.discovery.adversarial_parameter_decomposition.105_vpd", "run_vpd"),
    "circuitlens_weight_circuits": ("mechval.metrics.mechanistic_interpretability.methods.discovery.circuitlens_weight_circuits.119_circuitlens", "run_circuitlens"),

    # ── Mechanistic Interpretability: Methods / Steering ─────────────────
    "contrastive_activation_addition": ("mechval.metrics.mechanistic_interpretability.methods.steering.contrastive_activation_addition.93_caa", "run_caa"),
    "concept_erasure": ("mechval.metrics.mechanistic_interpretability.methods.steering.concept_erasure.99_concept_erasure", "run_concept_erasure"),
    "representation_engineering": ("mechval.metrics.mechanistic_interpretability.methods.steering.representation_engineering.100_representation_engineering", "run_representation_engineering"),
    "steering_reliability": ("mechval.metrics.mechanistic_interpretability.methods.steering.steering_reliability.102_steering_reliability", "run_steering_reliability"),
    "cross_model_transfer": ("mechval.metrics.mechanistic_interpretability.methods.steering.cross_model_transfer.111_cross_model_transfer", "run_cross_model_transfer"),

    # ── Mechanistic Interpretability: Benchmarks ─────────────────────────
    "axbench": ("mechval.metrics.mechanistic_interpretability.benchmarks.axbench.101_axbench", "run_axbench"),
    "saebench": ("mechval.metrics.mechanistic_interpretability.benchmarks.saebench.EX9_saebench", "run_saebench"),
    "ce_bench": ("mechval.metrics.mechanistic_interpretability.benchmarks.ce_bench.EX10_ce_bench", "run_ce_bench"),
    "mib_causal_variable": ("mechval.metrics.mechanistic_interpretability.benchmarks.mib_causal_variable.103_mib_causal_variable", "run_mib_causal_variable"),
    "failure_prediction": ("mechval.metrics.mechanistic_interpretability.benchmarks.failure_prediction.103_failure_prediction", "run_failure_prediction"),
    "eval_awareness_format_control": ("mechval.metrics.mechanistic_interpretability.benchmarks.eval_awareness.104_eval_awareness", "run_eval_awareness"),

    # ── Mechanistic Interpretability: Methods / Evaluation ───────────────
    "autointerp": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.autointerp.EX3_autointerp", "run_autointerp"),
    "feature_absorption": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.feature_absorption.EX4_feature_absorption", "run_feature_absorption"),
    "natural_language_autoencoder": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.natural_language_autoencoder.EX5_nla_reconstruction", "run_nla_reconstruction"),
    "rule_based_descriptions": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.rule_based_descriptions.EX6_rule_descriptions", "run_rule_descriptions"),
    "topk_scaling": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.topk_scaling.EX7_topk_scaling", "run_topk_scaling"),
    "rope_massive_value_filter": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.rope_massive_value_filter.EX8_rope_massive_value", "run_rope_massive_value"),
    "latent_self_consistency": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.latent_self_consistency.EX11_self_consistency", "run_self_consistency"),
    "cot_faithfulness": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.cot_faithfulness.108_cot_faithfulness", "run_cot_faithfulness"),
    "model_alignment_search": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.model_alignment_search.107_mas", "run_mas"),
    "activation_reasoning": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.activation_reasoning.109_activation_reasoning", "run_activation_reasoning"),
    "layer_navigator": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.layer_navigator.112_layer_navigator", "run_layer_navigator"),
    "gradsae_causal": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.gradsae_causal.113_gradsae_causal", "run_gradsae_causal"),
    "output_centric_description": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.output_centric_description.116_output_centric", "run_output_centric"),
    "functional_localizer": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.functional_localizer.121_functional_localizer", "run_functional_localizer"),
    "modcirc_modularity": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.modcirc_modularity.122_modcirc", "run_modcirc_modularity"),
    "latent_reasoning_validity": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.latent_reasoning_validity.123_latent_reasoning", "run_latent_reasoning_validity"),
    "semantic_hub_convergence": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.semantic_hub_convergence.124_semantic_hub", "run_semantic_hub_convergence"),
    "dual_mechanism_discriminant": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.dual_mechanism_discriminant.125_dual_mechanism", "run_dual_mechanism_discriminant"),
    "neuronpedia_agreement": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.neuronpedia_agreement.126_neuronpedia_agreement", "run_neuronpedia_agreement"),
    "clt_graph_faithfulness": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.clt_graph_faithfulness.EX29_clt_graph_faithfulness", "run_clt_graph_faithfulness"),
    "clt_cross_prompt_consistency": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.clt_cross_prompt_consistency.EX30_clt_cross_prompt_consistency", "run_clt_cross_prompt_consistency"),
    "clt_error_fraction": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.clt_error_fraction.EX31_clt_error_fraction", "run_clt_error_fraction"),
    "clt_missing_attention": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.clt_missing_attention.EX32_clt_missing_attention", "run_clt_missing_attention"),
    "clt_minimality_sensitivity": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.clt_minimality_sensitivity.EX33_clt_minimality_sensitivity", "run_clt_minimality_sensitivity"),
    "atlas_alignment": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.atlas_alignment.133_atlas_alignment", "run_atlas_alignment"),
    "mot_alignment": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.mot_alignment.134_mot_alignment", "run_mot_alignment"),
    "safety_subspace": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.safety_subspace.135_safety_subspace", "run_safety_subspace"),
    "safety_one_shot": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.safety_one_shot.137_safety_one_shot", "run_safety_one_shot"),
    "alignment_interpretability": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.alignment_interpretability.138_alignment_interpretability", "run_alignment_interpretability"),
    "transcoder_composability": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.transcoder_composability.139_transcoder_composability", "run_transcoder_composability"),
    "transcoder_sae_agreement": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.transcoder_sae_agreement.140_transcoder_sae_agreement", "run_transcoder_sae_agreement"),
    "transcoder_decomposition": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.transcoder_decomposition.141_transcoder_decomposition", "run_transcoder_decomposition"),
    "crosscoder_persistence": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.crosscoder_persistence.142_crosscoder_persistence", "run_crosscoder_persistence"),
    "crosscoder_model_diff": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.crosscoder_model_diff.143_crosscoder_model_diff", "run_crosscoder_model_diff"),
    "crosscoder_artifact_detection": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.crosscoder_artifact_detection.144_crosscoder_artifact_detection", "run_crosscoder_artifact_detection"),
    "nla_semantic_validity": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.nla_semantic_validity.130_nla_semantic_validity", "run_nla_semantic_validity"),
    "weight_sparse_circuit": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.weight_sparse_circuit.131_weight_sparse_circuit", "run_weight_sparse_circuit"),
    "assistant_axis": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.assistant_axis.132_assistant_axis", "run_assistant_axis"),

    # ── Mechanistic Interpretability: Methods / Evaluation (Part XI)
    "adversarial_ablation_verification": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.adversarial_ablation.127_adversarial_ablation", "run_adversarial_ablation_verification"),
    "actionability_score": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.actionability.128_actionability", "run_actionability_score"),
    "surprise_reduction": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.surprise_reduction.129_surprise_reduction", "run_surprise_reduction"),
    "behavior_capability_gap": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.behavior_capability_gap.130_behavior_capability_gap", "run_behavior_capability_gap"),
    "safety_claim_reliability": ("mechval.metrics.mechanistic_interpretability.methods.evaluation.safety_claim_reliability.131_safety_claim_reliability", "run_safety_claim_reliability"),

    # ── Philosophy of Science ────────────────────────────────────────────
    "sigma_ablation": ("mechval.metrics.philosophy_of_science.woodward.03_sigma_ablation", "run_sigma_ablation"),
    "resample_complement": ("mechval.metrics.philosophy_of_science.woodward.35_resample_complement", "run_resample_complement"),
    "misalignment": ("mechval.metrics.philosophy_of_science.woodward.37_misalignment_score", "run_misalignment"),
    "cate": ("mechval.metrics.philosophy_of_science.rubin_cate.06_cate", "run_cate"),
    "intervention_specificity": ("mechval.metrics.philosophy_of_science.rubin_cate.25_intervention_specificity", "run_intervention_specificity"),
    "replacement_test": ("mechval.metrics.philosophy_of_science.mdc_glennan.72_replacement_test", "run_replacement_test"),
    "composition_test": ("mechval.metrics.philosophy_of_science.mdc_glennan.78_composition_test", "run_composition_test"),
    "operation_specification": ("mechval.metrics.philosophy_of_science.mdc_glennan.70_operation_specification", "run_operation_specification"),
    "held_out_prediction": ("mechval.metrics.philosophy_of_science.mdc_glennan.71_held_out_prediction", "run_held_out_prediction"),
    "procedure_specification": ("mechval.metrics.philosophy_of_science.mdc_glennan.77_procedure_specification", "run_procedure_specification"),
    "logic_gates": ("mechval.metrics.philosophy_of_science.mdc_glennan.19_logic_gates", "run_logic_gates"),
    "minimality_class": ("mechval.metrics.philosophy_of_science.minimality.C4b_minimality_class", "run_minimality_class"),

    # ── Neuroscience ─────────────────────────────────────────────────────
    "mediation": ("mechval.metrics.neuroscience.mediation.05_mediation", "run_mediation"),
    "mediation_v2": ("mechval.metrics.neuroscience.mediation.05_mediation_v2", "run_mediation_v2"),
    "pse": ("mechval.metrics.neuroscience.mediation.24_pse", "run_pse"),
    "atp_star": ("mechval.metrics.neuroscience.atp_star.G2b_atp_star", "run_atp_star"),

    # ── Pharmacology ─────────────────────────────────────────────────────
    "dose_response": ("mechval.metrics.pharmacology.dose_response.95_dose_response", "run_dose_response"),
    "effect_size": ("mechval.metrics.pharmacology.effect_size.90_effect_size", "run_effect_size"),

    # ── Measurement Theory ───────────────────────────────────────────────
    "dprime": ("mechval.metrics.measurement_theory.signal_detection.EX1_dprime", "run_dprime"),
    "dif": ("mechval.metrics.measurement_theory.psychometrics.EX2_dif", "run_dif"),
    "weber_fechner": ("mechval.metrics.measurement_theory.psychophysics.EX11_weber_fechner", "run_weber_fechner"),
    "mib_faithfulness": ("mechval.metrics.measurement_theory.mib_faithfulness.MIB_faithfulness_curve", "run_mib_faithfulness"),
    "architecture_duality": ("mechval.metrics.measurement_theory.architecture_duality.110_architecture_duality", "run_architecture_duality"),
    "weightlens_convergence": ("mechval.metrics.measurement_theory.weightlens_convergence.114_weightlens", "run_weightlens_convergence"),
    "adaptive_sparsity": ("mechval.metrics.measurement_theory.adaptive_sparsity.120_adaptive_sparsity", "run_adaptive_sparsity"),
    "prism_polysemanticity": ("mechval.metrics.measurement_theory.prism_polysemanticity.117_prism", "run_prism_polysemanticity"),
    "matryoshka_consistency": ("mechval.metrics.measurement_theory.matryoshka_consistency.118_matryoshka", "run_matryoshka_consistency"),
    "core_stability": ("mechval.metrics.measurement_theory.core_stability.115_core_stability", "run_core_stability"),
    "superposition_regime": ("mechval.metrics.measurement_theory.superposition_regime.126_superposition_regime", "run_superposition_regime"),
    "saebench_audit": ("mechval.metrics.measurement_theory.saebench_reliability_audit.131_saebench_audit", "run_saebench_audit"),
    "reproducibility_check": ("mechval.metrics.measurement_theory.reproducibility_check.132_reproducibility", "run_reproducibility_check"),
    "safety_sve": ("mechval.metrics.measurement_theory.safety_sve.136_safety_sve", "run_safety_sve"),
    "nla_sae_convergence": ("mechval.metrics.measurement_theory.nla_sae_convergence.133_nla_sae_convergence", "run_nla_sae_convergence"),

    # ── Information Theory ───────────────────────────────────────────────
    "channel_capacity": ("mechval.metrics.information_theory.IT1_channel_capacity", "run_channel_capacity"),
    "rate_distortion": ("mechval.metrics.information_theory.IT2_rate_distortion", "run_rate_distortion"),
    "kolmogorov_complexity": ("mechval.metrics.information_theory.IT3_kolmogorov_complexity", "run_kolmogorov_complexity"),

    # ── Economics ─────────────────────────────────────────────────────────
    "mechanism_design": ("mechval.metrics.economics.ECON1_mechanism_design", "run_mechanism_design"),
    "attention_auction": ("mechval.metrics.economics.ECON2_attention_auction", "run_attention_auction"),
    "pairwise_synergy": ("mechval.metrics.economics.hedonic_pas.94_pairwise_synergy", "run_pairwise_synergy"),
    "shapley_interactions": ("mechval.metrics.economics.hedonic_pas.H1_shapley_interactions", "run_hedonic_synergy"),

    # ── Genetics ──────────────────────────────────────────────────────────
    "knock_in": ("mechval.metrics.genetics.GN1_knock_in", "run_knock_in"),
    "epistasis": ("mechval.metrics.genetics.GN2_epistasis", "run_epistasis"),
    "chimera": ("mechval.metrics.genetics.GN3_chimera", "run_chimera"),
    "convergent_evolution": ("mechval.metrics.genetics.GN4_convergent_evolution", "run_convergent_evolution"),
    "phylogenetic_tracking": ("mechval.metrics.genetics.GN5_phylogenetic_tracking", "run_phylogenetic_tracking"),

}


METRIC_METADATA: dict[str, dict[str, str | list[str]]] = {
    # ── Mechanistic Interpretability ─────────────────────────────────────
    "k_composition":              {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "copying_score":              {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "qk_norms":                   {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "cmd":                        {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "edge_jaccard":               {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "weight_eap_jaccard":         {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "network_motifs":             {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "motif_enrichment":           {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "capacity_utilization":       {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "k_alignment":                {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "weight_extended":            {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "spectral_svd":               {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "path_identification":        {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "edge_necessity":             {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "path_specificity":           {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I3"]},
    "compositional_sufficiency":  {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "graph_minimality":           {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C4"]},
    "attention_clustering":       {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "llc":                        {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "activation_patching":        {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "logit_diff":                 {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "role_ablation":              {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "causal_scrubbing":           {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "I5"]},
    "das_iia":                    {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "I3"]},
    "iia_variants":               {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I4"]},
    "path_patching":              {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1", "I3"]},
    "counterfactual_consistency": {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I4"]},
    "multi_axis_iia":             {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "I3"]},
    "corrupt_restore":            {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "intermediate_state_prediction": {"lens": "mechanistic_interpretability", "validity_type": "internal",  "criteria": ["I2"]},
    "eap":                        {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "cross_task_transfer":        {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "cross_model_invariance":     {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E6"]},
    "hyperparam_sensitivity":     {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M2"]},
    "pid":                        {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "ocse":                       {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "notears":                    {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "mutual_information":         {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "conditional_mi":             {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I3"]},
    "granger_causality":          {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "info_bottleneck":            {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C4"]},
    "o_information":              {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "transfer_entropy":           {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "attention_entropy":          {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2"]},
    "cka":                        {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "cka_cross_arch":             {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E6"]},
    "probe_decodability":         {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "causal_representation":      {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "I3"]},
    "ce_delta":                   {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "per_token_nll":              {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "calibration":                {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M5"]},
    "generalization_gap":         {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "mdl_compression":            {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C4"]},
    "subnetwork_probe":           {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "output_variants":            {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "output_variants_kl":         {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "output_variants_topk":       {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "corrupt_restore_behavioral": {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "mean_centered_logit":        {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "normative_account":          {"lens": "mechanistic_interpretability", "validity_type": "interpretive", "criteria": ["V2"]},
    "error_boundary":             {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "boundary_sweep":             {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C1"]},
    "epistemic_gradient":         {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C1"]},
    "cross_task_generalization":  {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    # Methods
    "sparse_feature_circuits":    {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1", "I2"]},
    "automatic_circuit_discovery": {"lens": "mechanistic_interpretability", "validity_type": "internal",    "criteria": ["I1"]},
    "relevance_patching":         {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "contextual_decomposition":   {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "information_bottleneck_circuit": {"lens": "mechanistic_interpretability", "validity_type": "construct","criteria": ["C4"]},
    "position_aware_eap":         {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1"]},
    "relp":                       {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M1", "M4"]},
    "adversarial_parameter_decomposition": {"lens": "mechanistic_interpretability", "validity_type": "internal", "criteria": ["I1", "I2"]},
    "circuitlens_weight_circuits": {"lens": "mechanistic_interpretability", "validity_type": "construct",  "criteria": ["C2", "C5"]},
    "contrastive_activation_addition": {"lens": "mechanistic_interpretability", "validity_type": "external","criteria": ["E1", "E3"]},
    "concept_erasure":            {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E3"]},
    "representation_engineering": {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E1"]},
    "steering_reliability":       {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "cross_model_transfer":       {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E6"]},
    "axbench":                    {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4", "M6"]},
    "saebench":                   {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4", "M6"]},
    "ce_bench":                   {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "mib_causal_variable":        {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "I3"]},
    "failure_prediction":         {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "eval_awareness_format_control": {"lens": "mechanistic_interpretability", "validity_type": "measurement", "criteria": ["M6"]},
    "autointerp":                 {"lens": "mechanistic_interpretability", "validity_type": "interpretive", "criteria": ["V3"]},
    "feature_absorption":         {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M6"]},
    "natural_language_autoencoder": {"lens": "mechanistic_interpretability", "validity_type": "measurement","criteria": ["M6"]},
    "rule_based_descriptions":    {"lens": "mechanistic_interpretability", "validity_type": "interpretive", "criteria": ["V3"]},
    "topk_scaling":               {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M2"]},
    "rope_massive_value_filter":  {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M3"]},
    "latent_self_consistency":    {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M1"]},
    "cot_faithfulness":           {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2"]},
    "activation_reasoning":       {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E5"]},
    "layer_navigator":            {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "gradsae_causal":             {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1", "I2"]},
    "model_alignment_search":     {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "output_centric_description": {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "C5"]},
    "functional_localizer":       {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I1", "I3", "C5"]},
    "modcirc_modularity":         {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5", "E2"]},
    "latent_reasoning_validity":  {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["E1", "I2", "C5"]},
    "semantic_hub_convergence":   {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "dual_mechanism_discriminant": {"lens": "mechanistic_interpretability", "validity_type": "construct",   "criteria": ["C4"]},
    "neuronpedia_agreement":      {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "clt_graph_faithfulness":     {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I2", "C2"]},
    "clt_cross_prompt_consistency": {"lens": "mechanistic_interpretability", "validity_type": "measurement", "criteria": ["M1", "M2"]},
    "clt_error_fraction":         {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M6", "I5"]},
    "clt_missing_attention":      {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["I5", "M6"]},
    "clt_minimality_sensitivity": {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C4", "M4"]},
    "atlas_alignment":            {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "mot_alignment":              {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "safety_subspace":            {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["E2", "I1"]},
    "safety_one_shot":            {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["E2"]},
    "alignment_interpretability": {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C4"]},
    "transcoder_composability":   {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["C2", "I2"]},
    "transcoder_sae_agreement":   {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C5"]},
    "transcoder_decomposition":   {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M6", "C2"]},
    "crosscoder_persistence":     {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["C2", "M2"]},
    "crosscoder_model_diff":      {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["C3", "M3"]},
    "crosscoder_artifact_detection": {"lens": "mechanistic_interpretability", "validity_type": "measurement", "criteria": ["M6", "I5"]},
    "nla_semantic_validity":      {"lens": "mechanistic_interpretability", "validity_type": "construct",    "criteria": ["E1", "C5"]},
    "weight_sparse_circuit":      {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["E2", "I1"]},
    "assistant_axis":             {"lens": "mechanistic_interpretability", "validity_type": "internal",     "criteria": ["E2", "M1", "C4"]},
    # ── Mechanistic Interpretability: Methods / Evaluation (Part XI)
    "adversarial_ablation_verification": {"lens": "mechanistic_interpretability", "validity_type": "internal", "criteria": ["I2"]},
    "actionability_score":        {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E1"]},
    "surprise_reduction":         {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M4"]},
    "behavior_capability_gap":    {"lens": "mechanistic_interpretability", "validity_type": "external",     "criteria": ["E3"]},
    "safety_claim_reliability":   {"lens": "mechanistic_interpretability", "validity_type": "measurement",  "criteria": ["M1"]},

    # ── Philosophy of Science ────────────────────────────────────────────
    "sigma_ablation":             {"lens": "philosophy_of_science", "validity_type": "internal",     "criteria": ["I1"]},
    "resample_complement":        {"lens": "philosophy_of_science", "validity_type": "internal",     "criteria": ["I5"]},
    "misalignment":               {"lens": "philosophy_of_science", "validity_type": "internal",     "criteria": ["I5"]},
    "cate":                       {"lens": "philosophy_of_science", "validity_type": "internal",     "criteria": ["I1"]},
    "intervention_specificity":   {"lens": "philosophy_of_science", "validity_type": "internal",     "criteria": ["I3"]},
    "replacement_test":           {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C2"]},
    "composition_test":           {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C2"]},
    "operation_specification":    {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C2"]},
    "held_out_prediction":        {"lens": "philosophy_of_science", "validity_type": "external",     "criteria": ["E5"]},
    "procedure_specification":    {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C2"]},
    "logic_gates":                {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C2"]},
    "minimality_class":           {"lens": "philosophy_of_science", "validity_type": "construct",    "criteria": ["C4"]},

    # ── Neuroscience ─────────────────────────────────────────────────────
    "mediation":                  {"lens": "neuroscience", "validity_type": "internal",     "criteria": ["I1", "I3"]},
    "mediation_v2":               {"lens": "neuroscience", "validity_type": "internal",     "criteria": ["I1", "I3"]},
    "pse":                        {"lens": "neuroscience", "validity_type": "internal",     "criteria": ["I1"]},
    "atp_star":                   {"lens": "neuroscience", "validity_type": "internal",     "criteria": ["I2"]},

    # ── Pharmacology ─────────────────────────────────────────────────────
    "dose_response":              {"lens": "pharmacology", "validity_type": "external",     "criteria": ["E2"]},
    "effect_size":                {"lens": "pharmacology", "validity_type": "external",     "criteria": ["E4"]},

    # ── Measurement Theory ───────────────────────────────────────────────
    "dprime":                     {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M4"]},
    "dif":                        {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M2"]},
    "weber_fechner":              {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M4"]},
    "mib_faithfulness":           {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M4", "M6"]},
    "architecture_duality":       {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M2", "M6"]},
    "weightlens_convergence":     {"lens": "measurement_theory", "validity_type": "construct",    "criteria": ["C5"]},
    "adaptive_sparsity":          {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["E1", "M6"]},
    "prism_polysemanticity":      {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M6", "E1"]},
    "matryoshka_consistency":     {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M1", "M2"]},
    "core_stability":             {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M1"]},
    "superposition_regime":       {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M6"]},
    "saebench_audit":             {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M1", "M2"]},
    "reproducibility_check":      {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M1"]},
    "safety_sve":                 {"lens": "measurement_theory", "validity_type": "measurement",  "criteria": ["M1", "M6"]},
    "nla_sae_convergence":        {"lens": "measurement_theory", "validity_type": "construct",    "criteria": ["C5"]},

    # ── Information Theory ───────────────────────────────────────────────
    "channel_capacity":           {"lens": "information_theory", "validity_type": "construct",    "criteria": ["C2"]},
    "rate_distortion":            {"lens": "information_theory", "validity_type": "construct",    "criteria": ["C4"]},
    "kolmogorov_complexity":      {"lens": "information_theory", "validity_type": "construct",    "criteria": ["C4"]},

    # ── Economics ─────────────────────────────────────────────────────────
    "mechanism_design":           {"lens": "economics", "validity_type": "construct",    "criteria": ["C2"]},
    "attention_auction":          {"lens": "economics", "validity_type": "construct",    "criteria": ["C2"]},
    "pairwise_synergy":           {"lens": "economics", "validity_type": "internal",     "criteria": ["I1"]},
    "shapley_interactions":       {"lens": "economics", "validity_type": "internal",     "criteria": ["I1"]},

    # ── Genetics ──────────────────────────────────────────────────────────
    "knock_in":                   {"lens": "genetics", "validity_type": "internal",     "criteria": ["I2"]},
    "epistasis":                  {"lens": "genetics", "validity_type": "internal",     "criteria": ["I1"]},
    "chimera":                    {"lens": "genetics", "validity_type": "external",     "criteria": ["E6"]},
    "convergent_evolution":       {"lens": "genetics", "validity_type": "external",     "criteria": ["E6"]},
    "phylogenetic_tracking":      {"lens": "genetics", "validity_type": "external",     "criteria": ["E6"]},

}


CALIBRATION_REGISTRY: dict[str, tuple[str, str]] = {
    "bootstrap": ("mechval.calibrations.bootstrap_stability.11_bootstrap", "run_bootstrap"),
    "seed_variance": ("mechval.calibrations.bootstrap_stability.30_seed_variance", "run_seed_variance"),
    "distributional_characterization": ("mechval.calibrations.bootstrap_stability.73_distributional_characterization", "run_distributional_characterization"),
    "distributional_stability": ("mechval.calibrations.bootstrap_stability.74_distributional_stability", "run_distributional_stability"),
    "convergent_validity": ("mechval.calibrations.convergent_validity.12_convergent_validity", "run_convergent_validity"),
    "nomological_validity": ("mechval.calibrations.convergent_validity.23_nomological_validity", "run_nomological_validity"),
    "incremental_validity": ("mechval.calibrations.convergent_validity.36_incremental_validity", "run_incremental_validity"),
    "discriminant_validity": ("mechval.calibrations.discriminant_validity.17_discriminant_validity", "run_discriminant_validity"),
    "measurement_invariance": ("mechval.calibrations.measurement_invariance.13_measurement_invariance", "run_measurement_invariance"),
    "ablation_invariance": ("mechval.calibrations.ablation_invariance.98_ablation_invariance", "run_ablation_invariance"),
    "method_invariance": ("mechval.calibrations.ablation_invariance.E1b_method_invariance", "run_ablation_method_invariance"),
    "certified_stability": ("mechval.calibrations.certified_stability.M3b_certified_stable", "run_certified_stability"),
    "reliability_suite": ("mechval.calibrations.test_retest.16_reliability_suite", "run_reliability_suite"),
    "internal_consistency": ("mechval.calibrations.internal_consistency.16_reliability_suite", "run_reliability_suite"),
}


def _resolve_lens(mod_path: str) -> str:
    parts = mod_path.split(".")
    if len(parts) >= 3:
        return parts[2]
    return "external"


def _resolve_family(mod_path: str) -> str:
    parts = mod_path.split(".")
    lens = _resolve_lens(mod_path)
    if lens == "mechanistic_interpretability" and len(parts) >= 4:
        sublevel = parts[3]
        if sublevel == "methods" and len(parts) >= 5:
            return f"mechanistic_interpretability/methods/{parts[4]}"
        return f"mechanistic_interpretability/{sublevel}"
    return lens


def _all_metrics() -> dict[str, tuple[str, str]]:
    """Merge built-in METRIC_REGISTRY with externally registered metrics.

    Built-in metrics take precedence over external ones with the same ID.
    """
    external = get_external_metrics()
    return {**external, **METRIC_REGISTRY}


METRIC_FAMILIES: dict[str, str] = {
    name: _resolve_family(mod) for name, (mod, _) in METRIC_REGISTRY.items()
}


def _all_metric_families() -> dict[str, str]:
    """Metric families including external registrations."""
    result = dict(METRIC_FAMILIES)
    for name, (mod, _) in get_external_metrics().items():
        if name not in result:
            result[name] = _resolve_family(mod)
    return result


def list_lenses() -> list[str]:
    """List all unique lenses from METRIC_METADATA."""
    return sorted({m["lens"] for m in METRIC_METADATA.values()})


def list_families() -> list[str]:
    return sorted(set(_all_metric_families().values()))


def list_metrics(
    family: str | None = None,
    *,
    lens: str | None = None,
    validity_type: str | None = None,
) -> list[str]:
    """List all metrics, optionally filtered by family, lens, or validity_type."""
    all_m = _all_metrics()
    result = set(all_m.keys())
    if family is not None:
        families = _all_metric_families()
        result &= {k for k in all_m if families.get(k) == family}
    if lens is not None:
        result &= {k for k, m in METRIC_METADATA.items() if m.get("lens") == lens}
    if validity_type is not None:
        result &= {k for k, m in METRIC_METADATA.items() if m.get("validity_type") == validity_type}
    return sorted(result)


def list_calibrations() -> list[str]:
    return sorted(CALIBRATION_REGISTRY.keys())


def get_metadata(name: str) -> dict:
    """Return metadata for a metric (lens, validity_type, criteria)."""
    return METRIC_METADATA.get(name, {})


def dispatch(registry: dict, name: str, **kwargs):
    if name in registry:
        mod_path, fn_name = registry[name]
    else:
        external = get_external_metrics()
        if name in external:
            mod_path, fn_name = external[name]
        else:
            raise KeyError(f"Unknown metric/calibration: {name!r}")

    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)

    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())

    if "model" in params and "device" not in params:
        if "model" not in kwargs or kwargs["model"] is None:
            from mechval.metrics.common import load_model
            kwargs["model"] = load_model(
                kwargs.pop("model_name", "gpt2"),
                kwargs.pop("device", "cpu"),
            )
        else:
            kwargs.pop("device", None)
            kwargs.pop("model_name", None)

    if "tasks" in params and "tasks" not in kwargs:
        from mechval.metrics.common import CIRCUIT_TASKS
        kwargs["tasks"] = CIRCUIT_TASKS

    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(**accepted)
