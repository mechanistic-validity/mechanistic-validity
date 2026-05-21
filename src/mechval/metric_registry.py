"""Metric and calibration registries.

Maps metric/calibration names to (module_path, function_name) tuples
for lazy import dispatch. External metrics registered via
``mechval.registry.register_metric()`` are merged automatically.
"""
import importlib
import inspect

from mechval.registry import get_external_metrics


METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    # ── Core: Structural (weight-space, no forward pass) ───────────────
    "k_composition": ("mechval.metrics.core.structural.k_composition.B10_k_composition", "run"),
    "copying_score": ("mechval.metrics.core.structural.copying_score.B11_copying_score", "run"),
    "qk_norms": ("mechval.metrics.core.structural.qk_norms.B12_qk_norms", "run"),
    "cmd": ("mechval.metrics.core.structural.template_distance.26_cmd", "run_cmd"),
    "edge_jaccard": ("mechval.metrics.core.structural.template_distance.27_edge_jaccard", "run_edge_jaccard"),
    "weight_eap_jaccard": ("mechval.metrics.core.structural.weight_alignment.28_weight_eap_jaccard", "run_weight_eap_jaccard"),
    "network_motifs": ("mechval.metrics.core.structural.network_motifs.97_network_motifs", "run_network_motifs"),
    "motif_enrichment": ("mechval.metrics.core.structural.network_motifs.G7_motif_enrichment", "run_motif_enrichment"),
    "capacity_utilization": ("mechval.metrics.core.structural.effective_rank.B13_capacity_utilization", "run"),
    "k_alignment": ("mechval.metrics.core.structural.effective_rank.B14_k_alignment", "run"),
    "weight_extended": ("mechval.metrics.core.structural.effective_rank.18_weight_extended", "run_weight_extended"),
    "spectral_svd": ("mechval.metrics.core.structural.spectral_svd.18_weight_extended", "run_weight_extended"),
    "path_identification": ("mechval.metrics.core.structural.edge_analysis.82_path_identification", "run_path_identification"),
    "edge_necessity": ("mechval.metrics.core.structural.edge_analysis.83_edge_necessity", "run_edge_necessity"),
    "path_specificity": ("mechval.metrics.core.structural.edge_analysis.84_path_specificity", "run_path_specificity"),
    "compositional_sufficiency": ("mechval.metrics.core.structural.edge_analysis.85_compositional_sufficiency", "run_compositional_sufficiency"),
    "graph_minimality": ("mechval.metrics.core.structural.edge_analysis.86_graph_minimality", "run_graph_minimality"),
    "attention_clustering": ("mechval.metrics.core.structural.attention_clustering.96_attention_clustering", "run_attention_clustering"),
    "llc": ("mechval.metrics.core.structural.llc_rllc.10_llc", "run_llc"),

    # ── Core: Causal (intervention-based) ──────────────────────────────
    "activation_patching": ("mechval.metrics.core.causal.scm_pearl.02_activation_patching", "run_activation_patching"),
    "logit_diff": ("mechval.metrics.core.causal.scm_pearl.logit_diff", "run_logit_diff"),
    "role_ablation": ("mechval.metrics.core.causal.scm_pearl.role_ablation", "run_role_ablation"),
    "causal_scrubbing": ("mechval.metrics.core.causal.scm_pearl.04_causal_scrubbing", "run_causal_scrubbing"),
    "sigma_ablation": ("mechval.metrics.core.causal.woodward.03_sigma_ablation", "run_sigma_ablation"),
    "resample_complement": ("mechval.metrics.core.causal.woodward.35_resample_complement", "run_resample_complement"),
    "misalignment": ("mechval.metrics.core.causal.woodward.37_misalignment_score", "run_misalignment"),
    "das_iia": ("mechval.metrics.core.causal.counterfactual_das.01_das_iia", "run_das_iia"),
    "iia_variants": ("mechval.metrics.core.causal.counterfactual_das.15_iia_variants", "run_iia_variants"),
    "path_patching": ("mechval.metrics.core.causal.counterfactual_das.33_path_patching", "run_path_patching"),
    "counterfactual_consistency": ("mechval.metrics.core.causal.counterfactual_das.34_counterfactual_consistency", "run_counterfactual_consistency"),
    "multi_axis_iia": ("mechval.metrics.core.causal.counterfactual_das.31_multi_axis_iia", "run_multi_axis_iia"),
    "corrupt_restore": ("mechval.metrics.core.causal.counterfactual_das.20_corrupt_restore", "run_corrupt_restore"),
    "mediation": ("mechval.metrics.core.causal.mediation.05_mediation", "run_mediation"),
    "mediation_v2": ("mechval.metrics.core.causal.mediation.05_mediation_v2", "run_mediation_v2"),
    "pse": ("mechval.metrics.core.causal.mediation.24_pse", "run_pse"),
    "cate": ("mechval.metrics.core.causal.rubin_cate.06_cate", "run_cate"),
    "intervention_specificity": ("mechval.metrics.core.causal.rubin_cate.25_intervention_specificity", "run_intervention_specificity"),
    "eap": ("mechval.metrics.core.causal.eap.91_eap", "run_eap"),
    "atp_star": ("mechval.metrics.core.causal.atp_star.G2b_atp_star", "run_atp_star"),
    "pairwise_synergy": ("mechval.metrics.core.causal.hedonic_pas.94_pairwise_synergy", "run_pairwise_synergy"),
    "shapley_interactions": ("mechval.metrics.core.causal.hedonic_pas.H1_shapley_interactions", "run_hedonic_synergy"),
    "minimality_class": ("mechval.metrics.core.causal.minimality.C4b_minimality_class", "run_minimality_class"),
    "replacement_test": ("mechval.metrics.core.causal.mdc_glennan.72_replacement_test", "run_replacement_test"),
    "composition_test": ("mechval.metrics.core.causal.mdc_glennan.78_composition_test", "run_composition_test"),
    "operation_specification": ("mechval.metrics.core.causal.mdc_glennan.70_operation_specification", "run_operation_specification"),
    "held_out_prediction": ("mechval.metrics.core.causal.mdc_glennan.71_held_out_prediction", "run_held_out_prediction"),
    "procedure_specification": ("mechval.metrics.core.causal.mdc_glennan.77_procedure_specification", "run_procedure_specification"),
    "logic_gates": ("mechval.metrics.core.causal.mdc_glennan.19_logic_gates", "run_logic_gates"),
    "intermediate_state_prediction": ("mechval.metrics.core.causal.counterfactual_das.79_intermediate_state_prediction", "run_intermediate_state_prediction"),
    "cross_task_transfer": ("mechval.metrics.core.causal.granger_te.32_cross_task_iia_transfer", "run_cross_task_transfer"),
    "cross_model_invariance": ("mechval.metrics.core.causal.transportability.38_cross_model_invariance", "run_cross_model_invariance"),
    "hyperparam_sensitivity": ("mechval.metrics.core.causal.mdl_slt.29_hyperparam_sensitivity", "run_hyperparam_sensitivity"),

    # ── Core: Information-theoretic ────────────────────────────────────
    "pid": ("mechval.metrics.core.information.pid.08_pid", "run_pid"),
    "ocse": ("mechval.metrics.core.information.osce.07_ocse", "run_ocse"),
    "notears": ("mechval.metrics.core.information.notears_dag.09_notears", "run_notears"),
    "mutual_information": ("mechval.metrics.core.information.mutual_info.54_mutual_information", "run_mutual_information"),
    "conditional_mi": ("mechval.metrics.core.information.conditional_mi.55_conditional_mi", "run_conditional_mi"),
    "granger_causality": ("mechval.metrics.core.information.granger.56_granger_causality", "run_granger"),
    "info_bottleneck": ("mechval.metrics.core.information.info_bottleneck.57_info_bottleneck", "run_info_bottleneck"),
    "o_information": ("mechval.metrics.core.information.synergistic_info.58_o_information", "run_o_information"),
    "transfer_entropy": ("mechval.metrics.core.information.transfer_entropy.53_transfer_entropy", "run_transfer_entropy"),

    # ── Core: Representational ─────────────────────────────────────────
    "attention_entropy": ("mechval.metrics.core.representational.attention_entropy.E11_attention_entropy", "run"),
    "cka": ("mechval.metrics.core.representational.cka.92_cka", "run_cka"),
    "cka_cross_arch": ("mechval.metrics.core.representational.cka.E6b_cka_cross_arch", "run_cka_analysis"),
    "probe_decodability": ("mechval.metrics.core.representational.linear_probe.75_probe_decodability", "run_probe_decodability"),
    "causal_representation": ("mechval.metrics.core.representational.linear_probe.76_causal_representation", "run_causal_representation"),

    # ── Core: Behavioral ───────────────────────────────────────────────
    "effect_size": ("mechval.metrics.core.behavioral.effect_size.90_effect_size", "run_effect_size"),
    "dose_response": ("mechval.metrics.core.behavioral.dose_response.95_dose_response", "run_dose_response"),
    "ce_delta": ("mechval.metrics.core.behavioral.ce_delta.43_ce_delta", "run_ce_delta"),
    "per_token_nll": ("mechval.metrics.core.behavioral.per_token_nll.44_per_token_nll", "run_per_token_nll"),
    "calibration": ("mechval.metrics.core.behavioral.calibration.45_calibration", "run_calibration"),
    "generalization_gap": ("mechval.metrics.core.behavioral.generalization_gap.46_generalization_gap", "run_generalization_gap"),
    "mdl_compression": ("mechval.metrics.core.behavioral.mdl_compression.47_mdl_compression", "run_mdl_compression"),
    "subnetwork_probe": ("mechval.metrics.core.behavioral.subnetwork_probe.48_subnetwork_probe", "run_subnetwork_probe"),
    "output_variants": ("mechval.metrics.core.behavioral.logit_diff_recovery.21_output_variants", "run_output_variants"),
    "output_variants_kl": ("mechval.metrics.core.behavioral.kl_divergence.21_output_variants", "run_output_variants"),
    "output_variants_topk": ("mechval.metrics.core.behavioral.topk_accuracy.21_output_variants", "run_output_variants"),
    "corrupt_restore_behavioral": ("mechval.metrics.core.behavioral.logit_diff_recovery.20_corrupt_restore", "run_corrupt_restore"),
    "mean_centered_logit": ("mechval.metrics.core.behavioral.logit_diff_recovery.22_mean_centered_logit", "run_mean_centered_logit"),
    "normative_account": ("mechval.metrics.core.behavioral.generalization_gap.80_normative_account", "run_normative_account"),
    "error_boundary": ("mechval.metrics.core.behavioral.generalization_gap.81_error_boundary_analysis", "run_error_boundary_analysis"),
    "boundary_sweep": ("mechval.metrics.core.behavioral.construct_boundary.RI3b_boundary_sweep", "run_boundary_sweep"),
    "epistemic_gradient": ("mechval.metrics.core.behavioral.minimal_pairs.A5_epistemic_gradient", "run_epistemic_gradient"),
    "cross_task_generalization": ("mechval.metrics.core.behavioral.cross_task_transfer.E6b_cross_task_generalization", "run_cross_task_generalization"),

    # ── Core: Measurement ─────────────────────────────────────────────
    "dprime": ("mechval.metrics.core.measurement.signal_detection.EX1_dprime", "run_dprime"),
    "dif": ("mechval.metrics.core.measurement.psychometrics.EX2_dif", "run_dif"),
    "weber_fechner": ("mechval.metrics.core.measurement.psychophysics.EX11_weber_fechner", "run_weber_fechner"),

    # ── Extended: economics ──────────────────────────────────────────────
    "mechanism_design": ("mechval.metrics.extended.economics.ECON1_mechanism_design", "run_mechanism_design"),
    "attention_auction": ("mechval.metrics.extended.economics.ECON2_attention_auction", "run_attention_auction"),
    # ── Extended: genetics ────────────────────────────────────────────────
    "knock_in": ("mechval.metrics.extended.genetics.GN1_knock_in", "run_knock_in"),
    "epistasis": ("mechval.metrics.extended.genetics.GN2_epistasis", "run_epistasis"),
    "chimera": ("mechval.metrics.extended.genetics.GN3_chimera", "run_chimera"),
    "convergent_evolution": ("mechval.metrics.extended.genetics.GN4_convergent_evolution", "run_convergent_evolution"),
    "phylogenetic_tracking": ("mechval.metrics.extended.genetics.GN5_phylogenetic_tracking", "run_phylogenetic_tracking"),
    # ── Extended: information theory ──────────────────────────────────────
    "channel_capacity": ("mechval.metrics.extended.information_theory.IT1_channel_capacity", "run_channel_capacity"),
    "rate_distortion": ("mechval.metrics.extended.information_theory.IT2_rate_distortion", "run_rate_distortion"),
    "kolmogorov_complexity": ("mechval.metrics.extended.information_theory.IT3_kolmogorov_complexity", "run_kolmogorov_complexity"),

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


def _resolve_family(mod_path: str) -> str:
    parts = mod_path.split(".")
    # Built-in: mechval.metrics.core.<family>.* or mechval.metrics.<family>.*
    if len(parts) >= 4 and parts[2] == "core":
        return parts[3]
    if len(parts) >= 3:
        return parts[2]
    return "external"


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


def list_families() -> list[str]:
    return sorted(set(_all_metric_families().values()))


def list_metrics(family: str | None = None) -> list[str]:
    """List all metrics (built-in + external), optionally filtered by family."""
    all_m = _all_metrics()
    if family is not None:
        families = _all_metric_families()
        return sorted(k for k in all_m if families.get(k) == family)
    return sorted(all_m)


def list_calibrations() -> list[str]:
    return sorted(CALIBRATION_REGISTRY.keys())


def dispatch(registry: dict, name: str, **kwargs):
    # Check the provided registry first, then fall back to external metrics
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
