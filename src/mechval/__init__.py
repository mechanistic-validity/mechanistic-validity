"""Mechanistic Validity — circuit evaluation framework.

3 Tracks + 4 Views + 4 Gates for evaluating mechanistic interpretability claims.

Quick start::

    import mechval as mv

    # Configure output once (or set MV_OUTPUT_DIR env var)
    mv.set_output_dir("./results")

    # Load a task (gym.make style)
    task = mv.load_task("ioi")
    circuit = task.get_circuit()

    # Run a metric
    results = mv.run("k_composition", tasks=["ioi"])

    # Run a calibration
    results = mv.calibrate("bootstrap", tasks=["ioi"])

    # Track 3: Causal Model Testing
    spec = mv.load_task("ioi").get_claim_spec()
    result = mv.verify(spec, device="cpu")

    # Views (scoring aggregations)
    mv.run_view("effect_estimation", tasks=["ioi"])

    # Gates (precondition checks)
    mv.check_gate("measurement_calibration", task="ioi")

    # List what's available
    mv.list_tasks()                         # all registered circuits
    mv.list_families()                      # ['behavioral', 'causal', ...]
    mv.list_metrics()                       # all 82 metrics
    mv.list_metrics(family="structural")    # just structural metrics
    mv.list_calibrations()                  # all 14 calibrations
    mv.list_domains()                       # linguistics taxonomy
    mv.list_experiment_groups()             # experiment-based grouping
    mv.status()                             # what results exist on disk

    # Tracing (optional, requires pip install mechanistic-validity[weave])
    mv.init_tracing("my-project")

    @mv.op
    def my_metric(model, task):
        ...
"""
from mechval.registry import list_tasks, load_task, list_experiment_groups, list_domains
from mechval.lib.tasks.spec import CircuitSpec
from mechval.lib.tasks.task import CircuitTask
from mechval.spec import MechanisticClaimSpec, SpecVerificationResult
from mechval.views import run_view
from mechval.gates import check_gate
from mechval.tracing import op, init as init_tracing

_METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    # ── Structural (weight-space, no forward pass) ──────────────────────
    "k_composition": ("mechval.metrics.structural.k_composition.B10_k_composition", "run"),
    "copying_score": ("mechval.metrics.structural.copying_score.B11_copying_score", "run"),
    "qk_norms": ("mechval.metrics.structural.qk_norms.B12_qk_norms", "run"),
    "cmd": ("mechval.metrics.structural.template_distance.26_cmd", "run_cmd"),
    "edge_jaccard": ("mechval.metrics.structural.template_distance.27_edge_jaccard", "run_edge_jaccard"),
    "weight_eap_jaccard": ("mechval.metrics.structural.weight_alignment.28_weight_eap_jaccard", "run_weight_eap_jaccard"),
    "network_motifs": ("mechval.metrics.structural.network_motifs.97_network_motifs", "run_network_motifs"),
    "motif_enrichment": ("mechval.metrics.structural.network_motifs.G7_motif_enrichment", "run_motif_enrichment"),
    "capacity_utilization": ("mechval.metrics.structural.effective_rank.B13_capacity_utilization", "run"),
    "k_alignment": ("mechval.metrics.structural.effective_rank.B14_k_alignment", "run"),
    "weight_extended": ("mechval.metrics.structural.effective_rank.18_weight_extended", "run_weight_extended"),
    "spectral_svd": ("mechval.metrics.structural.spectral_svd.18_weight_extended", "run_weight_extended"),
    "path_identification": ("mechval.metrics.structural.edge_analysis.82_path_identification", "run_path_identification"),
    "edge_necessity": ("mechval.metrics.structural.edge_analysis.83_edge_necessity", "run_edge_necessity"),
    "path_specificity": ("mechval.metrics.structural.edge_analysis.84_path_specificity", "run_path_specificity"),
    "compositional_sufficiency": ("mechval.metrics.structural.edge_analysis.85_compositional_sufficiency", "run_compositional_sufficiency"),
    "graph_minimality": ("mechval.metrics.structural.edge_analysis.86_graph_minimality", "run_graph_minimality"),
    "attention_clustering": ("mechval.metrics.structural.attention_clustering.96_attention_clustering", "run_attention_clustering"),
    "llc": ("mechval.metrics.structural.llc_rllc.10_llc", "run_llc"),

    # ── Causal (intervention-based) ─────────────────────────────────────
    "activation_patching": ("mechval.metrics.causal.scm_pearl.02_activation_patching", "run_activation_patching"),
    "logit_diff": ("mechval.metrics.causal.scm_pearl.logit_diff", "run_logit_diff"),
    "role_ablation": ("mechval.metrics.causal.scm_pearl.role_ablation", "run_role_ablation"),
    "causal_scrubbing": ("mechval.metrics.causal.scm_pearl.04_causal_scrubbing", "run_causal_scrubbing"),
    "sigma_ablation": ("mechval.metrics.causal.woodward.03_sigma_ablation", "run_sigma_ablation"),
    "resample_complement": ("mechval.metrics.causal.woodward.35_resample_complement", "run_resample_complement"),
    "misalignment": ("mechval.metrics.causal.woodward.37_misalignment_score", "run_misalignment"),
    "das_iia": ("mechval.metrics.causal.counterfactual_das.01_das_iia", "run_das_iia"),
    "iia_variants": ("mechval.metrics.causal.counterfactual_das.15_iia_variants", "run_iia_variants"),
    "path_patching": ("mechval.metrics.causal.counterfactual_das.33_path_patching", "run_path_patching"),
    "counterfactual_consistency": ("mechval.metrics.causal.counterfactual_das.34_counterfactual_consistency", "run_counterfactual_consistency"),
    "multi_axis_iia": ("mechval.metrics.causal.counterfactual_das.31_multi_axis_iia", "run_multi_axis_iia"),
    "corrupt_restore": ("mechval.metrics.causal.counterfactual_das.20_corrupt_restore", "run_corrupt_restore"),
    "mediation": ("mechval.metrics.causal.mediation.05_mediation", "run_mediation"),
    "mediation_v2": ("mechval.metrics.causal.mediation.05_mediation_v2", "run_mediation_v2"),
    "pse": ("mechval.metrics.causal.mediation.24_pse", "run_pse"),
    "cate": ("mechval.metrics.causal.rubin_cate.06_cate", "run_cate"),
    "intervention_specificity": ("mechval.metrics.causal.rubin_cate.25_intervention_specificity", "run_intervention_specificity"),
    "eap": ("mechval.metrics.causal.eap.91_eap", "run_eap"),
    "atp_star": ("mechval.metrics.causal.atp_star.G2b_atp_star", "run_atp_star"),
    "pairwise_synergy": ("mechval.metrics.causal.hedonic_pas.94_pairwise_synergy", "run_pairwise_synergy"),
    "shapley_interactions": ("mechval.metrics.causal.hedonic_pas.H1_shapley_interactions", "run_hedonic_synergy"),
    "minimality_class": ("mechval.metrics.causal.minimality.C4b_minimality_class", "run_minimality_class"),
    "replacement_test": ("mechval.metrics.causal.mdc_glennan.72_replacement_test", "run_replacement_test"),
    "composition_test": ("mechval.metrics.causal.mdc_glennan.78_composition_test", "run_composition_test"),
    "operation_specification": ("mechval.metrics.causal.mdc_glennan.70_operation_specification", "run_operation_specification"),
    "held_out_prediction": ("mechval.metrics.causal.mdc_glennan.71_held_out_prediction", "run_held_out_prediction"),
    "procedure_specification": ("mechval.metrics.causal.mdc_glennan.77_procedure_specification", "run_procedure_specification"),
    "logic_gates": ("mechval.metrics.causal.mdc_glennan.19_logic_gates", "run_logic_gates"),
    "intermediate_state_prediction": ("mechval.metrics.causal.counterfactual_das.79_intermediate_state_prediction", "run_intermediate_state_prediction"),
    "cross_task_transfer": ("mechval.metrics.causal.granger_te.32_cross_task_iia_transfer", "run_cross_task_transfer"),
    "cross_model_invariance": ("mechval.metrics.causal.transportability.38_cross_model_invariance", "run_cross_model_invariance"),
    "hyperparam_sensitivity": ("mechval.metrics.causal.mdl_slt.29_hyperparam_sensitivity", "run_hyperparam_sensitivity"),

    # ── Information-theoretic ───────────────────────────────────────────
    "pid": ("mechval.metrics.information.pid.08_pid", "run_pid"),
    "ocse": ("mechval.metrics.information.osce.07_ocse", "run_ocse"),
    "notears": ("mechval.metrics.information.notears_dag.09_notears", "run_notears"),
    "mutual_information": ("mechval.metrics.information.mutual_info.54_mutual_information", "run_mutual_information"),
    "conditional_mi": ("mechval.metrics.information.conditional_mi.55_conditional_mi", "run_conditional_mi"),
    "granger_causality": ("mechval.metrics.information.granger.56_granger_causality", "run_granger"),
    "info_bottleneck": ("mechval.metrics.information.info_bottleneck.57_info_bottleneck", "run_info_bottleneck"),
    "o_information": ("mechval.metrics.information.synergistic_info.58_o_information", "run_o_information"),
    "transfer_entropy": ("mechval.metrics.information.transfer_entropy.53_transfer_entropy", "run_transfer_entropy"),

    # ── Representational ────────────────────────────────────────────────
    "attention_entropy": ("mechval.metrics.representational.attention_entropy.E11_attention_entropy", "run"),
    "cka": ("mechval.metrics.representational.cka.92_cka", "run_cka"),
    "cka_cross_arch": ("mechval.metrics.representational.cka.E6b_cka_cross_arch", "run_cka_analysis"),
    "probe_decodability": ("mechval.metrics.representational.linear_probe.75_probe_decodability", "run_probe_decodability"),
    "causal_representation": ("mechval.metrics.representational.linear_probe.76_causal_representation", "run_causal_representation"),

    # ── Behavioral ──────────────────────────────────────────────────────
    "effect_size": ("mechval.metrics.behavioral.effect_size.90_effect_size", "run_effect_size"),
    "dose_response": ("mechval.metrics.behavioral.dose_response.95_dose_response", "run_dose_response"),
    "ce_delta": ("mechval.metrics.behavioral.ce_delta.43_ce_delta", "run_ce_delta"),
    "per_token_nll": ("mechval.metrics.behavioral.per_token_nll.44_per_token_nll", "run_per_token_nll"),
    "calibration": ("mechval.metrics.behavioral.calibration.45_calibration", "run_calibration"),
    "generalization_gap": ("mechval.metrics.behavioral.generalization_gap.46_generalization_gap", "run_generalization_gap"),
    "mdl_compression": ("mechval.metrics.behavioral.mdl_compression.47_mdl_compression", "run_mdl_compression"),
    "subnetwork_probe": ("mechval.metrics.behavioral.subnetwork_probe.48_subnetwork_probe", "run_subnetwork_probe"),
    "output_variants": ("mechval.metrics.behavioral.logit_diff_recovery.21_output_variants", "run_output_variants"),
    "output_variants_kl": ("mechval.metrics.behavioral.kl_divergence.21_output_variants", "run_output_variants"),
    "output_variants_topk": ("mechval.metrics.behavioral.topk_accuracy.21_output_variants", "run_output_variants"),
    "corrupt_restore_behavioral": ("mechval.metrics.behavioral.logit_diff_recovery.20_corrupt_restore", "run_corrupt_restore"),
    "mean_centered_logit": ("mechval.metrics.behavioral.logit_diff_recovery.22_mean_centered_logit", "run_mean_centered_logit"),
    "normative_account": ("mechval.metrics.behavioral.generalization_gap.80_normative_account", "run_normative_account"),
    "error_boundary": ("mechval.metrics.behavioral.generalization_gap.81_error_boundary_analysis", "run_error_boundary_analysis"),
    "boundary_sweep": ("mechval.metrics.behavioral.construct_boundary.RI3b_boundary_sweep", "run_boundary_sweep"),
    "epistemic_gradient": ("mechval.metrics.behavioral.minimal_pairs.A5_epistemic_gradient", "run_epistemic_gradient"),
    "cross_task_generalization": ("mechval.metrics.behavioral.cross_task_transfer.E6b_cross_task_generalization", "run_cross_task_generalization"),
}


_CALIBRATION_REGISTRY: dict[str, tuple[str, str]] = {
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


_METRIC_FAMILIES: dict[str, str] = {}
for _name, (_mod, _fn) in _METRIC_REGISTRY.items():
    _family = _mod.split(".")[2]  # mechval.metrics.<family>.*
    _METRIC_FAMILIES[_name] = _family


def list_families() -> list[str]:
    return sorted(set(_METRIC_FAMILIES.values()))


def list_metrics(family: str | None = None) -> list[str]:
    if family is not None:
        return sorted(k for k, f in _METRIC_FAMILIES.items() if f == family)
    return sorted(_METRIC_REGISTRY.keys())


def list_calibrations() -> list[str]:
    return sorted(_CALIBRATION_REGISTRY.keys())


def _dispatch(registry: dict, name: str, **kwargs):
    import importlib
    import inspect
    mod_path, fn_name = registry[name]
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


def run(metric: str, **kwargs):
    if metric not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric!r}. Available: {list_metrics()}")
    return _dispatch(_METRIC_REGISTRY, metric, **kwargs)


def calibrate(calibration: str, **kwargs):
    if calibration not in _CALIBRATION_REGISTRY:
        raise ValueError(f"Unknown calibration: {calibration!r}. Available: {list_calibrations()}")
    return _dispatch(_CALIBRATION_REGISTRY, calibration, **kwargs)


def set_output_dir(path: str) -> None:
    from mechval.metrics.common import set_data_dir
    set_data_dir(path)


def status() -> dict[str, dict]:
    import json
    from pathlib import Path
    from mechval.metrics.common import DATA_DIR

    result = {}
    for path in sorted(Path(DATA_DIR).glob("*.jsonl")):
        tasks_done = set()
        count = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                task = d.get("metadata", {}).get("task", "")
                if task:
                    tasks_done.add(task)
                count += 1
            except json.JSONDecodeError:
                continue
        result[path.stem] = {"results": count, "tasks": sorted(tasks_done), "path": str(path)}

    for path in sorted(Path(DATA_DIR).glob("*.json")):
        if path.stem in result:
            continue
        try:
            data = json.loads(path.read_text())
            items = data if isinstance(data, list) else data.get("results", [])
            tasks_done = set()
            for d in items:
                task = d.get("metadata", {}).get("task", "")
                if task:
                    tasks_done.add(task)
            result[path.stem] = {"results": len(items), "tasks": sorted(tasks_done), "path": str(path)}
        except (json.JSONDecodeError, AttributeError):
            continue

    return result


def verify(spec: MechanisticClaimSpec, **kwargs) -> SpecVerificationResult:
    """Track 3: Causal Model Testing — run pre-registered predictions from a spec.

    For each prediction, routes to role_ablation when the prediction has
    intervention/measurement targets (the common case). Falls back to
    the raw expected_metric for predictions that don't need targeted
    interventions (e.g., logit_diff on output).
    """
    results = []
    from mechval.spec import PredictionResult, PredictionVerdict
    from mechval.models import InterventionType

    intervention_to_ablation = {
        InterventionType.ablate: "zero",
        InterventionType.clamp: "mean",
    }

    for pred in spec.all_predictions():
        try:
            if (pred.intervention_target and pred.measurement_target
                    and pred.intervention_target != "random_non_circuit"
                    and "role_ablation" in _METRIC_REGISTRY):
                ablation_type = intervention_to_ablation.get(pred.intervention, "zero")
                raw = _dispatch(
                    _METRIC_REGISTRY, "role_ablation",
                    tasks=[spec.task_id],
                    intervention_target=pred.intervention_target,
                    measurement_target=pred.measurement_target,
                    ablation_type=ablation_type,
                    **kwargs,
                )
                metric_used = "role_ablation"
            elif pred.expected_metric in _METRIC_REGISTRY:
                raw = _dispatch(_METRIC_REGISTRY, pred.expected_metric, tasks=[spec.task_id], **kwargs)
                metric_used = pred.expected_metric
            else:
                results.append(PredictionResult(
                    prediction=pred,
                    measured_value=0.0,
                    verdict=PredictionVerdict.gap,
                    metric_used=pred.expected_metric,
                    metadata={"error": f"Metric {pred.expected_metric!r} not found in registry"},
                ))
                continue

            value = _extract_value(raw, spec.task_id)
            verdict = _evaluate_prediction(pred, value)
            results.append(PredictionResult(
                prediction=pred,
                measured_value=value,
                verdict=verdict,
                metric_used=metric_used,
            ))
        except Exception as e:
            results.append(PredictionResult(
                prediction=pred,
                measured_value=0.0,
                verdict=PredictionVerdict.gap,
                metric_used=pred.expected_metric,
                metadata={"error": str(e)},
            ))

    from mechval.spec import ModeVerdict
    from mechval.models import VerdictTier

    mode_verdicts = _compute_mode_verdicts(results)
    claim_ceiling = _compute_claim_ceiling(mode_verdicts)

    return SpecVerificationResult(
        spec_id=spec.task_id,
        task_id=spec.task_id,
        model_family=spec.model_family,
        prediction_results=results,
        mode_verdicts=mode_verdicts,
        claim_ceiling=claim_ceiling,
        verdict_tier=VerdictTier.proposed,
    )


def _extract_value(raw, task_id: str) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        return float(raw.get("value", raw.get("score", 0.0)))
    if isinstance(raw, list):
        for entry in raw:
            if hasattr(entry, "value") and hasattr(entry, "metadata"):
                t = entry.metadata.get("task", "")
                if t == task_id:
                    return float(entry.value)
            elif isinstance(entry, dict):
                t = entry.get("metadata", {}).get("task", "")
                if t == task_id:
                    return float(entry.get("value", entry.get("score", 0.0)))
        if raw:
            first = raw[0]
            if hasattr(first, "value"):
                return float(first.value)
            elif isinstance(first, dict):
                return float(first.get("value", first.get("score", 0.0)))
    if hasattr(raw, "value"):
        return float(raw.value)
    return 0.0


def _evaluate_prediction(pred, value: float):
    from mechval.spec import PredictionVerdict
    from mechval.models import PredictionDirection

    if pred.is_negative_control:
        if pred.expected_direction == PredictionDirection.invariant:
            if pred.expected_threshold is not None:
                return PredictionVerdict.pass_ if abs(value) < pred.expected_threshold else PredictionVerdict.fail
            return PredictionVerdict.pass_ if abs(value) < 0.1 else PredictionVerdict.fail
        return PredictionVerdict.pass_ if abs(value) < 0.1 else PredictionVerdict.fail

    if pred.expected_direction == PredictionDirection.decrease:
        if pred.expected_threshold is not None:
            return PredictionVerdict.pass_ if value <= -pred.expected_threshold else PredictionVerdict.partial
        return PredictionVerdict.pass_ if value < 0 else PredictionVerdict.fail
    elif pred.expected_direction == PredictionDirection.increase:
        if pred.expected_threshold is not None:
            return PredictionVerdict.pass_ if value >= pred.expected_threshold else PredictionVerdict.partial
        return PredictionVerdict.pass_ if value > 0 else PredictionVerdict.fail
    elif pred.expected_direction == PredictionDirection.invariant:
        if pred.expected_threshold is not None:
            return PredictionVerdict.pass_ if abs(value) < pred.expected_threshold else PredictionVerdict.fail
        return PredictionVerdict.pass_ if abs(value) < 0.1 else PredictionVerdict.fail

    return PredictionVerdict.gap


def _compute_mode_verdicts(results):
    from mechval.spec import ModeVerdict, PredictionVerdict
    from collections import defaultdict

    by_mode = defaultdict(lambda: {"pos_tested": 0, "pos_passed": 0, "neg_tested": 0, "neg_passed": 0})
    for r in results:
        mode = r.prediction.description_mode
        if r.prediction.is_negative_control:
            by_mode[mode]["neg_tested"] += 1
            if r.verdict == PredictionVerdict.pass_:
                by_mode[mode]["neg_passed"] += 1
        else:
            by_mode[mode]["pos_tested"] += 1
            if r.verdict == PredictionVerdict.pass_:
                by_mode[mode]["pos_passed"] += 1

    verdicts = []
    for mode, counts in sorted(by_mode.items(), key=lambda x: x[0].value):
        all_pass = (counts["pos_passed"] == counts["pos_tested"] and
                    counts["neg_passed"] == counts["neg_tested"])
        any_pass = counts["pos_passed"] > 0 or counts["neg_passed"] > 0
        verdict = PredictionVerdict.pass_ if all_pass else (PredictionVerdict.partial if any_pass else PredictionVerdict.fail)
        verdicts.append(ModeVerdict(
            mode=mode,
            predictions_tested=counts["pos_tested"],
            predictions_passed=counts["pos_passed"],
            negative_controls_tested=counts["neg_tested"],
            negative_controls_passed=counts["neg_passed"],
            verdict=verdict,
        ))
    return verdicts


def _compute_claim_ceiling(mode_verdicts):
    from mechval.spec import PredictionVerdict
    from mechval.models import DescriptionMode

    mode_order = [
        DescriptionMode.impl_topographic,
        DescriptionMode.impl_statistical,
        DescriptionMode.impl_connectomic,
        DescriptionMode.impl_functional,
        DescriptionMode.representational,
        DescriptionMode.algorithmic,
        DescriptionMode.computational,
    ]
    ceiling = None
    for mode in mode_order:
        matching = [v for v in mode_verdicts if v.mode == mode]
        if matching and matching[0].verdict == PredictionVerdict.pass_:
            ceiling = mode
        elif matching:
            break
    return ceiling


try:
    import mechval_lab as _lab  # noqa: F401 — registers sweep/status/results CLI commands
except ImportError:
    pass

__all__ = [
    "load_task",
    "list_tasks",
    "list_families",
    "list_metrics",
    "list_calibrations",
    "list_experiment_groups",
    "list_domains",
    "run",
    "calibrate",
    "verify",
    "run_view",
    "check_gate",
    "set_output_dir",
    "status",
    "op",
    "init_tracing",
    "CircuitSpec",
    "CircuitTask",
    "MechanisticClaimSpec",
    "SpecVerificationResult",
]
