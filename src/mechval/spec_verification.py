"""Spec verification — run pre-registered predictions from a MechanisticClaimSpec."""
from mechval.metric_registry import METRIC_REGISTRY, dispatch
from mechval.spec import MechanisticClaimSpec, SpecVerificationResult, PredictionResult, PredictionVerdict, ModeVerdict
from mechval.models import InterventionType, PredictionDirection, DescriptionMode, VerdictTier

from collections import defaultdict

INTERVENTION_TO_ABLATION = {
    InterventionType.ablate: "zero",
    InterventionType.clamp: "mean",
}


def verify(spec: MechanisticClaimSpec, **kwargs) -> SpecVerificationResult:
    results = []

    for pred in spec.all_predictions():
        try:
            if (pred.intervention_target and pred.measurement_target
                    and pred.intervention_target != "random_non_circuit"
                    and "role_ablation" in METRIC_REGISTRY):
                ablation_type = INTERVENTION_TO_ABLATION.get(pred.intervention, "zero")
                raw = dispatch(
                    METRIC_REGISTRY, "role_ablation",
                    tasks=[spec.task_id],
                    intervention_target=pred.intervention_target,
                    measurement_target=pred.measurement_target,
                    ablation_type=ablation_type,
                    **kwargs,
                )
                metric_used = "role_ablation"
            elif pred.expected_metric in METRIC_REGISTRY:
                raw = dispatch(METRIC_REGISTRY, pred.expected_metric, tasks=[spec.task_id], **kwargs)
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

            value = extract_value(raw, spec.task_id)
            verdict = evaluate_prediction(pred, value)
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

    mode_verdicts = compute_mode_verdicts(results)
    claim_ceiling = compute_claim_ceiling(mode_verdicts)

    return SpecVerificationResult(
        spec_id=spec.task_id,
        task_id=spec.task_id,
        model_family=spec.model_family,
        prediction_results=results,
        mode_verdicts=mode_verdicts,
        claim_ceiling=claim_ceiling,
        verdict_tier=VerdictTier.proposed,
    )


def extract_value(raw, task_id: str) -> float:
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


def evaluate_prediction(pred, value: float):
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


def compute_mode_verdicts(results):
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


def compute_claim_ceiling(mode_verdicts):
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
