"""Scoring views — aggregations over existing metrics.

Views are not separate measurement infrastructure. They group existing metrics
into causal-inference-grounded lenses:

  V1: Causal Effect Estimation    (Pearl/Rubin)
  V2: Causal Transportability     (Pearl/Bareinboim)
  V3: Counterfactual Verification (Pearl rung-3)
  V4: Mechanism Adjudication      (SEM equivalent-models)
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ViewResult(BaseModel):
    view: str
    task: str
    scores: dict[str, float] = Field(default_factory=dict)
    aggregate: float | None = None


VIEW_METRICS: dict[str, list[str]] = {
    "effect_estimation": [
        "mediation",
        "mediation_v2",
        "cate",
        "effect_size",
        "dose_response",
        "pse",
        "intervention_specificity",
    ],
    "transportability": [
        "cross_task_generalization",
        "cross_model_invariance",
        "generalization_gap",
    ],
    "counterfactual": [
        "das_iia",
        "iia_variants",
        "counterfactual_consistency",
        "corrupt_restore",
        "multi_axis_iia",
    ],
    "adjudication": [
        "discriminant_validity",
    ],
}

VIEW_NAMES = sorted(VIEW_METRICS.keys())


def list_views() -> list[str]:
    return VIEW_NAMES


def run_view(view: str, tasks: list[str] | None = None, **kwargs) -> list[ViewResult]:
    if view not in VIEW_METRICS:
        raise ValueError(f"Unknown view: {view!r}. Available: {VIEW_NAMES}")

    import mechval as mv

    metrics = VIEW_METRICS[view]
    available = set(mv.list_metrics())
    runnable = [m for m in metrics if m in available]

    if tasks is None:
        tasks = mv.list_tasks(has_circuit=True)

    results_by_task: dict[str, dict[str, float]] = {t: {} for t in tasks}

    for metric in runnable:
        try:
            raw = mv.run(metric, tasks=tasks, **kwargs)
            if isinstance(raw, list):
                for entry in raw:
                    task = entry.get("metadata", {}).get("task", "") if isinstance(entry, dict) else ""
                    val = entry.get("value", entry.get("score")) if isinstance(entry, dict) else None
                    if task in results_by_task and val is not None:
                        results_by_task[task][metric] = float(val)
        except Exception:
            continue

    out = []
    for task, scores in results_by_task.items():
        agg = sum(scores.values()) / len(scores) if scores else None
        out.append(ViewResult(view=view, task=task, scores=scores, aggregate=agg))
    return out
