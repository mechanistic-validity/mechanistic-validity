"""Mechanistic Validity — circuit evaluation framework.

Quick start::

    import mechval as mv

    mv.set_output_dir("./results")
    task = mv.load_task("ioi")
    results = mv.run("k_composition", tasks=["ioi"])
    results = mv.calibrate("bootstrap", tasks=["ioi"])

    spec = mv.load_task("ioi").get_claim_spec()
    result = mv.verify(spec, device="cpu")

    mv.run_view("effect_estimation", tasks=["ioi"])
    mv.check_gate("measurement_calibration", task="ioi")

    mv.list_tasks()
    mv.list_families()
    mv.list_metrics()
    mv.list_metrics(family="structural")
    mv.list_calibrations()
    mv.list_domains()
    mv.list_experiment_groups()
    mv.status()
"""
import json
from pathlib import Path

from mechval.registry import list_tasks, load_task, list_experiment_groups, list_domains
from mechval.lib.tasks.spec import CircuitSpec
from mechval.lib.tasks.task import CircuitTask
from mechval.spec import MechanisticClaimSpec, SpecVerificationResult
from mechval.views import run_view
from mechval.gates import check_gate
from mechval.tracing import op, init as init_tracing
from mechval.metric_registry import (
    METRIC_REGISTRY,
    CALIBRATION_REGISTRY,
    METRIC_FAMILIES,
    list_families,
    list_metrics,
    list_calibrations,
    dispatch,
)
from mechval.spec_verification import verify


def run(metric: str, **kwargs):
    if metric not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric!r}. Available: {list_metrics()}")
    return dispatch(METRIC_REGISTRY, metric, **kwargs)


def calibrate(calibration: str, **kwargs):
    if calibration not in CALIBRATION_REGISTRY:
        raise ValueError(f"Unknown calibration: {calibration!r}. Available: {list_calibrations()}")
    return dispatch(CALIBRATION_REGISTRY, calibration, **kwargs)


def set_output_dir(path: str) -> None:
    from mechval.metrics.common import set_data_dir
    set_data_dir(path)


def status() -> dict[str, dict]:
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


from importlib.metadata import entry_points as _entry_points

for _ep in _entry_points(group="mechval.plugins"):
    _ep.load()()

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
