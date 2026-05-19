"""Cyclopts CLI — `mechval` command.

Requires the 'cli' optional dependency group:
    pip install mechval[cli]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import cyclopts
except ImportError:
    print("CLI requires: pip install mechval[cli]", file=sys.stderr)
    sys.exit(1)

import mechval

app = cyclopts.App(name="mechval", help="Mechanistic validity — circuit evaluation framework")


@app.command
def run(
    metric: str,
    tasks: list[str] | None = None,
    device: str = "cpu",
    model_name: str = "gpt2",
    output_dir: str | None = None,
) -> None:
    """Run a metric across tasks."""
    if output_dir:
        mv.set_output_dir(output_dir)
    kwargs: dict = {"device": device, "model_name": model_name}
    if tasks:
        kwargs["tasks"] = tasks
    results = mv.run(metric, **kwargs)
    print(json.dumps(results, indent=2, default=str))


@app.command
def verify(
    spec_path: str,
    device: str = "cpu",
) -> None:
    """Run Track 3: Causal Model Testing on a MechanisticClaimSpec."""
    from mechval.spec import MechanisticClaimSpec

    spec = MechanisticClaimSpec.model_validate_json(Path(spec_path).read_text())
    result = mv.verify(spec, device=device)
    print(result.model_dump_json(indent=2))


@app.command
def calibrate(
    calibration: str,
    tasks: list[str] | None = None,
    device: str = "cpu",
) -> None:
    """Run a calibration check."""
    kwargs: dict = {"device": device}
    if tasks:
        kwargs["tasks"] = tasks
    results = mv.calibrate(calibration, **kwargs)
    print(json.dumps(results, indent=2, default=str))


@app.command
def tasks(
    source: str | None = None,
    domain: str | None = None,
    experiment_group: str | None = None,
    has_circuit: bool | None = None,
) -> None:
    """List registered tasks with optional filtering."""
    for t in mv.list_tasks(
        source=source,
        domain=domain,
        experiment_group=experiment_group,
        has_circuit=has_circuit,
    ):
        print(t)


@app.command
def metrics(family: str | None = None) -> None:
    """List available metrics."""
    for m in mv.list_metrics(family=family):
        print(m)


@app.command
def calibrations() -> None:
    """List available calibrations."""
    for c in mv.list_calibrations():
        print(c)


@app.command
def view(
    view_name: str,
    tasks: list[str] | None = None,
    device: str = "cpu",
) -> None:
    """Run a scoring view (V1-V4) across tasks."""
    results = mv.run_view(view_name, tasks=tasks, device=device)
    for r in results:
        print(r.model_dump_json(indent=2))


@app.command
def gate(
    gate_name: str,
    task: str | None = None,
    spec_path: str | None = None,
) -> None:
    """Check a precondition gate (G0-G3)."""
    spec = None
    if spec_path:
        from mechval.spec import MechanisticClaimSpec

        spec = MechanisticClaimSpec.model_validate_json(Path(spec_path).read_text())
    result = mv.check_gate(gate_name, task=task, spec=spec)
    print(result.model_dump_json(indent=2))


@app.command
def domains() -> None:
    """List available task domains."""
    for d in mv.list_domains():
        print(d)


@app.command
def experiment_groups() -> None:
    """List available experiment groups."""
    for g in mv.list_experiment_groups():
        print(g)


@app.command
def views() -> None:
    """List available scoring views."""
    from mechval.views import list_views

    for v in list_views():
        print(v)


@app.command
def gates() -> None:
    """List available precondition gates."""
    from mechval.gates import list_gates

    for g in list_gates():
        print(g)


@app.command
def status() -> None:
    """Show what results exist on disk."""
    s = mv.status()
    if not s:
        print("No results found.")
        return
    for name, info in s.items():
        print(f"{name}: {info['results']} results, {len(info['tasks'])} tasks")


if __name__ == "__main__":
    app()
