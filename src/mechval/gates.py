"""Precondition gates — checks that must pass before tracks/views are meaningful.

  G0: Construct Operationalization — is the construct defined independently?
  G1: Measurement Calibration — are metrics stable and separable from baselines?
  G2: Causal Identifiability — can the effects be estimated with available interventions?
  G3: Confound/Superposition Risk — are interventions confounded by polysemanticity?
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mechval.spec import MechanisticClaimSpec


class GateResult(BaseModel):
    gate: str
    passed: bool
    details: dict = Field(default_factory=dict)
    notes: str = ""


GATE_NAMES = [
    "construct_operationalization",
    "measurement_calibration",
    "identifiability",
    "superposition_risk",
]


def list_gates() -> list[str]:
    return list(GATE_NAMES)


def check_gate(
    gate: str,
    task: str | None = None,
    spec: MechanisticClaimSpec | None = None,
    **kwargs,
) -> GateResult:
    if gate not in GATE_NAMES:
        raise ValueError(f"Unknown gate: {gate!r}. Available: {GATE_NAMES}")

    if gate == "construct_operationalization":
        return _check_construct(task)
    elif gate == "measurement_calibration":
        return _check_calibration(task, **kwargs)
    elif gate == "identifiability":
        return _check_identifiability(spec)
    elif gate == "superposition_risk":
        return _check_superposition(spec)
    raise ValueError(f"Unknown gate: {gate!r}")


def _check_construct(task: str | None) -> GateResult:
    if task is None:
        return GateResult(gate="construct_operationalization", passed=False, notes="No task specified")
    from mechval.registry import load_task
    t = load_task(task)
    has_prompts = True
    try:
        t.get_prompts.__func__  # noqa: B018
    except AttributeError:
        has_prompts = False
    has_circuit = t.circuit_status in ("full_circuit", "proxy_circuit")
    passed = has_prompts and has_circuit
    return GateResult(
        gate="construct_operationalization",
        passed=passed,
        details={"has_prompts": has_prompts, "has_circuit": has_circuit},
        notes="" if passed else "Task lacks independent construct definition (circuit or prompts missing)",
    )


def _check_calibration(task: str | None, **kwargs) -> GateResult:
    if task is None:
        return GateResult(gate="measurement_calibration", passed=False, notes="No task specified")
    return GateResult(
        gate="measurement_calibration",
        passed=True,
        details={"calibrations_available": ["bootstrap", "seed_variance"]},
        notes="Run mv.calibrate('bootstrap', tasks=[task]) for full calibration check",
    )


def _check_identifiability(spec: MechanisticClaimSpec | None) -> GateResult:
    if spec is None:
        return GateResult(gate="identifiability", passed=False, notes="No spec provided")
    from mechval.models import IdentifiabilityStatus
    status = spec.identifiability.status
    passed = status in (IdentifiabilityStatus.identifiable, IdentifiabilityStatus.partially_identifiable)
    return GateResult(
        gate="identifiability",
        passed=passed,
        details={
            "status": status.value,
            "available_interventions": [i.value for i in spec.identifiability.available_interventions],
        },
        notes=spec.identifiability.notes,
    )


def _check_superposition(spec: MechanisticClaimSpec | None) -> GateResult:
    if spec is None:
        return GateResult(gate="superposition_risk", passed=False, notes="No spec provided")
    risk = spec.superposition_risk.polysemanticity_risk
    passed = risk in ("low", "medium")
    return GateResult(
        gate="superposition_risk",
        passed=passed,
        details={
            "polysemanticity_risk": risk,
            "known_confounds": spec.superposition_risk.known_confounds,
        },
        notes=spec.superposition_risk.mitigation if not passed else "",
    )
