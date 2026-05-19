"""Pydantic v2 models for mechanistic-validity I/O and enums.

These models define the shared vocabulary across tracks, views, and gates.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DescriptionMode(str, Enum):
    computational = "computational"
    algorithmic = "algorithmic"
    representational = "representational"
    impl_topographic = "implementational-topographic"
    impl_connectomic = "implementational-connectomic"
    impl_statistical = "implementational-statistical"
    impl_functional = "implementational-functional"


class VerdictTier(str, Enum):
    proposed = "proposed"
    causally_suggestive = "causally_suggestive"
    mechanistically_supported = "mechanistically_supported"
    triangulated = "triangulated"
    validated = "validated"
    underdetermined = "underdetermined"
    disconfirmed = "disconfirmed"


class ValidityType(str, Enum):
    construct = "construct"
    internal = "internal"
    external = "external"
    measurement = "measurement"
    interpretive = "interpretive"


class EvidenceFamily(str, Enum):
    causal = "causal"
    structural = "structural"
    representational = "representational"
    behavioral = "behavioral"
    information = "information"


class CircuitStatus(str, Enum):
    full_circuit = "full_circuit"
    proxy_circuit = "proxy_circuit"
    generator_only = "generator_only"
    planned = "planned"


class IdentifiabilityStatus(str, Enum):
    identifiable = "identifiable"
    partially_identifiable = "partially_identifiable"
    requires_intervention = "requires_intervention"
    not_identifiable = "not_identifiable"


class PredictionDirection(str, Enum):
    decrease = "decrease"
    increase = "increase"
    invariant = "invariant"


class PredictionVerdict(str, Enum):
    pass_ = "pass"
    partial = "partial"
    fail = "fail"
    gap = "gap"


class InterventionType(str, Enum):
    ablate = "ablate"
    patch = "patch"
    clamp = "clamp"
    resample = "resample"


class RunRequest(BaseModel):
    metric: str
    tasks: list[str] = []
    families: list[str] = []
    device: str = "cpu"
    model_name: str = "gpt2"
    n_prompts: int = 40
    seed: int = 42
    output_dir: str | None = None


class MetricResult(BaseModel):
    metric: str
    task: str
    value: float
    stderr: float | None = None
    evidence_family: EvidenceFamily | None = None
    description_mode: DescriptionMode | None = None
    metadata: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    metric: str
    results: list[MetricResult]
    model_name: str = "gpt2"
    device: str = "cpu"


class TaskInfo(BaseModel):
    task_id: str
    model_family: str
    source: str
    domain: str
    experiment_group: str
    circuit_status: CircuitStatus
    paper_ref: str | None = None
