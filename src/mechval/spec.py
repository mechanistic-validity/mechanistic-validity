"""MechanisticClaimSpec — Track 3: Causal Model Testing.

Pre-registered mechanistic hypotheses as Pydantic v2 models. A spec declares
a mechanism DAG (steps + edges), positive predictions, negative controls,
rival specs, and gate metadata. mv.verify(spec) runs the predictions and
produces a SpecVerificationResult with per-prediction verdicts, mode-level
aggregation, and a claim ceiling.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from mechval.models import (
    DescriptionMode,
    IdentifiabilityStatus,
    InterventionType,
    PredictionDirection,
    PredictionVerdict,
    VerdictTier,
)


class ComputationalStep(BaseModel):
    name: str
    category: str
    description: str
    input_type: str
    output_type: str
    position: str
    maps_to_role: str | None = None
    maps_to_heads: list[tuple[int, int]] = Field(default_factory=list)
    maps_to_mlps: list[int] = Field(default_factory=list)
    maps_to_neurons: list[tuple[int, int]] = Field(default_factory=list)
    maps_to_features: list[tuple[int, str, int]] = Field(default_factory=list)
    description_mode: DescriptionMode = DescriptionMode.impl_functional
    discovery_status: str = "hypothesized"

    @property
    def component_types(self) -> list[str]:
        types = []
        if self.maps_to_heads:
            types.append("attention")
        if self.maps_to_mlps:
            types.append("mlp")
        if self.maps_to_neurons:
            types.append("neuron")
        if self.maps_to_features:
            types.append("feature")
        return types


class ComputationalEdge(BaseModel):
    source: str
    target: str
    mechanism: str
    description: str = ""
    weight: float | None = None
    source_component: tuple[int, int] | None = None
    target_component: tuple[int, int] | None = None


class CausalPrediction(BaseModel):
    name: str
    claim: str
    intervention: InterventionType
    intervention_target: str
    measurement_target: str
    expected_direction: PredictionDirection
    expected_metric: str
    expected_threshold: float | None = None
    description_mode: DescriptionMode = DescriptionMode.impl_functional
    is_negative_control: bool = False
    result: float | None = None
    verdict: PredictionVerdict | None = None


class IdentifiabilityGate(BaseModel):
    status: IdentifiabilityStatus = IdentifiabilityStatus.identifiable
    available_interventions: list[InterventionType] = Field(default_factory=list)
    notes: str = ""


class SuperpositionGate(BaseModel):
    polysemanticity_risk: str = "low"
    known_confounds: list[str] = Field(default_factory=list)
    mitigation: str = ""


class MechanisticClaimSpec(BaseModel):
    task_id: str
    model_family: str = "gpt2"
    linguistic_claim: str

    steps: list[ComputationalStep]
    edges: list[ComputationalEdge]

    predictions: list[CausalPrediction]
    negative_controls: list[CausalPrediction] = Field(default_factory=list)

    rival_specs: list[str] = Field(default_factory=list)

    identifiability: IdentifiabilityGate = Field(default_factory=IdentifiabilityGate)
    superposition_risk: SuperpositionGate = Field(default_factory=SuperpositionGate)

    description_mode: DescriptionMode = DescriptionMode.impl_functional
    paper_ref: str | None = None
    author: str = ""

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def get_step(self, name: str) -> ComputationalStep:
        return next(s for s in self.steps if s.name == name)

    def all_predictions(self) -> list[CausalPrediction]:
        return self.predictions + self.negative_controls

    def untested_predictions(self) -> list[CausalPrediction]:
        return [p for p in self.all_predictions() if p.verdict is None]

    @computed_field
    @property
    def confirmation_rate(self) -> float | None:
        tested = [p for p in self.predictions if p.verdict is not None]
        if not tested:
            return None
        return sum(1 for p in tested if p.verdict == PredictionVerdict.pass_) / len(tested)

    @computed_field
    @property
    def negative_control_rate(self) -> float | None:
        tested = [p for p in self.negative_controls if p.verdict is not None]
        if not tested:
            return None
        return sum(1 for p in tested if p.verdict == PredictionVerdict.pass_) / len(tested)


class PredictionResult(BaseModel):
    prediction: CausalPrediction
    measured_value: float
    verdict: PredictionVerdict
    metric_used: str
    metadata: dict = Field(default_factory=dict)


class ModeVerdict(BaseModel):
    mode: DescriptionMode
    predictions_tested: int
    predictions_passed: int
    negative_controls_tested: int
    negative_controls_passed: int
    verdict: PredictionVerdict


class SpecVerificationResult(BaseModel):
    spec_id: str
    task_id: str
    model_family: str

    prediction_results: list[PredictionResult]

    mode_verdicts: list[ModeVerdict] = Field(default_factory=list)

    claim_ceiling: DescriptionMode | None = None
    verdict_tier: VerdictTier = VerdictTier.proposed

    gates_passed: dict[str, bool] = Field(default_factory=dict)

    effect_estimation_score: float | None = None
    transportability_score: float | None = None
    counterfactual_score: float | None = None
    adjudication_score: float | None = None

    @computed_field
    @property
    def confirmation_rate(self) -> float:
        positive = [r for r in self.prediction_results if not r.prediction.is_negative_control]
        if not positive:
            return 0.0
        return sum(1 for r in positive if r.verdict == PredictionVerdict.pass_) / len(positive)
