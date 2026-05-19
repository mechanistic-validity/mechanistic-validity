"""Greater-Than MechanisticClaimSpec — Hanna et al. 2023 (NeurIPS)."""
from __future__ import annotations

from mechval.spec import (
    MechanisticClaimSpec,
    ComputationalStep,
    ComputationalEdge,
    CausalPrediction,
    IdentifiabilityGate,
    SuperpositionGate,
)
from mechval.models import (
    DescriptionMode,
    PredictionDirection,
    InterventionType,
    IdentifiabilityStatus,
)

GREATER_THAN_SPEC = MechanisticClaimSpec(
    task_id="greater_than",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small compares year values in sentences like "
        "'The war lasted from 1732 to 17' via a two-stage circuit: "
        "early heads extract the reference year, late heads compute "
        "the threshold comparison to suppress smaller years."
    ),
    steps=[
        ComputationalStep(
            name="year_extraction",
            category="detection",
            description=(
                "Heads 5.1, 6.9, 8.11, 9.1 extract the reference year "
                "digits and propagate year-magnitude information."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_gt",
            maps_to_heads=[(5, 1), (6, 9), (8, 11), (9, 1)],
        ),
        ComputationalStep(
            name="threshold_comparison",
            category="output",
            description=(
                "Heads 5.5, 7.10, 8.8 suppress logits for years below "
                "the reference threshold, implementing the greater-than comparison."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="late_gt",
            maps_to_heads=[(5, 5), (7, 10), (8, 8)],
        ),
        ComputationalStep(
            name="mlp_year_processing",
            category="output",
            description=(
                "MLP layers 8, 9, 10, 11 perform the core year comparison computation. "
                "Hanna et al. show these MLPs are necessary for the threshold function."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_mlps=[8, 9, 10, 11],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="year_extraction",
            target="threshold_comparison",
            mechanism="residual_stream",
            description="Year magnitude signal flows from early to late heads.",
        ),
        ComputationalEdge(
            source="year_extraction",
            target="mlp_year_processing",
            mechanism="residual_stream",
            description="Year extraction feeds MLP layers that implement the threshold function.",
        ),
        ComputationalEdge(
            source="threshold_comparison",
            target="mlp_year_processing",
            mechanism="residual_stream",
            description="Late attention heads feed into MLP threshold computation.",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_early_gt_reduces_output",
            claim="Ablating early_gt heads reduces the model's ability to discriminate year magnitudes.",
            intervention=InterventionType.ablate,
            intervention_target="year_extraction",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_late_gt_reduces_output",
            claim="Ablating late_gt heads substantially impairs year comparison output.",
            intervention=InterventionType.ablate,
            intervention_target="threshold_comparison",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.3,
        ),
        CausalPrediction(
            name="ablate_early_gt_reduces_late_gt",
            claim="Ablating early_gt heads reduces threshold_comparison activation magnitude.",
            intervention=InterventionType.ablate,
            intervention_target="year_extraction",
            measurement_target="threshold_comparison",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_mlp_reduces_output",
            claim="Ablating MLP layers 8-11 substantially impairs year comparison.",
            intervention=InterventionType.ablate,
            intervention_target="mlp_year_processing",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.3,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_late_gt_does_not_affect_early_gt",
            claim="Ablating late_gt should not affect upstream year extraction.",
            intervention=InterventionType.ablate,
            intervention_target="threshold_comparison",
            measurement_target="year_extraction",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
    ],
    identifiability=IdentifiabilityGate(
        status=IdentifiabilityStatus.identifiable,
        available_interventions=[InterventionType.ablate, InterventionType.patch],
    ),
    superposition_risk=SuperpositionGate(
        polysemanticity_risk="medium",
        known_confounds=["MLP layers also contribute to year comparison"],
        mitigation="Attention-only ablation underestimates total effect; MLP ablation needed for completeness.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="https://arxiv.org/abs/2305.00586",
    author="Hanna et al. 2023",
)
