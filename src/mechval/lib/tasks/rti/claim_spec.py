"""RTI MechanisticClaimSpec — ours (weight-space discovery)."""
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

RTI_SPEC = MechanisticClaimSpec(
    task_id="rti",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small identifies repeated tokens via a 4-role circuit: "
        "backbone heads (L0) provide positional/token identity signal, "
        "a detector head (4.11) identifies repetition, copier heads (L4-L9) "
        "propagate the repeated token, and readout heads (L10-L11) write "
        "the prediction to output logits."
    ),
    steps=[
        ComputationalStep(
            name="backbone",
            category="detection",
            description="Heads 0.8, 0.9, 0.11 provide early positional and token identity signals.",
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="backbone",
            maps_to_heads=[(0, 8), (0, 9), (0, 11)],
        ),
        ComputationalStep(
            name="repetition_detector",
            category="detection",
            description="Head 4.11 detects that a token has appeared before in context.",
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="detector",
            maps_to_heads=[(4, 11)],
        ),
        ComputationalStep(
            name="token_copier",
            category="movement",
            description=(
                "Heads across L4-L9 copy the repeated token information "
                "forward through the residual stream to the readout layer."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="copier",
            maps_to_heads=[(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
        ),
        ComputationalStep(
            name="readout",
            category="output",
            description="Heads 10.11, 11.9, 11.11 project the repeated token to output logits.",
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="readout",
            maps_to_heads=[(10, 11), (11, 9), (11, 11)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="backbone",
            target="repetition_detector",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="backbone",
            target="token_copier",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="repetition_detector",
            target="token_copier",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="repetition_detector",
            target="readout",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="token_copier",
            target="readout",
            mechanism="residual_stream",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_readout_reduces_output",
            claim="Ablating readout heads reduces RTI logit diff.",
            intervention=InterventionType.ablate,
            intervention_target="readout",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.3,
        ),
        CausalPrediction(
            name="ablate_copier_reduces_output",
            claim="Ablating copier heads reduces RTI performance.",
            intervention=InterventionType.ablate,
            intervention_target="token_copier",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_detector_reduces_output",
            claim="Ablating the detector head reduces RTI performance.",
            intervention=InterventionType.ablate,
            intervention_target="repetition_detector",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_backbone_reduces_detector",
            claim="Ablating backbone heads reduces detector activation.",
            intervention=InterventionType.ablate,
            intervention_target="backbone",
            measurement_target="repetition_detector",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_copier_reduces_readout",
            claim="Ablating copier heads reduces readout head activation.",
            intervention=InterventionType.ablate,
            intervention_target="token_copier",
            measurement_target="readout",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_detector_reduces_copier",
            claim="Ablating the detector reduces copier head activation.",
            intervention=InterventionType.ablate,
            intervention_target="repetition_detector",
            measurement_target="token_copier",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_readout_does_not_affect_copier",
            claim="Ablating readout should not affect upstream copier heads.",
            intervention=InterventionType.ablate,
            intervention_target="readout",
            measurement_target="token_copier",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_readout_does_not_affect_detector",
            claim="Ablating readout should not affect upstream detector.",
            intervention=InterventionType.ablate,
            intervention_target="readout",
            measurement_target="repetition_detector",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_copier_does_not_affect_backbone",
            claim="Ablating copiers should not affect upstream backbone.",
            intervention=InterventionType.ablate,
            intervention_target="token_copier",
            measurement_target="backbone",
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
        known_confounds=["copier heads overlap with induction circuit"],
        mitigation="Compare RTI-specific prompts vs induction prompts to isolate.",
    ),
    description_mode=DescriptionMode.impl_functional,
    author="Tower et al. (weight-space discovery)",
)
