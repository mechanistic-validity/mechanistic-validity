"""Gendered Pronoun MechanisticClaimSpec — Mathwin 2023."""
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

GENDERED_PRONOUN_SPEC = MechanisticClaimSpec(
    task_id="gendered_pronoun",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small resolves gendered pronoun agreement via a 3-role, "
        "5-head circuit: an early gender agreement head (0.10) detects "
        "gender cues, late agreement heads (3.0, 5.8) propagate the "
        "gender signal, and name binding heads (6.6, 8.6) bind the "
        "gendered pronoun to the referent entity."
    ),
    steps=[
        ComputationalStep(
            name="early_gender_detection",
            category="detection",
            description=(
                "Head 0.10 detects early gender cues from name tokens "
                "and generates the initial gender agreement signal."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_ga",
            maps_to_heads=[(0, 10)],
        ),
        ComputationalStep(
            name="gender_propagation",
            category="movement",
            description=(
                "Heads 3.0 and 5.8 propagate the gender signal through "
                "mid layers, maintaining gender information across positions."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="late_ga",
            maps_to_heads=[(3, 0), (5, 8)],
        ),
        ComputationalStep(
            name="name_binding",
            category="output",
            description=(
                "Heads 6.6 and 8.6 bind the gender-marked pronoun to "
                "the correct referent and project to output logits."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="name_bind",
            maps_to_heads=[(6, 6), (8, 6)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="early_gender_detection",
            target="gender_propagation",
            mechanism="residual_stream",
            description="Gender cue signal flows from early detector to propagation heads.",
        ),
        ComputationalEdge(
            source="early_gender_detection",
            target="name_binding",
            mechanism="residual_stream",
            description="Direct path from gender detection to name binding (skip).",
        ),
        ComputationalEdge(
            source="gender_propagation",
            target="name_binding",
            mechanism="residual_stream",
            description="Propagated gender signal feeds name binding heads.",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_binding_reduces_output",
            claim="Ablating name binding heads reduces gendered pronoun resolution.",
            intervention=InterventionType.ablate,
            intervention_target="name_binding",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.15,
        ),
        CausalPrediction(
            name="ablate_propagation_reduces_output",
            claim="Ablating gender propagation heads reduces pronoun resolution.",
            intervention=InterventionType.ablate,
            intervention_target="gender_propagation",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_detection_reduces_output",
            claim="Ablating the early gender detection head reduces performance.",
            intervention=InterventionType.ablate,
            intervention_target="early_gender_detection",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_detection_reduces_propagation",
            claim="Ablating early detection reduces propagation head activation.",
            intervention=InterventionType.ablate,
            intervention_target="early_gender_detection",
            measurement_target="gender_propagation",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_binding_does_not_affect_detection",
            claim="Ablating name binding should not affect upstream gender detection.",
            intervention=InterventionType.ablate,
            intervention_target="name_binding",
            measurement_target="early_gender_detection",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_propagation_does_not_affect_detection",
            claim="Ablating propagation should not affect upstream detection.",
            intervention=InterventionType.ablate,
            intervention_target="gender_propagation",
            measurement_target="early_gender_detection",
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
        polysemanticity_risk="low",
        known_confounds=["Small circuit (5 heads) may be incomplete"],
        mitigation="Compare with IOI circuit overlap to check for shared components.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="Mathwin 2023",
    author="Mathwin 2023",
)
