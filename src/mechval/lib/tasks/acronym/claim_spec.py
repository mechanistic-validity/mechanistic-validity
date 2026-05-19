"""Acronym MechanisticClaimSpec — Garcia-Carrasco et al. AISTATS 2024."""
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

ACRONYM_SPEC = MechanisticClaimSpec(
    task_id="acronym",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small predicts acronym letters via a 3-role circuit: "
        "previous token heads (1.0, 2.2, 4.11) track positional context, "
        "a propagator head (5.8) relays information forward, and letter "
        "mover heads (8.11, 9.9, 10.10, 11.4) write the predicted letter "
        "to output logits."
    ),
    steps=[
        ComputationalStep(
            name="previous_token",
            category="detection",
            description=(
                "Heads 1.0, 2.2, and 4.11 attend to the previous token "
                "position, providing sequential context for letter prediction."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="PTH",
            maps_to_heads=[(1, 0), (2, 2), (4, 11)],
        ),
        ComputationalStep(
            name="propagator",
            category="movement",
            description=(
                "Head 5.8 propagates positional and letter information "
                "from early layers to the letter mover heads."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="propagator",
            maps_to_heads=[(5, 8)],
        ),
        ComputationalStep(
            name="letter_mover",
            category="output",
            description=(
                "Heads 8.11, 9.9, 10.10, and 11.4 predict the next letter "
                "in the acronym by writing to the output logit distribution."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="letter_mover",
            maps_to_heads=[(8, 11), (9, 9), (10, 10), (11, 4)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="previous_token",
            target="propagator",
            mechanism="residual_stream",
            description="Positional signal from PTH feeds propagator.",
        ),
        ComputationalEdge(
            source="previous_token",
            target="letter_mover",
            mechanism="residual_stream",
            description="Direct path from PTH to letter movers (skip).",
        ),
        ComputationalEdge(
            source="propagator",
            target="letter_mover",
            mechanism="residual_stream",
            description="Propagated letter information feeds mover heads.",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_mover_reduces_output",
            claim="Ablating letter mover heads substantially reduces acronym prediction.",
            intervention=InterventionType.ablate,
            intervention_target="letter_mover",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_propagator_reduces_output",
            claim="Ablating the propagator head reduces prediction performance.",
            intervention=InterventionType.ablate,
            intervention_target="propagator",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_pth_reduces_output",
            claim="Ablating previous token heads reduces acronym prediction.",
            intervention=InterventionType.ablate,
            intervention_target="previous_token",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_mover_does_not_affect_pth",
            claim="Ablating letter movers should not affect upstream PTH.",
            intervention=InterventionType.ablate,
            intervention_target="letter_mover",
            measurement_target="previous_token",
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
        known_confounds=[],
        mitigation="",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="Garcia-Carrasco et al. AISTATS 2024",
    author="Garcia-Carrasco et al. 2024",
)
