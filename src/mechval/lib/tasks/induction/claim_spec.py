"""Induction MechanisticClaimSpec — Olsson et al. 2022."""
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

INDUCTION_SPEC = MechanisticClaimSpec(
    task_id="induction",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small copies repeated sequences via a two-layer induction circuit: "
        "previous-token heads (L2, L4) attend to the token before the current one, "
        "and induction heads (L5-L7) compose with PTH via K-composition to copy "
        "the token that followed the previous occurrence."
    ),
    steps=[
        ComputationalStep(
            name="previous_token",
            category="detection",
            description=(
                "Heads 2.2 and 4.11 attend to the previous token position, "
                "writing positional information into the residual stream."
            ),
            input_type="position",
            output_type="attention_pattern",
            position="source",
            maps_to_role="PTH",
            maps_to_heads=[(2, 2), (4, 11)],
        ),
        ComputationalStep(
            name="induction_copy",
            category="movement",
            description=(
                "Heads 5.1, 5.5, 6.9, 7.2, 7.10 use K-composition with PTH "
                "output to attend to positions following previous occurrences, "
                "then copy the next token."
            ),
            input_type="token_identity",
            output_type="logit_boost",
            position="both",
            maps_to_role="IND",
            maps_to_heads=[(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="previous_token",
            target="induction_copy",
            mechanism="attention_composition",
            description="PTH output composes with induction head keys (K-composition).",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_pth_reduces_output",
            claim="Ablating previous-token heads reduces induction copying performance.",
            intervention=InterventionType.ablate,
            intervention_target="previous_token",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_ind_reduces_output",
            claim="Ablating induction heads substantially reduces copying performance.",
            intervention=InterventionType.ablate,
            intervention_target="induction_copy",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.5,
        ),
        CausalPrediction(
            name="ablate_pth_reduces_induction_output",
            claim="Ablating PTH heads reduces induction head activation magnitude (composition disrupted).",
            intervention=InterventionType.ablate,
            intervention_target="previous_token",
            measurement_target="induction_copy",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_ind_does_not_affect_pth",
            claim="Ablating induction heads should not affect upstream PTH attention patterns.",
            intervention=InterventionType.ablate,
            intervention_target="induction_copy",
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
        mitigation="Induction heads are well-characterized and relatively monosemantic.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="https://arxiv.org/abs/2209.11895",
    author="Olsson et al. 2022",
)
