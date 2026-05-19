"""Epistemic Framing (Tight) MechanisticClaimSpec — activation patching discovery."""
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

EPISTEMIC_TIGHT_SPEC = MechanisticClaimSpec(
    task_id="epistemic_tight",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small processes epistemic framing via a 5-role, 13-head circuit "
        "discovered by activation patching (|effect| > 0.20): early processors "
        "(L0) and early suppressors (L0, L2) handle initial token processing, "
        "mid composers (L3, L5) and mid suppressors (L4, L5) integrate the "
        "epistemic signal, and a late router (L10) directs output. This is a "
        "rival hypothesis to the 4-head core circuit, discovered by a different "
        "method (activation patching vs manual analysis)."
    ),
    steps=[
        ComputationalStep(
            name="early_processor",
            category="detection",
            description=(
                "Heads 0.0, 0.4, 0.8, and 2.10 perform initial token processing, "
                "extracting positional and identity features."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_processor",
            maps_to_heads=[(0, 0), (0, 4), (0, 8), (2, 10)],
        ),
        ComputationalStep(
            name="early_suppressor",
            category="inhibition",
            description=(
                "Heads 0.6, 0.7, 0.9, and 2.1 suppress irrelevant signals "
                "early in the circuit."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_suppressor",
            maps_to_heads=[(0, 6), (0, 7), (0, 9), (2, 1)],
        ),
        ComputationalStep(
            name="mid_composer",
            category="movement",
            description=(
                "Heads 3.8 and 5.3 compose epistemic and factual signals "
                "in mid layers."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="mid_composer",
            maps_to_heads=[(3, 8), (5, 3)],
        ),
        ComputationalStep(
            name="mid_suppressor",
            category="inhibition",
            description=(
                "Heads 4.6 and 5.2 suppress competing signals in mid layers."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="mid_suppressor",
            maps_to_heads=[(4, 6), (5, 2)],
        ),
        ComputationalStep(
            name="late_router",
            category="output",
            description="Head 10.0 routes the final epistemic signal to output.",
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="late_router",
            maps_to_heads=[(10, 0)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="early_processor",
            target="mid_composer",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="early_processor",
            target="late_router",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="early_suppressor",
            target="mid_suppressor",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="early_suppressor",
            target="mid_composer",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="mid_composer",
            target="late_router",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="mid_suppressor",
            target="late_router",
            mechanism="residual_stream",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_router_reduces_output",
            claim="Ablating the late router reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="late_router",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_composer_reduces_output",
            claim="Ablating mid composers reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="mid_composer",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_early_processor_reduces_output",
            claim="Ablating early processors reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="early_processor",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_early_processor_reduces_composer",
            claim="Ablating early processors reduces mid composer activation.",
            intervention=InterventionType.ablate,
            intervention_target="early_processor",
            measurement_target="mid_composer",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_router_does_not_affect_processor",
            claim="Ablating late router should not affect upstream early processors.",
            intervention=InterventionType.ablate,
            intervention_target="late_router",
            measurement_target="early_processor",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
    ],
    rival_specs=["epistemic_framing", "epistemic_eap", "epistemic_expanded"],
    identifiability=IdentifiabilityGate(
        status=IdentifiabilityStatus.identifiable,
        available_interventions=[InterventionType.ablate, InterventionType.patch],
    ),
    superposition_risk=SuperpositionGate(
        polysemanticity_risk="medium",
        known_confounds=[
            "L0 heads are highly polysemantic — early processor/suppressor distinction may not hold",
            "13 heads discovered by threshold cutoff — boundary is arbitrary",
        ],
        mitigation="Compare confirmation rates against the 4-head core circuit.",
    ),
    description_mode=DescriptionMode.impl_functional,
    author="Tower (2026, activation patching discovery)",
)
