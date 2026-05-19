"""Epistemic Framing (Expanded) MechanisticClaimSpec — broad activation patching discovery."""
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

EPISTEMIC_EXPANDED_SPEC = MechanisticClaimSpec(
    task_id="epistemic_expanded",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small processes epistemic framing via a broad 6-role, 32-head "
        "circuit discovered by activation patching (|effect| > 0.15): early "
        "processors and suppressors (L0-L2) feed mid composers and suppressors "
        "(L3-L5), which connect to late routers and suppressors (L6-L10). "
        "This maximally inclusive circuit captures both enhancing and inhibiting "
        "contributions, unlike the tighter variants."
    ),
    steps=[
        ComputationalStep(
            name="early_processor",
            category="detection",
            description=(
                "7 heads in L0-L2 that provide initial token processing. "
                "Includes heads 0.0, 0.4, 0.8, 0.11, 1.4, 2.0, 2.10."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_processor",
            maps_to_heads=[
                (0, 0), (0, 4), (0, 8), (0, 11),
                (1, 4),
                (2, 0), (2, 10),
            ],
        ),
        ComputationalStep(
            name="early_suppressor",
            category="inhibition",
            description=(
                "7 heads in L0-L2 with negative activation patching effects. "
                "Suppress competing interpretations early in processing."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_suppressor",
            maps_to_heads=[
                (0, 2), (0, 6), (0, 7), (0, 9),
                (2, 1), (2, 5), (2, 8),
            ],
        ),
        ComputationalStep(
            name="mid_composer",
            category="movement",
            description=(
                "8 heads in L3-L5 that compose early signals into epistemic "
                "representations."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="mid_composer",
            maps_to_heads=[
                (3, 8),
                (4, 0), (4, 1), (4, 4), (4, 7), (4, 9),
                (5, 3), (5, 7),
            ],
        ),
        ComputationalStep(
            name="mid_suppressor",
            category="inhibition",
            description=(
                "3 heads in L4-L5 that suppress competing mid-layer signals."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="mid_suppressor",
            maps_to_heads=[(4, 3), (4, 6), (5, 2)],
        ),
        ComputationalStep(
            name="late_router",
            category="output",
            description=(
                "4 heads in L6-L10 that route epistemic signal to output, "
                "including head 10.0 at the output layer."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="late_router",
            maps_to_heads=[(6, 5), (7, 3), (7, 9), (10, 0)],
        ),
        ComputationalStep(
            name="late_suppressor",
            category="inhibition",
            description=(
                "3 heads in L7-L8 that suppress competing outputs in the "
                "late layers."
            ),
            input_type="signal",
            output_type="signal",
            position="destination",
            maps_to_role="late_suppressor",
            maps_to_heads=[(7, 8), (8, 5), (8, 8)],
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
        ComputationalEdge(
            source="mid_composer",
            target="late_suppressor",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="late_suppressor",
            target="late_router",
            mechanism="residual_stream",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_late_router_reduces_output",
            claim="Ablating late router heads reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="late_router",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_mid_composer_reduces_output",
            claim="Ablating mid composers reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="mid_composer",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
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
        CausalPrediction(
            name="ablate_late_suppressor_increases_output",
            claim="Ablating late suppressors increases epistemic output (removes inhibition).",
            intervention=InterventionType.ablate,
            intervention_target="late_suppressor",
            measurement_target="output",
            expected_direction=PredictionDirection.increase,
            expected_metric="role_ablation",
        ),
        CausalPrediction(
            name="ablate_mid_composer_reduces_late_router",
            claim="Ablating mid composers reduces late router activation.",
            intervention=InterventionType.ablate,
            intervention_target="mid_composer",
            measurement_target="late_router",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_late_router_does_not_affect_early_processor",
            claim="Ablating late router should not affect upstream early processors.",
            intervention=InterventionType.ablate,
            intervention_target="late_router",
            measurement_target="early_processor",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_late_suppressor_does_not_affect_mid_suppressor",
            claim="Ablating late suppressors should not affect upstream mid suppressors.",
            intervention=InterventionType.ablate,
            intervention_target="late_suppressor",
            measurement_target="mid_suppressor",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
    ],
    rival_specs=["epistemic_framing", "epistemic_tight", "epistemic_eap"],
    identifiability=IdentifiabilityGate(
        status=IdentifiabilityStatus.identifiable,
        available_interventions=[InterventionType.ablate, InterventionType.patch],
    ),
    superposition_risk=SuperpositionGate(
        polysemanticity_risk="high",
        known_confounds=[
            "32 heads is 26% of all GPT-2 heads — risk of capturing task-unrelated components",
            "Early suppressor and early processor may be general-purpose, not epistemic-specific",
            "Head 7.3 appears in both this circuit and IOI S-Inhibition",
        ],
        mitigation="Compare with tighter circuits (3-head, 13-head, 15-head) to identify core vs peripheral.",
    ),
    description_mode=DescriptionMode.impl_functional,
    author="Tower (2026, expanded activation patching discovery)",
)
