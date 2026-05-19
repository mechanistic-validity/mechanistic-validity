"""Epistemic Framing (EAP) MechanisticClaimSpec — edge attribution patching discovery."""
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

EPISTEMIC_EAP_SPEC = MechanisticClaimSpec(
    task_id="epistemic_eap",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small processes epistemic framing via a 4-role, 15-head circuit "
        "discovered by Edge Attribution Patching: early relays (L1, L2) feed "
        "mid hubs (L4, L5), which connect to late integrators (L7-L9), which "
        "project through output heads (L10). This emphasizes edge connectivity "
        "over individual node effects, producing a different circuit than "
        "activation patching."
    ),
    steps=[
        ComputationalStep(
            name="early_relay",
            category="detection",
            description=(
                "Heads 1.3, 1.10, and 2.10 relay initial token information "
                "to mid-layer hubs."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="early_relay",
            maps_to_heads=[(1, 3), (1, 10), (2, 10)],
        ),
        ComputationalStep(
            name="mid_hub",
            category="movement",
            description=(
                "Heads 4.3, 4.7, 4.11, 5.9, and 5.10 act as central hubs "
                "with high edge involvement scores."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="mid_hub",
            maps_to_heads=[(4, 3), (4, 7), (4, 11), (5, 9), (5, 10)],
        ),
        ComputationalStep(
            name="late_integrator",
            category="movement",
            description=(
                "Heads 7.6, 8.8, 8.10, 9.5, and 9.6 integrate information "
                "from mid hubs before output projection."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="late_integrator",
            maps_to_heads=[(7, 6), (8, 8), (8, 10), (9, 5), (9, 6)],
        ),
        ComputationalStep(
            name="eap_output",
            category="output",
            description="Heads 10.0 and 10.7 project to output logits.",
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="output",
            maps_to_heads=[(10, 0), (10, 7)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="early_relay",
            target="mid_hub",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="early_relay",
            target="late_integrator",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="mid_hub",
            target="late_integrator",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="mid_hub",
            target="eap_output",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="late_integrator",
            target="eap_output",
            mechanism="residual_stream",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_output_reduces_performance",
            claim="Ablating output heads reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="eap_output",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_hub_reduces_output",
            claim="Ablating mid hubs reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="mid_hub",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_integrator_reduces_output",
            claim="Ablating late integrators reduces epistemic output.",
            intervention=InterventionType.ablate,
            intervention_target="late_integrator",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_relay_reduces_hub",
            claim="Ablating early relays reduces mid hub activation.",
            intervention=InterventionType.ablate,
            intervention_target="early_relay",
            measurement_target="mid_hub",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_output_does_not_affect_relay",
            claim="Ablating output should not affect upstream early relays.",
            intervention=InterventionType.ablate,
            intervention_target="eap_output",
            measurement_target="early_relay",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_integrator_does_not_affect_relay",
            claim="Ablating late integrators should not affect upstream relays.",
            intervention=InterventionType.ablate,
            intervention_target="late_integrator",
            measurement_target="early_relay",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
    ],
    rival_specs=["epistemic_framing", "epistemic_tight", "epistemic_expanded"],
    identifiability=IdentifiabilityGate(
        status=IdentifiabilityStatus.identifiable,
        available_interventions=[InterventionType.ablate, InterventionType.patch],
    ),
    superposition_risk=SuperpositionGate(
        polysemanticity_risk="medium",
        known_confounds=[
            "EAP emphasizes edges, not nodes — role assignments are derived from edge scores",
            "Head 4.11 (induction head) appears in mid_hub — may be shared with induction circuit",
        ],
        mitigation="Compare with activation patching circuit to identify method-specific artifacts.",
    ),
    description_mode=DescriptionMode.impl_functional,
    author="Tower (2026, EAP discovery)",
)
