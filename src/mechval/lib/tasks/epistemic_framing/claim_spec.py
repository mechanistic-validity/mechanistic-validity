"""Epistemic Framing MechanisticClaimSpec — Tower (2026, unpublished)."""
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

EPISTEMIC_SPEC = MechanisticClaimSpec(
    task_id="epistemic_framing",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small processes epistemic stance markers (think/believe/know) "
        "via a 3-role, 4-head circuit: a detector head (6.5) identifies the "
        "epistemic verb, integrator heads (9.2, 9.5) combine the epistemic "
        "signal with factual content, and an executor head (10.5) adjusts "
        "the output logit distribution. The circuit is truth-insensitive "
        "and subject-invariant (I/He/She/They)."
    ),
    steps=[
        ComputationalStep(
            name="epistemic_detector",
            category="detection",
            description=(
                "Head 6.5 detects epistemic verb tokens (think, believe, know) "
                "and generates the initial epistemic stance signal."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="detector",
            maps_to_heads=[(6, 5)],
        ),
        ComputationalStep(
            name="epistemic_integrator",
            category="movement",
            description=(
                "Heads 9.2 and 9.5 integrate the epistemic signal from the "
                "detector with factual content from the residual stream."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="integrator",
            maps_to_heads=[(9, 2), (9, 5)],
        ),
        ComputationalStep(
            name="epistemic_executor",
            category="output",
            description=(
                "Head 10.5 reads the integrated epistemic signal and adjusts "
                "the output token distribution accordingly."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="executor",
            maps_to_heads=[(10, 5)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="epistemic_detector",
            target="epistemic_integrator",
            mechanism="residual_stream",
            description="Epistemic verb signal flows from detector to integrators.",
        ),
        ComputationalEdge(
            source="epistemic_integrator",
            target="epistemic_executor",
            mechanism="residual_stream",
            description="Integrated epistemic+factual signal flows to executor.",
        ),
        ComputationalEdge(
            source="epistemic_detector",
            target="epistemic_executor",
            mechanism="residual_stream",
            description="Direct skip connection from detector to executor (L6H5→L10H5).",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_executor_reduces_output",
            claim="Ablating the executor head substantially reduces epistemic logit diff.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_executor",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.15,
        ),
        CausalPrediction(
            name="ablate_integrator_reduces_output",
            claim="Ablating integrator heads reduces epistemic framing performance.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_integrator",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_detector_reduces_output",
            claim="Ablating the detector head reduces epistemic framing performance.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_detector",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_detector_reduces_integrator",
            claim="Ablating the detector reduces integrator activation magnitude.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_detector",
            measurement_target="epistemic_integrator",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_executor_does_not_affect_detector",
            claim="Ablating executor should not affect upstream detector activation.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_executor",
            measurement_target="epistemic_detector",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_integrator_does_not_affect_detector",
            claim="Ablating integrators should not affect upstream detector.",
            intervention=InterventionType.ablate,
            intervention_target="epistemic_integrator",
            measurement_target="epistemic_detector",
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
        known_confounds=[
            "Head 6.5 may respond to other verb categories beyond epistemic markers",
            "Integrator heads may participate in general mid-layer composition",
        ],
        mitigation=(
            "Compare epistemic-framed vs bare prompts to isolate epistemic-specific "
            "activation; the bare prompts serve as a within-circuit control."
        ),
    ),
    rival_specs=["epistemic_tight", "epistemic_eap", "epistemic_expanded"],
    description_mode=DescriptionMode.impl_functional,
    author="Tower (2026, unpublished — batch1_mechval evaluation)",
)
