"""Copy Suppression MechanisticClaimSpec — McDougall et al. 2023."""
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

COPY_SUPPRESSION_SPEC = MechanisticClaimSpec(
    task_id="copy_suppression",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small suppresses naive token copying via a 3-role circuit: "
        "previous token heads (2.2, 4.11) and induction heads (5.1, 5.5, 6.9) "
        "generate a copying signal, which copy-suppression heads (10.7, 11.10) "
        "then suppress to prevent the model from over-predicting repeated tokens."
    ),
    steps=[
        ComputationalStep(
            name="previous_token",
            category="detection",
            description=(
                "Heads 2.2 and 4.11 attend to the previous token, providing "
                "positional context for the induction mechanism."
            ),
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="PTH",
            maps_to_heads=[(2, 2), (4, 11)],
        ),
        ComputationalStep(
            name="induction",
            category="movement",
            description=(
                "Heads 5.1, 5.5, and 6.9 implement induction: they copy "
                "tokens that followed similar contexts, generating the "
                "copying signal that will be suppressed."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="IND",
            maps_to_heads=[(5, 1), (5, 5), (6, 9)],
        ),
        ComputationalStep(
            name="copy_suppression",
            category="inhibition",
            description=(
                "Heads 10.7 and 11.10 suppress the copying signal from "
                "induction heads, preventing over-prediction of repeated tokens. "
                "L10H7 is the primary copy-suppression head (McDougall et al.)."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="copy_suppress",
            maps_to_heads=[(10, 7), (11, 10)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="previous_token",
            target="induction",
            mechanism="residual_stream",
            description="PTH provides positional signal to induction heads.",
        ),
        ComputationalEdge(
            source="previous_token",
            target="copy_suppression",
            mechanism="residual_stream",
            description="Direct path from PTH to suppression (skip connection).",
        ),
        ComputationalEdge(
            source="induction",
            target="copy_suppression",
            mechanism="residual_stream",
            description="Induction copying signal flows to suppression heads.",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_suppression_increases_output",
            claim=(
                "Ablating copy-suppression heads should increase copying "
                "(logit diff goes up because suppression is removed)."
            ),
            intervention=InterventionType.ablate,
            intervention_target="copy_suppression",
            measurement_target="output",
            expected_direction=PredictionDirection.increase,
            expected_metric="role_ablation",
            expected_threshold=None,
        ),
        CausalPrediction(
            name="ablate_induction_reduces_output",
            claim="Ablating induction heads reduces the copying signal.",
            intervention=InterventionType.ablate,
            intervention_target="induction",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_pth_reduces_induction",
            claim="Ablating PTH reduces induction head activation.",
            intervention=InterventionType.ablate,
            intervention_target="previous_token",
            measurement_target="induction",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_suppression_does_not_affect_induction",
            claim="Ablating suppression should not affect upstream induction.",
            intervention=InterventionType.ablate,
            intervention_target="copy_suppression",
            measurement_target="induction",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_suppression_does_not_affect_pth",
            claim="Ablating suppression should not affect upstream PTH.",
            intervention=InterventionType.ablate,
            intervention_target="copy_suppression",
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
        known_confounds=[
            "Copy suppression is template-sensitive (strongest on repeated bigrams)",
        ],
        mitigation="Use prompts with clear repeated token patterns.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="https://arxiv.org/abs/2310.04625",
    author="McDougall et al. 2023",
)
