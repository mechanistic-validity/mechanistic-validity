"""SVA MechanisticClaimSpec — Lazo et al. 2025."""
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

SVA_SPEC = MechanisticClaimSpec(
    task_id="sva",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small resolves subject-verb number agreement via a 4-stage circuit: "
        "embedding heads (L0) encode number features, encoder heads (L1-L2) build "
        "subject representations, router heads (L6, L9) propagate number info "
        "past attractors, and output heads (L10-L11) write the correct verb form."
    ),
    steps=[
        ComputationalStep(
            name="number_embedding",
            category="detection",
            description="Heads 0.4 and 0.8 encode number (singular/plural) features from the subject token.",
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="embed",
            maps_to_heads=[(0, 4), (0, 8)],
        ),
        ComputationalStep(
            name="subject_encoding",
            category="detection",
            description="Heads 1.0, 1.1, 2.1, 2.6 build a richer subject representation with number info.",
            input_type="signal",
            output_type="signal",
            position="source",
            maps_to_role="encode",
            maps_to_heads=[(1, 0), (1, 1), (2, 1), (2, 6)],
        ),
        ComputationalStep(
            name="number_routing",
            category="movement",
            description=(
                "Heads 6.0 and 9.4 route number information from the subject position "
                "to the verb position, maintaining it past prepositional phrase attractors."
            ),
            input_type="signal",
            output_type="signal",
            position="both",
            maps_to_role="route",
            maps_to_heads=[(6, 0), (9, 4)],
        ),
        ComputationalStep(
            name="verb_output",
            category="output",
            description="Heads 10.0, 11.4, 11.6, 11.7 write the correctly inflected verb form to output logits.",
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="output",
            maps_to_heads=[(10, 0), (11, 4), (11, 6), (11, 7)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="number_embedding",
            target="subject_encoding",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="number_embedding",
            target="number_routing",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="subject_encoding",
            target="number_routing",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="subject_encoding",
            target="verb_output",
            mechanism="residual_stream",
        ),
        ComputationalEdge(
            source="number_routing",
            target="verb_output",
            mechanism="residual_stream",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_output_reduces_performance",
            claim="Ablating output heads substantially reduces verb agreement accuracy.",
            intervention=InterventionType.ablate,
            intervention_target="verb_output",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.3,
        ),
        CausalPrediction(
            name="ablate_routing_reduces_output",
            claim="Ablating routing heads reduces number info reaching verb position.",
            intervention=InterventionType.ablate,
            intervention_target="number_routing",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_encoding_reduces_output",
            claim="Ablating subject encoding heads impairs agreement performance.",
            intervention=InterventionType.ablate,
            intervention_target="subject_encoding",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.1,
        ),
        CausalPrediction(
            name="ablate_embed_reduces_encoding",
            claim="Ablating embedding heads reduces encoder activation magnitude.",
            intervention=InterventionType.ablate,
            intervention_target="number_embedding",
            measurement_target="subject_encoding",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_output_does_not_affect_routing",
            claim="Ablating output heads should not affect upstream routing.",
            intervention=InterventionType.ablate,
            intervention_target="verb_output",
            measurement_target="number_routing",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_routing_does_not_affect_embedding",
            claim="Ablating routing heads should not affect upstream embedding.",
            intervention=InterventionType.ablate,
            intervention_target="number_routing",
            measurement_target="number_embedding",
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
        known_confounds=["attractor nouns can partially override number signal"],
        mitigation="Test with varying attractor counts to assess robustness.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="https://arxiv.org/abs/2506.22105",
    author="Lazo et al. 2025",
)
