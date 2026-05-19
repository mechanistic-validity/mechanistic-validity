"""IOI MechanisticClaimSpec — pre-registered hypothesis for Track 3.

Maps Wang et al. 2023's 6-role circuit into a computational DAG with
testable causal predictions and negative controls.
"""
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

IOI_SPEC = MechanisticClaimSpec(
    task_id="ioi",
    model_family="gpt2",
    linguistic_claim=(
        "GPT-2 small identifies the indirect object in sentences like "
        "'When Mary and John went to the store, John gave a drink to' "
        "via a 6-role circuit: duplicate token detection, previous token "
        "tracking, induction, S-inhibition, name movers, and negative "
        "name movers."
    ),
    steps=[
        ComputationalStep(
            name="duplicate_token_detection",
            category="detection",
            description="Heads 0.1 and 3.0 detect which name token appears twice in the context.",
            input_type="token_identity",
            output_type="signal",
            position="source",
            maps_to_role="DTH",
            maps_to_heads=[(0, 1), (3, 0)],
        ),
        ComputationalStep(
            name="previous_token_tracking",
            category="detection",
            description="Heads 2.2 and 4.11 track the previous token position, feeding induction.",
            input_type="position",
            output_type="attention_pattern",
            position="source",
            maps_to_role="PTH",
            maps_to_heads=[(2, 2), (4, 11)],
        ),
        ComputationalStep(
            name="induction",
            category="movement",
            description="Heads 5.5 and 6.9 compose with PTH to copy repeated token information.",
            input_type="token_identity",
            output_type="signal",
            position="both",
            maps_to_role="IND",
            maps_to_heads=[(5, 5), (6, 9)],
        ),
        ComputationalStep(
            name="s_inhibition",
            category="inhibition",
            description=(
                "Heads 7.3, 7.9, 8.6, 8.10 suppress the repeated (subject) name, "
                "allowing name movers to attend to the indirect object."
            ),
            input_type="signal",
            output_type="signal",
            position="destination",
            maps_to_role="S-Inh",
            maps_to_heads=[(7, 3), (7, 9), (8, 6), (8, 10)],
        ),
        ComputationalStep(
            name="name_mover",
            category="output",
            description="Heads 9.9, 9.6, 10.0 copy the indirect object name to the output logits.",
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="NM",
            maps_to_heads=[(9, 9), (9, 6), (10, 0)],
        ),
        ComputationalStep(
            name="negative_name_mover",
            category="output",
            description=(
                "Heads 10.7 and 11.10 suppress the indirect object name, acting as "
                "a calibration/backup mechanism opposing the name movers."
            ),
            input_type="signal",
            output_type="logit_boost",
            position="destination",
            maps_to_role="NegNM",
            maps_to_heads=[(10, 7), (11, 10)],
        ),
    ],
    edges=[
        ComputationalEdge(
            source="duplicate_token_detection",
            target="s_inhibition",
            mechanism="residual_stream",
            description="DTH signal propagates to S-Inh via residual stream.",
        ),
        ComputationalEdge(
            source="duplicate_token_detection",
            target="induction",
            mechanism="attention_composition",
            description="DTH composes with induction heads (Q-composition or K-composition).",
        ),
        ComputationalEdge(
            source="previous_token_tracking",
            target="induction",
            mechanism="attention_composition",
            description="PTH provides previous-token positional signal to induction heads.",
        ),
        ComputationalEdge(
            source="induction",
            target="s_inhibition",
            mechanism="residual_stream",
            description="Induction output feeds into S-Inh for name suppression.",
        ),
        ComputationalEdge(
            source="s_inhibition",
            target="name_mover",
            mechanism="residual_stream",
            description="S-Inh suppression signal causes NM to attend to the non-repeated name.",
        ),
        ComputationalEdge(
            source="s_inhibition",
            target="negative_name_mover",
            mechanism="residual_stream",
            description="S-Inh also feeds NegNM (opposing calibration pathway).",
        ),
    ],
    predictions=[
        CausalPrediction(
            name="ablate_dth_reduces_output",
            claim="Ablating duplicate token heads reduces logit diff (early detection is necessary).",
            intervention=InterventionType.ablate,
            intervention_target="duplicate_token_detection",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_induction_reduces_output",
            claim="Ablating induction heads substantially reduces logit diff.",
            intervention=InterventionType.ablate,
            intervention_target="induction",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.5,
        ),
        CausalPrediction(
            name="ablate_s_inh_kills_output",
            claim="Ablating S-inhibition heads drives logit diff to near zero.",
            intervention=InterventionType.ablate,
            intervention_target="s_inhibition",
            measurement_target="output",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.8,
        ),
        CausalPrediction(
            name="ablate_s_inh_reduces_name_mover_output",
            claim="Ablating S-inhibition heads reduces name mover activation magnitude.",
            intervention=InterventionType.ablate,
            intervention_target="s_inhibition",
            measurement_target="name_mover",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.2,
        ),
        CausalPrediction(
            name="ablate_neg_nm_increases_logit_diff",
            claim="Ablating negative name movers increases logit diff (removes opposing signal).",
            intervention=InterventionType.ablate,
            intervention_target="negative_name_mover",
            measurement_target="output",
            expected_direction=PredictionDirection.increase,
            expected_metric="role_ablation",
        ),
        CausalPrediction(
            name="ablate_dth_reduces_induction",
            claim="Ablating DTH reduces induction head activation (DTH→IND edge).",
            intervention=InterventionType.ablate,
            intervention_target="duplicate_token_detection",
            measurement_target="induction",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
        CausalPrediction(
            name="ablate_pth_reduces_induction",
            claim="Ablating PTH reduces induction head activation (PTH→IND edge).",
            intervention=InterventionType.ablate,
            intervention_target="previous_token_tracking",
            measurement_target="induction",
            expected_direction=PredictionDirection.decrease,
            expected_metric="role_ablation",
            expected_threshold=0.05,
        ),
    ],
    negative_controls=[
        CausalPrediction(
            name="ablate_nm_does_not_affect_s_inhibition",
            claim="Ablating name movers should not affect upstream S-inhibition selectivity.",
            intervention=InterventionType.ablate,
            intervention_target="name_mover",
            measurement_target="s_inhibition",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_neg_nm_does_not_affect_s_inhibition",
            claim="Ablating negative name movers should not affect upstream S-inhibition.",
            intervention=InterventionType.ablate,
            intervention_target="negative_name_mover",
            measurement_target="s_inhibition",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_pth_does_not_kill_output",
            claim="Ablating previous token heads alone should not destroy circuit output.",
            intervention=InterventionType.ablate,
            intervention_target="previous_token_tracking",
            measurement_target="output",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            expected_threshold=0.3,
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_s_inh_does_not_affect_dth",
            claim="Ablating S-inhibition should not affect upstream duplicate token detection.",
            intervention=InterventionType.ablate,
            intervention_target="s_inhibition",
            measurement_target="duplicate_token_detection",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
        CausalPrediction(
            name="ablate_nm_does_not_affect_neg_nm",
            claim="Ablating name movers should not affect negative name movers (parallel roles).",
            intervention=InterventionType.ablate,
            intervention_target="name_mover",
            measurement_target="negative_name_mover",
            expected_direction=PredictionDirection.invariant,
            expected_metric="role_ablation",
            is_negative_control=True,
        ),
    ],
    identifiability=IdentifiabilityGate(
        status=IdentifiabilityStatus.identifiable,
        available_interventions=[
            InterventionType.ablate,
            InterventionType.patch,
            InterventionType.resample,
        ],
        notes="All roles can be intervened on via activation patching at hook_result.",
    ),
    superposition_risk=SuperpositionGate(
        polysemanticity_risk="low",
        known_confounds=["backup name movers partially compensate for NM ablation"],
        mitigation="Use knockout + path patching to isolate direct vs backup pathways.",
    ),
    description_mode=DescriptionMode.impl_functional,
    paper_ref="https://arxiv.org/abs/2211.00593",
    author="Wang et al. 2023",
)
