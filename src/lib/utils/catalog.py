"""Exhaustive catalog of mechanistically-studied circuit tasks.

Every task that has published circuit-level analysis, a generator in this
codebase, or a strong mechanistic hypothesis from the literature is
cataloged here with full provenance, model support, and priority status.

The catalog is model-agnostic: each entry lists which models the task
has been validated on and which it can plausibly run on, so the same
catalog supports GPT-2 Small, Medium, Large, XL, Qwen, Gemma, Llama.
"""
from __future__ import annotations

from dataclasses import dataclass

from .citations import (
    Citation,
    CONMY_ACDC,
    FRIEDMAN_BRACKETS,
    GARCIA_CARRASCO_ACRONYM,
    GEVA_FACTUAL,
    HANNA_GT,
    HEIMERSHEIM_DOCSTRING,
    KIM_ENTITY,
    LAZO_SVA,
    LINZEN_SVA,
    MATHWIN_GENDER,
    MCDOUGALL_COPY_SUPP,
    MERULLO_COLORED,
    MILLER_A_AN,
    NANDA_MODULAR,
    OLSSON_INDUCTION,
    THIS_WORK,
    TIGGES_SENTIMENT,
    WANG_IOI,
    WARSTADT_BLIMP,
    WILCOX_FILLER,
)


@dataclass(frozen=True)
class TaskEntry:
    """One entry in the exhaustive task catalog."""

    # Identity
    task_id: str
    display_name: str
    description: str

    # Provenance
    citation: Citation | None
    source: str  # "published" | "this_work" | "blimp" | "roadmap"

    # Domain classification
    domain: str  # "entity_tracking" | "agreement" | "repetition" | "numerical"
                 # | "structural" | "factual" | "discourse"

    # Circuit status (on primary_model)
    circuit_status: str  # "full_circuit" | "partial" | "behavioral_only" | "untested"
    n_heads: int | None
    recovery_pct: float | None

    # Model support
    primary_model: str  # Model where circuit was discovered/validated
    supported_models: tuple[str, ...]  # All models where this task can run
    model_notes: str

    # Evaluation
    metric_type: str  # "logit_diff" | "prob_diff" | "accuracy" | "minimal_pair"

    # Generator
    has_generator: bool
    generator_module: str  # Python module path or "" if none

    # Priority for future circuit work
    priority: str  # "P0_done" | "P1_easy_win" | "P2_has_generator"
                   # | "P3_needs_work" | "P4_different_model"
    notes: str


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

TASK_CATALOG: dict[str, TaskEntry] = {}

# ===== ENTITY TRACKING / DISCOURSE =====

TASK_CATALOG["ioi"] = TaskEntry(
    task_id="ioi",
    display_name="IOI (Indirect Object Identification)",
    description="Predict which of two names receives the indirect object: 'A and B went... gave to ___'",
    citation=WANG_IOI,
    source="published",
    domain="entity_tracking",
    circuit_status="full_circuit",
    n_heads=26,
    recovery_pct=87.0,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl",
                      "qwen25_0.5b", "gemma2_2b", "llama31_8b"),
    model_notes="MIB benchmark task. Circuits discovered across all GPT-2 sizes (Paper B). "
                "MIB leaderboard covers Qwen/Gemma/Llama.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="The canonical mech-interp circuit. 7 roles in roles.py. See task_reference_baselines.py "
          "for MIB CMD scores across all models.",
)

TASK_CATALOG["copy_suppression"] = TaskEntry(
    task_id="copy_suppression",
    display_name="Copy Suppression",
    description="L10H7 suppresses over-confident copies from induction/copy heads",
    citation=MCDOUGALL_COPY_SUPP,
    source="published",
    domain="entity_tracking",
    circuit_status="full_circuit",
    n_heads=5,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="76.9% of L10H7 impact explained. Copy suppression heads exist in all GPT-2 sizes.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="3 roles in roles.py: PTH, IND, copy_suppress. "
          "Shares IOI surface form but different scoring.",
)

TASK_CATALOG["centering_theory"] = TaskEntry(
    task_id="centering_theory",
    display_name="Centering Theory",
    description="Backward-looking center (discourse salience) vs surface token repetition",
    citation=THIS_WORK,
    source="this_work",
    domain="entity_tracking",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="3 conditions: standard, recency, subject_prominence. 150 prompts.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Knockout profile shows high overlap with copy_suppression (J=0.500). "
          "Uses IOI circuit as proxy in roles.py.",
)

TASK_CATALOG["resumptive"] = TaskEntry(
    task_id="resumptive",
    display_name="Resumptive Repetition",
    description="Verbatim copy (Tier I) vs discourse-contextual continuation (Tier III)",
    citation=THIS_WORK,
    source="this_work",
    domain="entity_tracking",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="15 prompts. Tests boundary between Tier I (surface copy) and Tier III (discourse).",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Based on Norrick 1987 repetition theory.",
)

TASK_CATALOG["self_allo"] = TaskEntry(
    task_id="self_allo",
    display_name="Self vs Allo-Repetition",
    description="Tannen's distinction: same-speaker repetition vs cross-speaker echo",
    citation=THIS_WORK,
    source="this_work",
    domain="entity_tracking",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="20 prompts (10 self, 10 allo). Based on Tannen 1989.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Different discourse function for self vs allo repetition.",
)

TASK_CATALOG["mib_rti"] = TaskEntry(
    task_id="mib_rti",
    display_name="MIB RTI",
    description="RTI in MIB-format paired intervention style",
    citation=THIS_WORK,
    source="this_work",
    domain="entity_tracking",
    circuit_status="full_circuit",
    n_heads=15,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small",),
    model_notes="MIB-format wrapper around RTI task.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Uses RTI circuit from roles.py.",
)

TASK_CATALOG["colored_objects"] = TaskEntry(
    task_id="colored_objects",
    display_name="Colored Objects (Property Binding)",
    description="'The red ball, the blue cup... which is green?' — attribute-object binding",
    citation=MERULLO_COLORED,
    source="roadmap",
    domain="entity_tracking",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="78% overlap with IOI circuit. Same heads repurposed for binding.",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P1_easy_win",
    notes="Fast win: IOI heads already known, just need dataset + validation.",
)

TASK_CATALOG["entity_tracking"] = TaskEntry(
    task_id="entity_tracking",
    display_name="Entity State Tracking",
    description="Track entity state across sentences: 'put ball in box, take out ___'",
    citation=KIM_ENTITY,
    source="roadmap",
    domain="entity_tracking",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Novel circuit type: state tracking rather than pattern completion.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P3_needs_work",
    notes="Kim & Schuster 2023 identifies probes but not a full attention circuit.",
)

# ===== AGREEMENT =====

TASK_CATALOG["sva"] = TaskEntry(
    task_id="sva",
    display_name="SVA (Subject-Verb Agreement)",
    description="Subject-verb number agreement across intervening attractors",
    citation=LAZO_SVA,
    source="published",
    domain="agreement",
    circuit_status="full_circuit",
    n_heads=12,
    recovery_pct=93.0,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl",
                      "pythia_160m", "pythia_1.4b"),
    model_notes="Circuit scales with model size. Lazo et al. tested GPT-2 + Pythia. "
                "93% recovery on base, 80% on complex (125 heads).",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="4 roles in roles.py: embed, encode, route, output. "
          "BLiMP category accuracy: 0.864 (GPT-2 Large).",
)

TASK_CATALOG["gendered_pronoun"] = TaskEntry(
    task_id="gendered_pronoun",
    display_name="Gendered Pronoun",
    description="'The nurse... she/he' — anaphor gender agreement with occupational noun",
    citation=MATHWIN_GENDER,
    source="published",
    domain="agreement",
    circuit_status="full_circuit",
    n_heads=5,
    recovery_pct=100.0,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="ACDC-discovered circuit >= full model performance.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="3 roles in roles.py: early_ga, late_ga, name_bind. "
          "BLiMP anaphor gender accuracy: 0.994 (GPT-2 Large).",
)

TASK_CATALOG["sva_pp"] = TaskEntry(
    task_id="sva_pp",
    display_name="SVA across Prepositional Phrase",
    description="'The key to the cabinets is/are' — agreement with PP attractor",
    citation=LINZEN_SVA,
    source="roadmap",
    domain="agreement",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Simple extension of SVA with specific attractor type. "
                "Linzen et al. 2016 established the benchmark.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P2_has_generator",
    notes="Trivial to generate from existing SVA infrastructure. "
          "Tests whether SVA circuit generalizes to PP attractors.",
)

TASK_CATALOG["reflexive_anaphora"] = TaskEntry(
    task_id="reflexive_anaphora",
    display_name="Reflexive Anaphora (Principle A)",
    description="'The queen admired herself/himself' — reflexive gender must match subject",
    citation=WARSTADT_BLIMP,
    source="blimp",
    domain="agreement",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6 with 4 subtypes: short, distractor_pp, "
                "distractor_rc, control. ~1000 BLiMP-style minimal pairs.",
    metric_type="minimal_pair",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_reflexive_data",
    priority="P2_has_generator",
    notes="BLiMP anaphor_number accuracy: 0.992 (GPT-2 Large). "
          "Generator validates single-BPE constraint.",
)

TASK_CATALOG["npi_licensing"] = TaskEntry(
    task_id="npi_licensing",
    display_name="NPI Licensing",
    description="'Nobody ever/always did X' — negative polarity item requires licensor",
    citation=WARSTADT_BLIMP,
    source="roadmap",
    domain="agreement",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="BLiMP npi_present paradigm. No circuit published anywhere.",
    metric_type="minimal_pair",
    has_generator=False,
    generator_module="",
    priority="P3_needs_work",
    notes="BLiMP data available but generator not yet written.",
)

TASK_CATALOG["binding_principles"] = TaskEntry(
    task_id="binding_principles",
    display_name="Binding Principles A/B",
    description="Reflexive (Principle A) vs non-reflexive (Principle B) binding domains",
    citation=WARSTADT_BLIMP,
    source="roadmap",
    domain="agreement",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="BLiMP anaphor_number paradigm. Extends reflexive_anaphora "
                "with non-reflexive binding.",
    metric_type="minimal_pair",
    has_generator=False,
    generator_module="",
    priority="P3_needs_work",
    notes="More complex than reflexive_anaphora; requires clause boundary tracking.",
)

# ===== NUMERICAL =====

TASK_CATALOG["greater_than"] = TaskEntry(
    task_id="greater_than",
    display_name="Greater-Than",
    description="'The war lasted from 1743 to 17__' — predict valid later year digit",
    citation=HANNA_GT,
    source="published",
    domain="numerical",
    circuit_status="full_circuit",
    n_heads=7,
    recovery_pct=89.5,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="7 attention heads + 4 MLP layers. Generalizes to 'started/ended', "
                "'luxury goods', number sequences (88-99% recovery).",
    metric_type="prob_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="2 roles in roles.py: early_gt, late_gt. "
          "ACDC edge-level AUC: 0.883 (KL), 0.461 (task-specific).",
)

TASK_CATALOG["less_than"] = TaskEntry(
    task_id="less_than",
    display_name="Less-Than",
    description="Symmetric variant of greater-than: predict valid earlier year digit",
    citation=None,
    source="roadmap",
    domain="numerical",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Trivial extension of greater-than: reverse the comparison direction.",
    metric_type="prob_diff",
    has_generator=False,
    generator_module="",
    priority="P1_easy_win",
    notes="Modify greater_than_prompts to ask for earlier year.",
)

TASK_CATALOG["modular_arithmetic"] = TaskEntry(
    task_id="modular_arithmetic",
    display_name="Modular Arithmetic (Grokking)",
    description="(a + b) mod p on a trained toy 1-layer transformer",
    citation=NANDA_MODULAR,
    source="published",
    domain="numerical",
    circuit_status="full_circuit",
    n_heads=None,
    recovery_pct=None,
    primary_model="toy_1L",
    supported_models=("toy_1L",),
    model_notes="Full reverse engineering via discrete Fourier + trig identities. "
                "Not applicable to GPT-2 (requires specifically trained toy model).",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="Foundational grokking result. Separate trained model required.",
)

TASK_CATALOG["arithmetic"] = TaskEntry(
    task_id="arithmetic",
    display_name="Arithmetic (MIB)",
    description="Addition and subtraction in MIB benchmark format",
    citation=None,
    source="published",
    domain="numerical",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="llama31_8b",
    supported_models=("llama31_8b",),
    model_notes="MIB benchmark task. EAP_IG_act_CF CMD=0.00 (near perfect). "
                "Only evaluated on Llama 3.1 8B.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="MIB leaderboard: arith_add and arith_sub sub-tasks.",
)

# ===== REPETITION / PATTERN =====

TASK_CATALOG["induction"] = TaskEntry(
    task_id="induction",
    display_name="Induction",
    description="[A][B]...[A] -> [B]: in-context pattern completion via bigram statistics",
    citation=OLSSON_INDUCTION,
    source="published",
    domain="repetition",
    circuit_status="full_circuit",
    n_heads=7,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl",
                      "pythia_160m", "pythia_1.4b", "pythia_6.9b",
                      "qwen25_0.5b", "gemma2_2b", "llama31_8b"),
    model_notes="Universal across all transformer sizes and families. "
                "Forms at same training phase regardless of scale.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="2 roles in roles.py: PTH, IND. The foundational in-context learning circuit.",
)

TASK_CATALOG["rti"] = TaskEntry(
    task_id="rti",
    display_name="RTI (Repeated Token Interference)",
    description="Detect and suppress repeated tokens during generation to prevent degeneration",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="full_circuit",
    n_heads=15,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="15-head circuit in 4 tiers. Backbone ablation: degeneration 47.5% -> 87.3%. "
                "Cross-scale EAP validation: rho=0.916 (small-medium).",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part4_rigorous_circuit_finding.experiments.probes-rti-v2.run_probes_rti_v2",
    priority="P0_done",
    notes="4 roles in roles.py: backbone (L0), detector (L4H11), copier (8 heads), readout (3 heads). "
          "Paper A core discovery. Generator: make_rti_prompts() in run_probes_rti_v2.py.",
)

TASK_CATALOG["rti_pattern"] = TaskEntry(
    task_id="rti_pattern",
    display_name="RTI Pattern (Sentence Continuation)",
    description="'The cat sat on the mat. The cat ___' — continue from repeated sentence prefix",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Category 2 in 18-category repetition taxonomy. 200 prompts.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Knockout profile distinct from base RTI (J<0.20).",
)

TASK_CATALOG["sequence_internal"] = TaskEntry(
    task_id="sequence_internal",
    display_name="Sequence-Internal Pattern",
    description="Extract abstract slot structure: 'one fish two fish red fish ___'",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Category 3 in repetition taxonomy. 100 prompts. "
                "Uses induction circuit as proxy in roles.py.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Tests abstract pattern extraction beyond simple bigrams.",
)

TASK_CATALOG["alternating_pair"] = TaskEntry(
    task_id="alternating_pair",
    display_name="Alternating Pair / N-Cycle",
    description="'tick tock tick tock tick ___' — position tracking within repeating cycle",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Category 5 in repetition taxonomy. 100 prompts. "
                "Uses induction circuit as proxy in roles.py.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Modular position tracking, distinct from induction (no prefix-suffix matching).",
)

TASK_CATALOG["token_flood"] = TaskEntry(
    task_id="token_flood",
    display_name="Token Flood",
    description="'hello hello hello...' — single-token momentum vs anti-repetition",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Category 4 in repetition taxonomy. 40 prompts. "
                "Tests momentum vs anti-repetition balance.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Uses RTI circuit as proxy in roles.py.",
)

TASK_CATALOG["novel_song"] = TaskEntry(
    task_id="novel_song",
    display_name="Novel Song (Memorization Control)",
    description="Known songs vs novel songs with same structural template — memorization vs structure",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Tier V control. 15 prompts (5 known + 10 novel). "
                "Tests whether completion relies on memorized sequences.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Uses induction circuit as proxy in roles.py.",
)

TASK_CATALOG["buffalo"] = TaskEntry(
    task_id="buffalo",
    display_name="Buffalo (Garden-Path Semantic Trap)",
    description="'Buffalo buffalo buffalo buffalo' — grammatically valid repeated-word sentences",
    citation=THIS_WORK,
    source="this_work",
    domain="repetition",
    circuit_status="behavioral_only",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Category 8 in repetition taxonomy. Tests syntactic parsing under "
                "surface-level repetition.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="Uses RTI circuit as proxy in roles.py. "
          "Includes 'had had', 'that that', 'police police' variants.",
)

# ===== STRUCTURAL / SYNTACTIC =====

TASK_CATALOG["acronym"] = TaskEntry(
    task_id="acronym",
    display_name="Acronym Prediction",
    description="'The acronym for National Basketball Association is N___' — letter-mover heads",
    citation=GARCIA_CARRASCO_ACRONYM,
    source="published",
    domain="structural",
    circuit_status="full_circuit",
    n_heads=8,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="8-head circuit, 3 roles. Circuit-only slightly exceeds full model.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="tasks.task_prompts",
    priority="P0_done",
    notes="3 roles in roles.py: letter_mover, PTH, propagator. "
          "Overlaps with Paper E phonological operations.",
)

TASK_CATALOG["docstring"] = TaskEntry(
    task_id="docstring",
    display_name="Docstring Argument Prediction",
    description="Predict next argument name in Python docstring from function signature",
    citation=HEIMERSHEIM_DOCSTRING,
    source="published",
    domain="structural",
    circuit_status="full_circuit",
    n_heads=8,
    recovery_pct=None,
    primary_model="attn_only_4L",
    supported_models=("attn_only_4L", "gpt2_small"),
    model_notes="Circuit discovered on 4-layer attention-only transformer. "
                "Can run on GPT-2 Small but ground-truth circuit heads differ. "
                "ACDC AUC: 0.434 (KL), 0.972 (task-specific).",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P3_needs_work",
    notes="Adaptation to full GPT-2 non-trivial (MLPs participate).",
)

TASK_CATALOG["bracket_matching"] = TaskEntry(
    task_id="bracket_matching",
    display_name="Bracket Matching",
    description="Predict correct closing bracket in nested Python/JSON structure",
    citation=FRIEDMAN_BRACKETS,
    source="roadmap",
    domain="structural",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Structurally interesting: tests stack-like computation in attention. "
                "No published circuit for GPT-2.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P3_needs_work",
    notes="Tests whether L0 backbone generalizes to structural (non-semantic) tasks.",
)

TASK_CATALOG["filler_gap"] = TaskEntry(
    task_id="filler_gap",
    display_name="Filler-Gap Dependencies",
    description="'What did John eat ___?' — wh-movement tracking across clauses",
    citation=WILCOX_FILLER,
    source="roadmap",
    domain="structural",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6 (4-way factorial design). "
                "No published circuit.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_filler_gap_data",
    priority="P2_has_generator",
    notes="Linguistically rich syntactic movement phenomenon.",
)

TASK_CATALOG["negation"] = TaskEntry(
    task_id="negation",
    display_name="Negation Scope",
    description="'The book was not good, it was ___' — scope reversal predicts antonym",
    citation=None,
    source="this_work",
    domain="structural",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_negation_data",
    priority="P2_has_generator",
    notes="Tests semantic negation scope processing.",
)

TASK_CATALOG["conditional"] = TaskEntry(
    task_id="conditional",
    display_name="Conditional If-Then",
    description="'If the plan is ready, then ___' — conditional scope predicts 'then'",
    citation=None,
    source="this_work",
    domain="structural",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_conditional_data",
    priority="P2_has_generator",
    notes="Tests syntactic scope tracking (if -> then dependency).",
)

TASK_CATALOG["ellipsis"] = TaskEntry(
    task_id="ellipsis",
    display_name="Ellipsis Resolution",
    description="'Alice likes pizza, and Bob does too/not' — VP-ellipsis and gapping",
    citation=None,
    source="this_work",
    domain="structural",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_ellipsis_data",
    priority="P2_has_generator",
    notes="VP-ellipsis resolution requires discourse-level processing.",
)

# ===== FACTUAL / SEMANTIC =====

TASK_CATALOG["sentiment"] = TaskEntry(
    task_id="sentiment",
    display_name="Sentiment (Linear Direction)",
    description="SST sentiment classification via linear probe + causal ablation",
    citation=TIGGES_SENTIMENT,
    source="roadmap",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Single linear direction causally mediates 76% of accuracy. "
                "Summarized at comma/name positions. Small head + neuron subset.",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P2_has_generator",
    notes="Already has partial circuit characterization. SST dataset readily available.",
)

TASK_CATALOG["factual_recall"] = TaskEntry(
    task_id="factual_recall",
    display_name="Factual Recall (Country Capitals)",
    description="'Paris is the capital of ___' — factual association via 3-step mechanism",
    citation=GEVA_FACTUAL,
    source="roadmap",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_xl",
    supported_models=("gpt2_xl", "pythia_6.9b", "llama31_8b"),
    model_notes="3-step mechanism: early MLPs enrich subject, relation propagates, "
                "late attention heads extract attribute. Needs large model for factual knowledge.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="GPT-2 Small lacks factual knowledge for most relations. "
          "Run on XL or larger.",
)

TASK_CATALOG["a_an_prediction"] = TaskEntry(
    task_id="a_an_prediction",
    display_name="A/An Article Prediction",
    description="Predict 'a' vs 'an' based on following word's initial phoneme",
    citation=MILLER_A_AN,
    source="roadmap",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_large",
    supported_models=("gpt2_large", "gpt2_xl"),
    model_notes="Single neuron L1N2631 in GPT-2 Large. Phonological/orthographic. "
                "GPT-2 Small may lack the specialized neuron.",
    metric_type="logit_diff",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="Miller & Neo 2023. Interesting single-neuron characterization.",
)

# ===== DISCOURSE (part6 extras) =====

TASK_CATALOG["definiteness"] = TaskEntry(
    task_id="definiteness",
    display_name="Definiteness / Discourse Tracking",
    description="'I saw a cat. The cat was big' — new-reference (a->the) tracking",
    citation=None,
    source="this_work",
    domain="discourse",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_definiteness_data",
    priority="P2_has_generator",
    notes="Tests discourse-level reference tracking (a -> the alternation).",
)

TASK_CATALOG["but_reversal"] = TaskEntry(
    task_id="but_reversal",
    display_name="But Reversal",
    description="'The meal was bad, but it was ___' — 'but' reverses accumulated valence",
    citation=None,
    source="this_work",
    domain="discourse",
    circuit_status="untested",
    n_heads=None,
    recovery_pct=None,
    primary_model="gpt2_small",
    supported_models=("gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    model_notes="Generator exists in part6.",
    metric_type="logit_diff",
    has_generator=True,
    generator_module="part6_new_datasets_linguistics.experiments.initial.generate_but_reversal_data",
    priority="P2_has_generator",
    notes="Tests discourse connective processing (adversative 'but').",
)

# ===== MIB BENCHMARK TASKS (multi-model) =====

TASK_CATALOG["mcqa"] = TaskEntry(
    task_id="mcqa",
    display_name="Multiple Choice QA (MIB)",
    description="Multiple-choice question answering in MIB benchmark format",
    citation=None,
    source="published",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="qwen25_0.5b",
    supported_models=("qwen25_0.5b", "gemma2_2b", "llama31_8b"),
    model_notes="MIB benchmark task evaluated on Qwen/Gemma/Llama. "
                "Not available for GPT-2 family.",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="MIB leaderboard CMD scores available in task_reference_baselines.py.",
)

TASK_CATALOG["arc_easy"] = TaskEntry(
    task_id="arc_easy",
    display_name="ARC-Easy (MIB)",
    description="AI2 Reasoning Challenge (easy split) in MIB benchmark format",
    citation=None,
    source="published",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="gemma2_2b",
    supported_models=("gemma2_2b", "llama31_8b"),
    model_notes="MIB benchmark task. Not available for GPT-2 family.",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="MIB leaderboard CMD scores available in task_reference_baselines.py.",
)

TASK_CATALOG["arc_challenge"] = TaskEntry(
    task_id="arc_challenge",
    display_name="ARC-Challenge (MIB)",
    description="AI2 Reasoning Challenge (challenge split) in MIB benchmark format",
    citation=None,
    source="published",
    domain="factual",
    circuit_status="partial",
    n_heads=None,
    recovery_pct=None,
    primary_model="llama31_8b",
    supported_models=("llama31_8b",),
    model_notes="MIB benchmark task. Only evaluated on Llama 3.1 8B.",
    metric_type="accuracy",
    has_generator=False,
    generator_module="",
    priority="P4_different_model",
    notes="MIB leaderboard CMD scores available in task_reference_baselines.py.",
)
