"""Built-in task registrations for all shipped circuits.

Tasks are organized by experiment_group (which research effort produced them)
and tagged with domain (what linguistic phenomenon they test) and
circuit_status (full_circuit | proxy_circuit | generator_only | planned).
"""
from __future__ import annotations

from mechval.lib.tasks import TaskPrompt, TokenizerLike
from mechval.lib.tasks.task import CircuitTask, HeadCircuitTask

from mechval.lib.tasks.ioi import circuit as _ioi_circuit
from mechval.lib.tasks.greater_than import circuit as _gt_circuit
from mechval.lib.tasks.induction import circuit as _ind_circuit
from mechval.lib.tasks.sva import circuit as _sva_circuit
from mechval.lib.tasks.gendered_pronoun import circuit as _gp_circuit
from mechval.lib.tasks.acronym import circuit as _acr_circuit
from mechval.lib.tasks.copy_suppression import circuit as _cs_circuit
from mechval.lib.tasks.rti import circuit as _rti_circuit
from mechval.lib.tasks.epistemic_framing import (
    circuit as _ef_circuit,
    circuit_tight as _ef_tight_circuit,
    circuit_expanded as _ef_expanded_circuit,
    circuit_eap as _ef_eap_circuit,
)

from mechval.lib.tasks.ioi.prompts import (
    ioi_prompts, centering_theory_prompts, resumptive_prompts, self_allo_prompts,
)
from mechval.lib.tasks.ioi.claim_spec import IOI_SPEC
from mechval.lib.tasks.greater_than.claim_spec import GREATER_THAN_SPEC
from mechval.lib.tasks.induction.claim_spec import INDUCTION_SPEC
from mechval.lib.tasks.sva.claim_spec import SVA_SPEC
from mechval.lib.tasks.rti.claim_spec import RTI_SPEC
from mechval.lib.tasks.epistemic_framing.claim_spec import EPISTEMIC_SPEC
from mechval.lib.tasks.epistemic_framing.claim_spec_tight import EPISTEMIC_TIGHT_SPEC
from mechval.lib.tasks.epistemic_framing.claim_spec_eap import EPISTEMIC_EAP_SPEC
from mechval.lib.tasks.gendered_pronoun.claim_spec import GENDERED_PRONOUN_SPEC
from mechval.lib.tasks.copy_suppression.claim_spec import COPY_SUPPRESSION_SPEC
from mechval.lib.tasks.acronym.claim_spec import ACRONYM_SPEC
from mechval.lib.tasks.epistemic_framing.claim_spec_expanded import EPISTEMIC_EXPANDED_SPEC
from mechval.lib.tasks.greater_than.prompts import greater_than_prompts
from mechval.lib.tasks.induction.prompts import (
    induction_prompts, sequence_internal_prompts, alternating_pair_prompts, novel_song_prompts,
)
from mechval.lib.tasks.sva.prompts import sva_prompts
from mechval.lib.tasks.gendered_pronoun.prompts import gendered_pronoun_prompts
from mechval.lib.tasks.acronym.prompts import acronym_prompts
from mechval.lib.tasks.copy_suppression.prompts import copy_suppression_prompts
from mechval.lib.tasks.rti.prompts import (
    make_rti_prompts, rti_pattern_prompts, token_flood_prompts, buffalo_prompts,
)
from mechval.lib.tasks.epistemic_framing.prompts import epistemic_framing_prompts


# =============================================================================
# Published circuits — from peer-reviewed papers
# =============================================================================

class IOITask(HeadCircuitTask):
    task_id = "ioi"
    source = "published"
    domain = "linguistics_coreference"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://arxiv.org/abs/2211.00593"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(ioi_prompts)

    def get_baselines(self):
        return {
            "logit_diff_full_model": 3.56,
            "faithfulness_gap": 0.46,
            "recovery_pct": 87.0,
            "n_heads": 26,
            "mib_cmd_best": 0.02,
            "mib_cmd_best_method": "EActP_CF",
            "mib_cmd_random": 0.75,
            "acdc_auc_kl": 0.868,
            "cite": "Wang et al. 2023 (ICLR), Section 4",
            "mib_cite": "Mueller et al. 2025 (ICML), Table 2",
        }

    def get_claim_spec(self):
        return IOI_SPEC


class GreaterThanTask(HeadCircuitTask):
    task_id = "greater_than"
    source = "published"
    domain = "math"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://arxiv.org/abs/2305.00586"
    _circuit_module = _gt_circuit
    _prompt_fn = staticmethod(greater_than_prompts)

    def get_claim_spec(self):
        return GREATER_THAN_SPEC

    def get_baselines(self):
        return {
            "prob_diff_full_model": 0.817,
            "prob_diff_circuit": 0.727,
            "recovery_pct": 89.5,
            "n_attn_heads": 7,
            "n_mlps": 4,
            "necessity_score": -0.366,
            "acdc_auc_kl": 0.883,
            "cite": "Hanna et al. 2023 (NeurIPS), Section 3.2",
        }


class InductionTask(HeadCircuitTask):
    task_id = "induction"
    source = "published"
    domain = "patterns"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://arxiv.org/abs/2209.11895"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(induction_prompts)

    def get_claim_spec(self):
        return INDUCTION_SPEC

    def get_baselines(self):
        return {
            "n_induction_heads": 5,
            "n_previous_token_heads": 2,
            "cite": "Olsson et al. 2022, Table 5 Appendix",
        }


class SVATask(HeadCircuitTask):
    task_id = "sva"
    source = "published"
    domain = "linguistics_agreement"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://arxiv.org/abs/2506.22105"
    _circuit_module = _sva_circuit
    _prompt_fn = staticmethod(sva_prompts)

    def get_claim_spec(self):
        return SVA_SPEC

    def get_baselines(self):
        return {
            "logit_diff_acc_full_model": 0.70,
            "logit_diff_acc_circuit": 0.65,
            "recovery_pct": 93.0,
            "n_heads": 12,
            "cite": "Lazo et al. 2025, Table 3 (Section 4.2)",
        }


class GenderedPronounTask(HeadCircuitTask):
    task_id = "gendered_pronoun"
    source = "published"
    domain = "linguistics_agreement"
    experiment_group = "published"
    circuit_status = "full_circuit"
    _circuit_module = _gp_circuit
    _prompt_fn = staticmethod(gendered_pronoun_prompts)

    def get_baselines(self):
        return {
            "recovery_pct": 100.0,
            "cite": "Mathwin 2023 (MATS hackathon, unpublished)",
            "note": "ACDC circuit >= full model performance",
        }

    def get_claim_spec(self):
        return GENDERED_PRONOUN_SPEC


class AcronymTask(HeadCircuitTask):
    task_id = "acronym"
    source = "published"
    domain = "patterns"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://proceedings.mlr.press/v238/garcia-carrasco24a.html"
    _circuit_module = _acr_circuit
    _prompt_fn = staticmethod(acronym_prompts)

    def get_baselines(self):
        return {
            "n_heads": 8,
            "cite": "Garcia-Carrasco et al. 2024 (AISTATS)",
            "note": "circuit-only slightly exceeds full model",
        }

    def get_claim_spec(self):
        return ACRONYM_SPEC


class CopySuppressionTask(HeadCircuitTask):
    task_id = "copy_suppression"
    source = "published"
    domain = "linguistics_coreference"
    experiment_group = "published"
    circuit_status = "full_circuit"
    paper_ref = "https://arxiv.org/abs/2310.04625"
    _circuit_module = _cs_circuit
    _prompt_fn = staticmethod(copy_suppression_prompts)

    def get_claim_spec(self):
        return COPY_SUPPRESSION_SPEC


# =============================================================================
# RTI discovery — part 3-4, core RTI circuit + behavioral variants
# =============================================================================

class RTITask(CircuitTask):
    task_id = "rti"
    source = "ours"
    domain = "patterns"
    experiment_group = "rti_discovery"
    circuit_status = "full_circuit"
    _circuit_module = _rti_circuit

    def get_baselines(self):
        return {
            "n_heads": 15,
            "eap_auroc": 0.39,
        }

    def get_claim_spec(self):
        return RTI_SPEC

    def get_circuit(self):
        from mechval.lib.tasks.spec import CircuitSpec
        m = self._circuit_module
        return CircuitSpec(
            roles=m.ROLES, bands=m.BANDS, pathways=m.PATHWAYS,
            source=self.source, model_family=self.model_family,
        )

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        raw = make_rti_prompts(tokenizer, n=n_prompts, seed=seed)
        return [
            TaskPrompt(
                text=d["text"],
                target_correct=tokenizer.decode([d["correct_id"]]),
                target_incorrect=tokenizer.decode([d["wrong_id"]]),
                metadata={k: v for k, v in d.items() if k not in ("text", "correct_id", "wrong_id")},
            )
            for d in raw
        ]


class RTIPatternTask(HeadCircuitTask):
    task_id = "rti_pattern"
    source = "ours"
    domain = "patterns"
    experiment_group = "rti_discovery"
    circuit_status = "proxy_circuit"
    _circuit_module = _rti_circuit
    _prompt_fn = staticmethod(rti_pattern_prompts)


class TokenFloodTask(HeadCircuitTask):
    task_id = "token_flood"
    source = "ours"
    domain = "patterns"
    experiment_group = "rti_discovery"
    circuit_status = "proxy_circuit"
    _circuit_module = _rti_circuit
    _prompt_fn = staticmethod(token_flood_prompts)


class BuffaloTask(CircuitTask):
    task_id = "buffalo"
    source = "ours"
    domain = "patterns"
    experiment_group = "rti_discovery"
    circuit_status = "proxy_circuit"
    _circuit_module = _rti_circuit

    def get_circuit(self):
        from mechval.lib.tasks.spec import CircuitSpec
        m = self._circuit_module
        return CircuitSpec(
            roles=m.ROLES, bands=m.BANDS, pathways=m.PATHWAYS,
            source=self.source, model_family=self.model_family,
        )

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        return buffalo_prompts(tokenizer, seed=seed)[:n_prompts]


# =============================================================================
# IOI discourse variants — part 5, probing discourse phenomena via IOI circuit
# =============================================================================

class CenteringTheoryTask(HeadCircuitTask):
    task_id = "centering_theory"
    source = "ours"
    domain = "linguistics_coreference"
    experiment_group = "ioi_ablations"
    circuit_status = "proxy_circuit"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(centering_theory_prompts)


class ResumptiveTask(HeadCircuitTask):
    task_id = "resumptive"
    source = "ours"
    domain = "linguistics_coreference"
    experiment_group = "ioi_ablations"
    circuit_status = "proxy_circuit"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(resumptive_prompts)


class SelfAlloTask(HeadCircuitTask):
    task_id = "self_allo"
    source = "ours"
    domain = "linguistics_coreference"
    experiment_group = "ioi_ablations"
    circuit_status = "proxy_circuit"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(self_allo_prompts)


# =============================================================================
# Repetition taxonomy — part 8, 18-category classification
# =============================================================================

class SequenceInternalTask(HeadCircuitTask):
    task_id = "sequence_internal"
    source = "ours"
    domain = "patterns"
    experiment_group = "repetition_taxonomy"
    circuit_status = "proxy_circuit"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(sequence_internal_prompts)


class AlternatingPairTask(HeadCircuitTask):
    task_id = "alternating_pair"
    source = "ours"
    domain = "patterns"
    experiment_group = "repetition_taxonomy"
    circuit_status = "proxy_circuit"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(alternating_pair_prompts)


class NovelSongTask(HeadCircuitTask):
    task_id = "novel_song"
    source = "ours"
    domain = "patterns"
    experiment_group = "repetition_taxonomy"
    circuit_status = "proxy_circuit"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(novel_song_prompts)


# =============================================================================
# Epistemic study — part 15, epistemic framing analysis
# =============================================================================

class EpistemicFramingTask(HeadCircuitTask):
    task_id = "epistemic_framing"
    source = "experimental"
    domain = "linguistics_pragmatics"
    experiment_group = "epistemic_study"
    circuit_status = "full_circuit"
    _circuit_module = _ef_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)

    def get_baselines(self):
        return {
            "n_heads": 4,
        }

    def get_claim_spec(self):
        return EPISTEMIC_SPEC


class EpistemicExpandedTask(HeadCircuitTask):
    task_id = "epistemic_expanded"
    source = "experimental"
    domain = "linguistics_pragmatics"
    experiment_group = "epistemic_study"
    circuit_status = "full_circuit"
    _circuit_module = _ef_expanded_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)

    def get_claim_spec(self):
        return EPISTEMIC_EXPANDED_SPEC


class EpistemicTightTask(HeadCircuitTask):
    task_id = "epistemic_tight"
    source = "experimental"
    domain = "linguistics_pragmatics"
    experiment_group = "epistemic_study"
    circuit_status = "full_circuit"
    _circuit_module = _ef_tight_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)

    def get_claim_spec(self):
        return EPISTEMIC_TIGHT_SPEC


class EpistemicEAPTask(HeadCircuitTask):
    task_id = "epistemic_eap"
    source = "experimental"
    domain = "linguistics_pragmatics"
    experiment_group = "epistemic_study"
    circuit_status = "full_circuit"
    _circuit_module = _ef_eap_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)

    def get_claim_spec(self):
        return EPISTEMIC_EAP_SPEC


# =============================================================================
# Linguistic probes — part 6, initial linguistic phenomenon tests
# Generator-only: prompts exist in factorization-circuits but circuits
# need to be discovered via EAP.
# =============================================================================

class ReflexiveAnaphoraTask(CircuitTask):
    task_id = "reflexive_anaphora"
    source = "ours"
    domain = "linguistics_binding"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "reflexive_anaphora needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class FillerGapTask(CircuitTask):
    task_id = "filler_gap"
    source = "ours"
    domain = "linguistics_syntax"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"
    paper_ref = "https://doi.org/10.1162/tacl_a_00306"

    def get_circuit(self):
        raise NotImplementedError(
            "filler_gap needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class NegationTask(CircuitTask):
    task_id = "negation"
    source = "ours"
    domain = "linguistics_semantics"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "negation needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class ConditionalTask(CircuitTask):
    task_id = "conditional"
    source = "ours"
    domain = "linguistics_semantics"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "conditional needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class EllipsisTask(CircuitTask):
    task_id = "ellipsis"
    source = "ours"
    domain = "linguistics_semantics"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "ellipsis needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class DefinitenessTask(CircuitTask):
    task_id = "definiteness"
    source = "ours"
    domain = "linguistics_semantics"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "definiteness needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


class ButReversalTask(CircuitTask):
    task_id = "but_reversal"
    source = "ours"
    domain = "linguistics_semantics"
    experiment_group = "linguistic_probes"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError(
            "but_reversal needs circuit discovery via EAP. "
            "Generator exists in factorization-circuits part6."
        )


# =============================================================================
# Roadmap — catalog entries with no implementation yet
# =============================================================================

class LessThanTask(CircuitTask):
    task_id = "less_than"
    source = "roadmap"
    domain = "math"
    experiment_group = "roadmap"
    circuit_status = "planned"

    def get_circuit(self):
        raise NotImplementedError(
            "less_than needs circuit definition. "
            "Likely uses greater_than circuit (symmetric variant)."
        )


class SVAPPTask(CircuitTask):
    task_id = "sva_pp"
    source = "roadmap"
    domain = "linguistics_agreement"
    experiment_group = "roadmap"
    circuit_status = "planned"

    def get_circuit(self):
        raise NotImplementedError(
            "sva_pp needs circuit discovery. "
            "SVA variant with prepositional phrase attractors."
        )


class ColoredObjectsTask(CircuitTask):
    task_id = "colored_objects"
    source = "roadmap"
    domain = "linguistics_coreference"
    experiment_group = "roadmap"
    circuit_status = "planned"
    paper_ref = "https://arxiv.org/abs/2310.17191"

    def get_circuit(self):
        raise NotImplementedError(
            "colored_objects needs circuit validation. "
            "78% overlap with IOI circuit (Merullo et al. 2023)."
        )


class DocstringTask(CircuitTask):
    task_id = "docstring"
    source = "published"
    domain = "patterns"
    experiment_group = "roadmap"
    circuit_status = "planned"
    model_family = "attn_only_4L"
    paper_ref = "https://arxiv.org/abs/2304.14997"

    def get_circuit(self):
        raise NotImplementedError(
            "docstring circuit is for attn-only 4L model, not GPT-2. "
            "Adaptation to full GPT-2 non-trivial (MLPs participate)."
        )


class BracketMatchingTask(CircuitTask):
    task_id = "bracket_matching"
    source = "roadmap"
    domain = "linguistics_syntax"
    experiment_group = "roadmap"
    circuit_status = "planned"

    def get_circuit(self):
        raise NotImplementedError(
            "bracket_matching needs circuit discovery. "
            "Tests stack-like computation in attention."
        )


class NPILicensingTask(CircuitTask):
    task_id = "npi_licensing"
    source = "roadmap"
    domain = "linguistics_syntax"
    experiment_group = "roadmap"
    circuit_status = "planned"

    def get_circuit(self):
        raise NotImplementedError(
            "npi_licensing needs circuit discovery. "
            "BLiMP npi_present paradigm."
        )


class SentimentTask(CircuitTask):
    task_id = "sentiment"
    source = "roadmap"
    domain = "linguistics_semantics"
    experiment_group = "roadmap"
    circuit_status = "planned"
    paper_ref = "https://arxiv.org/abs/2310.15154"

    def get_circuit(self):
        raise NotImplementedError(
            "sentiment has partial circuit from Tigges et al. 2023. "
            "Single linear direction causally mediates 76% of accuracy."
        )


class MIBRTITask(CircuitTask):
    task_id = "mib_rti"
    source = "ours"
    domain = "patterns"
    experiment_group = "rti_discovery"
    circuit_status = "proxy_circuit"
    _circuit_module = _rti_circuit

    def get_circuit(self):
        from mechval.lib.tasks.spec import CircuitSpec
        m = self._circuit_module
        return CircuitSpec(
            roles=m.ROLES, bands=m.BANDS, pathways=m.PATHWAYS,
            source=self.source, model_family=self.model_family,
        )

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        raw = make_rti_prompts(tokenizer, n=n_prompts, seed=seed)
        return [
            TaskPrompt(
                text=d["text"],
                target_correct=tokenizer.decode([d["correct_id"]]),
                target_incorrect=tokenizer.decode([d["wrong_id"]]),
                metadata={k: v for k, v in d.items() if k not in ("text", "correct_id", "wrong_id")},
            )
            for d in raw
        ]


# =============================================================================
# BLiMP categories — generator_only, awaiting circuit discovery
# =============================================================================

class BlimpAnaphorAgreementTask(CircuitTask):
    task_id = "blimp_anaphor_agreement"
    source = "blimp"
    domain = "linguistics_binding"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpArgumentStructureTask(CircuitTask):
    task_id = "blimp_argument_structure"
    source = "blimp"
    domain = "linguistics_syntax"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpBindingTask(CircuitTask):
    task_id = "blimp_binding"
    source = "blimp"
    domain = "linguistics_binding"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpControlRaisingTask(CircuitTask):
    task_id = "blimp_control_raising"
    source = "blimp"
    domain = "linguistics_syntax"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpDeterminerNounTask(CircuitTask):
    task_id = "blimp_determiner_noun"
    source = "blimp"
    domain = "linguistics_agreement"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpEllipsisTask(CircuitTask):
    task_id = "blimp_ellipsis"
    source = "blimp"
    domain = "linguistics_semantics"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpFillerGapTask(CircuitTask):
    task_id = "blimp_filler_gap"
    source = "blimp"
    domain = "linguistics_syntax"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpIrregularFormsTask(CircuitTask):
    task_id = "blimp_irregular_forms"
    source = "blimp"
    domain = "linguistics_morphology"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpIslandEffectsTask(CircuitTask):
    task_id = "blimp_island_effects"
    source = "blimp"
    domain = "linguistics_syntax"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpNPILicensingTask(CircuitTask):
    task_id = "blimp_npi_licensing"
    source = "blimp"
    domain = "linguistics_syntax"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpQuantifiersTask(CircuitTask):
    task_id = "blimp_quantifiers"
    source = "blimp"
    domain = "linguistics_semantics"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


class BlimpSubjectVerbTask(CircuitTask):
    task_id = "blimp_subject_verb"
    source = "blimp"
    domain = "linguistics_agreement"
    experiment_group = "blimp"
    circuit_status = "generator_only"
    paper_ref = "https://arxiv.org/abs/1912.00582"

    def get_circuit(self):
        raise NotImplementedError("BLiMP category — circuit discovery needed via EAP.")


# =============================================================================
# Phonetic composition — generator_only, exploring phonological circuits
# =============================================================================

class PhoneticCompositionTask(CircuitTask):
    task_id = "phonetic_composition"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("phonetic_composition — circuit discovery needed via EAP.")


class HypocorismTask(CircuitTask):
    task_id = "hypocorism"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("hypocorism — circuit discovery needed via EAP.")


class PhoneticSequentialTask(CircuitTask):
    task_id = "phonetic_sequential"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("phonetic_sequential — circuit discovery needed via EAP.")


class DoubleShorteningTask(CircuitTask):
    task_id = "double_shortening"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("double_shortening — circuit discovery needed via EAP.")


class HomophoneRecognitionTask(CircuitTask):
    task_id = "homophone_recognition"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("homophone_recognition — circuit discovery needed via EAP.")


class ReverseDecompositionTask(CircuitTask):
    task_id = "reverse_decomposition"
    source = "ours"
    domain = "linguistics_phonology"
    experiment_group = "phonetic_composition"
    circuit_status = "generator_only"

    def get_circuit(self):
        raise NotImplementedError("reverse_decomposition — circuit discovery needed via EAP.")


# -- Registration list --------------------------------------------------------

BUILTIN_TASK_CLASSES: list[type[CircuitTask]] = [
    # Published circuits (7)
    IOITask, GreaterThanTask, InductionTask, SVATask,
    GenderedPronounTask, AcronymTask, CopySuppressionTask,
    # RTI discovery (5)
    RTITask, RTIPatternTask, TokenFloodTask, BuffaloTask, MIBRTITask,
    # IOI ablations (3)
    CenteringTheoryTask, ResumptiveTask, SelfAlloTask,
    # Repetition taxonomy (3)
    SequenceInternalTask, AlternatingPairTask, NovelSongTask,
    # Epistemic study (4)
    EpistemicFramingTask, EpistemicExpandedTask, EpistemicTightTask, EpistemicEAPTask,
    # Linguistic probes — generator_only (7)
    ReflexiveAnaphoraTask, FillerGapTask, NegationTask, ConditionalTask,
    EllipsisTask, DefinitenessTask, ButReversalTask,
    # BLiMP categories — generator_only (12)
    BlimpAnaphorAgreementTask, BlimpArgumentStructureTask, BlimpBindingTask,
    BlimpControlRaisingTask, BlimpDeterminerNounTask, BlimpEllipsisTask,
    BlimpFillerGapTask, BlimpIrregularFormsTask, BlimpIslandEffectsTask,
    BlimpNPILicensingTask, BlimpQuantifiersTask, BlimpSubjectVerbTask,
    # Phonetic composition — generator_only (6)
    PhoneticCompositionTask, HypocorismTask, PhoneticSequentialTask,
    DoubleShorteningTask, HomophoneRecognitionTask, ReverseDecompositionTask,
    # Roadmap — planned (7)
    LessThanTask, SVAPPTask, ColoredObjectsTask, DocstringTask,
    BracketMatchingTask, NPILicensingTask, SentimentTask,
]
