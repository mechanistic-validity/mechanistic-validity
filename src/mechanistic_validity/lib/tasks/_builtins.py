"""Built-in task registrations for all shipped circuits."""
from __future__ import annotations

from mechanistic_validity.lib.tasks import TaskPrompt, TokenizerLike
from mechanistic_validity.lib.tasks.task import CircuitTask, HeadCircuitTask

from mechanistic_validity.lib.tasks.ioi import circuit as _ioi_circuit
from mechanistic_validity.lib.tasks.greater_than import circuit as _gt_circuit
from mechanistic_validity.lib.tasks.induction import circuit as _ind_circuit
from mechanistic_validity.lib.tasks.sva import circuit as _sva_circuit
from mechanistic_validity.lib.tasks.gendered_pronoun import circuit as _gp_circuit
from mechanistic_validity.lib.tasks.acronym import circuit as _acr_circuit
from mechanistic_validity.lib.tasks.copy_suppression import circuit as _cs_circuit
from mechanistic_validity.lib.tasks.rti import circuit as _rti_circuit
from mechanistic_validity.lib.tasks.epistemic_framing import (
    circuit as _ef_circuit,
    circuit_tight as _ef_tight_circuit,
    circuit_expanded as _ef_expanded_circuit,
    circuit_eap as _ef_eap_circuit,
)

from mechanistic_validity.lib.tasks.ioi.prompts import (
    ioi_prompts, centering_theory_prompts, resumptive_prompts, self_allo_prompts,
)
from mechanistic_validity.lib.tasks.greater_than.prompts import greater_than_prompts
from mechanistic_validity.lib.tasks.induction.prompts import (
    induction_prompts, sequence_internal_prompts, alternating_pair_prompts, novel_song_prompts,
)
from mechanistic_validity.lib.tasks.sva.prompts import sva_prompts
from mechanistic_validity.lib.tasks.gendered_pronoun.prompts import gendered_pronoun_prompts
from mechanistic_validity.lib.tasks.acronym.prompts import acronym_prompts
from mechanistic_validity.lib.tasks.copy_suppression.prompts import copy_suppression_prompts
from mechanistic_validity.lib.tasks.rti.prompts import (
    make_rti_prompts, rti_pattern_prompts, token_flood_prompts, buffalo_prompts,
)
from mechanistic_validity.lib.tasks.epistemic_framing.prompts import epistemic_framing_prompts


# -- Published circuits (7) --------------------------------------------------

class IOITask(HeadCircuitTask):
    task_id = "ioi"
    source = "published"
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


class GreaterThanTask(HeadCircuitTask):
    task_id = "greater_than"
    source = "published"
    paper_ref = "https://arxiv.org/abs/2305.00586"
    _circuit_module = _gt_circuit
    _prompt_fn = staticmethod(greater_than_prompts)

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
    paper_ref = "https://arxiv.org/abs/2209.11895"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(induction_prompts)

    def get_baselines(self):
        return {
            "n_induction_heads": 5,
            "n_previous_token_heads": 2,
            "cite": "Olsson et al. 2022, Table 5 Appendix",
        }


class SVATask(HeadCircuitTask):
    task_id = "sva"
    source = "published"
    paper_ref = "https://arxiv.org/abs/2506.22105"
    _circuit_module = _sva_circuit
    _prompt_fn = staticmethod(sva_prompts)

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
    _circuit_module = _gp_circuit
    _prompt_fn = staticmethod(gendered_pronoun_prompts)

    def get_baselines(self):
        return {
            "recovery_pct": 100.0,
            "cite": "Mathwin 2023 (MATS hackathon, unpublished)",
            "note": "ACDC circuit >= full model performance",
        }


class AcronymTask(HeadCircuitTask):
    task_id = "acronym"
    source = "published"
    paper_ref = "https://proceedings.mlr.press/v238/garcia-carrasco24a.html"
    _circuit_module = _acr_circuit
    _prompt_fn = staticmethod(acronym_prompts)

    def get_baselines(self):
        return {
            "n_heads": 8,
            "cite": "Garcia-Carrasco et al. 2024 (AISTATS)",
            "note": "circuit-only slightly exceeds full model",
        }


class CopySuppressionTask(HeadCircuitTask):
    task_id = "copy_suppression"
    source = "published"
    paper_ref = "https://arxiv.org/abs/2310.04625"
    _circuit_module = _cs_circuit
    _prompt_fn = staticmethod(copy_suppression_prompts)


# -- RTI + aliases (ours) ----------------------------------------------------

class RTITask(CircuitTask):
    task_id = "rti"
    source = "ours"
    _circuit_module = _rti_circuit

    def get_baselines(self):
        return {
            "n_heads": 15,
            "eap_auroc": 0.39,
        }

    def get_circuit(self):
        from mechanistic_validity.lib.tasks.spec import CircuitSpec
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
    _circuit_module = _rti_circuit
    _prompt_fn = staticmethod(rti_pattern_prompts)


class TokenFloodTask(HeadCircuitTask):
    task_id = "token_flood"
    source = "ours"
    _circuit_module = _rti_circuit
    _prompt_fn = staticmethod(token_flood_prompts)


class BuffaloTask(CircuitTask):
    task_id = "buffalo"
    source = "ours"
    _circuit_module = _rti_circuit

    def get_circuit(self):
        from mechanistic_validity.lib.tasks.spec import CircuitSpec
        m = self._circuit_module
        return CircuitSpec(
            roles=m.ROLES, bands=m.BANDS, pathways=m.PATHWAYS,
            source=self.source, model_family=self.model_family,
        )

    def get_prompts(self, tokenizer: TokenizerLike, n_prompts: int = 40, seed: int = 42) -> list[TaskPrompt]:
        return buffalo_prompts(tokenizer, seed=seed)[:n_prompts]


# -- IOI alias tasks ----------------------------------------------------------

class CenteringTheoryTask(HeadCircuitTask):
    task_id = "centering_theory"
    source = "ours"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(centering_theory_prompts)


class ResumptiveTask(HeadCircuitTask):
    task_id = "resumptive"
    source = "ours"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(resumptive_prompts)


class SelfAlloTask(HeadCircuitTask):
    task_id = "self_allo"
    source = "ours"
    _circuit_module = _ioi_circuit
    _prompt_fn = staticmethod(self_allo_prompts)


# -- Induction alias tasks ----------------------------------------------------

class SequenceInternalTask(HeadCircuitTask):
    task_id = "sequence_internal"
    source = "ours"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(sequence_internal_prompts)


class AlternatingPairTask(HeadCircuitTask):
    task_id = "alternating_pair"
    source = "ours"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(alternating_pair_prompts)


class NovelSongTask(HeadCircuitTask):
    task_id = "novel_song"
    source = "ours"
    _circuit_module = _ind_circuit
    _prompt_fn = staticmethod(novel_song_prompts)


# -- Epistemic framing (experimental) ----------------------------------------

class EpistemicFramingTask(HeadCircuitTask):
    task_id = "epistemic_framing"
    source = "experimental"
    _circuit_module = _ef_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)

    def get_baselines(self):
        return {
            "n_heads": 4,
        }


class EpistemicExpandedTask(HeadCircuitTask):
    task_id = "epistemic_expanded"
    source = "experimental"
    _circuit_module = _ef_expanded_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)


class EpistemicTightTask(HeadCircuitTask):
    task_id = "epistemic_tight"
    source = "experimental"
    _circuit_module = _ef_tight_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)


class EpistemicEAPTask(HeadCircuitTask):
    task_id = "epistemic_eap"
    source = "experimental"
    _circuit_module = _ef_eap_circuit
    _prompt_fn = staticmethod(epistemic_framing_prompts)


# -- Registration list --------------------------------------------------------

BUILTIN_TASK_CLASSES: list[type[CircuitTask]] = [
    IOITask, GreaterThanTask, InductionTask, SVATask,
    GenderedPronounTask, AcronymTask, CopySuppressionTask,
    RTITask, RTIPatternTask, TokenFloodTask, BuffaloTask,
    CenteringTheoryTask, ResumptiveTask, SelfAlloTask,
    SequenceInternalTask, AlternatingPairTask, NovelSongTask,
    EpistemicFramingTask, EpistemicExpandedTask, EpistemicTightTask, EpistemicEAPTask,
]
