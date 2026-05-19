"""Task prompt registry — imports from per-task prompt modules.

Each task's prompts live in lib/tasks/<task>/prompts.py.
This module re-exports them as TASK_REGISTRY for common.py.
"""
from __future__ import annotations

from typing import Any

from mechval.lib.tasks import TaskBuilder, TaskPrompt, TokenizerLike

from mechval.lib.tasks.ioi.prompts import (
    centering_theory_prompts,
    ioi_prompts,
    resumptive_prompts,
    self_allo_prompts,
)
from mechval.lib.tasks.greater_than.prompts import greater_than_prompts
from mechval.lib.tasks.copy_suppression.prompts import copy_suppression_prompts
from mechval.lib.tasks.induction.prompts import (
    alternating_pair_prompts,
    induction_prompts,
    novel_song_prompts,
    sequence_internal_prompts,
)
from mechval.lib.tasks.gendered_pronoun.prompts import gendered_pronoun_prompts
from mechval.lib.tasks.sva.prompts import sva_prompts
from mechval.lib.tasks.acronym.prompts import acronym_prompts
from mechval.lib.tasks.rti.prompts import (
    buffalo_prompts,
    rti_pattern_prompts,
    token_flood_prompts,
)
from mechval.lib.tasks.epistemic_framing.prompts import epistemic_framing_prompts

TASK_REGISTRY: dict[str, TaskBuilder] = {
    "ioi": ioi_prompts,
    "greater_than": greater_than_prompts,
    "copy_suppression": copy_suppression_prompts,
    "induction": induction_prompts,
    "gendered_pronoun": gendered_pronoun_prompts,
    "sva": sva_prompts,
    "acronym": acronym_prompts,
    "rti_pattern": rti_pattern_prompts,
    "sequence_internal": sequence_internal_prompts,
    "alternating_pair": alternating_pair_prompts,
    "novel_song": novel_song_prompts,
    "centering_theory": centering_theory_prompts,
    "resumptive": resumptive_prompts,
    "self_allo": self_allo_prompts,
    "token_flood": token_flood_prompts,
    "buffalo": buffalo_prompts,
    "epistemic_framing": epistemic_framing_prompts,
    "epistemic_expanded": epistemic_framing_prompts,
    "epistemic_tight": epistemic_framing_prompts,
    "epistemic_eap": epistemic_framing_prompts,
}


def build_task(
    task_name: str,
    tokenizer: TokenizerLike,
    **kwargs: Any,
) -> list[TaskPrompt]:
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {task_name!r}. Known: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_name](tokenizer, **kwargs)


def task_best_layers(task_name: str, n_layers: int) -> list[int]:
    fractions = {
        "ioi": (0.75, 0.92),
        "sva": (0.58, 0.75),
        "greater_than": (0.67, 0.92),
        "copy_suppression": (0.83, 0.92),
        "induction": (0.42, 0.58),
        "gendered_pronoun": (0.00, 0.67),
        "acronym": (0.08, 0.92),
    }.get(task_name, (0.5, 0.9))
    lo = max(0, int(round(fractions[0] * (n_layers - 1))))
    hi = min(n_layers - 1, int(round(fractions[1] * (n_layers - 1))))
    return list(range(lo, hi + 1))


def resolve_target_token_ids(
    tokenizer: TokenizerLike,
    target: str,
    scoring: str = "string",
) -> int:
    if scoring == "by_id":
        return int(target)
    ids = tokenizer.encode(target, add_special_tokens=False) if hasattr(
        tokenizer, "encode"
    ) else tokenizer(target)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenizer returned no ids for {target!r}")
    return int(ids[0])
