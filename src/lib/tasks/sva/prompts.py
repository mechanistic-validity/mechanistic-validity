"""Subject-verb agreement prompts (Lazo et al. 2025)."""
from __future__ import annotations

import logging
import random

from lib.tasks import TaskPrompt, TokenizerLike

logger = logging.getLogger(__name__)

_SVA_PAIRS = [
    ("The key", "The keys", " is", " are"),
    ("The book", "The books", " is", " are"),
    ("The cat", "The cats", " is", " are"),
    ("The child", "The children", " is", " are"),
    ("My friend", "My friends", " was", " were"),
    ("The author", "The authors", " writes", " write"),
    ("The runner", "The runners", " runs", " run"),
]
_SVA_DISTRACTORS = [
    " on the table",
    " near the window",
    " of the teacher",
    " with the cover",
    " from the shelf",
]


def sva_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
    use_hf: bool = True,
) -> list[TaskPrompt]:
    if use_hf:
        try:
            from datasets import load_dataset
            prompts_out: list[TaskPrompt] = []
            for config in ["regular_plural_subject_verb_agreement_1",
                           "regular_plural_subject_verb_agreement_2",
                           "distractor_agreement_relational_noun",
                           "distractor_agreement_relative_clause"]:
                ds = load_dataset("nyu-mll/blimp", config, split="train")
                for row in ds:
                    good_words = row["sentence_good"].split()
                    bad_words = row["sentence_bad"].split()
                    diverge = 0
                    for i, (g, b) in enumerate(zip(good_words, bad_words)):
                        if g != b:
                            diverge = i
                            break
                    prefix = " ".join(good_words[:diverge])
                    if not prefix:
                        continue
                    prompts_out.append(TaskPrompt(
                        text=prefix,
                        target_correct=" " + good_words[diverge],
                        target_incorrect=" " + bad_words[diverge],
                        metadata={"source": f"blimp/{config}"},
                    ))
            rng = random.Random(seed)
            rng.shuffle(prompts_out)
            if prompts_out:
                return prompts_out[:n_prompts]
        except Exception as e:
            logger.info("BLiMP SVA unavailable (%s); using synthetic.", e)

    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for (sg_subj, pl_subj, sg_verb, pl_verb) in _SVA_PAIRS:
        for distractor in _SVA_DISTRACTORS:
            plural_dist = distractor.replace("table", "tables").replace(
                "window", "windows").replace("teacher", "teachers").replace(
                "cover", "covers").replace("shelf", "shelves")
            prompts.append(TaskPrompt(
                text=f"{sg_subj}{plural_dist}",
                target_correct=sg_verb,
                target_incorrect=pl_verb,
                metadata={"subject_num": "sg", "distractor_num": "pl", "source": "synthetic"},
            ))
            prompts.append(TaskPrompt(
                text=f"{pl_subj}{distractor}",
                target_correct=pl_verb,
                target_incorrect=sg_verb,
                metadata={"subject_num": "pl", "distractor_num": "sg", "source": "synthetic"},
            ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
