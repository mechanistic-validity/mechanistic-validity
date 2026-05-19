"""Gendered pronoun agreement prompts (Mathwin 2023)."""
from __future__ import annotations

import logging
import random

from mechval.lib.tasks import TaskPrompt, TokenizerLike

logger = logging.getLogger(__name__)

_GENDERED_NAMES = [
    ("Mary", "she", "he"), ("Alice", "she", "he"), ("Sarah", "she", "he"),
    ("Emma", "she", "he"), ("Lisa", "she", "he"),
    ("John", "he", "she"), ("Bob", "he", "she"), ("Tom", "he", "she"),
    ("David", "he", "she"), ("Mark", "he", "she"),
]


def gendered_pronoun_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
    use_hf: bool = True,
) -> list[TaskPrompt]:
    if use_hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("nyu-mll/blimp", "anaphor_gender_agreement", split="train")
            prompts_out: list[TaskPrompt] = []
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
                    metadata={"source": "blimp/anaphor_gender_agreement"},
                ))
            rng = random.Random(seed)
            rng.shuffle(prompts_out)
            if prompts_out:
                return prompts_out[:n_prompts]
        except Exception as e:
            logger.info("BLiMP gender unavailable (%s); using synthetic.", e)

    templates = [
        "So {name} is a nurse. Yesterday",
        "{name} went to the store because",
        "Everyone agreed that {name} was tired, so",
        "After a long day, {name} decided that",
        "The teacher praised {name} because",
    ]
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for (name, correct, incorrect) in _GENDERED_NAMES:
        for t in templates:
            prompts.append(TaskPrompt(
                text=t.format(name=name),
                target_correct=" " + correct,
                target_incorrect=" " + incorrect,
                metadata={"name": name, "source": "synthetic"},
            ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
