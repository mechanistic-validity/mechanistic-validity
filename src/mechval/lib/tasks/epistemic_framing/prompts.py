"""Epistemic framing prompts.

Each prompt has a bare factual statement and an epistemic-framed version.
target_correct = next token of bare version (the factual completion)
target_incorrect = a wrong completion

The circuit should activate MORE on epistemic prompts vs bare prompts.
"""
from __future__ import annotations

import random

from mechval.lib.tasks import TaskPrompt, TokenizerLike

EPISTEMIC_PAIRS = [
    ("The capital of France is", "I think the capital of France is", " Paris", " London"),
    ("Dogs are commonly known as", "I believe dogs are commonly known as", " man", " fish"),
    ("Water boils at one hundred", "I know water boils at one hundred", " degrees", " miles"),
    ("The sun is a", "I think the sun is a", " star", " planet"),
    ("Shakespeare wrote many famous", "I believe Shakespeare wrote many famous", " plays", " songs"),
    ("Mathematics is the study of", "I think mathematics is the study of", " numbers", " animals"),
    ("The largest ocean is the", "I know the largest ocean is the", " Pacific", " Atlantic"),
    ("Gravity pulls objects", "I think gravity pulls objects", " down", " up"),
    ("The human heart has four", "I believe the human heart has four", " chambers", " legs"),
    ("Oxygen is essential for", "I know oxygen is essential for", " life", " death"),
    ("The Earth revolves around the", "I think the Earth revolves around the", " sun", " moon"),
    ("Plants convert sunlight into", "I think plants convert sunlight into", " energy", " water"),
]


def epistemic_framing_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 40,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts = []
    pairs = EPISTEMIC_PAIRS * ((n_prompts // len(EPISTEMIC_PAIRS)) + 1)
    rng.shuffle(pairs)
    for bare, epistemic, correct, incorrect in pairs[:n_prompts]:
        prompts.append(TaskPrompt(
            text=epistemic,
            target_correct=correct,
            target_incorrect=incorrect,
            metadata={"bare": bare, "epistemic": epistemic},
        ))
    return prompts
