"""Copy-suppression prompts (reuses IOI template structure)."""
from __future__ import annotations

import random

from lib.tasks import TaskPrompt, TokenizerLike
from lib.tasks.ioi.prompts import NAME_PAIRS, TEMPLATES


def copy_suppression_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for (A, B) in NAME_PAIRS:
        for (prefix, connector, tail) in TEMPLATES[:5]:
            for (first, second, io) in [(A, B, A), (B, A, B)]:
                text = (
                    f"{prefix.format(A=first, B=second)} "
                    f"{second} {connector} {tail}"
                )
                prompts.append(TaskPrompt(
                    text=text,
                    target_correct=" " + io,
                    target_incorrect=" " + second,
                    metadata={"repeated": second, "expected": io},
                ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
