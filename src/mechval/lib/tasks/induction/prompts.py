"""Induction prompts (Olsson et al. 2022)."""
from __future__ import annotations

import random

from mechval.lib.tasks import TaskPrompt, TokenizerLike


def induction_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
    seq_len: int = 32,
    vocab_size_hint: int = 50000,
) -> list[TaskPrompt]:
    """Random-token bigram induction: [A][B] ... [A][?] -> [?]=B."""
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    vocab_size = getattr(tokenizer, "vocab_size", vocab_size_hint)
    for _ in range(n_prompts):
        ids = [rng.randrange(100, vocab_size - 100) for _ in range(seq_len - 2)]
        a, b = ids[0], ids[1]
        ids[-1] = a
        try:
            text = tokenizer.decode(ids) if hasattr(tokenizer, "decode") else ""
        except Exception:
            text = ""
        incorrect = (b + rng.randrange(1, 1000)) % vocab_size
        prompts.append(TaskPrompt(
            text=text,
            target_correct=str(b),
            target_incorrect=str(incorrect),
            metadata={"token_ids": ids, "trigger": a, "expected_id": b,
                      "incorrect_id": incorrect, "scoring": "by_id"},
        ))
    return prompts
