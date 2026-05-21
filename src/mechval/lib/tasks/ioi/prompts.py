"""IOI (indirect object identification) prompts."""
from __future__ import annotations

import logging
import random

from mechval.lib.tasks import TaskPrompt, TokenizerLike

logger = logging.getLogger(__name__)

NAME_PAIRS: list[tuple[str, str]] = [
    ("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"),
    ("David", "Emma"), ("Lisa", "Mark"), ("James", "Kate"),
    ("Alex", "Chris"), ("Mike", "Sophie"), ("Rachel", "Dan"),
    ("Nick", "Laura"),
]
TEMPLATES: list[tuple[str, str, str]] = [
    ("When {A} and {B} went to the store,", "gave a drink", "to"),
    ("After {A} met {B} at the park,", "handed the ball", "to"),
    ("When {A} and {B} went to the office,", "gave a file", "to"),
    ("After {A} met {B} at the cafe,", "passed the cup", "to"),
    ("When {A} and {B} went to the restaurant,", "gave a menu", "to"),
    ("After {A} met {B} at the library,", "lent a book", "to"),
    ("When {A} and {B} went to the gym,", "threw a towel", "to"),
    ("After {A} met {B} at the party,", "handed the gift", "to"),
    ("When {A} and {B} went to the meeting,", "gave the report", "to"),
    ("After {A} met {B} at the hotel,", "sent the key", "to"),
]


def ioi_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 200,
    seed: int = 0,
    use_mib: bool = True,
) -> list[TaskPrompt]:
    if use_mib:
        try:
            from datasets import load_dataset
            ds = load_dataset("mib-bench/ioi", split="train")
            prompts_out: list[TaskPrompt] = []
            for row in ds:
                choices = row["choices"]
                answer_idx = row["answerKey"]
                incorrect_idx = 1 - answer_idx if len(choices) == 2 else 0
                meta = row.get("metadata", {})
                prompts_out.append(TaskPrompt(
                    text=row["prompt"],
                    target_correct=" " + choices[answer_idx],
                    target_incorrect=" " + choices[incorrect_idx],
                    metadata={
                        "source": "mib",
                        "io": meta.get("indirect_object", choices[answer_idx]),
                        "s": meta.get("subject", choices[incorrect_idx]),
                    },
                ))
                if len(prompts_out) >= n_prompts:
                    break
            if prompts_out:
                return prompts_out
        except Exception as e:
            logger.info("MIB IOI dataset unavailable (%s); using synthetic generator.", e)

    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for (A, B) in NAME_PAIRS:
        for (prefix, connector, tail) in TEMPLATES:
            for (first, second, io) in [(A, B, A), (B, A, B)]:
                text = (
                    f"{prefix.format(A=first, B=second)} "
                    f"{second} {connector} {tail}"
                )
                s_name = second
                prompts.append(TaskPrompt(
                    text=text,
                    target_correct=" " + io,
                    target_incorrect=" " + s_name,
                    metadata={"io": io, "s": s_name, "source": "synthetic"},
                ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
