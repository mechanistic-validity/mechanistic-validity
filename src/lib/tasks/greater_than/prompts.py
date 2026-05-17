"""Greater-than year prompts (Hanna et al. 2023)."""
from __future__ import annotations

import logging
import random

from lib.tasks import TaskPrompt, TokenizerLike

logger = logging.getLogger(__name__)


def greater_than_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
    use_hf: bool = True,
) -> list[TaskPrompt]:
    if use_hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("mwhanna/greater-than", split="train")
            rng = random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            prompts_out: list[TaskPrompt] = []
            for i in indices[:n_prompts]:
                row = ds[i]
                year = int(row["year"])
                year_tens = (year % 100) // 10
                correct_digit = str(min(9, year_tens + 1))
                incorrect_digit = str(max(0, year_tens - 2))
                text = row["clean"].rsplit(" ", 1)[0] + " "
                prompts_out.append(TaskPrompt(
                    text=text,
                    target_correct=correct_digit,
                    target_incorrect=incorrect_digit,
                    metadata={"year": year, "source": "hf/mwhanna"},
                ))
            if prompts_out:
                return prompts_out
        except Exception as e:
            logger.info("HF greater-than unavailable (%s); using synthetic.", e)

    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for start_year in range(1702, 1798, 3):
        for delta in (5, 10, 20, 40):
            end_year = start_year + delta
            if end_year > 1799:
                continue
            start_tens = (start_year % 100) // 10
            end_tens = (end_year % 100) // 10
            if end_tens <= start_tens:
                continue
            correct_digit = str(end_tens)
            incorrect_digit = str(max(0, start_tens - 1))
            if correct_digit == incorrect_digit:
                continue
            text = f"The war lasted from the year {start_year} to the year 17"
            prompts.append(TaskPrompt(
                text=text,
                target_correct=correct_digit,
                target_incorrect=incorrect_digit,
                metadata={"start_year": start_year, "end_year": end_year, "source": "synthetic"},
            ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
