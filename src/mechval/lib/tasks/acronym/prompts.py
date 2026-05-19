"""Acronym prediction prompts (Garcia-Carrasco et al. AISTATS 2024)."""
from __future__ import annotations

import random

from mechval.lib.tasks import TaskPrompt, TokenizerLike

_ACRONYM_EXAMPLES: list[tuple[str, str, str]] = [
    ("the United Nations", "U", "N"),
    ("the World Health Organization", "W", "H"),
    ("the North Atlantic Treaty Organization", "N", "A"),
    ("the Federal Bureau of Investigation", "F", "B"),
    ("the Central Intelligence Agency", "C", "I"),
    ("the National Aeronautics and Space Administration", "N", "A"),
    ("the International Monetary Fund", "I", "M"),
    ("the United States of America", "U", "S"),
    ("the European Union", "E", "U"),
    ("the United Kingdom", "U", "K"),
    ("the International Space Station", "I", "S"),
    ("the World Trade Organization", "W", "T"),
    ("the Department of Defense", "D", "O"),
    ("the National Football League", "N", "F"),
    ("the National Basketball Association", "N", "B"),
    ("the Department of Energy", "D", "O"),
    ("the Securities and Exchange Commission", "S", "E"),
    ("the American Medical Association", "A", "M"),
    ("the Internal Revenue Service", "I", "R"),
    ("the Environmental Protection Agency", "E", "P"),
]


def acronym_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for org_name, first_letter, second_letter in _ACRONYM_EXAMPLES:
        text = f"The acronym for {org_name} is {first_letter}"
        wrong_letters = [chr(c) for c in range(65, 91) if chr(c) != second_letter]
        incorrect = rng.choice(wrong_letters)
        prompts.append(TaskPrompt(
            text=text,
            target_correct=second_letter,
            target_incorrect=incorrect,
            metadata={"org": org_name, "first": first_letter, "expected": second_letter,
                      "source": "synthetic"},
        ))
        text_alt = f"{org_name} ({first_letter}"
        prompts.append(TaskPrompt(
            text=text_alt,
            target_correct=second_letter,
            target_incorrect=incorrect,
            metadata={"org": org_name, "first": first_letter, "expected": second_letter,
                      "template": "parens", "source": "synthetic"},
        ))
    rng.shuffle(prompts)
    return prompts[:n_prompts]
