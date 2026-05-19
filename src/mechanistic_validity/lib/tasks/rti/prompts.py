"""RTI (Repeated Token Identification) prompts + RTI-circuit alias tasks.

Core RTI prompts extracted from run_probes_rti_v2.py.

Alias tasks that reuse the RTI circuit:
  rti_pattern — sentence-pattern continuation
  token_flood — single-token flood
  buffalo — semantic trap sentences
"""
from __future__ import annotations

import random
from typing import Any

from mechanistic_validity.lib.tasks import TaskPrompt, TokenizerLike

NAMES_A = [
    "Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate",
    "Mary", "Nick", "Paul", "Sarah", "Tom", "Anna", "Luke", "Jane",
]
NAMES_B = [
    "Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia",
    "Kevin", "Laura", "Mike", "Nancy", "Oscar", "Peter", "Quinn", "Ruth",
]
TEMPLATES = [
    "Then {D} and {C} went to the store. {D} gave a drink to",
    "{D} told {C} a story. {D} then handed the book to",
    "{D} met {C} at the park. {D} passed the ball to",
    "When {D} and {C} arrived, {D} gave the keys to",
    "{D} helped {C} move. {D} gave the boxes to",
    "{D} invited {C} to dinner. {D} served food to",
]


def make_rti_prompts(tokenizer: Any, n: int = 400, seed: int = 42) -> list[dict]:
    """Raw RTI prompt dicts (wrapped into TaskPrompt by RTITask)."""
    rng = random.Random(seed)
    n_pairs = min(len(NAMES_A), len(NAMES_B))
    prompts: list[dict] = []
    count = 0
    while count < n:
        i = rng.randint(0, n_pairs - 1)
        tmpl = rng.choice(TEMPLATES)
        d_name, c_name = NAMES_A[i], NAMES_B[i]
        text = tmpl.format(D=d_name, C=c_name)

        correct_tok = tokenizer.encode(" " + c_name)
        wrong_tok = tokenizer.encode(" " + d_name)
        if len(correct_tok) != 1 or len(wrong_tok) != 1:
            continue

        tokens = tokenizer.encode(text)
        correct_id = correct_tok[0]
        wrong_id = wrong_tok[0]

        d_token = tokenizer.encode(" " + d_name)
        c_token = tokenizer.encode(" " + c_name)
        if len(d_token) != 1 or len(c_token) != 1:
            continue

        d_tok_id = d_token[0]
        c_tok_id = c_token[0]

        d_positions = [j for j, t in enumerate(tokens) if t == d_tok_id]
        c_positions = [j for j, t in enumerate(tokens) if t == c_tok_id]

        if len(d_positions) < 2 or len(c_positions) < 1:
            continue

        prompts.append({
            "text": text,
            "tokens": tokens,
            "correct_id": correct_id,
            "wrong_id": wrong_id,
            "d_name": d_name,
            "c_name": c_name,
            "d_positions": d_positions,
            "c_positions": c_positions,
            "d_first_pos": d_positions[0],
            "d_second_pos": d_positions[1] if len(d_positions) > 1 else d_positions[0],
            "pair_idx": i,
        })
        count += 1

    return prompts


# ---------------------------------------------------------------------------
# RTI pattern: sentence-pattern continuation (RTI-circuit alias)
# ---------------------------------------------------------------------------

_RTI_SUBJECTS = [
    "The cat", "The dog", "A bird", "The fish", "A mouse",
    "The boy", "The girl", "A man", "The child", "A woman",
    "The teacher", "The doctor", "A farmer", "The student", "A cook",
    "The king", "The queen", "A knight", "The prince", "A wizard",
]
_RTI_VERBS = [
    "sat on", "ran to", "jumped over", "walked past", "climbed up",
    "fell from", "looked at", "hid behind", "swam in", "flew over",
    "stood near", "danced around", "slept under", "played with", "ate from",
    "crawled through", "rolled down", "leaned against", "stared at", "sang to",
]
_RTI_OBJECTS = [
    "the mat", "the hill", "the fence", "the tree", "the wall",
    "the bridge", "the lake", "the roof", "the bench", "the rock",
    "the garden", "the river", "the tower", "the gate", "the table",
    "the chair", "the box", "the path", "the door", "the window",
]
_RTI_WRONG_VERBS = [
    "was", "is", "went", "had", "did",
    "could", "would", "saw", "got", "came",
]


def rti_pattern_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 200,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    seen: set[tuple[str, str, str]] = set()

    while len(prompts) < n_prompts:
        subj = rng.choice(_RTI_SUBJECTS)
        verb = rng.choice(_RTI_VERBS)
        obj = rng.choice(_RTI_OBJECTS)
        key = (subj, verb, obj)
        if key in seen:
            continue
        seen.add(key)

        sentence = f"{subj} {verb} {obj}"
        correct = " " + verb.split()[0]
        incorrect = " " + rng.choice(_RTI_WRONG_VERBS)
        while incorrect == correct:
            incorrect = " " + rng.choice(_RTI_WRONG_VERBS)

        prompts.append(TaskPrompt(
            text=f"{sentence}. {subj}",
            target_correct=correct,
            target_incorrect=incorrect,
            metadata={"category": "sentence_pattern", "sentence": sentence},
        ))

    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Single-token flood (RTI-circuit alias)
# ---------------------------------------------------------------------------

_FLOOD_TOKENS = [
    "hello", "the", "cat", "apple", "yes", "no", "very",
    "good", "bad", "run", "stop", "go", "red", "blue",
    "one", "two", "and", "but", "not", "is",
]


def token_flood_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 40,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    for token in _FLOOD_TOKENS[:n_prompts]:
        n_reps = rng.randint(8, 15)
        text = " ".join([token] * n_reps)
        wrong_pool = [t for t in _FLOOD_TOKENS if t != token]
        prompts.append(TaskPrompt(
            text=text,
            target_correct=" " + token,
            target_incorrect=" " + rng.choice(wrong_pool),
            metadata={"token": token, "n_reps": n_reps},
        ))
    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Buffalo / semantic trap sentences (RTI-circuit alias)
# ---------------------------------------------------------------------------

def buffalo_prompts(
    tokenizer: TokenizerLike,
    seed: int = 42,
    **_: Any,
) -> list[TaskPrompt]:
    prompts: list[TaskPrompt] = []

    for n in range(3, 9):
        words = []
        for i in range(n):
            words.append("Buffalo" if i % 3 == 0 else "buffalo")
        sentence = " ".join(words)
        prefix_len = max(2, n // 2)
        prefix_words = []
        for i in range(prefix_len):
            prefix_words.append("Buffalo" if i % 3 == 0 else "buffalo")
        prefix = " ".join(prefix_words)
        next_word = "Buffalo" if prefix_len % 3 == 0 else "buffalo"
        wrong = "buffalo" if next_word == "Buffalo" else "Buffalo"
        text = f'{sentence}. {prefix}'
        prompts.append(TaskPrompt(
            text=text,
            target_correct=" " + next_word,
            target_incorrect=" " + wrong,
            metadata={"family": "buffalo", "n_reps": n},
        ))

    for n in range(4, 12):
        hads = " ".join(["had"] * n)
        text = f"James while John {hads} had a better effect on the teacher. James while John {hads}"
        prompts.append(TaskPrompt(
            text=text,
            target_correct=" had",
            target_incorrect=" was",
            metadata={"family": "had_had", "n_reps": n},
        ))

    for n in range(3, 9):
        thats = " ".join(["that"] * n)
        text = f"He said {thats} student used was incorrect. He said {thats}"
        prompts.append(TaskPrompt(
            text=text,
            target_correct=" student",
            target_incorrect=" that",
            metadata={"family": "that_that", "n_reps": n},
        ))

    for n in range(4, 12):
        text = " ".join(["police"] * n) + ". The " + " ".join(["police"] * max(2, n // 2))
        prompts.append(TaskPrompt(
            text=text,
            target_correct=" police",
            target_incorrect=" officer",
            metadata={"family": "police", "n_reps": n},
        ))

    return prompts
