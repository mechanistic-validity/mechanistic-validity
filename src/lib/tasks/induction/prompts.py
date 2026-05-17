"""Induction prompts (Olsson et al. 2022) + induction-circuit alias tasks.

Alias tasks that reuse the induction circuit:
  sequence_internal — slot-structured pattern completion
  alternating_pair — modular position tracking in 2-cycles
  novel_song — memorization vs structure control
"""
from __future__ import annotations

import random

from lib.tasks import TaskPrompt, TokenizerLike


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


# ---------------------------------------------------------------------------
# Sequence-internal pattern completion (induction-circuit alias)
# ---------------------------------------------------------------------------

_SEQ_NOUNS = [
    "fish", "cat", "dog", "bird", "frog", "bear", "duck",
    "hat", "ball", "star", "bell", "box", "cup", "drum",
]
_SEQ_SLOT_SETS: dict[str, list[str]] = {
    "colors": ["red", "blue", "green", "gold", "pink", "gray", "white", "black"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    "sizes": ["big", "small", "tall", "short", "fat", "thin", "wide", "long"],
    "feelings": ["happy", "sad", "mad", "glad", "shy", "bold", "calm", "wild"],
}


def sequence_internal_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    seen: set[tuple[str, ...]] = set()
    slot_names = list(_SEQ_SLOT_SETS.keys())

    while len(prompts) < n_prompts:
        noun = rng.choice(_SEQ_NOUNS)
        slot_set_name = rng.choice(slot_names)
        pool = _SEQ_SLOT_SETS[slot_set_name]
        n_slots = rng.choice([3, 4, 4, 4])
        if len(pool) < n_slots:
            continue
        slots = rng.sample(pool, n_slots)
        key = (noun, *slots)
        if key in seen:
            continue
        seen.add(key)

        full = " ".join(f"{s} {noun}" for s in slots)
        prefix = " ".join(f"{s} {noun}" for s in slots[:-1])
        correct = " " + slots[-1]
        wrong_pool = [s for s in pool if s != slots[-1]]
        incorrect = " " + rng.choice(wrong_pool)

        prompts.append(TaskPrompt(
            text=f"{full}. {prefix}",
            target_correct=correct,
            target_incorrect=incorrect,
            metadata={"category": "sequence_internal", "noun": noun,
                      "slots": slots, "slot_set": slot_set_name},
        ))

    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Alternating pair (induction-circuit alias)
# ---------------------------------------------------------------------------

_ALT_PAIRS = [
    ("tick", "tock"), ("red", "blue"), ("yes", "no"), ("up", "down"),
    ("left", "right"), ("hot", "cold"), ("big", "small"), ("day", "night"),
    ("in", "out"), ("on", "off"), ("high", "low"), ("fast", "slow"),
    ("black", "white"), ("good", "bad"), ("old", "new"), ("east", "west"),
    ("push", "pull"), ("stop", "go"), ("give", "take"), ("buy", "sell"),
    ("win", "lose"), ("sit", "stand"), ("rise", "fall"), ("open", "close"),
    ("first", "last"), ("top", "bottom"), ("front", "back"), ("north", "south"),
    ("true", "false"), ("odd", "even"), ("sweet", "sour"), ("thick", "thin"),
    ("loud", "quiet"), ("rich", "poor"), ("dark", "light"), ("hard", "soft"),
    ("wet", "dry"), ("rough", "smooth"), ("plus", "minus"), ("near", "far"),
]


def alternating_pair_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 100,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    seen: set[tuple[str, str, int, bool]] = set()

    while len(prompts) < n_prompts:
        a, b = rng.choice(_ALT_PAIRS)
        n_reps = rng.randint(4, 8)
        end_on_a = rng.choice([True, False])
        key = (a, b, n_reps, end_on_a)
        if key in seen:
            continue
        seen.add(key)

        seq = " ".join(f"{a} {b}" for _ in range(n_reps))
        if end_on_a:
            seq += f" {a}"
            correct, incorrect = " " + b, " " + a
        else:
            correct, incorrect = " " + a, " " + b

        prompts.append(TaskPrompt(
            text=seq,
            target_correct=correct,
            target_incorrect=incorrect,
            metadata={"category": "alternating_pair", "pair": [a, b],
                      "n_reps": n_reps, "end_on_a": end_on_a},
        ))

    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Novel songs (induction-circuit alias)
# ---------------------------------------------------------------------------

_KNOWN_SONGS = [
    ("Baa, baa, black sheep, have you any wool? Yes sir, yes sir, three bags full. Baa, baa, black sheep,",
     " have", " are"),
    ("Twinkle, twinkle, little star, how I wonder what you are. Up above the world so high, like a diamond in the sky. Twinkle, twinkle, little star,",
     " how", " up"),
    ("London Bridge is falling down, falling down, falling down. London Bridge is falling down, my fair lady. London Bridge is falling down,",
     " falling", " my"),
    ("Row, row, row your boat, gently down the stream. Merrily, merrily, merrily, merrily, life is but a dream. Row, row, row your boat,",
     " gently", " mer"),
    ("Jingle bells, jingle bells, jingle all the way. Oh what fun it is to ride in a one-horse open sleigh. Jingle bells, jingle bells,",
     " jingle", " oh"),
]

_NOVEL_SONGS = [
    ("Shimmer, shimmer, tiny flame, burning bright without a name. Dancing high above the ground, spinning light without a sound. Shimmer, shimmer, tiny flame,",
     " burning", " dancing"),
    ("Wobble, wobble, round and round, bouncing gently off the ground. Up and down and side to side, what a funny bumpy ride. Wobble, wobble, round and round,",
     " bouncing", " up"),
    ("The old stone tower is crumbling down, crumbling down, crumbling down. The old stone tower is crumbling down, beside the quiet river. The old stone tower is crumbling down,",
     " crumbling", " beside"),
    ("Bubble, bubble, copper pot, steaming gently, nice and hot. Pour the water, fill the cup, drink it slowly, bottom up. Bubble, bubble, copper pot,",
     " steaming", " pour"),
    ("March, march, march along, marching to a steady song. Left and right and left and right, marching from the morning light. March, march, march along,",
     " marching", " left"),
    ("Clap, clap, golden crown, will you share your jewels around? Yes please, yes please, seven gems bright. Clap, clap, golden crown,",
     " will", " yes"),
    ("Splash, splash, splash the oar, drifting past the sandy shore. Peacefully, peacefully, peacefully, peacefully, floating to the ocean floor. Splash, splash, splash the oar,",
     " drifting", " peace"),
    ("Lantern glow, lantern glow, lantern shining down below. See the light that guides the way through the darkness, night to day. Lantern glow, lantern glow,",
     " lantern", " see"),
    ("Dig, dig, little mole, tunneling beneath the knoll. Under roots and under stones, through the dark and dirt alone. Dig, dig, little mole,",
     " tunneling", " under"),
    ("The old brass clock is winding down, winding down, winding down. The old brass clock is winding down, upon the dusty mantle. The old brass clock is winding down,",
     " winding", " upon"),
]


def novel_song_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 15,
    seed: int = 0,
) -> list[TaskPrompt]:
    prompts: list[TaskPrompt] = []
    for text, correct, incorrect in _KNOWN_SONGS:
        prompts.append(TaskPrompt(text=text, target_correct=correct,
                                  target_incorrect=incorrect,
                                  metadata={"is_novel": False}))
    for text, correct, incorrect in _NOVEL_SONGS:
        prompts.append(TaskPrompt(text=text, target_correct=correct,
                                  target_incorrect=incorrect,
                                  metadata={"is_novel": True}))
    return prompts[:n_prompts]
