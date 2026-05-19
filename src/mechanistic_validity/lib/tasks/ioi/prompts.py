"""IOI (indirect object identification) prompts + IOI-circuit alias tasks.

Alias tasks that reuse the IOI circuit:
  centering_theory — manipulate entity salience
  resumptive — topic resumption after digression
  self_allo — self vs allo-repetition
"""
from __future__ import annotations

import logging
import random

from mechanistic_validity.lib.tasks import TaskPrompt, TokenizerLike

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


# ---------------------------------------------------------------------------
# Centering theory manipulation (IOI-circuit alias)
# ---------------------------------------------------------------------------

_CENTER_NAMES_A = [
    "John", "David", "James", "Michael", "Robert", "Thomas", "Daniel", "William",
    "Henry", "Edward", "George", "Charles", "Frank", "Peter", "Andrew", "Paul",
]
_CENTER_NAMES_B = [
    "Mary", "Sarah", "Emma", "Alice", "Grace", "Helen", "Kate", "Laura",
    "Anna", "Jane", "Rose", "Lily", "Claire", "Diana", "Fiona", "Ruth",
]
_CENTER_ACTIONS = [
    ("gave a book", "to"), ("passed the ball", "to"), ("handed a letter", "to"),
    ("sent a gift", "to"), ("threw the keys", "to"), ("offered a seat", "to"),
    ("gave the report", "to"), ("lent a pen", "to"), ("showed the map", "to"),
    ("brought flowers", "to"),
]
_CENTER_CTX_STANDARD = [
    "After {A} and {B} went to the store,",
    "When {A} met {B} at the park,",
    "After {A} and {B} arrived at the office,",
    "When {A} saw {B} at the cafe,",
    "After {A} and {B} went to the library,",
]
_CENTER_CTX_RECENCY = [
    "{A} called {B} earlier. Later, {B} visited {A} at home. Then",
    "{A} wrote to {B} last week. Yesterday, {B} replied to {A}. Today",
    "{A} invited {B} to dinner. {B} then invited {A} to lunch. After that",
    "First {A} greeted {B}. Then {B} greeted {A} back. Next",
    "{A} saw {B} on Monday. {B} saw {A} on Tuesday. On Wednesday",
]
_CENTER_CTX_SUBJECT = [
    "{B} noticed {A} at the door. When {A} walked in,",
    "{B} recognized {A} from the meeting. Later",
    "{B} watched {A} arrive at the station. Then",
    "{B} heard {A} at the front gate. After that",
    "{B} spotted {A} across the room. Then",
]


def centering_theory_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 150,
    seed: int = 0,
) -> list[TaskPrompt]:
    rng = random.Random(seed)
    prompts: list[TaskPrompt] = []
    conditions = [
        ("standard", _CENTER_CTX_STANDARD),
        ("recency", _CENTER_CTX_RECENCY),
        ("subject_prominence", _CENTER_CTX_SUBJECT),
    ]
    n_per = n_prompts // len(conditions)

    for condition, contexts in conditions:
        seen: set[tuple[str, str, str]] = set()
        count = 0
        while count < n_per:
            a = rng.choice(_CENTER_NAMES_A)
            b = rng.choice(_CENTER_NAMES_B)
            ctx = rng.choice(contexts)
            action, prep = rng.choice(_CENTER_ACTIONS)
            key = (a, b, ctx)
            if key in seen:
                continue
            seen.add(key)
            text = f"{ctx.format(A=a, B=b)} {a} {action} {prep}"
            prompts.append(TaskPrompt(
                text=text, target_correct=" " + b, target_incorrect=" " + a,
                metadata={"condition": condition},
            ))
            count += 1

    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Resumptive repetition (IOI-circuit alias)
# ---------------------------------------------------------------------------

_RESUMPTIVE = [
    ("The old king ruled the land with wisdom.", "Many years passed and the seasons changed. New roads were built and towns grew larger.", "The old king", " ruled", " watched"),
    ("The merchant sold fine cloth at the market.", "Winter came early that year. The river froze and trade slowed to a halt.", "The merchant", " sold", " waited"),
    ("The children played in the garden every afternoon.", "One day a storm destroyed the fence and flooded the paths. Workers came to fix the damage.", "The children", " played", " watched"),
    ("The professor lectured on ancient history.", "A fire alarm interrupted the class. Everyone evacuated the building and waited outside for twenty minutes.", "The professor", " lectured", " resumed"),
    ("The baker made fresh bread every morning.", "The oven broke down last Tuesday. A repairman came and replaced the heating element.", "The baker", " made", " tested"),
    ("The captain steered the ship through calm waters.", "Dark clouds gathered on the horizon. The wind picked up and waves began crashing against the hull.", "The captain", " steered", " gripped"),
    ("The gardener tended the roses with great care.", "A new neighbor moved in next door. They brought two large dogs that kept digging under the fence.", "The gardener", " tended", " noticed"),
    ("The pianist practiced scales every evening.", "Her apartment flooded when the upstairs pipes burst. She had to move everything to a temporary place.", "The pianist", " practiced", " missed"),
    ("The watchmaker repaired clocks in his small shop.", "A new shopping mall opened across the street. Many of the old shops lost their customers.", "The watchmaker", " repaired", " struggled"),
    ("The farmer harvested wheat from the north field.", "A drought hit the region in July. The wells dried up and the government declared an emergency.", "The farmer", " harvested", " lost"),
    ("The teacher read stories to the class after lunch.", "The school was closed for a week due to flooding. Parents had to arrange childcare at short notice.", "The teacher", " read", " returned"),
    ("The doctor examined patients at the village clinic.", "An epidemic swept through the neighboring county. Medical supplies ran dangerously low everywhere.", "The doctor", " examined", " worked"),
    ("The writer finished a chapter every two weeks.", "Her publisher went bankrupt and the contract was voided. She spent months looking for a new deal.", "The writer", " finished", " started"),
    ("The astronomer observed the stars through a large telescope.", "Construction began on a new highway nearby. The light pollution ruined the viewing conditions for months.", "The astronomer", " observed", " struggled"),
    ("The tailor sewed suits for the wealthy families in town.", "Fashion trends shifted rapidly that year. Casual wear became the norm and formal events grew rare.", "The tailor", " sewed", " adapted"),
]


def resumptive_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 15,
    seed: int = 0,
) -> list[TaskPrompt]:
    prompts: list[TaskPrompt] = []
    for setup, digression, prefix, copy_target, discourse_target in _RESUMPTIVE:
        prompts.append(TaskPrompt(
            text=f"{setup} {digression} {prefix}",
            target_correct=copy_target,
            target_incorrect=discourse_target,
            metadata={"setup": setup, "digression_len": len(digression.split())},
        ))
    return prompts[:n_prompts]


# ---------------------------------------------------------------------------
# Self vs allo-repetition (IOI-circuit alias)
# ---------------------------------------------------------------------------

_SELF_ALLO = [
    ('John said "I need to leave." John said "I need to leave." John said "I need to',
     'John said "I need to leave." Mary said "I need to leave." John said "I need to',
     " leave", " go"),
    ('She whispered "the door is open." She whispered "the door is open." She whispered "the door is',
     'She whispered "the door is open." He whispered "the door is open." She whispered "the door is',
     " open", " closed"),
    ('The report concluded that prices will rise. The report concluded that prices will rise. The report concluded that prices will',
     'The first report concluded that prices will rise. The second report concluded that prices will rise. The first report concluded that prices will',
     " rise", " fall"),
    ('Tom asked "where is the key?" Tom asked "where is the key?" Tom asked "where is the',
     'Tom asked "where is the key?" Anna asked "where is the key?" Tom asked "where is the',
     " key", " door"),
    ('"We should wait," the leader warned. "We should wait," the leader warned. "We should wait," the leader',
     '"We should wait," the leader warned. "We should wait," the general agreed. "We should wait," the leader',
     " warned", " said"),
    ('The witness stated that the car was blue. The witness stated that the car was blue. The witness stated that the car was',
     'The witness stated that the car was blue. The officer confirmed that the car was blue. The witness stated that the car was',
     " blue", " red"),
    ('He promised "I will return by dawn." He promised "I will return by dawn." He promised "I will return by',
     'He promised "I will return by dawn." She hoped "I will return by dawn." He promised "I will return by',
     " dawn", " night"),
    ('The sign read "the bridge is closed." The sign read "the bridge is closed." The sign read "the bridge is',
     'The sign read "the bridge is closed." The guard confirmed "the bridge is closed." The sign read "the bridge is',
     " closed", " open"),
    ('The doctor said "take the medicine twice." The doctor said "take the medicine twice." The doctor said "take the medicine',
     'The doctor said "take the medicine twice." The nurse repeated "take the medicine twice." The doctor said "take the medicine',
     " twice", " daily"),
    ('Lisa wrote "the answer is seven." Lisa wrote "the answer is seven." Lisa wrote "the answer is',
     'Lisa wrote "the answer is seven." Mark wrote "the answer is seven." Lisa wrote "the answer is',
     " seven", " eight"),
]


def self_allo_prompts(
    tokenizer: TokenizerLike,
    n_prompts: int = 20,
    seed: int = 0,
) -> list[TaskPrompt]:
    prompts: list[TaskPrompt] = []
    for self_text, allo_text, correct, incorrect in _SELF_ALLO:
        prompts.append(TaskPrompt(
            text=self_text, target_correct=correct, target_incorrect=incorrect,
            metadata={"repetition_type": "self"},
        ))
        prompts.append(TaskPrompt(
            text=allo_text, target_correct=correct, target_incorrect=incorrect,
            metadata={"repetition_type": "allo"},
        ))
    return prompts[:n_prompts]
