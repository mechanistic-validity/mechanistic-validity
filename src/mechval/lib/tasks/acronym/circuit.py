"""Acronym circuit — Garcia-Carrasco et al. AISTATS 2024.

8 heads, 3 roles. Letter prediction in known acronyms.
"""

ROLES = {
    "letter_mover": [(8, 11), (9, 9), (10, 10), (11, 4)],
    "PTH": [(1, 0), (2, 2), (4, 11)],
    "propagator": [(5, 8)],
}

BANDS = {
    "early": (range(1, 5), ["PTH"]),
    "mid": (range(5, 6), ["propagator"]),
    "late": (range(8, 12), ["letter_mover"]),
}

PATHWAYS = [
    ("PTH", "propagator"), ("PTH", "letter_mover"),
    ("propagator", "letter_mover"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
