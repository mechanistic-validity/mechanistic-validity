"""Gendered Pronoun circuit — Mathwin 2023.

5 heads, 3 empirical roles. Gender agreement resolution.
"""

ROLES = {
    "early_ga": [(0, 10)],
    "late_ga": [(3, 0), (5, 8)],
    "name_bind": [(6, 6), (8, 6)],
}

BANDS = {
    "very_early": (range(0, 2), ["early_ga"]),
    "early_mid": (range(2, 6), ["late_ga"]),
    "mid_late": (range(6, 9), ["name_bind"]),
}

PATHWAYS = [
    ("early_ga", "late_ga"), ("early_ga", "name_bind"),
    ("late_ga", "name_bind"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
