"""Induction circuit — Olsson et al. 2022 (arXiv:2209.11895).

7 heads, 2 roles: previous-token heads feed induction heads.
"""

ROLES = {
    "PTH": [(2, 2), (4, 11)],
    "IND": [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)],
}

BANDS = {
    "early": (range(0, 5), ["PTH"]),
    "mid": (range(5, 8), ["IND"]),
}

PATHWAYS = [("PTH", "IND")]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
