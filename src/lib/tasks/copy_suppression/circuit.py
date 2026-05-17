"""Copy Suppression circuit — McDougall et al. 2023 (arXiv:2310.04625).

L10H7 is the primary copy-suppression head. PTH and induction
heads provide the copying signal that gets suppressed.
Smallest circuit in the taxonomy (5 heads); most template-sensitive.
"""

ROLES = {
    "PTH": [(2, 2), (4, 11)],
    "IND": [(5, 1), (5, 5), (6, 9)],
    "copy_suppress": [(10, 7), (11, 10)],
}

BANDS = {
    "early": (range(0, 5), ["PTH"]),
    "mid": (range(5, 7), ["IND"]),
    "late": (range(10, 12), ["copy_suppress"]),
}

PATHWAYS = [
    ("PTH", "IND"), ("PTH", "copy_suppress"),
    ("IND", "copy_suppress"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
