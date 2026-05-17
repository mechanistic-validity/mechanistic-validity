"""IOI circuit — Wang et al. 2023 (ICLR, arXiv:2211.00593).

15 core heads across 6 functional roles. The most thoroughly
validated circuit in the mech-interp literature.
"""

ROLES = {
    "DTH": [(0, 1), (3, 0)],
    "PTH": [(2, 2), (4, 11)],
    "IND": [(5, 5), (6, 9)],
    "S-Inh": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "NM": [(9, 9), (9, 6), (10, 0)],
    "NegNM": [(10, 7), (11, 10)],
}

BANDS = {
    "early": (range(0, 5), ["DTH", "PTH"]),
    "mid": (range(5, 7), ["IND"]),
    "midlate": (range(7, 9), ["S-Inh"]),
    "late": (range(9, 12), ["NM", "NegNM"]),
}

PATHWAYS = [
    ("DTH", "S-Inh"), ("DTH", "IND"), ("PTH", "IND"),
    ("IND", "S-Inh"), ("S-Inh", "NM"), ("S-Inh", "NegNM"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
