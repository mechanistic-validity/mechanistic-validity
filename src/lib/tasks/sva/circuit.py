"""SVA (Subject-Verb Agreement) circuit — Lazo et al. 2025.

12 heads, 4 empirical roles. Number agreement across attractors.
"""

ROLES = {
    "embed": [(0, 4), (0, 8)],
    "encode": [(1, 0), (1, 1), (2, 1), (2, 6)],
    "route": [(6, 0), (9, 4)],
    "output": [(10, 0), (11, 4), (11, 6), (11, 7)],
}

BANDS = {
    "early": (range(0, 3), ["embed", "encode"]),
    "mid": (range(3, 10), ["route"]),
    "late": (range(10, 12), ["output"]),
}

PATHWAYS = [
    ("embed", "encode"), ("embed", "route"), ("embed", "output"),
    ("encode", "route"), ("encode", "output"),
    ("route", "output"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
