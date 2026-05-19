"""Epistemic framing tight circuit — GPT-2 small.

13 heads identified via activation patching scan (|effect| > 0.20).
Similar size to IOI (15 heads). Focused on the strongest contributors.

Source: activation patching discovery (2026-05). Not published.
"""

ROLES = {
    "early_processor": [
        (0, 0), (0, 4), (0, 8),
        (2, 10),
    ],
    "early_suppressor": [
        (0, 6), (0, 7), (0, 9),
        (2, 1),
    ],
    "mid_composer": [
        (3, 8),
        (5, 3),
    ],
    "mid_suppressor": [
        (4, 6),
        (5, 2),
    ],
    "late_router": [
        (10, 0),
    ],
}

BANDS = {
    "early": (range(0, 3), ["early_processor", "early_suppressor"]),
    "mid": (range(3, 6), ["mid_composer", "mid_suppressor"]),
    "late": (range(6, 12), ["late_router"]),
}

PATHWAYS = [
    ("early_processor", "mid_composer"),
    ("early_processor", "late_router"),
    ("early_suppressor", "mid_suppressor"),
    ("early_suppressor", "mid_composer"),
    ("mid_composer", "late_router"),
    ("mid_suppressor", "late_router"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "experimental"
