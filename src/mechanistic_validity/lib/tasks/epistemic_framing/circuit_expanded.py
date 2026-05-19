"""Epistemic framing expanded circuit — GPT-2 small.

32 heads identified via activation patching scan (|effect| > 0.15).
Roles assigned by layer depth and effect direction.

Source: activation patching discovery (2026-05). Not published.
"""

ROLES = {
    "early_processor": [
        (0, 0), (0, 4), (0, 8), (0, 11),
        (1, 4),
        (2, 0), (2, 10),
    ],
    "early_suppressor": [
        (0, 2), (0, 6), (0, 7), (0, 9),
        (2, 1), (2, 5), (2, 8),
    ],
    "mid_composer": [
        (3, 8),
        (4, 0), (4, 1), (4, 4), (4, 7), (4, 9),
        (5, 3), (5, 7),
    ],
    "mid_suppressor": [
        (4, 3), (4, 6),
        (5, 2),
    ],
    "late_router": [
        (6, 5),
        (7, 3), (7, 9),
        (10, 0),
    ],
    "late_suppressor": [
        (7, 8),
        (8, 5), (8, 8),
    ],
}

BANDS = {
    "early": (range(0, 3), ["early_processor", "early_suppressor"]),
    "mid": (range(3, 6), ["mid_composer", "mid_suppressor"]),
    "midlate": (range(6, 9), ["late_router", "late_suppressor"]),
    "late": (range(9, 12), ["late_router"]),
}

PATHWAYS = [
    ("early_processor", "mid_composer"),
    ("early_processor", "late_router"),
    ("early_suppressor", "mid_suppressor"),
    ("early_suppressor", "mid_composer"),
    ("mid_composer", "late_router"),
    ("mid_suppressor", "late_router"),
    ("mid_composer", "late_suppressor"),
    ("late_suppressor", "late_router"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "experimental"
