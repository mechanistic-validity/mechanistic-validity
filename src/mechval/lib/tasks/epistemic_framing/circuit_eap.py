"""Epistemic framing EAP-derived circuit — GPT-2 small.

15 heads identified via Edge Attribution Patching (top by total edge
involvement score). Discovers a different set of components than
activation patching, emphasizing edge connectivity over individual
node effects.

Source: EAP discovery (2026-05). Not published.
"""

ROLES = {
    "early_relay": [
        (1, 3), (1, 10),
        (2, 10),
    ],
    "mid_hub": [
        (4, 3), (4, 7), (4, 11),
        (5, 9), (5, 10),
    ],
    "late_integrator": [
        (7, 6),
        (8, 8), (8, 10),
        (9, 5), (9, 6),
    ],
    "output": [
        (10, 0), (10, 7),
    ],
}

BANDS = {
    "early": (range(0, 3), ["early_relay"]),
    "mid": (range(3, 6), ["mid_hub"]),
    "midlate": (range(6, 10), ["late_integrator"]),
    "late": (range(10, 12), ["output"]),
}

PATHWAYS = [
    ("early_relay", "mid_hub"),
    ("early_relay", "late_integrator"),
    ("mid_hub", "late_integrator"),
    ("mid_hub", "output"),
    ("late_integrator", "output"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "experimental"
