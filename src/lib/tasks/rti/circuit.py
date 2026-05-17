"""RTI (Repeated Token Identification) circuit — OURS.

NOT from published literature. Discovered via weight-space classification.
15 heads in a three-tier architecture:
  - L0 backbone writes fixed directions
  - L4H11 detector reads them to attend to repeated tokens
  - Mid-layer copiers amplify non-repeated signal
  - Downstream induction heads do final prediction
"""

ROLES = {
    "backbone": [(0, 8), (0, 9), (0, 11)],
    "detector": [(4, 11)],
    "copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "readout": [(10, 11), (11, 9), (11, 11)],
}

BANDS = {
    "early": (range(0, 1), ["backbone"]),
    "mid_early": (range(4, 5), ["detector"]),
    "mid": (range(4, 10), ["copier"]),
    "late": (range(10, 12), ["readout"]),
}

PATHWAYS = [
    ("backbone", "detector"), ("backbone", "copier"), ("backbone", "readout"),
    ("detector", "copier"), ("detector", "readout"),
    ("copier", "readout"),
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "ours"
