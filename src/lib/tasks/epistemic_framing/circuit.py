"""Epistemic framing circuit — GPT-2 small.

4 core heads processing epistemic stance markers (think/believe/know).
Processes epistemic framing regardless of subject (I/He/She/They).
Truth-insensitive: pure syntactic stance marker.

Source: batch1_mechval evaluation (2026-05). Not published.
"""

ROLES = {
    "detector": [(6, 5)],       # L6H5: detects epistemic verb tokens
    "integrator": [(9, 2), (9, 5)],  # L9H2, L9H5: integrate epistemic signal
    "executor": [(10, 5)],      # L10H5: adjusts output distribution
}

BANDS = {
    "early": (range(0, 8), ["detector"]),
    "mid": (range(8, 10), ["integrator"]),
    "late": (range(10, 12), ["executor"]),
}

PATHWAYS = [
    ("detector", "integrator"),
    ("integrator", "executor"),
    ("detector", "executor"),   # direct skip connection L6H5→L10H5
]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "experimental"
