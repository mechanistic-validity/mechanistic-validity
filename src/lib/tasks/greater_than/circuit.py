"""Greater-Than circuit — Hanna et al. 2023 (NeurIPS, arXiv:2305.00586).

7 heads in 2 empirical roles. Year comparison task.
"""

ROLES = {
    "early_gt": [(5, 1), (6, 9), (8, 11), (9, 1)],
    "late_gt": [(5, 5), (7, 10), (8, 8)],
}

BANDS = {
    "mid": (range(5, 7), ["early_gt"]),
    "midlate": (range(7, 10), ["late_gt"]),
}

PATHWAYS = [("early_gt", "late_gt")]

HEADS = {h for heads in ROLES.values() for h in heads}
SOURCE = "published"
