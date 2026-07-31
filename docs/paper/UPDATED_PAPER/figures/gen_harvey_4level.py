"""Generate the 4-level Harvey ball matrix figure.

13 circuits × 5 validity dimensions, with per-criterion slices colored by
evidence status: Confirmed (green), Semi-confirmed (light green),
Mixed/negative (light pink), Disconfirmed (red), Untested (white).

CVS scores shown as colored text next to circuit names (tier color).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig_harvey_4level.png"

# ---------------------------------------------------------------------------
# 4-level evidence codes
# ---------------------------------------------------------------------------
C, S, M, D, U = 1.0, 0.7, 0.3, -1.0, 0.0  # Confirmed, Semi, Mixed, Disconfirmed, Untested

# Colors — green/red spectrum, no amber
COLOR_CONFIRMED      = "#16a34a"
COLOR_SEMI           = "#86efac"
COLOR_MIXED          = "#fecaca"   # light pink/red instead of amber
COLOR_DISCONFIRMED   = "#ef4444"
COLOR_UNTESTED       = "#ffffff"

def status_color(val):
    if val == C: return COLOR_CONFIRMED
    if val == S: return COLOR_SEMI
    if val == M: return COLOR_MIXED
    if val == D: return COLOR_DISCONFIRMED
    return COLOR_UNTESTED

# ---------------------------------------------------------------------------
# Per-criterion data (extracted from 13 case study markdown files)
# ---------------------------------------------------------------------------
criteria_4level = {
    "Induction Heads": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":C,
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":U,
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":S,"M6":C,
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":C,
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":C,
    },
    "Grokking": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":C,
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":C,
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":C,"M6":C,
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":S,
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":C,
    },
    "Superposition": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":S,
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":C,
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":C,"M6":S,
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":S,
    },
    "IOI Circuit": {
        "C1":C,"C2":C,"C3":U,"C4":S,"C5":S,
        "I1":C,"I2":C,"I3":U,"I4":S,"I5":U,
        "M1":U,"M2":S,"M3":C,"M4":U,"M5":U,"M6":S,
        "E1":U,"E2":S,"E3":U,"E4":C,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":M,"V5":S,
    },
    "Greater-Than": {
        "C1":C,"C2":C,"C3":S,"C4":C,"C5":S,
        "I1":C,"I2":S,"I3":S,"I4":S,"I5":U,
        "M1":U,"M2":S,"M3":C,"M4":C,"M5":U,"M6":C,
        "E1":U,"E2":S,"E3":S,"E4":C,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Copy Suppression": {
        "C1":C,"C2":C,"C3":S,"C4":C,"C5":S,
        "I1":C,"I2":S,"I3":C,"I4":S,"I5":U,
        "M1":U,"M2":S,"M3":C,"M4":C,"M5":U,"M6":C,
        "E1":S,"E2":U,"E3":C,"E4":S,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Successor Heads": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":S,
        "I1":C,"I2":S,"I3":S,"I4":S,"I5":U,
        "M1":U,"M2":C,"M3":C,"M4":C,"M5":U,"M6":C,
        "E1":U,"E2":S,"E3":S,"E4":S,"E5":C,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Docstring Circuit": {
        "C1":C,"C2":S,"C3":U,"C4":S,"C5":S,
        "I1":C,"I2":S,"I3":U,"I4":S,"I5":U,
        "M1":U,"M2":S,"M3":C,"M4":U,"M5":U,"M6":S,
        "E1":U,"E2":U,"E3":U,"E4":S,"E5":S,"E6":U,
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":S,
    },
    "SAE Features": {
        "C1":M,"C2":S,"C3":U,"C4":M,"C5":M,
        "I1":S,"I2":S,"I3":U,"I4":M,"I5":U,
        "M1":M,"M2":U,"M3":S,"M4":U,"M5":U,"M6":M,
        "E1":S,"E2":S,"E3":U,"E4":M,"E5":U,"E6":U,
        "V1":C,"V2":M,"V3":M,"V4":U,"V5":D,
    },
    "Othello World Model": {
        "C1":C,"C2":S,"C3":C,"C4":U,"C5":S,
        "I1":S,"I2":S,"I3":S,"I4":S,"I5":M,
        "M1":U,"M2":S,"M3":S,"M4":U,"M5":U,"M6":S,
        "E1":C,"E2":U,"E3":U,"E4":S,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":S,"V4":M,"V5":S,
    },
    "Knowledge Neurons": {
        "C1":C,"C2":S,"C3":S,"C4":M,"C5":S,
        "I1":C,"I2":C,"I3":M,"I4":S,"I5":M,
        "M1":S,"M2":M,"M3":S,"M4":U,"M5":U,"M6":S,
        "E1":S,"E2":U,"E3":S,"E4":C,"E5":S,"E6":S,
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":S,
    },
    "Probing Classifiers": {
        "C1":S,"C2":M,"C3":M,"C4":C,"C5":M,
        "I1":U,"I2":U,"I3":U,"I4":S,"I5":M,
        "M1":S,"M2":M,"M3":M,"M4":M,"M5":U,"M6":S,
        "E1":U,"E2":C,"E3":C,"E4":C,"E5":M,"E6":S,
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":D,
    },
    "Gender Bias Circuits": {
        "C1":S,"C2":S,"C3":M,"C4":M,"C5":M,
        "I1":S,"I2":D,"I3":M,"I4":M,"I5":M,
        "M1":M,"M2":M,"M3":S,"M4":U,"M5":U,"M6":M,
        "E1":S,"E2":M,"E3":M,"E4":M,"E5":M,"E6":S,
        "V1":S,"V2":S,"V3":M,"V4":M,"V5":D,
    },
}

# Dimension → criteria mapping
dimensions = {
    "Construct":    ["C1","C2","C3","C4","C5"],
    "Internal":     ["I1","I2","I3","I4","I5"],
    "Measurement":  ["M1","M2","M3","M4","M5","M6"],
    "External":     ["E1","E2","E3","E4","E5","E6"],
    "Interpretive": ["V1","V2","V3","V4","V5"],
}

dim_names = list(dimensions.keys())

# CVS scores and tier colors
cvs_scores = {
    "Induction Heads": 8.9, "Grokking": 9.4, "Superposition": 6.1,
    "IOI Circuit": 6.7, "Greater-Than": 6.9, "Copy Suppression": 6.9,
    "Successor Heads": 5.6, "Docstring Circuit": 4.2, "SAE Features": 3.3,
    "Othello World Model": 4.4, "Knowledge Neurons": 4.2,
    "Probing Classifiers": 1.9, "Gender Bias Circuits": 1.4,
}

tier_colors = {
    "Proposed":          "#ef4444",
    "Caus. Suggestive":  "#f59e0b",
    "Mech. Supported":   "#10b981",
    "Triangulated":      "#3b82f6",
    "Validated":         "#7c3aed",
}

def get_tier(score):
    if score >= 8.0: return "Validated"
    if score >= 6.0: return "Triangulated"
    if score >= 4.0: return "Mech. Supported"
    if score >= 2.0: return "Caus. Suggestive"
    return "Proposed"

circuit_order = list(criteria_4level.keys())

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
n_rows = len(circuit_order)
n_cols = len(dim_names)

fig_w = 11
row_h = 0.62
top_margin = 1.1
bot_margin = 0.9
fig_h = top_margin + n_rows * row_h + bot_margin

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# Layout params
left_text_x = 0.15          # circuit names far left
score_x_offset = 0.12       # CVS score right after name
circle_start_x = 4.0        # where circles begin
circle_spacing = 1.45
radius = 0.22

# Column headers
for j, dname in enumerate(dim_names):
    cx = circle_start_x + j * circle_spacing
    cy = fig_h - top_margin + 0.55
    ax.text(cx, cy, dname, ha="center", va="bottom", fontsize=11,
            fontweight="bold", fontstyle="italic", rotation=35)

# Rows
for i, circuit in enumerate(circuit_order):
    y = fig_h - top_margin - i * row_h - row_h / 2
    cdata = criteria_4level[circuit]
    score = cvs_scores[circuit]
    tier = get_tier(score)
    tc = tier_colors[tier]

    # Circuit name — bold italic
    ax.text(left_text_x, y, circuit, ha="left", va="center",
            fontsize=11, fontweight="bold", fontstyle="italic")

    # CVS score in tier color — fixed position so they all align
    ax.text(3.45, y, f"{score:.1f}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=tc)

    # Harvey balls per dimension
    for j, dname in enumerate(dim_names):
        cx = circle_start_x + j * circle_spacing
        crits = dimensions[dname]
        n = len(crits)
        angle_each = 360.0 / n

        # Draw slices
        for k, crit in enumerate(crits):
            val = cdata[crit]
            fc = status_color(val)
            start_angle = 90 - k * angle_each
            wedge = mpatches.Wedge(
                (cx, y), radius, start_angle - angle_each, start_angle,
                facecolor=fc, edgecolor="#9ca3af", linewidth=0.6
            )
            ax.add_patch(wedge)

        # Circle outline
        outline = plt.Circle((cx, y), radius, fill=False,
                             edgecolor="#6b7280", linewidth=1.0)
        ax.add_patch(outline)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_y = 0.45
legend_items = [
    (COLOR_CONFIRMED,    "Confirmed"),
    (COLOR_SEMI,         "Semi-confirmed"),
    (COLOR_MIXED,        "Mixed"),
    (COLOR_DISCONFIRMED, "Disconfirmed"),
    (COLOR_UNTESTED,     "Untested"),
]

legend_x_start = circle_start_x - 1.5
legend_spacing = 1.8
for idx, (color, label) in enumerate(legend_items):
    lx = legend_x_start + idx * legend_spacing
    circ = plt.Circle((lx, legend_y), 0.12, facecolor=color,
                      edgecolor="#6b7280", linewidth=0.8)
    ax.add_patch(circ)
    ax.text(lx + 0.2, legend_y, label, ha="left", va="center", fontsize=9)

# Tier legend below
tier_y = legend_y - 0.35
tier_items = [
    ("Proposed", tier_colors["Proposed"]),
    ("Caus. Suggestive", tier_colors["Caus. Suggestive"]),
    ("Mech. Supported", tier_colors["Mech. Supported"]),
    ("Triangulated", tier_colors["Triangulated"]),
]
tier_x_start = legend_x_start + 0.3
tier_spacing = 2.2
for idx, (label, color) in enumerate(tier_items):
    tx = tier_x_start + idx * tier_spacing
    ax.text(tx, tier_y, label, ha="left", va="center",
            fontsize=8, fontweight="bold", color=color)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}")
