"""Harvey 4-level v5: Superposition→Mech.Supported, name in tier color, (CVS: X.X) format, shift left."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig_harvey_4level_v10.png"

C, S, M, D, U = 1.0, 0.7, 0.3, -1.0, 0.0

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
        # C1 falsifiability: testable predictions regardless → C
        # C2 structural plaus: "Pass (in toy models)" → S
        # C3/C4 N/A (general theory, not circuit) → C
        # C5 convergent: "Partial" → S
        "C1":C,"C2":S,"C3":C,"C4":C,"C5":S,
        # I1-I3,I5: "Pass (toy)" → S; I4 consistency: "Pass" (robust across seeds) → C
        "I1":S,"I2":S,"I3":S,"I4":C,"I5":S,
        # M1-M2,M4-M5: "Pass (toy)" → S; M3 baseline sep: "Pass" (sharp) → C; M6: "Partial" → S
        "M1":S,"M2":S,"M3":C,"M4":S,"M5":S,"M6":S,
        # E1-E5: "Pass (toy)" → S; E6: "The critical gap" → U
        "E1":S,"E2":S,"E3":S,"E4":S,"E5":S,"E6":U,
        # V1 level decl, V2 match, V3 narrative: unconditional → C; V4,V5: toy-qualified → S
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":S,
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

cvs_scores = {
    "Induction Heads": 8.9, "Grokking": 9.4, "Superposition": 6.1,
    "IOI Circuit": 6.7, "Greater-Than": 6.9, "Copy Suppression": 6.9,
    "Successor Heads": 5.6, "Docstring Circuit": 4.2, "SAE Features": 3.3,
    "Othello World Model": 4.4, "Knowledge Neurons": 4.2,
    "Probing Classifiers": 1.9, "Gender Bias Circuits": 1.4,
}

tier_map = {
    "Induction Heads": "Triangulated", "Grokking": "Triangulated",
    "IOI Circuit": "Mech. Supported", "Greater-Than": "Mech. Supported",
    "Copy Suppression": "Mech. Supported", "Superposition": "Mech. Supported",
    "Successor Heads": "Caus. Suggestive", "Docstring Circuit": "Caus. Suggestive",
    "SAE Features": "Caus. Suggestive", "Othello World Model": "Caus. Suggestive",
    "Knowledge Neurons": "Caus. Suggestive", "Probing Classifiers": "Proposed",
    "Gender Bias Circuits": "Proposed",
}
tier_colors = {
    "Proposed": "#ef4444", "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981", "Triangulated": "#3b82f6",
}

# v3 change: light pink for Mixed instead of amber
status_colors = {
    1.0:  "#16a34a",  # Confirmed — green
    0.7:  "#86efac",  # Semi-confirmed — light green
    0.3:  "#fecaca",  # Mixed — light pink/red (was #fdba74 amber)
    -1.0: "#ef4444",  # Disconfirmed — red
    0.0:  "white",    # Untested
}
status_alpha = {1.0: 0.9, 0.7: 0.75, 0.3: 0.75, -1.0: 0.85, 0.0: 1.0}

dim_criteria = [
    ["C1","C2","C3","C4","C5"], ["I1","I2","I3","I4","I5"],
    ["M1","M2","M3","M4","M5","M6"], ["E1","E2","E3","E4","E5","E6"],
    ["V1","V2","V3","V4","V5"],
]
dim_names = ["Construct", "Internal", "Measurement", "External", "Interpretive"]
circuit_order = sorted(criteria_4level.keys(), key=lambda c: -cvs_scores[c])

n_rows = len(circuit_order)
col_spacing = 1.4
row_spacing = 1.2
radius = 0.42

fig_w = 14
fig_h = n_rows * row_spacing + 3.8
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(-11.5, (5 - 1) * col_spacing + radius + 0.5)
ax.set_ylim(-2.8, n_rows * row_spacing + 1.2)
ax.set_aspect('equal')
ax.axis('off')

# --- Top-left tier legend (vertical) ---
tier_legend_items = [
    ("Triangulated", "#3b82f6"),
    ("Mech. Supported", "#10b981"),
    ("Caus. Suggestive", "#f59e0b"),
    ("Proposed", "#ef4444"),
]
legend_x = -11.0
legend_top_y = (n_rows - 1) * row_spacing + 0.3
for idx, (tname, tcol) in enumerate(tier_legend_items):
    ly = legend_top_y - idx * 0.7
    sq_size = 0.28
    ax.add_patch(mpatches.FancyBboxPatch(
        (legend_x, ly - sq_size/2), sq_size, sq_size,
        boxstyle='round,pad=0.02', facecolor=tcol, edgecolor='none'))
    ax.text(legend_x + sq_size + 0.15, ly, tname, fontsize=10, va='center',
            fontweight='bold', color=tcol)

# Column headers
for j, dim in enumerate(dim_names):
    cx = j * col_spacing
    header_y = n_rows * row_spacing - 0.35
    ax.text(cx, header_y, dim, fontsize=12, fontweight='bold', color='#1e293b',
            ha='left', va='bottom', rotation=45)

for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    cdata = criteria_4level[name]
    cvs = cvs_scores[name]
    tc = tier_colors[tier]

    # Name in tier color, then tier-colored square, then score
    score_x = -1.2
    sq_size = 0.25
    ax.add_patch(mpatches.FancyBboxPatch(
        (score_x - sq_size/2, y - sq_size/2), sq_size, sq_size,
        boxstyle='round,pad=0.02', facecolor=tc, edgecolor='none'))
    # Score right of square
    ax.text(score_x + sq_size/2 + 0.12, y, f"{cvs:.1f}",
            fontsize=12, ha='left', va='center',
            fontweight='bold', fontstyle='italic', color=tc)
    # Name left of square
    ax.text(score_x - sq_size/2 - 0.15, y, name,
            fontsize=12, ha='right', va='center',
            fontweight='bold', fontstyle='italic', color=tc)

    for dim in range(5):
        cx = dim * col_spacing
        keys = dim_criteria[dim]
        n_crit = len(keys)
        angle_span = 360 / n_crit
        for seg, ckey in enumerate(keys):
            theta1 = 90 - (seg + 1) * angle_span
            theta2 = 90 - seg * angle_span
            val = cdata[ckey]
            fc = status_colors[val]
            alpha = status_alpha[val]
            wedge = mpatches.Wedge((cx, y), radius, theta1, theta2,
                                    facecolor=fc, edgecolor='white',
                                    linewidth=1.8, alpha=alpha)
            ax.add_patch(wedge)
        circle = plt.Circle((cx, y), radius, facecolor='none',
                              edgecolor='#cbd5e1', linewidth=1.0)
        ax.add_patch(circle)

# Bottom legend — pie slice wedges, spread 1.5x wider
ly = -1.3
items = [
    (1.0, "Confirmed"), (0.7, "Semi-confirmed"), (0.3, "Mixed"),
    (-1.0, "Disconfirmed"), (0.0, "Untested"),
]
positions = np.linspace(-3.5, 5.8, 5)
wedge_r = 0.25
for px, (val, label) in zip(positions, items):
    fc = status_colors[val]
    al = status_alpha[val]
    # Pie slice wedge (partial circle, not full)
    w = mpatches.Wedge((px, ly), wedge_r, 30, 150,
                        facecolor=fc, alpha=al, edgecolor='#cbd5e1', linewidth=1)
    ax.add_patch(w)
    ax.text(px + wedge_r + 0.15, ly, label, fontsize=10, va='center',
            color='#475569', fontweight='bold')

plt.tight_layout(pad=0.5)
plt.savefig(str(OUT), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {OUT}")
