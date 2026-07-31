"""Harvey 4-level v19: v18 + vertical column separators clipped to pie area."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent

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
        "C1":C,"C2":S,"C3":C,"C4":C,"C5":S,
        "I1":S,"I2":S,"I3":S,"I4":C,"I5":S,
        "M1":S,"M2":S,"M3":C,"M4":S,"M5":S,"M6":S,
        "E1":S,"E2":S,"E3":S,"E4":S,"E5":S,"E6":U,
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
Tier_colors = {
    "Proposed": "#ef4444", "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981", "Triangulated": "#3b82f6",
}

tier_colors = {
    "Proposed": "#ef4444", "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981", "Triangulated": "#3b82f6",
}

status_colors = {
    1.0:  "#16a34a",
    0.7:  "#86efac",
    0.3:  "#fecaca",
    -1.0: "#ef4444",
    0.0:  "white",
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

last_col_x = 4 * col_spacing
grid_x_left = 0 * col_spacing - radius - 0.15
grid_x_right = last_col_x + radius + 0.15


def draw_legend_box(ax, box_x, box_top_y):
    box_w = 3.2
    box_h = 6.2
    box_y = box_top_y - box_h

    ax.add_patch(mpatches.FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle='round,pad=0.15', facecolor='#f8fafc',
        edgecolor='#cbd5e1', linewidth=1.2))

    vy = box_top_y - 0.35
    ax.text(box_x + 0.3, vy, "Verdicts", fontsize=11, fontweight='bold',
            color='#1e293b', va='center')

    tier_items = [
        ("Triangulated", "#3b82f6"),
        ("Mech. Supported", "#10b981"),
        ("Caus. Suggestive", "#f59e0b"),
        ("Proposed", "#ef4444"),
    ]
    for idx, (tname, tcol) in enumerate(tier_items):
        ty = vy - 0.55 - idx * 0.55
        sq = 0.25
        ax.add_patch(mpatches.FancyBboxPatch(
            (box_x + 0.3, ty - sq/2), sq, sq,
            boxstyle='round,pad=0.02', facecolor=tcol, edgecolor='none'))
        ax.text(box_x + 0.3 + sq + 0.15, ty, tname, fontsize=10,
                va='center', fontweight='bold', color=tcol)

    cy = vy - 0.55 - 4 * 0.55 - 0.3
    ax.text(box_x + 0.3, cy, "Criteria", fontsize=11, fontweight='bold',
            color='#1e293b', va='center')

    crit_items = [
        (1.0, "Confirmed"), (0.7, "Semi-confirmed"), (0.3, "Mixed"),
        (-1.0, "Disconfirmed"), (0.0, "Untested"),
    ]
    wedge_r = 0.2
    for idx, (val, label) in enumerate(crit_items):
        iy = cy - 0.5 - idx * 0.5
        fc = status_colors[val]
        al = status_alpha[val]
        wx = box_x + 0.45
        w = mpatches.Wedge((wx, iy), wedge_r, 60, 120,
                            facecolor=fc, alpha=al,
                            edgecolor='#94a3b8', linewidth=0.8)
        ax.add_patch(w)
        ax.text(wx + wedge_r + 0.15, iy, label, fontsize=10,
                va='center', color='#475569', fontweight='bold')


fig_w = 14
fig_h = n_rows * row_spacing + 2.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

circles_right = last_col_x + radius + 0.5
label_left = -5.5
ax.set_xlim(label_left, circles_right + 4.5)
ax.set_ylim(-1.0, n_rows * row_spacing + 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Legend box — right, vertically centered
box_h = 6.2
mid = (n_rows - 1) * row_spacing / 2
legend_top = mid + box_h / 2
draw_legend_box(ax, circles_right + 0.5, legend_top)

# --- Clipped horizontal grid lines (stop at pie area edges) ---
grid_color = '#e2e8f0'
grid_lw = 0.6

# Top rule (under headers)
top_gy = (n_rows - 1) * row_spacing + row_spacing / 2
ax.plot([grid_x_left, grid_x_right], [top_gy, top_gy],
        color=grid_color, linewidth=grid_lw, zorder=0)

# Row separators + bottom closing rule
for i in range(1, n_rows + 1):
    if i < n_rows:
        gy = (n_rows - 1 - i) * row_spacing + row_spacing / 2
    else:
        gy = -row_spacing / 2
    ax.plot([grid_x_left, grid_x_right], [gy, gy],
            color=grid_color, linewidth=grid_lw, zorder=0)

# Vertical column separators (clipped to pie area top/bottom)
col_positions = [j * col_spacing for j in range(5)]
grid_y_top = top_gy
grid_y_bot = -row_spacing / 2
for j in range(1, 5):
    gx = (col_positions[j - 1] + col_positions[j]) / 2
    ax.plot([gx, gx], [grid_y_bot, grid_y_top],
            color=grid_color, linewidth=grid_lw, zorder=0)

# Column headers
for j, dim in enumerate(dim_names):
    cx = j * col_spacing
    header_y = n_rows * row_spacing - 0.35
    ax.text(cx, header_y, dim, fontsize=12, fontweight='bold', color='#1e293b',
            ha='left', va='bottom', rotation=45)

# Rows
for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    cdata = criteria_4level[name]
    cvs = cvs_scores[name]
    tc = tier_colors[tier]

    score_x = -1.4
    sq_size = 0.25
    ax.add_patch(mpatches.FancyBboxPatch(
        (score_x - sq_size/2, y - sq_size/2), sq_size, sq_size,
        boxstyle='round,pad=0.02', facecolor=tc, edgecolor='none'))
    ax.text(score_x + sq_size/2 + 0.12, y, f"{cvs:.1f}",
            fontsize=12, ha='left', va='center',
            fontweight='bold', fontstyle='italic', color=tc)
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

plt.tight_layout(pad=0.5)
out = OUTDIR / "fig_harvey_4level_v19.png"
plt.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved -> {out}")
