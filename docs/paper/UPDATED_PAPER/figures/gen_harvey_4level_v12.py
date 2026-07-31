"""Harvey 4-level v12: right-middle legend, gridlines, verdict stripe,
header rule, mini-pie legend icons, black titles with colored square+number."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig_harvey_4level_v15.png"

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
tier_colors = {
    "Proposed": "#ef4444", "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981", "Triangulated": "#3b82f6",
}

status_colors = {
    C:  "#16a34a",   # Confirmed
    S:  "#86efac",   # Semi-confirmed
    M:  "#fecaca",   # Mixed — light pink
    D:  "#ef4444",   # Disconfirmed
    U:  "white",     # Untested
}
status_alpha = {C: 0.9, S: 0.75, M: 0.75, D: 0.85, U: 1.0}

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

# Layout coordinates
label_right_x = -1.4      # right edge of label area (square sits here)
circle_x0 = 0.0           # first column of circles
circles_right = (4) * col_spacing + radius + 0.3
stripe_x = -5.0            # left edge of verdict stripe (close to names)
row_left = stripe_x       # leftmost point of data area
header_y_base = n_rows * row_spacing - 0.35

# ── figure ──────────────────────────────────────────────────────────────
fig_w = 15
fig_h = n_rows * row_spacing + 2.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(stripe_x - 1.5, circles_right + 5.5)
ax.set_ylim(-1.0, n_rows * row_spacing + 1.5)
ax.set_aspect('equal')
ax.axis('off')

# ── (4) Header separator rule ───────────────────────────────────────────
rule_y = header_y_base - 0.55
ax.plot([row_left, circles_right], [rule_y, rule_y],
        color='#d0d0d0', linewidth=1.0, zorder=0)

# ── Column headers ──────────────────────────────────────────────────────
for j, dim in enumerate(dim_names):
    cx = j * col_spacing
    ax.text(cx, header_y_base, dim, fontsize=12, fontweight='bold',
            color='#1e293b', ha='left', va='bottom', rotation=45)

# ── Rows ────────────────────────────────────────────────────────────────
for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    cdata = criteria_4level[name]
    cvs = cvs_scores[name]
    tc = tier_colors[tier]

    # (2) Light row tint — 5% opacity verdict color band
    row_top = y + row_spacing / 2
    row_bot = y - row_spacing / 2
    from matplotlib.colors import to_rgba
    tint = to_rgba(tc, alpha=0.06)
    ax.fill_between([row_left, circles_right], row_bot, row_top,
                    color=tint, zorder=0)

    # (2 alt) Thin left verdict stripe — 4px colored bar
    stripe_w = 0.08
    ax.add_patch(mpatches.Rectangle(
        (stripe_x, row_bot), stripe_w, row_spacing,
        facecolor=tc, edgecolor='none', zorder=1))

    # (1) Row gridline
    if i < n_rows - 1:
        grid_y = row_bot
        ax.plot([row_left, circles_right], [grid_y, grid_y],
                color='#e8e8e8', linewidth=0.5, zorder=0)

    # Circuit name — BLACK, bold italic
    sq_x = label_right_x
    sq_size = 0.25
    ax.add_patch(mpatches.FancyBboxPatch(
        (sq_x - sq_size/2, y - sq_size/2), sq_size, sq_size,
        boxstyle='round,pad=0.02', facecolor=tc, edgecolor='none', zorder=2))
    # Score in tier color
    ax.text(sq_x + sq_size/2 + 0.12, y, f"{cvs:.1f}",
            fontsize=12, ha='left', va='center',
            fontweight='bold', fontstyle='italic', color=tc)
    # Name in black
    ax.text(sq_x - sq_size/2 - 0.15, y, name,
            fontsize=12, ha='right', va='center',
            fontweight='bold', fontstyle='italic', color='#1e293b')

    # Harvey balls
    for dim_idx in range(5):
        cx = dim_idx * col_spacing
        keys = dim_criteria[dim_idx]
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
                                    linewidth=1.8, alpha=alpha, zorder=2)
            ax.add_patch(wedge)
        circle = plt.Circle((cx, y), radius, facecolor='none',
                              edgecolor='#cbd5e1', linewidth=1.0, zorder=3)
        ax.add_patch(circle)

# (1) Column gridlines — between columns, slightly heavier
for j in range(1, 5):
    gx = j * col_spacing - col_spacing / 2
    ax.plot([gx, gx], [-0.6, (n_rows - 1) * row_spacing + row_spacing/2],
            color='#d0d0d0', linewidth=0.5, zorder=0)

# ── Left-side tier group labels ─────────────────────────────────────────
# Figure out which rows belong to which tier and place label at group midpoint
tier_groups = {}
for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    tier_groups.setdefault(tier, []).append(y)

for tier, ys in tier_groups.items():
    mid = (max(ys) + min(ys)) / 2
    tc = tier_colors[tier]
    ax.text(stripe_x - 0.15, mid, tier, fontsize=9, ha='right', va='center',
            fontweight='bold', color=tc, rotation=90)

# ── (3) Legend box — right-middle, Criteria only ────────────────────────
box_h = 3.8
box_w = 3.8
mid_y = (n_rows - 1) * row_spacing / 2
legend_top = mid_y + box_h / 2
box_x = circles_right + 0.7
box_y = legend_top - box_h

ax.add_patch(mpatches.FancyBboxPatch(
    (box_x, box_y), box_w, box_h,
    boxstyle='round,pad=0.15', facecolor='#f8fafc',
    edgecolor='#cbd5e1', linewidth=1.2, zorder=4))

# Criteria section header
cy = legend_top - 0.4
ax.text(box_x + 0.3, cy, "Criteria", fontsize=11, fontweight='bold',
        color='#1e293b', va='center')

crit_items = [
    (C, "Confirmed"),
    (S, "Semi-confirmed"),
    (M, "Mixed"),
    (D, "Disconfirmed"),
    (U, "Untested"),
]
mini_r = 0.2
slice_angle = 60  # 1/6th triangle slice
for idx, (val, label) in enumerate(crit_items):
    iy = cy - 0.55 - idx * 0.55
    ix = box_x + 0.48
    fc = status_colors[val]
    al = status_alpha[val]
    # White background circle
    bg = plt.Circle((ix, iy), mini_r, facecolor='white',
                     edgecolor='#94a3b8', linewidth=0.8, zorder=5)
    ax.add_patch(bg)
    # One triangle slice filled in status color
    w = mpatches.Wedge((ix, iy), mini_r, 90 - slice_angle, 90,
                        facecolor=fc, alpha=al,
                        edgecolor='white', linewidth=1.0, zorder=6)
    ax.add_patch(w)
    # Outer ring on top
    ring = plt.Circle((ix, iy), mini_r, facecolor='none',
                        edgecolor='#94a3b8', linewidth=0.8, zorder=7)
    ax.add_patch(ring)
    ax.text(ix + mini_r + 0.18, iy, label, fontsize=10,
            va='center', color='#475569', fontweight='bold', zorder=7)

plt.tight_layout(pad=0.5)
plt.savefig(str(OUT), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {OUT}")
