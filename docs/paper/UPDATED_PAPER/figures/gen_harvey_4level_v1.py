import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 4-level + untested encoding:
# 1.0 = Confirmed, 0.7 = Semi-confirmed, 0.3 = Mixed, -1.0 = Disconfirmed, 0.0 = Untested
C, S, M, D, U = 1.0, 0.7, 0.3, -1.0, 0.0

criteria_4level = {
    "Induction Heads": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":C,    # all strong pass, C3 NA→C
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":U,     # I5 not directly tested
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":S,"M6":C,  # M5 partial calibration
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":C,
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":C,
    },
    "Grokking": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":C,
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":C,
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":C,"M6":C,
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":S,  # weak cross-arch
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":C,
    },
    "Superposition": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":S,    # C3/C4 NA→C, C5 partial convergent
        "I1":C,"I2":C,"I3":C,"I4":C,"I5":C,     # all pass in toy
        "M1":C,"M2":C,"M3":C,"M4":C,"M5":C,"M6":S,  # M6 partial coverage
        "E1":C,"E2":C,"E3":C,"E4":C,"E5":C,"E6":U,  # E6 the critical gap = untested
        "V1":C,"V2":C,"V3":C,"V4":C,"V5":S,     # V5 mostly honest
    },
    "IOI Circuit": {
        "C1":C,"C2":C,"C3":U,"C4":S,"C5":S,     # C3 not tested, C4/C5 partial
        "I1":C,"I2":C,"I3":U,"I4":S,"I5":U,      # I3/I5 not tested
        "M1":U,"M2":S,"M3":C,"M4":U,"M5":U,"M6":S,  # M1/M4/M5 not reported
        "E1":U,"E2":S,"E3":U,"E4":C,"E5":S,"E6":U,  # mostly untested
        "V1":C,"V2":C,"V3":C,"V4":M,"V5":S,      # V4 weak alt exclusion
    },
    "Greater-Than": {
        "C1":C,"C2":C,"C3":S,"C4":C,"C5":S,
        "I1":C,"I2":S,"I3":S,"I4":S,"I5":U,      # I5 not tested
        "M1":U,"M2":S,"M3":C,"M4":C,"M5":U,"M6":C,  # M1/M5 not reported
        "E1":U,"E2":S,"E3":S,"E4":C,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Copy Suppression": {
        "C1":C,"C2":C,"C3":S,"C4":C,"C5":S,
        "I1":C,"I2":S,"I3":C,"I4":S,"I5":U,      # I5 not tested
        "M1":U,"M2":S,"M3":C,"M4":C,"M5":U,"M6":C,
        "E1":S,"E2":U,"E3":C,"E4":S,"E5":S,"E6":U,  # E2 not tested
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Successor Heads": {
        "C1":C,"C2":C,"C3":C,"C4":C,"C5":S,     # C3 NA→C
        "I1":C,"I2":S,"I3":S,"I4":S,"I5":U,
        "M1":U,"M2":C,"M3":C,"M4":C,"M5":U,"M6":C,
        "E1":U,"E2":S,"E3":S,"E4":S,"E5":C,"E6":U,
        "V1":C,"V2":C,"V3":C,"V4":S,"V5":C,
    },
    "Docstring Circuit": {
        "C1":C,"C2":S,"C3":U,"C4":S,"C5":S,     # C3 not tested
        "I1":C,"I2":S,"I3":U,"I4":S,"I5":U,      # I3/I5 not tested
        "M1":U,"M2":S,"M3":C,"M4":U,"M5":U,"M6":S,
        "E1":U,"E2":U,"E3":U,"E4":S,"E5":S,"E6":U,  # mostly untested
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":S,      # V4 alternatives unexcluded
    },
    "SAE Features": {
        "C1":M,"C2":S,"C3":U,"C4":M,"C5":M,     # C1 circularity, C4 open question
        "I1":S,"I2":S,"I3":U,"I4":M,"I5":U,      # I4 weak consistency
        "M1":M,"M2":U,"M3":S,"M4":U,"M5":U,"M6":M,  # M1/M6 weak
        "E1":S,"E2":S,"E3":U,"E4":M,"E5":U,"E6":U,  # E4 variable
        "V1":C,"V2":M,"V3":M,"V4":U,"V5":D,      # V5 often missing = disconfirmed
    },
    "Othello World Model": {
        "C1":C,"C2":S,"C3":C,"C4":U,"C5":S,     # C4 not tested
        "I1":S,"I2":S,"I3":S,"I4":S,"I5":M,      # I5 weak confound control
        "M1":U,"M2":S,"M3":S,"M4":U,"M5":U,"M6":S,
        "E1":C,"E2":U,"E3":U,"E4":S,"E5":S,"E6":U,
        "V1":C,"V2":C,"V3":S,"V4":M,"V5":S,     # V4 alternatives not excluded
    },
    "Knowledge Neurons": {
        "C1":C,"C2":S,"C3":S,"C4":M,"C5":S,     # C4 unclear minimality
        "I1":C,"I2":C,"I3":M,"I4":S,"I5":M,      # I3 critical gap, I5 weak
        "M1":S,"M2":M,"M3":S,"M4":U,"M5":U,"M6":S,  # M2 weak invariance
        "E1":S,"E2":U,"E3":S,"E4":C,"E5":S,"E6":S,
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":S,     # V4 alternatives not excluded
    },
    "Probing Classifiers": {
        "C1":S,"C2":M,"C3":M,"C4":C,"C5":M,     # C2 weak structural, C3/C5 variable
        "I1":U,"I2":U,"I3":U,"I4":S,"I5":M,      # I1-I3 not tested, I5 critical gap
        "M1":S,"M2":M,"M3":M,"M4":M,"M5":U,"M6":S,  # M3 critical gap w/o controls
        "E1":U,"E2":C,"E3":C,"E4":C,"E5":M,"E6":S,  # E2-E4 NA→C
        "V1":C,"V2":S,"V3":S,"V4":M,"V5":D,     # V5 often violated = disconfirmed
    },
    "Gender Bias Circuits": {
        "C1":S,"C2":S,"C3":M,"C4":M,"C5":M,     # C3 weak, C4 unclear, C5 weak
        "I1":S,"I2":D,"I3":M,"I4":M,"I5":M,      # I2 not demonstrated = disconfirmed
        "M1":M,"M2":M,"M3":S,"M4":U,"M5":U,"M6":M,
        "E1":S,"E2":M,"E3":M,"E4":M,"E5":M,"E6":S,
        "V1":S,"V2":S,"V3":M,"V4":M,"V5":D,     # V5 often violated = disconfirmed
    },
}

tier_map = {
    "Induction Heads": "Triangulated", "Grokking": "Triangulated",
    "Superposition": "Triangulated", "IOI Circuit": "Mech. Supported",
    "Greater-Than": "Mech. Supported", "Copy Suppression": "Mech. Supported",
    "Successor Heads": "Caus. Suggestive", "Docstring Circuit": "Caus. Suggestive",
    "SAE Features": "Caus. Suggestive", "Othello World Model": "Caus. Suggestive",
    "Knowledge Neurons": "Caus. Suggestive", "Probing Classifiers": "Proposed",
    "Gender Bias Circuits": "Proposed",
}
tier_colors = {
    "Proposed": "#ef4444", "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981", "Triangulated": "#3b82f6",
}

# Status colors (universal, not tier-dependent)
status_colors = {
    1.0:  "#16a34a",   # Confirmed — green
    0.7:  "#86efac",   # Semi-confirmed — light green
    0.3:  "#fdba74",   # Mixed — amber
    -1.0: "#ef4444",   # Disconfirmed — red
    0.0:  "white",     # Untested — white
}
status_alpha = {1.0: 0.9, 0.7: 0.75, 0.3: 0.75, -1.0: 0.85, 0.0: 1.0}

dim_criteria = [
    ["C1","C2","C3","C4","C5"], ["I1","I2","I3","I4","I5"],
    ["M1","M2","M3","M4","M5","M6"], ["E1","E2","E3","E4","E5","E6"],
    ["V1","V2","V3","V4","V5"],
]
dim_names = ["Construct", "Internal", "Measurement", "External", "Interpretive"]
circuit_order = list(criteria_4level.keys())

# Print counts
for name in circuit_order:
    cdata = criteria_4level[name]
    vals = list(cdata.values())
    cc = sum(1 for v in vals if v == 1.0)
    sc = sum(1 for v in vals if v == 0.7)
    mx = sum(1 for v in vals if v == 0.3)
    dc = sum(1 for v in vals if v == -1.0)
    ut = sum(1 for v in vals if v == 0.0)
    print(f"{name:25s}  C={cc:2d} S={sc:2d} M={mx:2d} D={dc:2d} U={ut:2d}")

n_rows = len(circuit_order)
col_spacing = 1.3
row_spacing = 1.1
radius = 0.38

fig_w = 11
fig_h = n_rows * row_spacing + 3.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(-7.0, (5 - 1) * col_spacing + radius + 0.5)
ax.set_ylim(-2.5, n_rows * row_spacing + 1.5)
ax.set_aspect('equal')
ax.axis('off')

for j, dim in enumerate(dim_names):
    cx = j * col_spacing
    header_y = n_rows * row_spacing - 0.15
    ax.text(cx, header_y, dim, fontsize=10, fontweight='bold', color='#1e293b',
            ha='left', va='bottom', rotation=45)

for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    cdata = criteria_4level[name]

    square_x = -0.85
    ax.add_patch(mpatches.FancyBboxPatch(
        (square_x, y - 0.13), 0.26, 0.26,
        boxstyle='round,pad=0.02', facecolor=tier_colors[tier], edgecolor='none'))
    ax.text(square_x - 0.2, y, name, fontsize=10, ha='right', va='center',
            fontweight='bold', color='#374151', fontstyle='italic')

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
                                    linewidth=1.5, alpha=alpha)
            ax.add_patch(wedge)
        circle = plt.Circle((cx, y), radius, facecolor='none',
                              edgecolor='#cbd5e1', linewidth=1.0)
        ax.add_patch(circle)

# Legend — 5 levels spread across bottom
ly = -1.1
items = [
    (1.0, "Confirmed"), (0.7, "Semi-confirmed"), (0.3, "Mixed"),
    (-1.0, "Disconfirmed"), (0.0, "Untested"),
]
positions = np.linspace(-1.8, 5.0, 5)
for px, (val, label) in zip(positions, items):
    fc = status_colors[val]
    al = status_alpha[val]
    ax.add_patch(mpatches.Wedge((px, ly), 0.2, 0, 360,
                 facecolor=fc, alpha=al, edgecolor='#cbd5e1', linewidth=1))
    ax.text(px + 0.35, ly, label, fontsize=8.5, va='center', color='#475569', fontweight='bold')

# Tier legend
ty = ly - 0.6
tiers = [("Proposed","#ef4444"), ("Caus. Suggestive","#f59e0b"),
         ("Mech. Supported","#10b981"), ("Triangulated","#3b82f6")]
for idx, (tname, tcol) in enumerate(tiers):
    tx = -1.8 + idx * 2.3
    ax.add_patch(mpatches.FancyBboxPatch(
        (tx - 0.1, ty - 0.1), 0.2, 0.2,
        boxstyle='round,pad=0.02', facecolor=tcol, edgecolor='none'))
    ax.text(tx + 0.2, ty, tname, fontsize=8, va='center', color='#475569')

plt.tight_layout(pad=0.5)
plt.savefig('fig_harvey_4level.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved fig_harvey_4level.png')
