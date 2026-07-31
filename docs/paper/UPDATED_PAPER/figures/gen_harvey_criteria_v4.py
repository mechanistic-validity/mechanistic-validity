import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Real per-criterion data extracted from 13 case study markdown files
# YES=1, PARTIAL=0.5, NO=0, NA=-1
V = {"YES": 1, "PARTIAL": 0.5, "NO": 0, "NA": -1}

criteria_raw = {
    "Induction Heads": {
        "C1":1,"C2":1,"C3":-1,"C4":1,"C5":1,
        "I1":1,"I2":1,"I3":1,"I4":1,"I5":0,
        "M1":1,"M2":1,"M3":1,"M4":1,"M5":.5,"M6":1,
        "E1":1,"E2":1,"E3":1,"E4":1,"E5":1,"E6":1,
        "V1":1,"V2":1,"V3":1,"V4":1,"V5":1,
    },
    "Grokking": {
        "C1":1,"C2":1,"C3":1,"C4":1,"C5":1,
        "I1":1,"I2":1,"I3":1,"I4":1,"I5":1,
        "M1":1,"M2":1,"M3":1,"M4":1,"M5":1,"M6":1,
        "E1":1,"E2":1,"E3":1,"E4":1,"E5":1,"E6":.5,
        "V1":1,"V2":1,"V3":1,"V4":1,"V5":1,
    },
    "Superposition": {
        "C1":1,"C2":1,"C3":-1,"C4":-1,"C5":.5,
        "I1":1,"I2":1,"I3":1,"I4":1,"I5":1,
        "M1":1,"M2":1,"M3":1,"M4":1,"M5":1,"M6":.5,
        "E1":1,"E2":1,"E3":1,"E4":1,"E5":1,"E6":0,
        "V1":1,"V2":1,"V3":1,"V4":1,"V5":.5,
    },
    "IOI Circuit": {
        "C1":1,"C2":1,"C3":0,"C4":.5,"C5":.5,
        "I1":1,"I2":1,"I3":0,"I4":.5,"I5":0,
        "M1":0,"M2":.5,"M3":1,"M4":0,"M5":0,"M6":.5,
        "E1":0,"E2":.5,"E3":0,"E4":1,"E5":.5,"E6":0,
        "V1":1,"V2":1,"V3":1,"V4":.5,"V5":.5,
    },
    "Greater-Than": {
        "C1":1,"C2":1,"C3":.5,"C4":1,"C5":.5,
        "I1":1,"I2":.5,"I3":.5,"I4":.5,"I5":0,
        "M1":0,"M2":.5,"M3":1,"M4":1,"M5":0,"M6":1,
        "E1":0,"E2":.5,"E3":.5,"E4":1,"E5":.5,"E6":0,
        "V1":1,"V2":1,"V3":1,"V4":.5,"V5":1,
    },
    "Copy Suppression": {
        "C1":1,"C2":1,"C3":.5,"C4":1,"C5":.5,
        "I1":1,"I2":.5,"I3":1,"I4":.5,"I5":0,
        "M1":0,"M2":.5,"M3":1,"M4":1,"M5":0,"M6":1,
        "E1":.5,"E2":0,"E3":1,"E4":.5,"E5":.5,"E6":0,
        "V1":1,"V2":1,"V3":1,"V4":.5,"V5":1,
    },
    "Successor Heads": {
        "C1":1,"C2":1,"C3":-1,"C4":1,"C5":.5,
        "I1":1,"I2":.5,"I3":.5,"I4":.5,"I5":0,
        "M1":0,"M2":1,"M3":1,"M4":1,"M5":0,"M6":1,
        "E1":0,"E2":.5,"E3":.5,"E4":.5,"E5":1,"E6":0,
        "V1":1,"V2":1,"V3":1,"V4":.5,"V5":1,
    },
    "Docstring Circuit": {
        "C1":1,"C2":.5,"C3":0,"C4":.5,"C5":.5,
        "I1":1,"I2":.5,"I3":0,"I4":.5,"I5":0,
        "M1":0,"M2":.5,"M3":1,"M4":0,"M5":0,"M6":.5,
        "E1":0,"E2":0,"E3":0,"E4":.5,"E5":.5,"E6":0,
        "V1":1,"V2":.5,"V3":.5,"V4":0,"V5":.5,
    },
    "SAE Features": {
        "C1":.5,"C2":.5,"C3":0,"C4":.5,"C5":0,
        "I1":.5,"I2":.5,"I3":0,"I4":0,"I5":0,
        "M1":0,"M2":0,"M3":.5,"M4":0,"M5":0,"M6":0,
        "E1":.5,"E2":.5,"E3":0,"E4":.5,"E5":0,"E6":0,
        "V1":1,"V2":0,"V3":.5,"V4":0,"V5":0,
    },
    "Othello World Model": {
        "C1":1,"C2":.5,"C3":1,"C4":0,"C5":.5,
        "I1":.5,"I2":.5,"I3":.5,"I4":.5,"I5":.5,
        "M1":0,"M2":.5,"M3":.5,"M4":0,"M5":0,"M6":.5,
        "E1":1,"E2":0,"E3":0,"E4":.5,"E5":.5,"E6":0,
        "V1":1,"V2":1,"V3":.5,"V4":.5,"V5":.5,
    },
    "Knowledge Neurons": {
        "C1":1,"C2":.5,"C3":.5,"C4":.5,"C5":.5,
        "I1":1,"I2":1,"I3":.5,"I4":.5,"I5":.5,
        "M1":.5,"M2":.5,"M3":.5,"M4":0,"M5":0,"M6":.5,
        "E1":.5,"E2":0,"E3":.5,"E4":1,"E5":.5,"E6":.5,
        "V1":1,"V2":.5,"V3":.5,"V4":.5,"V5":.5,
    },
    "Probing Classifiers": {
        "C1":.5,"C2":.5,"C3":.5,"C4":-1,"C5":.5,
        "I1":0,"I2":0,"I3":0,"I4":.5,"I5":.5,
        "M1":.5,"M2":.5,"M3":0,"M4":.5,"M5":0,"M6":.5,
        "E1":0,"E2":-1,"E3":-1,"E4":-1,"E5":.5,"E6":.5,
        "V1":1,"V2":.5,"V3":.5,"V4":.5,"V5":0,
    },
    "Gender Bias Circuits": {
        "C1":.5,"C2":.5,"C3":.5,"C4":.5,"C5":.5,
        "I1":.5,"I2":0,"I3":.5,"I4":.5,"I5":.5,
        "M1":.5,"M2":.5,"M3":.5,"M4":0,"M5":0,"M6":.5,
        "E1":.5,"E2":.5,"E3":.5,"E4":.5,"E5":.5,"E6":.5,
        "V1":.5,"V2":.5,"V3":.5,"V4":.5,"V5":0,
    },
}

tier_map = {
    "Induction Heads": "Triangulated",
    "Grokking": "Triangulated",
    "Superposition": "Triangulated",
    "IOI Circuit": "Mech. Supported",
    "Greater-Than": "Mech. Supported",
    "Copy Suppression": "Mech. Supported",
    "Successor Heads": "Caus. Suggestive",
    "Docstring Circuit": "Caus. Suggestive",
    "SAE Features": "Caus. Suggestive",
    "Othello World Model": "Caus. Suggestive",
    "Knowledge Neurons": "Caus. Suggestive",
    "Probing Classifiers": "Proposed",
    "Gender Bias Circuits": "Proposed",
}

tier_colors = {
    "Proposed": "#ef4444",
    "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981",
    "Triangulated": "#3b82f6",
}

# Dimension → criteria keys in standard order
dim_criteria = [
    ["C1","C2","C3","C4","C5"],
    ["I1","I2","I3","I4","I5"],
    ["M1","M2","M3","M4","M5","M6"],
    ["E1","E2","E3","E4","E5","E6"],
    ["V1","V2","V3","V4","V5"],
]
dim_names = ["Construct", "Internal", "Measurement", "External", "Interpretive"]
circuit_order = list(criteria_raw.keys())

n_rows = len(circuit_order)
n_dims = 5

# V2 layout params
col_spacing = 1.3
row_spacing = 1.1
radius = 0.38

fig_w = 11
fig_h = n_rows * row_spacing + 3.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.set_xlim(-6.5, (n_dims - 1) * col_spacing + radius + 0.5)
ax.set_ylim(-2.0, n_rows * row_spacing + 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Column headers — black, diagonal, close to circles
for j, dim in enumerate(dim_names):
    cx = j * col_spacing
    header_y = n_rows * row_spacing - 0.15
    ax.text(cx, header_y, dim, fontsize=10, fontweight='bold', color='#1e293b',
            ha='left', va='bottom', rotation=45)

# Draw rows
for i, name in enumerate(circuit_order):
    tier = tier_map[name]
    y = (n_rows - 1 - i) * row_spacing
    fill_color = tier_colors[tier]
    cdata = criteria_raw[name]

    # Left side — v2 style: italic bold label + tier square
    square_x = -0.6
    ax.add_patch(mpatches.FancyBboxPatch(
        (square_x, y - 0.13), 0.26, 0.26,
        boxstyle='round,pad=0.02', facecolor=fill_color, edgecolor='none'))
    ax.text(square_x - 0.2, y, name, fontsize=10, ha='right', va='center',
            fontweight='bold', color='#374151', fontstyle='italic')

    for dim in range(n_dims):
        cx = dim * col_spacing
        criteria_keys = dim_criteria[dim]
        n_crit = len(criteria_keys)
        angle_span = 360 / n_crit

        for seg, ckey in enumerate(criteria_keys):
            theta1 = 90 - (seg + 1) * angle_span
            theta2 = 90 - seg * angle_span
            val = cdata[ckey]

            if val == 1:       # YES — full color
                fc = fill_color
                alpha = 0.85
            elif val == 0.5:   # PARTIAL — lighter
                fc = fill_color
                alpha = 0.3
            elif val == -1:    # NA — light gray
                fc = '#e2e8f0'
                alpha = 0.5
            else:              # NO — white
                fc = 'white'
                alpha = 1.0

            wedge = mpatches.Wedge((cx, y), radius, theta1, theta2,
                                    facecolor=fc, edgecolor='white',
                                    linewidth=1.5, alpha=alpha)
            ax.add_patch(wedge)

        circle = plt.Circle((cx, y), radius, facecolor='none',
                              edgecolor='#cbd5e1', linewidth=1.0)
        ax.add_patch(circle)

# Legend
ly = -1.1
# YES
ax.add_patch(mpatches.Wedge((-2.0, ly), 0.2, 0, 360,
             facecolor='#94a3b8', edgecolor='white', linewidth=1, alpha=0.85))
ax.text(-1.6, ly, 'YES', fontsize=9, va='center', color='#475569', fontweight='bold')

# PARTIAL
ax.add_patch(mpatches.Wedge((-.5, ly), 0.2, 0, 360,
             facecolor='#94a3b8', edgecolor='#cbd5e1', linewidth=1, alpha=0.3))
ax.text(-0.1, ly, 'PARTIAL', fontsize=9, va='center', color='#475569', fontweight='bold')

# NO
ax.add_patch(plt.Circle((1.5, ly), 0.2,
             facecolor='white', edgecolor='#cbd5e1', linewidth=1))
ax.text(1.9, ly, 'NO', fontsize=9, va='center', color='#475569', fontweight='bold')

# NA
ax.add_patch(mpatches.Wedge((3.0, ly), 0.2, 0, 360,
             facecolor='#e2e8f0', edgecolor='#cbd5e1', linewidth=1, alpha=0.5))
ax.text(3.4, ly, 'N/A', fontsize=9, va='center', color='#475569', fontweight='bold')

# Tier legend
ty = ly - 0.55
tiers = [("Proposed","#ef4444"), ("Caus. Suggestive","#f59e0b"),
         ("Mech. Supported","#10b981"), ("Triangulated","#3b82f6")]
for idx, (tname, tcol) in enumerate(tiers):
    tx = -2.0 + idx * 2.3
    ax.add_patch(mpatches.FancyBboxPatch(
        (tx - 0.1, ty - 0.1), 0.2, 0.2,
        boxstyle='round,pad=0.02', facecolor=tcol, edgecolor='none'))
    ax.text(tx + 0.2, ty, tname, fontsize=8, va='center', color='#475569')

plt.tight_layout(pad=0.5)
plt.savefig('fig_harvey_criteria_all.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved fig_harvey_criteria_all.png')
