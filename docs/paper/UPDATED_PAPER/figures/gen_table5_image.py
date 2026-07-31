#!/usr/bin/env python3
"""
Generate an image rendering of Table 5 with wrapped text for legibility.
Run: python3 gen_table5_image.py
Outputs: fig_table5_matplotlib.png in the same folder.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import os

# --- Table content (update as needed to match Table 5) ---
cols = ["Category", "Description"]
rows = [
    ("Causal Inference", "Includes ATE/CATE, transportability, counterfactual checks, mechanism adjudication"),
    ("Neuroscience", "Perturbation quant., circuit mapping, functional localization, circuit dissection"),
    ("Psychometrics", "Reliability, measurement invariance, effect size estimation, construct validity"),
    ("Pharmacology", "Dose-response, target discovery, phase III generalization"),
    ("Philosophy", "Abduction, novel predictions, crucial experiments"),
]

# Wrap long text for nicer table layout in the image
max_width = 40
wrapped_rows = [(r[0], textwrap.fill(r[1], width=max_width)) for r in rows]
cell_text = [[r[0], r[1]] for r in wrapped_rows]

nrows = len(cell_text)

# Figure sizing: width fixed, height scales with rows
fig_width = 9
row_height = 0.6
fig_height = max(2.0, row_height * (nrows + 1))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# Create table; use matplotlib.table for full control
table = ax.table(cellText=cell_text, colLabels=cols, cellLoc='left', colLoc='left', loc='center')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)

# Scale columns roughly and add borders
table.scale(1, 1.2)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.6)
    # Header style
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f7f7f7')

# Tight layout and save
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig_table5_matplotlib.png")
plt.savefig(out, dpi=200, bbox_inches='tight')
print("Saved", out)
