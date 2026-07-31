"""Foundations pyramid — construct validity as wide base, interpretive as narrow top."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent

# Bottom to top: construct (widest) → interpretive (narrowest)
layers = [
    ("Philosophy of Science", "Construct Validity", '#6366f1', '#eef2ff'),
    ("Psychometrics", "Measurement Validity", '#14b8a6', '#f0fdfa'),
    ("Causal Inference  /  Neuroscience  /  Genetics", "Internal Validity", '#3b82f6', '#eff6ff'),
    ("Pharmacology", "External Validity", '#f59e0b', '#fffbeb'),
    ("Mechanistic Interpretability", "Interpretive Validity", '#ef4444', '#fef2f2'),
]

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

n = len(layers)
max_width = 9.0
min_width = 6.5
bar_h = 0.85
gap = 0.2
center_x = 5.0

for i, (field, vtype, color, light) in enumerate(layers):
    # Bottom layer (i=0) is widest, top (i=n-1) is narrowest
    frac = i / (n - 1) if n > 1 else 0
    w = max_width - frac * (max_width - min_width)
    y = i * (bar_h + gap)
    x = center_x - w / 2

    box = mpatches.FancyBboxPatch(
        (x, y), w, bar_h,
        boxstyle='round,pad=0.08', facecolor=light, edgecolor=color,
        linewidth=2.0)
    ax.add_patch(box)

    # Number on the left
    ax.text(x + 0.35, y + bar_h / 2, f"{i + 1}.",
            ha='left', va='center', fontsize=13, fontweight='900',
            color='#1e293b', fontfamily='sans-serif')

    # Validity type left-center (dark, readable)
    ax.text(x + 0.85, y + bar_h / 2, vtype,
            ha='left', va='center', fontsize=11, fontweight='bold',
            color='#1e293b', fontfamily='sans-serif')

    # Field name right-aligned (colored, bold)
    ax.text(x + w - 0.35, y + bar_h / 2, field,
            ha='right', va='center', fontsize=11, fontweight='bold',
            color=color, fontfamily='sans-serif')

# Arrow on the left side
# Curved arrow on the left following the pyramid slope
arrow_bottom_x = center_x - max_width / 2 - 0.4
arrow_bottom_y = 0 + bar_h / 2
arrow_top_x = center_x - min_width / 2 - 0.4
arrow_top_y = (n - 1) * (bar_h + gap) + bar_h / 2
ax.annotate('', xy=(arrow_top_x, arrow_top_y), xytext=(arrow_bottom_x, arrow_bottom_y),
            arrowprops=dict(arrowstyle='-|>', color='#1e293b', lw=2.5,
                            mutation_scale=18,
                            connectionstyle='arc3,rad=-0.2'))
mid_x = (arrow_bottom_x + arrow_top_x) / 2 - 0.8
mid_y = (arrow_bottom_y + arrow_top_y) / 2
ax.text(mid_x, mid_y, 'Dependency\norder',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#1e293b',
        fontfamily='sans-serif', rotation=80)

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.1)
fig.savefig(str(OUTDIR / 'foundations-pyramid.pdf'), bbox_inches='tight', pad_inches=0.05, facecolor='white')
fig.savefig(str(OUTDIR / 'foundations-pyramid.png'), bbox_inches='tight', pad_inches=0.05, dpi=300, facecolor='white')
plt.close()
print('Saved foundations-pyramid')
