"""Theoretical foundations as a stacked hierarchy — no arrows, layered bars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent

fields = [
    ("Philosophy of Science", "Construct Validity"),
    ("Causal Inference", "Internal Validity"),
    ("Neuroscience", "Internal Validity"),
    ("Psychometrics", "Measurement Validity"),
    ("Pharmacology", "External Validity"),
    ("Mechanistic Interpretability", "Interpretive Validity"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ef4444']
light = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fef2f2']

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

n = len(fields)
bar_w = 7.0
bar_h = 0.7
x_left = 0.5
gap = 0.25

for i, (field, vtype) in enumerate(fields):
    y = (n - 1 - i) * (bar_h + gap)
    box = mpatches.FancyBboxPatch(
        (x_left, y), bar_w, bar_h,
        boxstyle='round,pad=0.08', facecolor=light[i], edgecolor=colors[i],
        linewidth=2.0)
    ax.add_patch(box)

    ax.text(x_left + 0.3, y + bar_h / 2, f"{i + 1}.",
            ha='left', va='center', fontsize=14, fontweight='900',
            color=colors[i], fontfamily='sans-serif')

    ax.text(x_left + 0.8, y + bar_h / 2, field,
            ha='left', va='center', fontsize=13, fontweight='bold',
            color='#1e293b', fontfamily='sans-serif')

    ax.text(x_left + bar_w - 0.3, y + bar_h / 2, vtype,
            ha='right', va='center', fontsize=12, fontweight='bold',
            color=colors[i], fontfamily='sans-serif')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.1)
fig.savefig(str(OUTDIR / 'foundations-hierarchy.pdf'), bbox_inches='tight', pad_inches=0.05, facecolor='white')
fig.savefig(str(OUTDIR / 'foundations-hierarchy.png'), bbox_inches='tight', pad_inches=0.05, dpi=300, facecolor='white')
plt.close()
print('Saved foundations-hierarchy')
