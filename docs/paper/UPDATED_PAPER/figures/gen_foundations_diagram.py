"""Theoretical foundations diagram — six fields mapped to five validity types."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent

fields = [
    ("Philosophy\nof Science", "Construct\nValidity"),
    ("Causal\nInference", "Internal\nValidity"),
    ("Neuroscience", "Internal\nValidity"),
    ("Psychometrics", "Measurement\nValidity"),
    ("Pharmacology", "External\nValidity"),
    ("Mechanistic\nInterpretability", "Interpretive\nValidity"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ef4444']
light = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fef2f2']

fig, ax = plt.subplots(figsize=(18, 4.2))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-1.2, 1.7)
ax.set_aspect('equal')
ax.axis('off')

n = len(fields)
spacing = 11.0 / (n - 1)
x_positions = [0.5 + i * spacing for i in range(n)]
cy = 0.5
r = 0.48

for i, (field, vtype) in enumerate(fields):
    x = x_positions[i]
    circle = plt.Circle((x, cy), r, facecolor=light[i], edgecolor=colors[i],
                         linewidth=2.5, zorder=2)
    ax.add_patch(circle)
    ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
            fontsize=22, fontweight='900', color=colors[i], zorder=3,
            fontfamily='sans-serif')

    ax.text(x, cy + r + 0.15, field, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#1e293b', fontfamily='sans-serif',
            linespacing=1.2)

    ax.text(x, cy - r - 0.15, vtype, ha='center', va='top',
            fontsize=11, fontweight='bold', color=colors[i], fontfamily='sans-serif',
            linespacing=1.2)

    if i < n - 1:
        x_next = x_positions[i + 1]
        ax.annotate('', xy=(x_next - r - 0.08, cy), xytext=(x + r + 0.08, cy),
                    arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5,
                                    mutation_scale=18, shrinkA=0, shrinkB=0))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.1)
fig.savefig(str(OUTDIR / 'foundations-diagram.pdf'), bbox_inches='tight', pad_inches=0.05, facecolor='white')
fig.savefig(str(OUTDIR / 'foundations-diagram.png'), bbox_inches='tight', pad_inches=0.05, dpi=300, facecolor='white')
plt.close()
print('Saved foundations-diagram')
