"""Side-by-side: same data, two stories. Cherry-picking vs ClaimSpec.
Clean, minimal — just the numbers and the verdict."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_box(ax, x, y, w, h, label, color='#e2e8f0', text_color='#1e293b', fontsize=9):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.10', facecolor=color, edgecolor='#94a3b8', linewidth=1.3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, linespacing=1.3)


fig, axes = plt.subplots(1, 2, figsize=(13, 7))

# ========== LEFT: Post-hoc ==========
ax = axes[0]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(3, 7.5, 'Without ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#991b1b')

# Experiment results
draw_box(ax, 3, 6.2, 5, 0.5, 'Run 3 ablation methods on discovered circuit', color='#dbeafe', fontsize=9)

# Three results
draw_box(ax, 1, 4.8, 1.8, 0.7, 'Mean\nablation\n87%', color='#bbf7d0', fontsize=9)
draw_box(ax, 3, 4.8, 1.8, 0.7, 'Resample\nablation\n40%', color='#fecaca', fontsize=9, text_color='#94a3b8')
draw_box(ax, 5, 4.8, 1.8, 0.7, 'Zero\nablation\n65%', color='#fef3c7', fontsize=9, text_color='#94a3b8')

# Arrows from experiment to results
for rx in [1, 3, 5]:
    ax.annotate('', xy=(rx, 5.15), xytext=(rx, 5.93),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.2, shrinkA=5, shrinkB=5))

# Cherry-pick arrow
ax.annotate('', xy=(1, 3.6), xytext=(1, 4.43),
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=2.5, shrinkA=5, shrinkB=5))
ax.text(3, 3.9, 'Report only\nthe best one', fontsize=9, ha='center', color='#991b1b',
        fontstyle='italic')

# Reported result
draw_box(ax, 1, 3.1, 2.5, 0.6, '"87% faithfulness"', color='#bbf7d0', fontsize=10)

# Name roles
ax.annotate('', xy=(3, 2.0), xytext=(1.5, 2.78),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, shrinkA=5, shrinkB=5))
draw_box(ax, 3, 1.6, 3, 0.6, 'Name roles after\nobserved behavior', color='#fef3c7',
         text_color='#92400e', fontsize=9)

# Verdict
ax.annotate('', xy=(3, 0.4), xytext=(3, 1.28),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, shrinkA=5, shrinkB=5))
draw_box(ax, 3, -0.1, 2.5, 0.6, '"Confirmed"', color='#bbf7d0', fontsize=12)
ax.text(3, -0.7, 'No prediction could\nhave failed', fontsize=8, ha='center',
        color='#991b1b', fontstyle='italic')


# ========== RIGHT: With ClaimSpec ==========
ax = axes[1]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(3, 7.5, 'With ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#16a34a')

# Pre-registered predictions
draw_box(ax, 3, 6.2, 5.2, 0.5, 'Pre-register: faithfulness >70%\nunder all 3 methods',
         color='#dbeafe', fontsize=9)

# Same three results
draw_box(ax, 1, 4.8, 1.8, 0.7, 'Mean\nablation\n87%  ✓', color='#bbf7d0', fontsize=9)
draw_box(ax, 3, 4.8, 1.8, 0.7, 'Resample\nablation\n40%  ✗', color='#fecaca', fontsize=9)
draw_box(ax, 5, 4.8, 1.8, 0.7, 'Zero\nablation\n65%  ✗', color='#fecaca', fontsize=9)

for rx in [1, 3, 5]:
    ax.annotate('', xy=(rx, 5.15), xytext=(rx, 5.93),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.2, shrinkA=5, shrinkB=5))

# All results matter
ax.annotate('', xy=(3, 3.6), xytext=(3, 4.43),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2.5, shrinkA=5, shrinkB=5))
ax.text(3, 3.95, '2 of 3 methods\nfail threshold', fontsize=9, ha='center', color='#ef4444',
        fontweight='bold')

# Diagnosis
draw_box(ax, 3, 3.1, 3.5, 0.6, 'E1 Intervention reach: FAIL\nResult is method-conditional',
         color='#fecaca', text_color='#991b1b', fontsize=9)

# What to do next
ax.annotate('', xy=(3, 2.0), xytext=(3, 2.78),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, shrinkA=5, shrinkB=5))
draw_box(ax, 3, 1.6, 3.5, 0.6, 'Path forward: investigate why\nresample ablation fails',
         color='#dbeafe', fontsize=9)

# Verdict
ax.annotate('', xy=(3, 0.4), xytext=(3, 1.28),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, shrinkA=5, shrinkB=5))
draw_box(ax, 3, -0.1, 2.8, 0.6, '"Causally Suggestive"', color='#fef3c7',
         text_color='#92400e', fontsize=11)
ax.text(3, -0.7, 'The spec told us exactly\nwhat failed and why', fontsize=8, ha='center',
        color='#16a34a', fontstyle='italic')


plt.tight_layout(w_pad=2)
out = OUTDIR / 'fig_posthoc_vs_claimspec_v3.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
