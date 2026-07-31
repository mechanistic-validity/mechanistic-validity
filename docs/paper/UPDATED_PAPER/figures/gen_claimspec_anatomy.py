"""Anatomy of a ClaimSpec graph — reference figure showing all edge types."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_node(ax, x, y, label, color='#e2e8f0', text_color='#1e293b', fontsize=9,
              width=1.8, height=0.6):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.10', facecolor=color, edgecolor='#94a3b8', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, linespacing=1.2)


def draw_edge(ax, x1, y1, x2, y2, color='#16a34a', linewidth=2.0, linestyle='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=12, shrinkB=12))


def add_label(ax, x1, y1, x2, y2, text, color='#475569', fontsize=7, offset=(0, 0.2)):
    mid_x = (x1 + x2) / 2 + offset[0]
    mid_y = (y1 + y2) / 2 + offset[1]
    ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))


fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(-2, 12)
ax.set_ylim(-2.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(5, 8.0, 'Anatomy of a ClaimSpec Graph', fontsize=16, fontweight='bold',
        ha='center', color='#0f172a')

# ---- Nodes ----
draw_node(ax, 2, 6, 'Role A\n(e.g., Detector)', color='#dbeafe', width=2.0)
draw_node(ax, 8, 6, 'Role B\n(e.g., Inhibitor)', color='#dbeafe', width=2.0)
draw_node(ax, 5, 3, 'Role C\n(e.g., Mover)', color='#dbeafe', width=2.0)
draw_node(ax, 5, 0.5, 'Output\nbehavior', color='#f1f5f9', width=1.8)

# ---- Edge 1: A → C (positive prediction, PASSED) ----
draw_edge(ax, 2.7, 5.6, 4.3, 3.4, color='#16a34a')
add_label(ax, 2.7, 5.6, 4.3, 3.4, '"Ablate A → C activation\ndecreases by ≥0.2"  ✓',
          color='#16a34a', offset=(-0.8, 0.1))

# ---- Edge 2: B → C (positive prediction, PASSED) ----
draw_edge(ax, 7.3, 5.6, 5.7, 3.4, color='#16a34a')
add_label(ax, 7.3, 5.6, 5.7, 3.4, '"Ablate B → C activation\ndecreases by ≥0.3"  ✓',
          color='#16a34a', offset=(0.8, 0.1))

# ---- Edge 3: C → Output (positive prediction, PASSED) ----
draw_edge(ax, 5, 2.7, 5, 0.82, color='#16a34a')
add_label(ax, 5, 2.7, 5, 0.82, '"Ablate C → output\ndegrades by ≥0.5"  ✓',
          color='#16a34a', offset=(1.6, 0))

# ---- Non-edge 4: C ✗→ A (negative control, PASSED — independence confirmed) ----
draw_edge(ax, 4.3, 3.4, 2.7, 5.6, color='#16a34a', linestyle='--', linewidth=1.5)
add_label(ax, 4.3, 3.4, 2.7, 5.6, '"Ablate C → A activation\nchanges by <0.05"  ✓',
          color='#16a34a', offset=(-1.8, -0.2))

# ---- Non-edge 5: C ✗→ B (negative control, UNTESTED) ----
draw_edge(ax, 5.7, 3.4, 7.3, 5.6, color='#f59e0b', linestyle='--', linewidth=1.5)
mid_x5 = (5.7 + 7.3) / 2
mid_y5 = (3.4 + 5.6) / 2
ax.text(mid_x5 + 0.15, mid_y5 + 0.15, '?', fontsize=13, ha='center', va='center',
        color='#f59e0b', fontweight='bold')
add_label(ax, 5.7, 3.4, 7.3, 5.6, '"Ablate C → B activation\nchanges by <0.05"  ?',
          color='#f59e0b', offset=(1.8, -0.2))

# ---- Non-edge 6: A ✗→ Output directly (negative control, FAILED) ----
draw_edge(ax, 2, 5.38, 4.2, 0.82, color='#ef4444', linestyle='--', linewidth=1.5)
mid_x6 = (2 + 4.2) / 2
mid_y6 = (5.38 + 0.82) / 2
ax.text(mid_x6 + 0.15, mid_y6 + 0.15, '✗', fontsize=13, ha='center', va='center',
        color='#ef4444', fontweight='bold')
add_label(ax, 2, 5.38, 4.2, 0.82, '"Ablate A → output does\nNOT degrade directly"  ✗',
          color='#ef4444', offset=(-1.6, 0))

# ---- Legend ----
legend_y = -1.5
legend_items = [
    (0, '━━━', '#16a34a', 'Positive prediction (passed)'),
    (3, '╌╌╌ ✓', '#16a34a', 'Negative control (passed — independence confirmed)'),
    (7, '╌╌╌ ?', '#f59e0b', 'Negative control (untested)'),
    (10, '╌╌╌ ✗', '#ef4444', 'Negative control (failed)'),
]
for lx, sym, col, desc in legend_items:
    ax.text(lx, legend_y, sym, fontsize=10, color=col, fontweight='bold', va='center')
    ax.text(lx + 0.6, legend_y, desc, fontsize=8, color='#475569', va='center')

# ---- Verdict box ----
verdict_box = mpatches.FancyBboxPatch(
    (7.5, -0.2), 3.5, 1.4,
    boxstyle='round,pad=0.15', facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=1.5)
ax.add_patch(verdict_box)
ax.text(9.25, 0.8, 'Verdict', fontsize=9, fontweight='bold', ha='center', color='#92400e')
ax.text(9.25, 0.35, 'Causally Suggestive', fontsize=10, fontweight='bold', ha='center', color='#f59e0b')
ax.text(9.25, -0.05, 'One non-edge untested (I3)\nOne non-edge failed (specificity)', fontsize=7,
        ha='center', color='#92400e')

plt.tight_layout()
out = OUTDIR / 'fig_claimspec_anatomy_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
