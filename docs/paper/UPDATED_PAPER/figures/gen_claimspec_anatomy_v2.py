"""Anatomy of a ClaimSpec graph v2 — curved edges, clear roles, no overlap."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
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


def draw_straight(ax, x1, y1, x2, y2, color='#16a34a', linewidth=2.0, linestyle='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=14, shrinkB=14))


def draw_curved(ax, x1, y1, x2, y2, color='#f59e0b', linewidth=1.5, linestyle='--',
                connectionstyle='arc3,rad=0.3'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=14, shrinkB=14,
                                connectionstyle=connectionstyle))


def add_label(ax, x, y, text, color='#475569', fontsize=7.5):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9))


fig, ax = plt.subplots(figsize=(11, 9))
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 8.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(5, 8.0, 'Anatomy of a ClaimSpec Graph', fontsize=16, fontweight='bold',
        ha='center', color='#0f172a')

# ---- Nodes ----
draw_node(ax, 2.5, 5.5, 'Role A\nFeature Detector', color='#dbeafe', width=2.2)
draw_node(ax, 7.5, 5.5, 'Role B\nCopier', color='#dbeafe', width=2.0)
draw_node(ax, 5, 2.5, 'Task Output\n(e.g., IOI)', color='#f1f5f9', width=2.0)
draw_node(ax, 9.5, 2.5, 'Unrelated Task\n(e.g., SVA)', color='#f1f5f9', width=2.0, text_color='#94a3b8')

# ---- Edge 1: A → B (positive, passed) ----
draw_straight(ax, 3.6, 5.5, 6.5, 5.5, color='#16a34a')
add_label(ax, 5, 5.9, '"Ablate A → B activation\ndecreases by ≥0.2"  ✓', color='#16a34a')

# ---- Edge 2: B → Output (positive, passed) ----
draw_straight(ax, 7.5, 5.17, 5.5, 2.83, color='#16a34a')
add_label(ax, 7.2, 3.9, '"Ablate B → task output\ndegrades by ≥0.5"  ✓', color='#16a34a')

# ---- Edge 3: A → Output (positive, passed) ----
draw_straight(ax, 2.5, 5.17, 4.5, 2.83, color='#16a34a')
add_label(ax, 2.8, 3.9, '"Ablate A → task output\ndegrades by ≥0.3"  ✓', color='#16a34a')

# ---- Non-edge 4: B → A (negative control, PASSED) — curved to avoid overlap with edge 1 ----
draw_curved(ax, 6.5, 5.3, 3.6, 5.3, color='#16a34a', linestyle='--',
            connectionstyle='arc3,rad=-0.4')
add_label(ax, 5, 4.4, '"Ablate B → A unchanged"  ✓', color='#16a34a')

# ---- Non-edge 5: A → Unrelated (negative control, UNTESTED) — curved ----
draw_curved(ax, 3.2, 5.17, 8.5, 2.83, color='#f59e0b', linestyle='--',
            connectionstyle='arc3,rad=-0.25')
ax.text(6.5, 3.0, '?', fontsize=14, ha='center', va='center',
        color='#f59e0b', fontweight='bold')
add_label(ax, 6.5, 2.5, '"Ablate A → SVA\nunchanged"  ?', color='#f59e0b')

# ---- Non-edge 6: B → Unrelated (negative control, FAILED) ----
draw_straight(ax, 7.8, 5.17, 9.2, 2.83, color='#ef4444', linestyle='--', linewidth=1.5)
ax.text(8.8, 4.2, '✗', fontsize=14, ha='center', va='center',
        color='#ef4444', fontweight='bold')
add_label(ax, 9.8, 4.0, '"Ablate B → SVA\nunchanged"  ✗\nSVA also degrades!', color='#ef4444')

# ---- Legend ----
legend_items = [
    (-0.5, -1.2, '━━━', '#16a34a', 'Positive prediction (passed)'),
    (2.8, -1.2, '╌╌╌ ✓', '#16a34a', 'Negative control (passed)'),
    (6.0, -1.2, '╌╌╌ ?', '#f59e0b', 'Untested non-edge'),
    (8.8, -1.2, '╌╌╌ ✗', '#ef4444', 'Failed non-edge'),
]
for lx, ly, sym, col, desc in legend_items:
    ax.text(lx, ly, sym, fontsize=10, color=col, fontweight='bold', va='center')
    ax.text(lx + 0.55, ly, desc, fontsize=8, color='#475569', va='center')

# ---- Verdict box ----
verdict_box = mpatches.FancyBboxPatch(
    (0, -0.5), 3.0, 1.2,
    boxstyle='round,pad=0.12', facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=1.5)
ax.add_patch(verdict_box)
ax.text(1.5, 0.35, 'Verdict: Causally Suggestive', fontsize=9, fontweight='bold',
        ha='center', color='#92400e')
ax.text(1.5, -0.05, 'I3 untested (A→SVA)\nI3 failed (B breaks SVA)', fontsize=7,
        ha='center', color='#92400e')

plt.tight_layout()
out = OUTDIR / 'fig_claimspec_anatomy_v2.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
