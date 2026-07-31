"""Three-panel ClaimSpec figure: Model → Claimed Mechanism → ClaimSpec graph.
Uses Knowledge Neurons as the example because the failure is concrete and
the community already agrees something is wrong."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_node(ax, x, y, label, color='#e2e8f0', text_color='#1e293b', fontsize=9,
              width=1.8, height=0.55):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.10', facecolor=color, edgecolor='#94a3b8', linewidth=1.3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, linespacing=1.3)


def draw_edge(ax, x1, y1, x2, y2, color='#475569', linewidth=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                shrinkA=10, shrinkB=10))


def draw_non_edge(ax, x1, y1, x2, y2, color='#ef4444', linewidth=1.5, symbol='✗'):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle='--', shrinkA=10, shrinkB=10))
    ax.text(mid_x, mid_y + 0.18, symbol, fontsize=13, ha='center', va='center',
            color=color, fontweight='bold')


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# ========== Panel A: Model Architecture ==========
ax = axes[0]
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A.  Model Architecture', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

draw_node(ax, 2, 3.2, 'Input\nembedding', color='#f1f5f9', width=1.6)
draw_node(ax, 2, 2.0, 'MLP Layer 17', color='#e2e8f0', width=1.6)
draw_node(ax, 2, 0.8, 'Output', color='#f1f5f9', width=1.6)
draw_edge(ax, 2, 2.92, 2, 2.28)
draw_edge(ax, 2, 1.72, 2, 1.08)

ax.text(2, -0.2, 'No interpretation.\nJust architecture.', fontsize=8, ha='center',
        color='#94a3b8', fontstyle='italic')

# ========== Panel B: Claimed Mechanism ==========
ax = axes[1]
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('B.  Claimed Mechanism', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

draw_node(ax, 2, 3.2, '"Knowledge\nneuron" 3847', color='#bbf7d0', width=1.8)
draw_node(ax, 2, 1.5, '"Eiffel Tower\nis in ___"', color='#dbeafe', width=1.8, fontsize=8)
draw_edge(ax, 2, 2.92, 2, 1.78, color='#16a34a')

ax.text(2.8, 2.35, 'Stores this\nspecific fact', fontsize=8, color='#16a34a', ha='left')
ax.text(2, -0.2, 'The narrative:\nthis neuron stores this fact.', fontsize=8, ha='center',
        color='#94a3b8', fontstyle='italic')

# ========== Panel C: ClaimSpec Graph ==========
ax = axes[2]
ax.set_xlim(-1.0, 5.0)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('C.  ClaimSpec Graph', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

draw_node(ax, 2, 3.2, '"Knowledge\nneuron" 3847', color='#bbf7d0', width=1.8)

# Positive prediction (green) — left
draw_node(ax, 0.3, 1.5, '"Eiffel Tower\nis in ___"', color='#dbeafe', width=1.6, fontsize=7)
draw_edge(ax, 1.3, 2.92, 0.5, 1.78, color='#16a34a')
ax.text(-0.3, 2.4, 'Edit →\nchanges ✓', fontsize=7, color='#16a34a', ha='center')

# Failed non-edge (red) — right
draw_node(ax, 3.7, 1.5, '"What country\nis the Eiffel\nTower in?"', color='#fecaca', width=1.6,
          fontsize=7, height=0.7)
draw_non_edge(ax, 2.7, 2.92, 3.5, 1.88, color='#ef4444')
ax.text(4.3, 2.4, 'Edit →\nalso changes ✗', fontsize=7, color='#ef4444', ha='center')

ax.text(2, -0.2, 'The non-edge fails: editing one\nfact corrupts related facts.', fontsize=8,
        ha='center', color='#ef4444', fontstyle='italic')

# Bottom legend spanning all panels
fig.text(0.5, 0.02,
         '━━  Positive prediction (confirmed)          '
         '╌╌  Non-edge (failed — ripple effects refute the "stores one fact" claim)',
         ha='center', fontsize=9, color='#475569')

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = OUTDIR / 'fig_claimspec_threepanel_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
