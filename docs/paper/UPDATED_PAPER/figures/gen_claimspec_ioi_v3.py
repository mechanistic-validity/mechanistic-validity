"""IOI Circuit ClaimSpec graph v3 — fixed arrows, curved non-edges, no overlap."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def draw_edge(ax, x1, y1, x2, y2, color='#475569', linewidth=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                shrinkA=12, shrinkB=12))


def draw_curved(ax, x1, y1, x2, y2, color='#f59e0b', linewidth=1.5, linestyle='--',
                rad=0.3, symbol='?'):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=12, shrinkB=12,
                                connectionstyle=f'arc3,rad={rad}'))
    offset_x = rad * (y2 - y1) * 0.3
    offset_y = -rad * (x2 - x1) * 0.3
    ax.text(mid_x + offset_x, mid_y + offset_y + 0.15, symbol, fontsize=12,
            ha='center', va='center', color=color, fontweight='bold')


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-1, 11)
ax.set_ylim(-1.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(5, 6.2, 'IOI Circuit — ClaimSpec Graph', fontsize=15, fontweight='bold',
        ha='center', color='#0f172a')
ax.text(5, 5.7, '6 roles, 7 positive predictions, 5 negative controls',
        fontsize=10, ha='center', color='#64748b')

# Nodes — spread out more, 3 tiers
# Top tier: DTH, PTH
draw_node(ax, 1.5, 4.5, 'DTH\n(Duplicate\nToken Det.)', color='#dbeafe', height=0.8, width=2.0)
draw_node(ax, 5.5, 4.5, 'PTH\n(Previous\nToken)', color='#dbeafe', height=0.8, width=2.0)

# Middle tier: IND, S-Inh
draw_node(ax, 3.0, 2.5, 'IND\n(Induction)', color='#dbeafe', width=1.8)
draw_node(ax, 7.5, 2.5, 'S-Inh\n(S-Inhibition)', color='#dbeafe', width=2.0)

# Bottom tier: NegNM, NM
draw_node(ax, 2.0, 0.5, 'NegNM\n(Neg. Name\nMover)', color='#fecaca', height=0.8, width=2.0)
draw_node(ax, 8.5, 0.5, 'NM\n(Name Mover)', color='#bbf7d0', width=2.0)

# === Positive edges (solid dark) ===
# DTH → IND
draw_edge(ax, 1.5, 4.1, 2.7, 2.82)
# PTH → IND
draw_edge(ax, 5.5, 4.1, 3.3, 2.82)
# IND → S-Inh
draw_edge(ax, 3.9, 2.5, 6.5, 2.5)
# S-Inh → NM
draw_edge(ax, 8.0, 2.18, 8.5, 0.82)

# === Untested non-edges (orange dashed, curved to avoid overlap) ===
# NM → S-Inh (curved right, away from the positive S-Inh → NM edge)
draw_curved(ax, 9.0, 0.82, 8.2, 2.18, color='#f59e0b', rad=-0.5, symbol='?')

# NegNM → S-Inh (curved left)
draw_curved(ax, 2.8, 0.9, 6.8, 2.18, color='#f59e0b', rad=-0.3, symbol='?')

# === Legend ===
legend_y = -1.0
ax.annotate('', xy=(1.5, legend_y), xytext=(0.5, legend_y),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))
ax.text(1.7, legend_y, 'Positive prediction (edge)', fontsize=9, va='center', color='#475569')

ax.annotate('', xy=(5.5, legend_y), xytext=(4.5, legend_y),
            arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.5, linestyle='--'))
ax.text(4.9, legend_y + 0.15, '?', fontsize=11, ha='center', color='#f59e0b', fontweight='bold')
ax.text(5.7, legend_y, 'Untested non-edge', fontsize=9, va='center', color='#f59e0b')

ax.annotate('', xy=(9.0, legend_y), xytext=(8.0, legend_y),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5, linestyle='--'))
ax.text(8.4, legend_y + 0.15, '✗', fontsize=11, ha='center', color='#ef4444', fontweight='bold')
ax.text(9.2, legend_y, 'Failed non-edge', fontsize=9, va='center', color='#ef4444')

plt.tight_layout()
out = OUTDIR / 'fig_claimspec_ioi_v3.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
