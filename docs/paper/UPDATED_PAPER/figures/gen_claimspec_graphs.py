"""Generate ClaimSpec graph visualizations for IOI, Knowledge Neurons, and Gender Bias."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_node(ax, x, y, label, color='#e2e8f0', text_color='#1e293b', fontsize=9, width=1.6, height=0.5):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.12', facecolor=color, edgecolor='#94a3b8', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color)


def draw_edge(ax, x1, y1, x2, y2, color='#475569', style='->', linewidth=1.8, linestyle='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=12, shrinkB=12))


def draw_non_edge(ax, x1, y1, x2, y2, color='#ef4444', linewidth=1.5):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle='--', shrinkA=12, shrinkB=12))
    ax.text(mid_x, mid_y + 0.15, '✗', fontsize=14, ha='center', va='center',
            color=color, fontweight='bold')


# ========== 1. IOI Circuit ==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-1.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(4.5, 5.2, 'IOI Circuit — ClaimSpec Graph', fontsize=14, fontweight='bold',
        ha='center', color='#0f172a')
ax.text(4.5, 4.7, '6 roles, 7 positive predictions, 5 negative controls',
        fontsize=10, ha='center', color='#64748b')

# Nodes
draw_node(ax, 1.0, 3.5, 'DTH\n(Duplicate\nToken Det.)', color='#dbeafe', height=0.8)
draw_node(ax, 4.0, 3.5, 'PTH\n(Previous\nToken)', color='#dbeafe', height=0.8)
draw_node(ax, 2.5, 2.0, 'IND\n(Induction)', color='#dbeafe')
draw_node(ax, 5.5, 2.0, 'S-Inh\n(S-Inhibition)', color='#dbeafe')
draw_node(ax, 7.0, 0.5, 'NM\n(Name Mover)', color='#bbf7d0')
draw_node(ax, 3.5, 0.5, 'NegNM\n(Neg. Name\nMover)', color='#fecaca', height=0.8)

# Positive edges (solid)
draw_edge(ax, 1.0, 3.1, 2.5, 2.3)
draw_edge(ax, 4.0, 3.1, 2.5, 2.3)
draw_edge(ax, 2.5, 1.75, 5.5, 2.0)
draw_edge(ax, 5.5, 1.75, 7.0, 0.75)

# Non-edges (orange = untested)
draw_non_edge(ax, 7.0, 0.75, 5.5, 1.75, color='#f59e0b')  # NM -/-> S-Inh
draw_non_edge(ax, 3.5, 0.9, 5.5, 1.75, color='#f59e0b')   # NegNM -/-> S-Inh

# Legend
legend_y = -0.8
ax.annotate('', xy=(1.5, legend_y), xytext=(0.5, legend_y),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))
ax.text(1.7, legend_y, 'Positive prediction (edge)', fontsize=9, va='center', color='#475569')

ax.annotate('', xy=(5.5, legend_y), xytext=(4.5, legend_y),
            arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.5, linestyle='--'))
ax.text(4.9, legend_y + 0.15, '?', fontsize=12, ha='center', color='#f59e0b', fontweight='bold')
ax.text(5.7, legend_y, 'Untested non-edge', fontsize=9, va='center', color='#f59e0b')

ax.annotate('', xy=(8.5, legend_y), xytext=(7.5, legend_y),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5, linestyle='--'))
ax.text(7.9, legend_y + 0.15, '✗', fontsize=12, ha='center', color='#ef4444', fontweight='bold')
ax.text(8.7, legend_y, 'Failed non-edge', fontsize=9, va='center', color='#ef4444')

plt.tight_layout()
fig.savefig(str(OUTDIR / 'fig_claimspec_ioi_v2.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_claimspec_ioi_v2.png')


# ========== 2. Knowledge Neurons ==========
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(-1.5, 9.5)
ax.set_ylim(-1.5, 4.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(4, 4.2, 'Knowledge Neurons — ClaimSpec Graph', fontsize=14, fontweight='bold',
        ha='center', color='#0f172a')
ax.text(4, 3.7, 'Single node, strong intervention, untested specificity',
        fontsize=10, ha='center', color='#64748b')

# Central node
draw_node(ax, 4, 2.8, 'Neuron 3847\n(Layer 17)', color='#bbf7d0', width=2.0, height=0.7)

# Positive prediction — straight down
draw_node(ax, 4, 0.8, '"Eiffel Tower\nis in ___"', color='#dbeafe', width=2.0, height=0.6, fontsize=8)
draw_edge(ax, 4, 2.45, 4, 1.1, color='#16a34a')
ax.text(3.7, 1.7, 'Ablate → recall\ndegraded  ✓', fontsize=8, color='#16a34a', ha='right')

# Non-edges — spread out, arrows from box edges not center
targets = [
    (0.0, 2.8, '"Unrelated\nfact recall"',   3.0, 2.8, 0.9, 2.8),   # left of neuron → left box
    (8.0, 2.8, '"Other MLP\nneurons"',        5.0, 2.8, 7.1, 2.8),   # right of neuron → right box
    (0.5, 0.8, '"Capital of\nGermany is ___"', 3.0, 2.45, 1.3, 1.1),  # bottom-left
    (7.5, 0.8, '"General\nfluency"',           5.0, 2.45, 6.7, 1.1),  # bottom-right
]
for tx, ty, tlabel, ax1, ay1, ax2, ay2 in targets:
    draw_node(ax, tx, ty, tlabel, color='#f1f5f9', width=1.6, height=0.55, fontsize=7, text_color='#94a3b8')
    draw_non_edge(ax, ax1, ay1, ax2, ay2, color='#f59e0b')

ax.text(4, -0.5, 'Non-edges untested: is this a "knowledge neuron" or just\n'
        'a "middle-layer neuron that disrupts everything when ablated"?',
        fontsize=9, ha='center', color='#f59e0b', fontstyle='italic')

plt.tight_layout()
fig.savefig(str(OUTDIR / 'fig_claimspec_knowledge_v2.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_claimspec_knowledge_v2.png')


# ========== 3. Gender Bias Circuits ==========
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1, 9)
ax.set_ylim(-1.5, 4.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(4, 4.2, 'Gender Bias Circuits — ClaimSpec Graph', fontsize=14, fontweight='bold',
        ha='center', color='#0f172a')
ax.text(4, 3.7, 'Construct incoherence: the required non-edge does not exist',
        fontsize=10, ha='center', color='#64748b')

# Two nodes that should be separable
draw_node(ax, 1.5, 2.5, '"Gender bias"\ncircuit', color='#fecaca', width=2.0, height=0.7)
draw_node(ax, 6.5, 2.5, '"Gender knowledge"\ncircuit', color='#bbf7d0', width=2.2, height=0.7)

# The required non-edge — between right edge of bias and left edge of knowledge
draw_non_edge(ax, 2.5, 2.5, 5.4, 2.5, color='#ef4444')

ax.text(4.0, 3.2, 'Required: separable\n(ablate bias ≠ ablate knowledge)',
        fontsize=9, ha='center', color='#ef4444')

# But they're the same heads — arrows from bottom edges of each box to top of same-heads box
draw_node(ax, 4.0, 0.5, 'Same heads\n(7.8, 8.11, 9.6, ...)', color='#fef3c7', width=2.4, height=0.7)
draw_edge(ax, 1.5, 2.15, 3.2, 0.85)
draw_edge(ax, 6.5, 2.15, 4.8, 0.85)

ax.text(4, -0.5, 'The claim graph collapses: you cannot ablate "bias" without\n'
        'ablating "knowledge." The construct itself is incoherent (C4 failure).',
        fontsize=9, ha='center', color='#ef4444', fontstyle='italic')

plt.tight_layout()
fig.savefig(str(OUTDIR / 'fig_claimspec_gender_v2.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_claimspec_gender_v2.png')
