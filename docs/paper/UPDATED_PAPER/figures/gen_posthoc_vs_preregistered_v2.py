"""Two-row horizontal figure: Post-hoc narrative vs Pre-registered ClaimSpec.
Same visual language as other ClaimSpec figures — nodes, edges, non-edges."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_node(ax, x, y, label, color='#e2e8f0', text_color='#1e293b', fontsize=8,
              width=1.6, height=0.55):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.10', facecolor=color, edgecolor='#94a3b8', linewidth=1.3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, linespacing=1.2)


def draw_edge(ax, x1, y1, x2, y2, color='#475569', linewidth=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                shrinkA=8, shrinkB=8))


def draw_non_edge(ax, x1, y1, x2, y2, color='#ef4444', linewidth=1.5, symbol='✗'):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle='--', shrinkA=8, shrinkB=8))
    ax.text(mid_x, mid_y + 0.22, symbol, fontsize=11, ha='center', va='center',
            color=color, fontweight='bold')


fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# ========== Top row: Post-hoc ==========
ax = axes[0]
ax.set_xlim(-1, 15)
ax.set_ylim(-0.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(-0.5, 1.5, 'Post-hoc narrative', fontsize=13, fontweight='bold', color='#991b1b', va='center')

draw_node(ax, 1.5, 0.5, 'Run\nexperiment', color='#dbeafe', width=1.5)
draw_edge(ax, 2.25, 0.5, 3.05, 0.5)

draw_node(ax, 3.8, 0.5, 'Ablation\ndegrades\nbehavior', color='#dbeafe', width=1.5, height=0.65)
draw_edge(ax, 4.55, 0.5, 5.35, 0.5)

draw_node(ax, 6.1, 0.5, 'Try 3 methods;\nreport best\n(87%)', color='#fef3c7', text_color='#92400e', width=1.5, height=0.65)
draw_edge(ax, 6.85, 0.5, 7.65, 0.5)

draw_node(ax, 8.4, 0.5, 'Name roles\nafter what\nthey do', color='#fef3c7', text_color='#92400e', width=1.5, height=0.65)
draw_edge(ax, 9.15, 0.5, 9.95, 0.5)

draw_node(ax, 10.7, 0.5, 'Never test\noff-target\neffects', color='#fecaca', text_color='#991b1b', width=1.5, height=0.65)
draw_edge(ax, 11.45, 0.5, 12.25, 0.5)

draw_node(ax, 13.0, 0.5, '"Confirmed"', color='#bbf7d0', width=1.5, fontsize=10)

ax.text(13.0, -0.2, 'No prediction\ncould have failed', fontsize=7, ha='center', color='#991b1b', fontstyle='italic')

# ========== Bottom row: Pre-registered ==========
ax = axes[1]
ax.set_xlim(-1, 15)
ax.set_ylim(-0.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(-0.5, 1.5, 'Pre-registered ClaimSpec', fontsize=13, fontweight='bold', color='#16a34a', va='center')

draw_node(ax, 1.5, 0.5, 'Write\nClaimSpec\nwith controls', color='#dbeafe', width=1.5, height=0.65)
draw_edge(ax, 2.25, 0.5, 3.05, 0.5)

draw_node(ax, 3.8, 0.5, 'Commit:\n>70% under\n≥2 methods', color='#dbeafe', width=1.5, height=0.65)
draw_edge(ax, 4.55, 0.5, 5.35, 0.5)

draw_node(ax, 6.1, 0.5, 'Run same\nexperiment', color='#e2e8f0', width=1.5)
draw_edge(ax, 6.85, 0.5, 7.65, 0.5)

draw_node(ax, 8.4, 0.5, 'Mean: 87% ✓\nResample:\n40% ✗', color='#bbf7d0', width=1.5, height=0.65, fontsize=7)
ax.text(8.4, -0.15, 'E1 fails', fontsize=7, color='#ef4444', ha='center', fontweight='bold')
draw_edge(ax, 9.15, 0.5, 9.95, 0.5)

draw_node(ax, 10.7, 0.5, 'Method-\nconditional:\nnot robust', color='#fecaca', text_color='#991b1b', width=1.5, height=0.65)
draw_edge(ax, 11.45, 0.5, 12.25, 0.5)

draw_node(ax, 13.0, 0.5, '"Causally\nSuggestive"', color='#fef3c7', text_color='#92400e', width=1.5, height=0.55, fontsize=9)

ax.text(13.0, -0.2, 'Spec told us exactly\nwhat failed and why', fontsize=7, ha='center', color='#16a34a', fontstyle='italic')

plt.tight_layout(h_pad=1.5)
out = OUTDIR / 'fig_posthoc_vs_preregistered_v2.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
