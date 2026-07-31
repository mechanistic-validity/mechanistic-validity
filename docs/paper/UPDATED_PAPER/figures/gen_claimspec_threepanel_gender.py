"""Three-panel ClaimSpec figure for Gender Bias Circuits.
Panel A: Model architecture. Panel B: Claimed mechanism. Panel C: ClaimSpec graph."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def draw_non_edge(ax, x1, y1, x2, y2, color='#ef4444', linewidth=1.5):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle='--', shrinkA=10, shrinkB=10))
    ax.text(mid_x, mid_y + 0.18, '✗', fontsize=13, ha='center', va='center',
            color=color, fontweight='bold')


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# ========== Panel A: Model Architecture ==========
ax = axes[0]
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A.  Model Architecture', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

draw_node(ax, 2, 3.2, 'Input', color='#f1f5f9', width=1.4)
draw_node(ax, 2, 2.0, 'Heads 7.8,\n8.11, 9.6', color='#e2e8f0', width=1.6, height=0.6)
draw_node(ax, 2, 0.8, 'Output', color='#f1f5f9', width=1.4)
draw_edge(ax, 2, 2.92, 2, 2.32)
draw_edge(ax, 2, 1.68, 2, 1.08)

ax.text(2, -0.2, 'No interpretation.\nJust architecture.', fontsize=9, ha='center',
        color='#475569', fontstyle='italic')

# ========== Panel B: Claimed Mechanism ==========
ax = axes[1]
ax.set_xlim(-1.0, 5.0)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('B.  Claimed Mechanism', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

draw_node(ax, 1.0, 3.2, '"Gender\nbias" circuit', color='#fecaca', width=1.6, height=0.6)
draw_node(ax, 3.0, 3.2, '"Gender\nknowledge"\ncircuit', color='#bbf7d0', width=1.6, height=0.7)
draw_node(ax, 2.0, 1.2, 'Gendered\npredictions', color='#dbeafe', width=1.6, height=0.55)

draw_edge(ax, 1.0, 2.88, 1.5, 1.5, color='#ef4444')
draw_edge(ax, 3.0, 2.83, 2.5, 1.5, color='#16a34a')

ax.text(0.3, 2.3, 'Produces\nbias', fontsize=8, color='#ef4444', ha='center', fontweight='bold')
ax.text(3.5, 2.3, 'Produces\nknowledge', fontsize=8, color='#16a34a', ha='center', fontweight='bold')

ax.text(2, -0.2, 'The narrative: bias and knowledge\nare separate, removable circuits.', fontsize=9,
        ha='center', color='#475569', fontstyle='italic')

# ========== Panel C: ClaimSpec Graph ==========
ax = axes[2]
ax.set_xlim(-1.0, 5.0)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('C.  ClaimSpec Graph', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

# Shared heads at top
draw_node(ax, 2.0, 3.2, 'Same heads\n(7.8, 8.11, 9.6)', color='#fef3c7', width=2.0, height=0.6)

# Two constructs below
draw_node(ax, 0.6, 1.5, '"Gender\nbias"', color='#fecaca', width=1.4, height=0.55)
draw_node(ax, 3.4, 1.5, '"Gender\nknowledge"', color='#bbf7d0', width=1.4, height=0.55)

# Shared heads feed both
draw_edge(ax, 1.4, 2.88, 0.8, 1.8)
draw_edge(ax, 2.6, 2.88, 3.2, 1.8)

# Failed non-edge between the two constructs
draw_non_edge(ax, 1.3, 1.5, 2.7, 1.5, color='#ef4444')
ax.text(2.0, 1.95, 'Required:\nseparable', fontsize=7, color='#ef4444', ha='center', fontweight='bold')

ax.text(2, -0.2, 'Construct collapses: cannot ablate\nbias without ablating knowledge.', fontsize=9,
        ha='center', color='#ef4444', fontstyle='italic')

# Bottom legend
fig.text(0.5, 0.02,
         '━━  Confirmed edge          '
         '╌╌  Failed non-edge (the required separability does not exist)',
         ha='center', fontsize=9, color='#475569')

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = OUTDIR / 'fig_claimspec_threepanel_gender_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
