"""Two-row horizontal figure for the Sutter et al. nonlinear DAS example.
Post-hoc: high IIA looks great. Pre-registered: random model baseline reveals vacuousness."""
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
ax.set_xlim(-1, 14)
ax.set_ylim(-0.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(-0.5, 1.5, 'Without baseline control', fontsize=13, fontweight='bold', color='#991b1b', va='center')

draw_node(ax, 1.5, 0.5, 'Train nonlinear\nalignment map', color='#dbeafe', width=1.7, height=0.55)
draw_edge(ax, 2.35, 0.5, 3.25, 0.5)

draw_node(ax, 4.1, 0.5, 'Achieve\n95% IIA', color='#bbf7d0', width=1.5, fontsize=9)
draw_edge(ax, 4.85, 0.5, 5.75, 0.5)

draw_node(ax, 6.6, 0.5, '"The model\nrepresents this\nvariable here"', color='#bbf7d0', width=1.7, height=0.65, fontsize=7)
draw_edge(ax, 7.45, 0.5, 8.35, 0.5)

draw_node(ax, 9.2, 0.5, 'No baseline\ntested', color='#fecaca', text_color='#991b1b', width=1.5)
draw_edge(ax, 9.95, 0.5, 10.85, 0.5)

draw_node(ax, 11.7, 0.5, '"Confirmed"', color='#bbf7d0', width=1.5, fontsize=10)

ax.text(11.7, -0.2, 'Looks great — but is\nthe map doing the work?', fontsize=7, ha='center',
        color='#991b1b', fontstyle='italic')

# ========== Bottom row: With baseline ==========
ax = axes[1]
ax.set_xlim(-1, 14)
ax.set_ylim(-0.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(-0.5, 1.5, 'With baseline control (M2)', fontsize=13, fontweight='bold', color='#16a34a', va='center')

draw_node(ax, 1.5, 0.5, 'Train same\nalignment map', color='#dbeafe', width=1.7, height=0.55)
draw_edge(ax, 2.35, 0.5, 3.25, 0.5)

draw_node(ax, 4.1, 0.5, 'Trained model:\n95% IIA', color='#bbf7d0', width=1.5)
draw_edge(ax, 4.85, 0.5, 5.75, 0.5)

draw_node(ax, 6.6, 0.5, 'Random model\nbaseline:\nalso 95% IIA', color='#fecaca', text_color='#991b1b', width=1.7, height=0.65, fontsize=7)
ax.text(6.6, -0.15, 'M2 fails', fontsize=7, color='#ef4444', ha='center', fontweight='bold')
draw_edge(ax, 7.45, 0.5, 8.35, 0.5)

draw_node(ax, 9.2, 0.5, 'Map is too\nexpressive:\nresult is\nuninformative', color='#fecaca', text_color='#991b1b', width=1.6, height=0.75, fontsize=7)
draw_edge(ax, 10.0, 0.5, 10.85, 0.5)

draw_node(ax, 11.7, 0.5, '"Proposed"', color='#fecaca', text_color='#991b1b', width=1.5, fontsize=10)

ax.text(11.7, -0.2, 'Baseline revealed the\npositive result was vacuous', fontsize=7, ha='center',
        color='#16a34a', fontstyle='italic')

plt.tight_layout(h_pad=1.5)
out = OUTDIR / 'fig_claimspec_sutter_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
