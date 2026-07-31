"""Two-panel figure: Post-hoc narrative vs Pre-registered ClaimSpec.
Same experimental results, different verdicts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_node(ax, x, y, label, color='#e2e8f0', text_color='#1e293b', fontsize=9,
              width=2.0, height=0.5):
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
    ax.text(mid_x + 0.15, mid_y + 0.15, symbol, fontsize=12, ha='center', va='center',
            color=color, fontweight='bold')


def draw_step(ax, x, y, label, color='#f1f5f9', text_color='#475569', fontsize=8, width=2.2, height=0.4):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.08', facecolor=color, edgecolor='#cbd5e1', linewidth=1.0)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, linespacing=1.2)


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# ========== Left: Post-hoc ==========
ax = axes[0]
ax.set_xlim(-1.5, 5.5)
ax.set_ylim(-1.0, 8.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Post-hoc Narrative', fontsize=14, fontweight='bold', pad=14, color='#0f172a')

# Steps going down
y = 7.5
draw_step(ax, 2, y, 'Run experiment', color='#dbeafe', text_color='#1e293b', fontsize=9)
draw_edge(ax, 2, y-0.22, 2, y-0.68)

y = 6.5
draw_step(ax, 2, y, 'See results: ablation\ndegrades behavior', color='#dbeafe', text_color='#1e293b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 5.3
draw_step(ax, 2, y, 'Try 3 ablation methods;\nreport the best one (87%)', color='#fef3c7', text_color='#92400e', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 4.1
draw_step(ax, 2, y, 'Name roles after what\nthey appear to do', color='#fef3c7', text_color='#92400e', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 2.9
draw_step(ax, 2, y, 'Never test off-target\neffects on other tasks', color='#fecaca', text_color='#991b1b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 1.7
draw_step(ax, 2, y, 'Write narrative\nthat fits all results', color='#fecaca', text_color='#991b1b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

# Verdict
draw_node(ax, 2, 0.5, '"Confirmed"', color='#bbf7d0', width=2.0, height=0.5, fontsize=11)
ax.text(2, -0.3, 'Unfalsifiable: no prediction\ncould have failed.', fontsize=9,
        ha='center', color='#991b1b', fontstyle='italic')

# ========== Right: Pre-registered ==========
ax = axes[1]
ax.set_xlim(-1.5, 5.5)
ax.set_ylim(-1.0, 8.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Pre-registered ClaimSpec', fontsize=14, fontweight='bold', pad=14, color='#0f172a')

y = 7.5
draw_step(ax, 2, y, 'Write ClaimSpec graph\nwith predictions + controls', color='#dbeafe', text_color='#1e293b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 6.3
draw_step(ax, 2, y, 'Commit: faithfulness >70%\nunder ≥2 ablation methods', color='#dbeafe', text_color='#1e293b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 5.1
draw_step(ax, 2, y, 'Commit: ablating circuit\ndoes NOT break other tasks', color='#dbeafe', text_color='#1e293b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 3.9
draw_step(ax, 2, y, 'Run experiment:\nsame results as left', color='#e2e8f0', text_color='#1e293b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 2.7
draw_step(ax, 2, y, 'Mean ablation: 87% ✓\nResample ablation: 40% ✗', color='#bbf7d0', text_color='#1e293b', fontsize=8, height=0.5)
ax.text(3.4, 2.85, '✓', fontsize=14, color='#16a34a', fontweight='bold')
ax.text(3.4, 2.55, '✗', fontsize=14, color='#ef4444', fontweight='bold')
draw_edge(ax, 2, y-0.27, 2, y-0.73)

y = 1.5
draw_step(ax, 2, y, 'E1 (intervention reach)\nfails: method-conditional', color='#fecaca', text_color='#991b1b', fontsize=8, height=0.5)
draw_edge(ax, 2, y-0.27, 2, y-0.73)

# Verdict
draw_node(ax, 2, 0.3, '"Causally Suggestive"', color='#fef3c7', width=2.2, height=0.5, fontsize=10, text_color='#92400e')
ax.text(2, -0.4, 'Falsifiable: the spec told us\nexactly what failed and why.', fontsize=9,
        ha='center', color='#16a34a', fontstyle='italic')

plt.tight_layout(w_pad=3)
out = OUTDIR / 'fig_posthoc_vs_preregistered_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
