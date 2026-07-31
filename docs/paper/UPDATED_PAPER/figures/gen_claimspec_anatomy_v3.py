"""Anatomy of a ClaimSpec graph v3 — 3 roles + output, vertical layout,
realistic failure (direct path bypasses claimed mechanism)."""
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


def draw_straight(ax, x1, y1, x2, y2, color='#16a34a', linewidth=2.0, linestyle='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=14, shrinkB=14))


def draw_curved(ax, x1, y1, x2, y2, color='#f59e0b', linewidth=1.5, linestyle='--',
                rad=0.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle=linestyle, shrinkA=14, shrinkB=14,
                                connectionstyle=f'arc3,rad={rad}'))


def add_label(ax, x, y, text, color='#475569', fontsize=7.5):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9))


fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(-2.5, 11.5)
ax.set_ylim(-2.5, 9)
ax.set_aspect('equal')
ax.axis('off')

ax.text(4.5, 8.5, 'Anatomy of a ClaimSpec Graph', fontsize=16, fontweight='bold',
        ha='center', color='#0f172a')
ax.text(4.5, 7.9, 'Example: Greater-Than circuit (Hanna et al., 2023)', fontsize=10,
        ha='center', color='#64748b')

# ---- Nodes (vertical: A top, B middle, outputs at bottom) ----
draw_node(ax, 3, 6.5, 'Year Attention\nHeads 9.1, 10.7', color='#dbeafe', width=2.2)
draw_node(ax, 3, 3.8, 'Comparison\nHead 11.2', color='#dbeafe', width=2.0)
draw_node(ax, 1.5, 1.0, 'Greater-Than\nPrediction', color='#f1f5f9', width=2.0)
draw_node(ax, 7, 1.0, 'Other Numerical\nTasks', color='#f1f5f9', width=2.2, text_color='#94a3b8')

# ==== POSITIVE PREDICTIONS (solid green) ====

# Year Attn → Comparison: passed
draw_straight(ax, 3, 6.17, 3, 4.13)
add_label(ax, 5.0, 5.2, '"Ablate year heads →\ncomparison activation drops"  ✓', color='#16a34a')

# Comparison → GT Output: passed
draw_straight(ax, 2.5, 3.47, 1.8, 1.33)
add_label(ax, 0.5, 2.4, '"Ablate comparison head →\nGT prediction degrades"  ✓', color='#16a34a')

# ==== FAILED NON-EDGE (year heads → GT output directly, bypassing comparison) ====
draw_curved(ax, 1.8, 6.17, 1.0, 1.33, color='#ef4444', linewidth=1.8, linestyle='--', rad=0.5)
ax.text(-0.8, 3.8, '✗', fontsize=15, ha='center', va='center',
        color='#ef4444', fontweight='bold')
add_label(ax, -1.2, 3.0, '"Year heads do NOT\naffect GT directly"\nFAILS: direct path\nbypasses comparison', color='#ef4444')

# ==== NEGATIVE CONTROLS ====

# Year Attn → Other Numerical: passed (year heads are GT-specific)
draw_curved(ax, 4.1, 6.17, 6.5, 1.33, color='#16a34a', linewidth=1.5, linestyle='--', rad=-0.2)
add_label(ax, 6.5, 4.5, '"Ablate year heads →\nother numerical tasks\nunchanged"  ✓', color='#16a34a')

# Comparison → Other Numerical: untested
draw_straight(ax, 3.8, 3.5, 6.0, 1.33, color='#f59e0b', linestyle='--', linewidth=1.5)
ax.text(5.2, 2.7, '?', fontsize=14, ha='center', va='center',
        color='#f59e0b', fontweight='bold')
add_label(ax, 5.8, 2.1, '"Ablate comparison →\nother numerical unchanged"\n?  untested', color='#f59e0b')

# ==== LEGEND ====
ly = -1.5
legend = [
    (-1.5, '━━  ✓', '#16a34a', 'Positive prediction (passed)'),
    (2.2, '╌╌  ✓', '#16a34a', 'Negative control (passed)'),
    (5.5, '╌╌  ?', '#f59e0b', 'Untested non-edge'),
    (8.3, '╌╌  ✗', '#ef4444', 'Failed prediction'),
]
for lx, sym, col, desc in legend:
    ax.text(lx, ly, sym, fontsize=10, color=col, fontweight='bold', va='center')
    ax.text(lx + 0.7, ly, desc, fontsize=8, color='#475569', va='center')

# ==== VERDICT ====
vbox = mpatches.FancyBboxPatch(
    (6.5, 3.0), 4.2, 1.5,
    boxstyle='round,pad=0.12', facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=1.5)
ax.add_patch(vbox)
ax.text(8.6, 4.05, 'Verdict: Causally Suggestive', fontsize=9, fontweight='bold',
        ha='center', color='#92400e')
ax.text(8.6, 3.5, 'Specificity untested (comparison → numerical)\nMechanism incomplete (year heads bypass comparison)',
        fontsize=7, ha='center', color='#92400e')

plt.tight_layout()
out = OUTDIR / 'fig_claimspec_anatomy_v3.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
