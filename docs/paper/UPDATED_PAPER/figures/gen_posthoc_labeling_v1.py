"""Side-by-side: Post-hoc role labeling vs ClaimSpec pre-registered labels.
And: Post-hoc circuit redefinition vs ClaimSpec pre-committed membership."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent


def draw_box(ax, x, y, w, h, label, color='#e2e8f0', text_color='#1e293b', fontsize=9):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.10', facecolor=color, edgecolor='#94a3b8', linewidth=1.3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color='#475569', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, shrinkA=5, shrinkB=5))


# ============================================================
# FIGURE 1: Post-hoc Role Labeling
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 7))

# --- LEFT: Post-hoc ---
ax = axes[0]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.text(3, 7.5, 'Without ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#991b1b')

draw_box(ax, 3, 6.3, 4.5, 0.5, 'Ablate head 9.9, observe effect', color='#dbeafe')
arrow(ax, 3, 6.03, 3, 5.5)

draw_box(ax, 3, 5.0, 4.5, 0.6, 'Head 9.9 increases logit for\nindirect object name', color='#dbeafe')
arrow(ax, 3, 4.68, 3, 4.1)

draw_box(ax, 3, 3.6, 3.5, 0.6, 'Label it "name mover"\nbased on what it did', color='#fef3c7',
         text_color='#92400e')
arrow(ax, 3, 3.28, 3, 2.7)

draw_box(ax, 3, 2.2, 4.0, 0.6, 'But does it also move\nnon-name tokens? Never tested.', color='#fecaca',
         text_color='#991b1b', fontsize=8)
arrow(ax, 3, 1.88, 3, 1.3)

draw_box(ax, 3, 0.8, 3.0, 0.6, 'The label becomes\nthe explanation', color='#fecaca',
         text_color='#991b1b')

draw_box(ax, 3, -0.2, 2.5, 0.5, '"Confirmed"', color='#bbf7d0', fontsize=12)
arrow(ax, 3, 0.48, 3, 0.07)
ax.text(3, -0.7, 'The label was chosen to fit.\nIt cannot be wrong.', fontsize=8,
        ha='center', color='#991b1b', fontstyle='italic')

# --- RIGHT: With ClaimSpec ---
ax = axes[1]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.text(3, 7.5, 'With ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#16a34a')

draw_box(ax, 3, 6.3, 5.0, 0.5, 'Pre-register: "Head 9.9 moves\nname tokens to output position"',
         color='#dbeafe', fontsize=9)
arrow(ax, 3, 6.03, 3, 5.5)

draw_box(ax, 3, 5.0, 5.0, 0.6, 'Prediction 1: ablate 9.9 →\nIO name logit drops  ✓', color='#bbf7d0')
arrow(ax, 3, 4.68, 3, 4.1)

draw_box(ax, 3, 3.6, 5.0, 0.6, 'Prediction 2: ablate 9.9 →\nnon-IO tokens unaffected  ?',
         color='#fef3c7', text_color='#92400e')
ax.text(3, 3.05, 'Does 9.9 specifically move names,\nor does it boost any attended token?',
        fontsize=8, ha='center', color='#92400e', fontstyle='italic')
arrow(ax, 3, 2.78, 3, 2.2)

draw_box(ax, 3, 1.8, 4.5, 0.5, 'Label "name mover" is testable:\nspecificity is part of the claim',
         color='#dbeafe', fontsize=9)
arrow(ax, 3, 1.53, 3, 1.0)

draw_box(ax, 3, 0.5, 4.0, 0.6, 'If non-IO tokens also boosted:\nlabel should be "general copier"',
         color='#fecaca', text_color='#991b1b', fontsize=8)

draw_box(ax, 3, -0.5, 3.0, 0.5, '"Causally Suggestive"', color='#fef3c7',
         text_color='#92400e', fontsize=10)
arrow(ax, 3, 0.18, 3, -0.23)
ax.text(3, -1.0, 'The label is a prediction.\nIt can be wrong.', fontsize=8,
        ha='center', color='#16a34a', fontstyle='italic')

plt.tight_layout(w_pad=2)
out = OUTDIR / 'fig_posthoc_labeling_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')


# ============================================================
# FIGURE 2: Post-hoc Circuit Redefinition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 7))

# --- LEFT: Post-hoc ---
ax = axes[0]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.text(3, 7.5, 'Without ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#991b1b')

draw_box(ax, 3, 6.3, 4.5, 0.5, 'Discover 26 candidate heads', color='#dbeafe')
arrow(ax, 3, 6.03, 3, 5.5)

draw_box(ax, 3, 5.0, 4.5, 0.6, 'Measure faithfulness:\n26 heads → 78%', color='#fef3c7',
         text_color='#92400e')
arrow(ax, 3, 4.68, 3, 4.1)

draw_box(ax, 3, 3.6, 4.5, 0.6, 'Drop 4 heads that hurt:\n22 heads → 87%', color='#fef3c7',
         text_color='#92400e')
ax.text(3, 3.05, 'Circuit membership adjusted\nto maximize the metric', fontsize=8,
        ha='center', color='#991b1b', fontstyle='italic')
arrow(ax, 3, 2.78, 3, 2.2)

draw_box(ax, 3, 1.8, 4.0, 0.5, 'Report "22-head circuit\nwith 87% faithfulness"', color='#bbf7d0')
arrow(ax, 3, 1.53, 3, 1.0)

draw_box(ax, 3, 0.5, 2.5, 0.5, '"Confirmed"', color='#bbf7d0', fontsize=12)
ax.text(3, -0.1, 'The circuit was defined\nby the metric it optimizes.', fontsize=8,
        ha='center', color='#991b1b', fontstyle='italic')

# --- RIGHT: With ClaimSpec ---
ax = axes[1]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.text(3, 7.5, 'With ClaimSpec', fontsize=14, fontweight='bold',
        ha='center', color='#16a34a')

draw_box(ax, 3, 6.3, 5.0, 0.5, 'Pre-register: "These 26 heads\nimplement IOI"', color='#dbeafe')
arrow(ax, 3, 6.03, 3, 5.5)

draw_box(ax, 3, 5.0, 5.0, 0.6, 'Prediction: 26-head circuit\nachieves >70% faithfulness  ✓',
         color='#bbf7d0')
arrow(ax, 3, 4.68, 3, 4.1)

draw_box(ax, 3, 3.6, 5.0, 0.7, 'Prediction: all 26 heads\ncontribute — removing any one\ndegrades faithfulness  ✗',
         color='#fecaca', text_color='#991b1b', fontsize=8)
ax.text(3, 2.95, '4 heads are redundant or harmful.\nThe circuit is over-specified.',
        fontsize=8, ha='center', color='#ef4444', fontstyle='italic')
arrow(ax, 3, 2.68, 3, 2.1)

draw_box(ax, 3, 1.7, 4.5, 0.5, 'Diagnosis: claimed circuit\ncontains non-functional heads',
         color='#fecaca', text_color='#991b1b', fontsize=9)
arrow(ax, 3, 1.43, 3, 0.85)

draw_box(ax, 3, 0.4, 3.0, 0.5, '"Causally Suggestive"', color='#fef3c7',
         text_color='#92400e', fontsize=10)
ax.text(3, -0.2, 'The spec caught over-specification.\nThe 22-head version needs re-testing.',
        fontsize=8, ha='center', color='#16a34a', fontstyle='italic')

plt.tight_layout(w_pad=2)
out = OUTDIR / 'fig_posthoc_redefinition_v1.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
