import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

layers = [
    ("LAYER 6", "Verdict", "What verdict does the evidence warrant?", '#7c3aed', '#ede9fe'),
    ("LAYER 5", "Description Modes", "Which level of description is your claim making?", '#8b5cf6', '#f5f3ff'),
    ("LAYER 4", "Validity Types", "Which dimensions does the evidence address?", '#2563eb', '#eff6ff'),
    ("LAYER 3", "Criteria", "Does the evidence meet the required conditions?", '#16a34a', '#f0fdf4'),
    ("LAYER 2", "Evidence Families", "Which kind of signal is relevant to your claim?", '#d97706', '#fffbeb'),
    ("LAYER 1", "Metrics", "Which tests can measure your claim?", '#dc2626', '#fef2f2'),
]

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.3, len(layers) * 1.1 + 1.0)
ax.set_aspect('auto')
ax.axis('off')

box_w = 7.5
box_h = 0.75
box_x = 2.5
gap = 1.1

for i, (label, title, desc, color, bg) in enumerate(layers):
    y = i * gap + 0.5

    rect = mpatches.FancyBboxPatch((box_x, y), box_w, box_h, boxstyle="round,pad=0.08",
                                    facecolor=bg, edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)

    ax.text(box_x + 0.3, y + 0.50, label, fontsize=10, fontweight='700',
            color=color, fontfamily='sans-serif', verticalalignment='center')

    ax.text(box_x + 1.75, y + 0.50, title, fontsize=13, fontweight='700',
            color='#1e293b', fontfamily='sans-serif', verticalalignment='center')

    ax.text(box_x + 0.3, y + 0.20, desc, fontsize=10.5, color='#334155',
            fontfamily='sans-serif', verticalalignment='center')

for i in range(len(layers) - 1):
    y_bot = i * gap + 0.5 + box_h + 0.04
    y_top = (i + 1) * gap + 0.5 - 0.04
    mid_x = box_x + box_w / 2
    ax.annotate('', xy=(mid_x, y_top), xytext=(mid_x, y_bot),
                arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1.8,
                                mutation_scale=15))

arrow_x_left = 0.8
arrow_x_right = box_x + box_w + 1.2
y_bottom = 0.5 + box_h / 2
y_top_arrow = (len(layers) - 1) * gap + 0.5 + box_h / 2

ax.annotate('', xy=(arrow_x_left, y_top_arrow), xytext=(arrow_x_left, y_bottom),
            arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1.8,
                            mutation_scale=18, shrinkA=8, shrinkB=8))
ax.text(arrow_x_left, y_bottom - 0.3, 'Audit', fontsize=13, fontweight='700',
        color='#475569', ha='center', fontfamily='sans-serif')

ax.annotate('', xy=(arrow_x_right, y_bottom), xytext=(arrow_x_right, y_top_arrow),
            arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1.8,
                            mutation_scale=18, shrinkA=8, shrinkB=8))
ax.text(arrow_x_right, y_top_arrow + 0.3, 'Experiment', fontsize=13, fontweight='700',
        color='#475569', ha='center', fontfamily='sans-serif')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.5)

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER/figures'
fig.savefig(f'{outdir}/hierarchy.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{outdir}/hierarchy.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("Saved hierarchy.pdf and .png")
