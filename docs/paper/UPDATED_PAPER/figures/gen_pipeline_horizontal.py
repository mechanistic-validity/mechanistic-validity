import matplotlib.pyplot as plt

stages = [
    ("Description Mode", "7 modes", "Which level of description\nis your claim making?"),
    ("Evidence Family", "6 families", "Which kind of signal\nis relevant to your claim?"),
    ("Metrics", "58 metrics", "Which tests can\nmeasure your claim?"),
    ("Criteria", "27 criteria", "Does the evidence meet\nthe required conditions?"),
    ("Validity Type", "5 dimensions", "Which dimensions does\nthe evidence address?"),
    ("Verdict", "7 tiers", "What verdict does\nthe evidence warrant?"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ef4444']
light_colors = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fef2f2']

fig, ax = plt.subplots(figsize=(18, 3.8))
ax.set_xlim(-0.5, 12)
ax.set_ylim(-1.2, 2.0)
ax.set_aspect('equal')
ax.axis('off')

n = len(stages)
spacing = 10.5 / (n - 1)
x_positions = [0.5 + i * spacing for i in range(n)]
cy = 0.5
r = 0.48

ax.plot([x_positions[0], x_positions[-1]], [cy, cy],
        color='#e2e8f0', linewidth=3, zorder=1, solid_capstyle='round')

for i, (title, count, question) in enumerate(stages):
    x = x_positions[i]

    circle = plt.Circle((x, cy), r,
                         facecolor=light_colors[i], edgecolor=colors[i],
                         linewidth=2.5, zorder=2)
    ax.add_patch(circle)

    ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
            fontsize=18, fontweight='bold', color=colors[i], zorder=3,
            fontfamily='sans-serif')

    ax.text(x, cy + r + 0.2, title, ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#1e293b',
            fontfamily='sans-serif')

    ax.text(x, cy - r - 0.14, question, ha='center', va='top',
            fontsize=10, color='#475569', fontstyle='italic',
            fontfamily='sans-serif', linespacing=1.4)

    ax.text(x, cy - r - 0.62, count, ha='center', va='top',
            fontsize=9.5, color='#64748b', fontfamily='sans-serif')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.3)

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER/figures'
fig.savefig(f'{outdir}/pipeline-horizontal.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{outdir}/pipeline-horizontal.png', bbox_inches='tight', dpi=300, facecolor='white')
print("Saved pipeline-horizontal.pdf and .png")
