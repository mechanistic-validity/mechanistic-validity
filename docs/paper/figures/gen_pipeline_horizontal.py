import matplotlib.pyplot as plt

stages = [
    ("Description Mode", "7 modes", "Which level of description\nis your claim making?"),
    ("Evidence Family", "4 sources", "Which source of signal\nis relevant to your claim?"),
    ("Metrics", "84 metrics", "Which tests can\nmeasure your claim?"),
    ("Criteria", "30 criteria", "Does the evidence meet\nthe required conditions?"),
    ("Validity Type", "5 types", "Which dimensions does\nthe evidence address?"),
    ("Synthesis", "9 methods", "How should evidence\nbe aggregated?"),
    ("Verdict", "7 tiers", "What verdict does\nthe evidence warrant?"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ea580c', '#ef4444']
light_colors = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fff7ed', '#fef2f2']

fig, ax = plt.subplots(figsize=(20, 3.8))
ax.set_xlim(-0.5, 14)
ax.set_ylim(-1.2, 2.0)
ax.set_aspect('equal')
ax.axis('off')

n = len(stages)
spacing = 12.5 / (n - 1)
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
            fontsize=10, color='#1e293b', fontweight='600',
            fontfamily='sans-serif', linespacing=1.4)

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.3)

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER_3_5/figures'
fig.savefig(f'{outdir}/pipeline-horizontal.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{outdir}/pipeline-horizontal.png', bbox_inches='tight', dpi=300, facecolor='white')
print("Saved pipeline-horizontal.pdf and .png")
