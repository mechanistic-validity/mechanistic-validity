import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

stages = [
    ("Description\nMode", "7 modes"),
    ("Evidence\nFamily", "4 sources"),
    ("Metrics", "84 metrics"),
    ("Criteria", "30 criteria"),
    ("Validity\nType", "5 types"),
    ("Synthesis", "9 methods"),
    ("Verdict", "7 tiers"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ea580c', '#ef4444']
light = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fff7ed', '#fef2f2']

fig, ax = plt.subplots(figsize=(20, 5.5))
ax.set_xlim(-1.5, 15.5)
ax.set_ylim(-2.5, 4.0)
ax.set_aspect('equal')
ax.axis('off')

n = len(stages)
spacing = 12.5 / (n - 1)
x_positions = [0.5 + i * spacing for i in range(n)]
cy = 0.5
r = 0.52

# arrows between circles
arrow_color = '#94a3b8'
for i in range(n - 1):
    x_from = x_positions[i] + r + 0.08
    x_to = x_positions[i + 1] - r - 0.08
    ax.annotate('', xy=(x_to, cy), xytext=(x_from, cy),
                arrowprops=dict(arrowstyle='-|>', color=arrow_color, lw=2,
                                mutation_scale=14))

# circles with labels
for i, (title, count) in enumerate(stages):
    x = x_positions[i]
    circle = plt.Circle((x, cy), r, facecolor=light[i], edgecolor=colors[i],
                         linewidth=2.5, zorder=2)
    ax.add_patch(circle)
    ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
            fontsize=20, fontweight='bold', color=colors[i], zorder=3,
            fontfamily='sans-serif')
    ax.text(x, cy + r + 0.18, title, ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#1e293b', fontfamily='sans-serif',
            linespacing=1.15)
    ax.text(x, cy - r - 0.15, count, ha='center', va='top',
            fontsize=12, fontweight='600', color='#334155', fontfamily='sans-serif')

# Experiment: curved arrow above (left to right)
exp_start_y = cy + r + 0.95
exp_arc = mpatches.FancyArrowPatch(
    (x_positions[0], exp_start_y),
    (x_positions[-1], exp_start_y - 0.2),
    connectionstyle="arc3,rad=-0.25",
    arrowstyle='-|>',
    color=arrow_color, lw=2.2, mutation_scale=18, zorder=1)
ax.add_patch(exp_arc)
ax.text((x_positions[0] + x_positions[-1]) / 2, exp_start_y + 1.45,
        'Experiment', fontsize=14, fontweight='700', color='#475569',
        ha='center', va='bottom', fontfamily='sans-serif')

# Audit: curved arrow below (right to left), label ABOVE the arrow
audit_start_y = cy - r - 0.55
audit_arc = mpatches.FancyArrowPatch(
    (x_positions[-1], audit_start_y),
    (x_positions[0], audit_start_y),
    connectionstyle="arc3,rad=-0.25",
    arrowstyle='-|>',
    color=arrow_color, lw=2.2, mutation_scale=18, zorder=1)
ax.add_patch(audit_arc)
ax.text((x_positions[0] + x_positions[-1]) / 2, audit_start_y - 0.85,
        'Audit', fontsize=14, fontweight='700', color='#475569',
        ha='center', va='top', fontfamily='sans-serif')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.3)

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER_3_5/figures'
fig.savefig(f'{outdir}/pipeline-final.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{outdir}/pipeline-final.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("Saved pipeline-final.pdf and .png")
