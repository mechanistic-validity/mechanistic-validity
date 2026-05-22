import matplotlib.pyplot as plt
import matplotlib.patches as patches

stages = [
    ("Description Mode", "Which level of description is your claim making?",
     "Implementational — Functional", "Names a component and attributes a function (name-moving)"),
    ("Evidence Families", "Which kind of signal is relevant to your claim?",
     "Causal + Structural", "Activation patching (causal) and QK/OV composition (structural)"),
    ("Evidence", "Run metrics, calibrations, protocols.",
     "Activation patching, path patching, mean ablation", "No calibrations run — no bootstrap, no random-vector baseline"),
    ("Criteria", "Score against 5 validity types.",
     "I1 Necessity passes; I3, M1, C1 weak", "Specificity untested, single seed, circular construct"),
    ("Verdict", "What verdict does the evidence warrant?",
     "Causally suggestive [implementational–functional]", "Necessity established; sufficiency method-conditional (Miller et al. 2024)"),
]

colors = ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ef4444']
light_colors = ['#eef2ff', '#eff6ff', '#ecfdf5', '#fffbeb', '#fef2f2']

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-0.2, 14)
ax.set_ylim(-0.5, 12)
ax.set_aspect('equal')
ax.axis('off')
ax.invert_yaxis()

n = len(stages)
cx = 9.2
r = 0.55
spacing = 2.2

for i, (title, question, ex_title, ex_desc) in enumerate(stages):
    y = 0.8 + i * spacing

    if i < n - 1:
        next_y = 0.8 + (i + 1) * spacing
        ax.plot([cx, cx], [y + r + 0.05, next_y - r - 0.05],
                color='#e2e8f0', linewidth=2.5, zorder=1, solid_capstyle='round')

    circle = plt.Circle((cx, y), r,
                         facecolor=light_colors[i], edgecolor=colors[i],
                         linewidth=2.5, zorder=2)
    ax.add_patch(circle)

    ax.text(cx, y, str(i + 1), ha='center', va='center',
            fontsize=22, fontweight='bold', color=colors[i], zorder=3,
            fontfamily='sans-serif')

    ax.text(cx + r + 0.35, y - 0.13, title, ha='left', va='center',
            fontsize=16, fontweight='bold', color='#1e293b',
            fontfamily='sans-serif')

    ax.text(cx + r + 0.35, y + 0.30, question, ha='left', va='center',
            fontsize=12, color='#475569', fontstyle='italic',
            fontfamily='sans-serif')

    ex_box_w = 7.7
    ex_box_h = 1.1
    ex_box_x = cx - r - 0.6 - ex_box_w
    ex_box_y = y - 0.55

    rect = patches.FancyBboxPatch(
        (ex_box_x, ex_box_y), ex_box_w, ex_box_h,
        boxstyle="round,pad=0.15",
        facecolor=light_colors[i], edgecolor=colors[i],
        linewidth=1.2, zorder=1)
    ax.add_patch(rect)

    ax.text(ex_box_x + 0.35, y - 0.18, ex_title, ha='left', va='center',
            fontsize=14, fontweight='bold', color='#1e293b',
            fontfamily='sans-serif')

    ax.text(ex_box_x + 0.35, y + 0.28, ex_desc, ha='left', va='center',
            fontsize=12, color='#475569', fontstyle='italic',
            fontfamily='sans-serif')

ax.text(cx - r - 0.6 - ex_box_w, -0.1,
        "Example: activation patching L9H9 for IOI task (GPT-2 Small)",
        ha='left', va='bottom',
        fontsize=12, fontweight='bold', color='#475569',
        fontfamily='sans-serif')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.tight_layout(pad=0.5)

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/public/figures/v2'
fig.savefig(f'{outdir}/pipeline-vert-ioi.png', bbox_inches='tight', dpi=300, facecolor='white')
print("Saved pipeline-vert-ioi.png")
