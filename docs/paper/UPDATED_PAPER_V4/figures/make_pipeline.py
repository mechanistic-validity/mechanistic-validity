import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 4))
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

steps = [
    {"name": "Description Mode", "num": 1, "color": "#5B6FB5",
     "desc": "Which level of description\nis your claim making?", "count": "7 modes"},
    {"name": "Evidence Family", "num": 2, "color": "#5B8EC4",
     "desc": "Which source of signal\nis relevant to your claim?", "count": "4 sources"},
    {"name": "Metrics", "num": 3, "color": "#3BBFA0",
     "desc": "Which tests can\nmeasure your claim?", "count": "84 metrics"},
    {"name": "Criteria", "num": 4, "color": "#2CC4A8",
     "desc": "Does the evidence meet\nthe required conditions?", "count": "30 criteria"},
    {"name": "Validity Type", "num": 5, "color": "#F5A623",
     "desc": "Which dimensions does\nthe evidence address?", "count": "5 types"},
    {"name": "Synthesis", "num": 6, "color": "#E8834A",
     "desc": "How should evidence\nbe aggregated?", "count": "9 methods"},
    {"name": "Verdict", "num": 7, "color": "#E25B5B",
     "desc": "What verdict does\nthe evidence warrant?", "count": "7 tiers"},
]

xs = np.linspace(1.2, 17.3, 7)
y = 0.3

# Connecting line
ax.plot([xs[0], xs[-1]], [y, y], color='#E0E0E0', linewidth=2, zorder=0)

radius = 0.55

for i, (x, step) in enumerate(zip(xs, steps)):
    color = step["color"]

    # Circle
    circle = plt.Circle((x, y), radius, fill=False, edgecolor=color,
                         linewidth=2.5, zorder=2)
    ax.add_patch(circle)

    # Number inside circle
    ax.text(x, y, str(step["num"]), ha='center', va='center',
            fontsize=18, fontweight='bold', color=color,
            fontfamily='sans-serif', zorder=3)

    # Title above
    ax.text(x, y + 1.05, step["name"], ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=color,
            fontfamily='sans-serif')

    # Description below
    ax.text(x, y - 0.95, step["desc"], ha='center', va='top',
            fontsize=7.5, color='#888888', fontfamily='sans-serif',
            linespacing=1.4)

    # Count at bottom
    ax.text(x, y - 1.75, step["count"], ha='center', va='top',
            fontsize=8, fontweight='bold', color='#AAAAAA',
            fontfamily='sans-serif')

plt.tight_layout()
plt.savefig('/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER_3_5/figures/pipeline_figure_v3.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER_3_5/figures/pipeline_figure_v3.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Done")
