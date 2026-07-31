import matplotlib.pyplot as plt

stages_6 = [
    ("Description Mode", "7 modes", "What level of description\nis the claim making?"),
    ("Evidence Family", "6 families", "What kind of signal\nis relevant to the claim?"),
    ("Metrics", "84 definitions", "Which tests can\nmeasure the claim?"),
    ("Criteria", "27 criteria", "Does the evidence meet\nthe required conditions?"),
    ("Validity Type", "5 dimensions", "Which dimensions does\nthe evidence address?"),
    ("Verdict", "7 tiers", "What verdict does\nthe evidence warrant?"),
]

colors_6 = ['#6366f1', '#3b82f6', '#10b981', '#14b8a6', '#f59e0b', '#ef4444']
light_6 = ['#eef2ff', '#eff6ff', '#ecfdf5', '#f0fdfa', '#fffbeb', '#fef2f2']

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER/figures'


def variant_b():
    """Title + count only, no questions."""
    fig, ax = plt.subplots(figsize=(18, 3.5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1.0, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')

    n = len(stages_6)
    spacing = 10.5 / (n - 1)
    x_positions = [0.5 + i * spacing for i in range(n)]
    cy = 0.5
    r = 0.48

    ax.plot([x_positions[0], x_positions[-1]], [cy, cy],
            color='#e2e8f0', linewidth=3, zorder=1, solid_capstyle='round')

    for i, (title, count, question) in enumerate(stages_6):
        x = x_positions[i]
        circle = plt.Circle((x, cy), r, facecolor=light_6[i], edgecolor=colors_6[i],
                             linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
                fontsize=18, fontweight='bold', color=colors_6[i], zorder=3,
                fontfamily='sans-serif')
        ax.text(x, cy + r + 0.15, title, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        ax.text(x, cy - r - 0.15, count, ha='center', va='top',
                fontsize=12, fontweight='600', color='#334155', fontfamily='sans-serif')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{outdir}/pipeline-B.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{outdir}/pipeline-B.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("B: title + count only")


def variant_d():
    """2-row paired layout, fixed rendering."""
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(-1.0, 11.5)
    ax.set_ylim(-2.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')

    r = 0.45
    col_x = [1.0, 4.0, 7.0, 10.0]

    pairs = [
        [("Description\nMode", "7 modes", '#6366f1', '#eef2ff', 1),
         ("Evidence\nFamily", "6 families", '#3b82f6', '#eff6ff', 2)],
        [("Metrics", "58 metrics", '#10b981', '#ecfdf5', 3)],
        [("Criteria", "27 criteria", '#14b8a6', '#f0fdfa', 4),
         ("Validity\nType", "5 dimensions", '#f59e0b', '#fffbeb', 5)],
        [("Verdict", "7 tiers", '#ef4444', '#fef2f2', 6)],
    ]

    top_y = 1.5
    bot_y = -0.5
    mid_y = 0.5

    for col_idx, group in enumerate(pairs):
        cx = col_x[col_idx]
        if len(group) == 2:
            for row_idx, (title, count, color, light, num) in enumerate(group):
                y = top_y if row_idx == 0 else bot_y
                circle = plt.Circle((cx, y), r, facecolor=light, edgecolor=color,
                                     linewidth=2.5, zorder=2)
                ax.add_patch(circle)
                ax.text(cx, y - 0.01, str(num), ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color, zorder=3,
                        fontfamily='sans-serif')
                ax.text(cx, y + r + 0.15, title, ha='center', va='bottom',
                        fontsize=11.5, fontweight='bold', color='#1e293b',
                        fontfamily='sans-serif', linespacing=1.2)
                ax.text(cx, y - r - 0.12, count, ha='center', va='top',
                        fontsize=10.5, fontweight='600', color='#334155', fontfamily='sans-serif')
            ax.plot([cx, cx], [top_y - r - 0.05, bot_y + r + 0.05], color='#cbd5e1',
                    linewidth=1.5, linestyle='--', zorder=1)
        else:
            title, count, color, light, num = group[0]
            y = mid_y
            circle = plt.Circle((cx, y), r, facecolor=light, edgecolor=color,
                                 linewidth=2.5, zorder=2)
            ax.add_patch(circle)
            ax.text(cx, y - 0.01, str(num), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color, zorder=3,
                    fontfamily='sans-serif')
            ax.text(cx, y + r + 0.15, title, ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
            ax.text(cx, y - r - 0.12, count, ha='center', va='top',
                    fontsize=10.5, fontweight='600', color='#334155', fontfamily='sans-serif')

    arrow_kw = dict(arrowstyle='->', color='#94a3b8', lw=2)
    ax.annotate('', xy=(col_x[1] - r - 0.2, mid_y), xytext=(col_x[0] + r + 0.2, mid_y), arrowprops=arrow_kw)
    ax.annotate('', xy=(col_x[2] - r - 0.2, mid_y), xytext=(col_x[1] + r + 0.2, mid_y), arrowprops=arrow_kw)
    ax.annotate('', xy=(col_x[3] - r - 0.2, mid_y), xytext=(col_x[2] + r + 0.2, mid_y), arrowprops=arrow_kw)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{outdir}/pipeline-D.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{outdir}/pipeline-D.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("D: 2-row paired layout (fixed)")


def variant_e():
    """B layout + questions below, same bold size as count."""
    fig, ax = plt.subplots(figsize=(18, 5.0))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')

    n = len(stages_6)
    spacing = 10.5 / (n - 1)
    x_positions = [0.5 + i * spacing for i in range(n)]
    cy = 0.5
    r = 0.48

    ax.plot([x_positions[0], x_positions[-1]], [cy, cy],
            color='#e2e8f0', linewidth=3, zorder=1, solid_capstyle='round')

    for i, (title, count, question) in enumerate(stages_6):
        x = x_positions[i]
        circle = plt.Circle((x, cy), r, facecolor=light_6[i], edgecolor=colors_6[i],
                             linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
                fontsize=18, fontweight='bold', color=colors_6[i], zorder=3,
                fontfamily='sans-serif')
        ax.text(x, cy + r + 0.15, title, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        ax.text(x, cy - r - 0.15, count, ha='center', va='top',
                fontsize=12, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        ax.text(x, cy - r - 0.48, question, ha='center', va='top',
                fontsize=12, fontweight='600', color='#334155', fontfamily='sans-serif',
                linespacing=1.3)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{outdir}/pipeline-E.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{outdir}/pipeline-E.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("E: B layout + bold questions, same size as count")


def variant_f():
    """E layout but with arrows between circles and bolder numbers."""
    fig, ax = plt.subplots(figsize=(18, 4.2))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1.55, 1.7)
    ax.set_aspect('equal')
    ax.axis('off')

    n = len(stages_6)
    spacing = 10.5 / (n - 1)
    x_positions = [0.5 + i * spacing for i in range(n)]
    cy = 0.5
    r = 0.48

    for i, (title, count, question) in enumerate(stages_6):
        x = x_positions[i]
        circle = plt.Circle((x, cy), r, facecolor=light_6[i], edgecolor=colors_6[i],
                             linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, cy - 0.01, str(i + 1), ha='center', va='center',
                fontsize=22, fontweight='900', color=colors_6[i], zorder=3,
                fontfamily='sans-serif')
        ax.text(x, cy + r + 0.15, title, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        ax.text(x, cy - r - 0.15, count, ha='center', va='top',
                fontsize=12, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        ax.text(x, cy - r - 0.48, question, ha='center', va='top',
                fontsize=12, fontweight='600', color='#334155', fontfamily='sans-serif',
                linespacing=1.3)

        if i < n - 1:
            x_next = x_positions[i + 1]
            ax.annotate('', xy=(x_next - r - 0.08, cy), xytext=(x + r + 0.08, cy),
                        arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5,
                                        mutation_scale=18, shrinkA=0, shrinkB=0))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.1)
    fig.savefig(f'{outdir}/pipeline-F.pdf', bbox_inches='tight', pad_inches=0.05, facecolor='white')
    fig.savefig(f'{outdir}/pipeline-F.png', bbox_inches='tight', pad_inches=0.05, dpi=300, facecolor='white')
    plt.close()
    print("F: E layout + arrows + bolder numbers")


variant_f()
