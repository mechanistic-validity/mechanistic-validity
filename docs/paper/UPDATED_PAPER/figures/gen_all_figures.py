import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

outdir = '/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER/figures'

# ============================================================
# Data
# ============================================================

# 13 case studies: (name, verdict, construct, internal, measurement, external, interpretive)
# Scores 0-3 per dimension, estimated from case study descriptions
case_studies = [
    ("Induction Heads",      "Triangulated",      3, 3, 2, 3, 2),
    ("Grokking",             "Triangulated",      3, 3, 3, 2, 3),
    ("Superposition",        "Triangulated",      2, 2, 2, 1, 2),
    ("IOI Circuit",          "Mech. Supported",   2, 2, 2, 2, 2),
    ("Greater-Than",         "Mech. Supported",   3, 2, 2, 1, 2),
    ("Copy Suppression",     "Mech. Supported",   2, 3, 2, 1, 2),
    ("Successor Heads",      "Caus. Suggestive",  2, 2, 1, 2, 1),
    ("Docstring Circuit",    "Caus. Suggestive",  1, 2, 1, 1, 1),
    ("SAE Features",         "Caus. Suggestive",  1, 1, 1, 1, 1),
    ("Othello World Model",  "Caus. Suggestive",  2, 2, 1, 1, 0),
    ("Knowledge Neurons",    "Caus. Suggestive",  1, 2, 1, 1, 1),
    ("Probing Classifiers",  "Proposed",          1, 0, 1, 0, 1),
    ("Gender Bias Circuits", "Proposed",          0, 1, 1, 0, 0),
]

# 27 criteria pass rates (estimated from 13 circuits)
criteria_data = [
    ("C1 Falsifiability",      10),
    ("C2 Structural plaus.",    9),
    ("C3 Task specificity",     7),
    ("C4 Minimality",           4),
    ("C5 Convergent validity",  3),
    ("I1 Necessity",           10),
    ("I2 Sufficiency",          5),
    ("I3 Specificity",          3),
    ("I4 Consistency",          4),
    ("I5 Confound control",     2),
    ("M1 Reliability",          2),
    ("M2 Invariance",           3),
    ("M3 Baseline separation",  8),
    ("M4 Sensitivity",          3),
    ("M5 Calibration",          2),
    ("M6 Construct coverage",   5),
    ("E1 Intervention reach",   6),
    ("E2 Graded response",      3),
    ("E3 Selectivity",          4),
    ("E4 Effect magnitude",     7),
    ("E5 Robustness",           3),
    ("E6 Cross-architecture",   2),
    ("V1 Level declaration",    8),
    ("V2 Level-evidence match", 6),
    ("V3 Narrative coherence",  7),
    ("V4 Alternative exclusion",3),
    ("V5 Scope honesty",        5),
]

# 9 automated papers: (name, raw_cvs, debiased_cvs, predicted_tier, expected_tier, off_by)
auto_papers = [
    ("Probing",           2.8, 1.4, "Proposed",         "Proposed",          0),
    ("Gender Bias",       5.0, 3.3, "Mech. Supported",  "Proposed",         +1),
    ("Successor Heads",   4.4, 3.1, "Caus. Suggestive", "Caus. Suggestive",  0),
    ("Greater-Than",      7.2, 5.6, "Mech. Supported",  "Caus. Suggestive", +1),
    ("Othello",           6.1, 4.4, "Mech. Supported",  "Caus. Suggestive", +1),
    ("IOI",               7.2, 5.6, "Mech. Supported",  "Mech. Supported",   0),
    ("Copy Suppression",  7.2, 5.6, "Mech. Supported",  "Mech. Supported",   0),
    ("Induction",         8.1, 6.4, "Triangulated",     "Mech. Supported",  +1),
    ("Grokking",          9.4, 8.3, "Validated",         "Triangulated",     +1),
]

tier_colors = {
    "Proposed": "#ef4444",
    "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981",
    "Triangulated": "#3b82f6",
    "Validated": "#7c3aed",
}

tier_order = ["Proposed", "Caus. Suggestive", "Mech. Supported", "Triangulated", "Validated"]
dim_names = ["Construct", "Internal", "Measurement", "External", "Interpretive"]

# ============================================================
# Figure 1: Validity heatmap (13 circuits × 5 dimensions)
# ============================================================
def fig1_heatmap():
    names = [c[0] for c in case_studies]
    verdicts = [c[1] for c in case_studies]
    data = np.array([c[2:] for c in case_studies])

    cell_colors = {
        0: ('#fef2f2', '#dc2626'),
        1: ('#fffbeb', '#b45309'),
        2: ('#ecfdf5', '#059669'),
        3: ('#dcfce7', '#15803d'),
    }

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.8, len(names))
    ax.axis('off')

    cell_w = 1.0
    cell_h = 0.75
    x_start = 2.8
    y_gap = 0.04

    for j, dname in enumerate(dim_names):
        ax.text(x_start + j * cell_w + cell_w/2, len(names) - 0.15, dname,
                ha='center', va='bottom', fontsize=10, fontweight='700',
                color='#1e293b', fontfamily='sans-serif')

    for i in range(len(names)):
        y = len(names) - 1 - i
        ax.text(x_start - 0.15, y * (cell_h + y_gap) + cell_h/2, names[i],
                ha='right', va='center', fontsize=10, color='#1e293b', fontfamily='sans-serif')

        v = verdicts[i]
        vc = tier_colors.get(v, '#64748b')
        pill_x = x_start + 5 * cell_w + 0.2
        pill = mpatches.FancyBboxPatch((pill_x, y * (cell_h + y_gap) + 0.08), 1.9, cell_h - 0.16,
                                        boxstyle='round,pad=0.08', facecolor=vc + '18',
                                        edgecolor=vc, linewidth=1.2)
        ax.add_patch(pill)
        ax.text(pill_x + 0.95, y * (cell_h + y_gap) + cell_h/2, v,
                ha='center', va='center', fontsize=8, fontweight='600',
                color=vc, fontfamily='sans-serif')

        for j in range(5):
            val = data[i, j]
            bg, tc = cell_colors[val]
            rect = mpatches.FancyBboxPatch(
                (x_start + j * cell_w + 0.03, y * (cell_h + y_gap) + 0.03),
                cell_w - 0.06, cell_h - 0.06,
                boxstyle='round,pad=0.06', facecolor=bg, edgecolor='#e2e8f0', linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x_start + j * cell_w + cell_w/2, y * (cell_h + y_gap) + cell_h/2,
                    str(val), ha='center', va='center',
                    fontsize=14, fontweight='700', color=tc, fontfamily='sans-serif')

    legend_y = -0.55
    scale_labels = ['None', 'Weak', 'Moderate', 'Strong']
    for val in range(4):
        bg, tc = cell_colors[val]
        lx = x_start + val * 1.3
        rect = mpatches.FancyBboxPatch((lx, legend_y), 0.4, 0.35,
                                        boxstyle='round,pad=0.04', facecolor=bg,
                                        edgecolor='#e2e8f0', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(lx + 0.2, legend_y + 0.175, str(val), ha='center', va='center',
                fontsize=11, fontweight='700', color=tc, fontfamily='sans-serif')
        ax.text(lx + 0.55, legend_y + 0.175, scale_labels[val], ha='left', va='center',
                fontsize=9, color='#475569', fontfamily='sans-serif')
    ax.text(x_start - 0.15, legend_y + 0.175, 'Score:', ha='right', va='center',
            fontsize=9, fontweight='600', color='#64748b', fontfamily='sans-serif')

    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{outdir}/fig1_heatmap.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("1: Validity heatmap")


# ============================================================
# Figure 2: Criteria pass rate bar chart
# ============================================================
def fig2_criteria_bars():
    names = [c[0] for c in criteria_data]
    counts = [c[1] for c in criteria_data]
    pcts = [c / 13 * 100 for c in counts]

    group_colors = {
        'C': '#6366f1', 'I': '#10b981', 'M': '#06b6d4',
        'E': '#f59e0b', 'V': '#8b5cf6'
    }
    colors = [group_colors[n[0]] for n in names]

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, pcts, color=colors, height=0.7, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, fontfamily='sans-serif')
    ax.set_xlabel('% of 13 circuits passing', fontsize=11, fontfamily='sans-serif')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (pct, count) in enumerate(zip(pcts, counts)):
        ax.text(pct + 1.5, i, f'{count}/13', va='center', fontsize=8.5,
                color='#475569', fontfamily='sans-serif')

    # Group labels
    group_ranges = [("Construct", 0, 4), ("Internal", 5, 9), ("Measurement", 10, 15),
                    ("External", 16, 21), ("Interpretive", 22, 26)]
    for label, start, end in group_ranges:
        mid = (start + end) / 2
        ax.text(-2, mid, label, ha='right', va='center', fontsize=8, fontweight='600',
                color=group_colors[label[0]], fontfamily='sans-serif', rotation=0)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig2_criteria_bars.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("2: Criteria pass rates")


# ============================================================
# Figure 3: Slope chart (before/after debiasing)
# ============================================================
def fig3_slope():
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, raw, debiased, pred, exp, off in auto_papers:
        color = tier_colors.get(exp, '#64748b')
        ax.plot([0, 1], [raw, debiased], '-o', color=color, linewidth=2,
                markersize=7, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        ax.text(-0.08, raw, name, ha='right', va='center', fontsize=9,
                color='#334155', fontfamily='sans-serif')
        ax.text(1.08, debiased, f'{debiased}', ha='left', va='center', fontsize=9,
                color='#334155', fontweight='600', fontfamily='sans-serif')

    ax.set_xlim(-0.6, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Raw (single run)', 'Debiased (3-run min)'],
                        fontsize=12, fontweight='600', fontfamily='sans-serif')
    ax.set_ylabel('CVS Score', fontsize=11, fontfamily='sans-serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 10)
    ax.axhline(y=2, color='#fecaca', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=4, color='#fef3c7', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=6, color='#d1fae5', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=8, color='#dbeafe', linewidth=0.8, linestyle='--', alpha=0.5)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig3_slope.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("3: Slope chart (debiasing)")


# ============================================================
# Figure 4: Pipeline architecture flowchart
# ============================================================
def fig4_pipeline_arch():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    box_style = dict(boxstyle="round,pad=0.4", linewidth=1.8)

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = mpatches.FancyBboxPatch((x, y), w, h, facecolor=bg, edgecolor=color, **box_style)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=11, fontweight='700', color='#1e293b', fontfamily='sans-serif')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                    fontsize=8.5, color='#475569', fontfamily='sans-serif')

    # Paper input
    draw_box(0.2, 1.0, 1.8, 1.2, "Paper\nPDF", None, '#64748b', '#f1f5f9')

    # 3 extraction calls
    for i, label in enumerate(["Run 1", "Run 2", "Run 3"]):
        draw_box(3.0, 0.1 + i * 1.1, 1.6, 0.9, "Extract", label, '#3b82f6', '#eff6ff')

    # Arrows from paper to extractions
    for i in range(3):
        ax.annotate('', xy=(3.0, 0.55 + i * 1.1), xytext=(2.0, 1.6),
                    arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5, mutation_scale=12))

    # Minimum vote
    draw_box(5.5, 1.0, 1.8, 1.2, "Minimum\nVote", "per criterion", '#f59e0b', '#fffbeb')
    for i in range(3):
        ax.annotate('', xy=(5.5, 1.6), xytext=(4.6, 0.55 + i * 1.1),
                    arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5, mutation_scale=12))

    # Evidence audit
    draw_box(8.2, 1.0, 1.8, 1.2, "Evidence\nAudit", "no paper text", '#ef4444', '#fef2f2')
    ax.annotate('', xy=(8.2, 1.6), xytext=(7.3, 1.6),
                arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5, mutation_scale=12))

    # Score
    draw_box(10.9, 1.0, 1.6, 1.2, "CVS\nScore", "27 criteria → 5 dims", '#10b981', '#ecfdf5')
    ax.annotate('', xy=(10.9, 1.6), xytext=(10.0, 1.6),
                arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5, mutation_scale=12))

    # Verdict
    draw_box(13.0, 1.0, 0.8, 1.2, "Tier", None, '#7c3aed', '#ede9fe')
    ax.annotate('', xy=(13.0, 1.6), xytext=(12.5, 1.6),
                arrowprops=dict(arrowstyle='-|>', color='#94a3b8', lw=1.5, mutation_scale=12))

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig4_pipeline_arch.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("4: Pipeline architecture")


# ============================================================
# Figure 5: Confusion matrix (automated vs reference)
# ============================================================
def fig5_confusion():
    tier_labels = ["Proposed", "Caus.\nSuggestive", "Mech.\nSupported", "Triangulated", "Validated"]
    matrix = np.zeros((5, 5), dtype=int)

    tier_idx = {"Proposed": 0, "Caus. Suggestive": 1, "Mech. Supported": 2,
                "Triangulated": 3, "Validated": 4}

    for name, raw, debiased, pred, exp, off in auto_papers:
        pi = tier_idx[pred]
        ei = tier_idx[exp]
        matrix[ei, pi] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3)

    ax.set_xticks(range(5))
    ax.set_xticklabels(tier_labels, fontsize=9, fontfamily='sans-serif', ha='center')
    ax.set_yticks(range(5))
    ax.set_yticklabels(tier_labels, fontsize=9, fontfamily='sans-serif')
    ax.set_xlabel('Predicted Tier', fontsize=12, fontweight='600', fontfamily='sans-serif')
    ax.set_ylabel('Reference Tier', fontsize=12, fontweight='600', fontfamily='sans-serif')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if val >= 2 else '#1e293b', fontfamily='sans-serif')

    # Diagonal line
    ax.plot([-0.5, 4.5], [-0.5, 4.5], '--', color='#ef4444', linewidth=1, alpha=0.5)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig5_confusion.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("5: Confusion matrix")


# ============================================================
# Figure 6: Radar/spider chart (3 circuits)
# ============================================================
def fig6_radar():
    from matplotlib.lines import Line2D

    circuits = [
        ("Induction Heads",      [3, 3, 2, 3, 2], '#3b82f6'),
        ("IOI Circuit",          [2, 2, 2, 2, 2], '#10b981'),
        ("Knowledge Neurons",    [1, 1, 1, 0, 1], '#ef4444'),
    ]

    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([])
    ax.set_ylim(0, 3.6)
    ax.set_rlabel_position(180)

    for r in [1, 2, 3]:
        theta_ring = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_ring, [r]*100, color='#d1d5db', linewidth=0.8, zorder=1)
    ax.plot(np.linspace(0, 2*np.pi, 100), [3]*100, color='#9ca3af', linewidth=1.2, zorder=1)
    for a in angles[:-1]:
        ax.plot([a, a], [0, 3], color='#d1d5db', linewidth=0.8, zorder=1)

    ring_labels = {1: 'Weak', 2: 'Moderate', 3: 'Strong'}
    for r, label in ring_labels.items():
        ax.text(angles[0], r, f'  {r} — {label}', ha='left', va='center',
                fontsize=9, color='#6b7280', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines['polar'].set_visible(False)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    label_pad = 0.7
    for i, name in enumerate(dim_names):
        angle = angles[i]
        x_cart = np.cos(np.pi/2 - angle)
        ha = 'center'
        if x_cart > 0.3:
            ha = 'left'
        elif x_cart < -0.3:
            ha = 'right'
        ax.text(angle, 3 + label_pad, name, ha=ha, va='center',
                fontsize=14, fontweight='700', color='#1e293b', fontfamily='sans-serif')

    for name, values, color in circuits:
        vals = values + values[:1]
        ax.plot(angles, vals, '-', linewidth=3, color=color, zorder=3)
        ax.plot(angles, vals, 'o', color=color, markersize=9,
                markeredgecolor='white', markeredgewidth=2.5, zorder=4)
        ax.fill(angles, vals, alpha=0.08, color=color, zorder=2)

    handles = [Line2D([0], [0], marker='o', color=c[2], linewidth=2.5, markersize=9,
                       markeredgecolor='white', markeredgewidth=2, label=c[0]) for c in circuits]
    leg = fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=12,
                     frameon=True, facecolor='white', edgecolor='#e2e8f0',
                     borderpad=0.7, columnspacing=2.0, handletextpad=0.6,
                     bbox_to_anchor=(0.5, -0.01))
    for text in leg.get_texts():
        text.set_fontfamily('sans-serif')
        text.set_color('#334155')

    fig.patch.set_facecolor('white')
    fig.subplots_adjust(bottom=0.06, top=0.94, left=0.04, right=0.96)
    fig.savefig(f'{outdir}/fig6_radar.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("6: Radar chart")


# ============================================================
# Figure 7: Evidence family × validity type matrix
# ============================================================
def fig6_radar_by_tier():
    # Generate a grid of small radar charts, one per case study, grouped by tier.
    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Layout: 4 columns
    cols = 4
    rows = int(np.ceil(len(case_studies) / cols))
    fig_w = 3.2 * cols
    fig_h = 2.6 * rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Rotation tweak to make bottom edge flatter: offset by one sector
    theta_offset = np.pi/2 + np.pi / N

    for idx, (name, verdict, *vals) in enumerate(case_studies):
        row = idx // cols
        col = idx % cols
        ax = plt.subplot(rows, cols, idx + 1, polar=True)
        ax.set_theta_offset(theta_offset)
        ax.set_theta_direction(-1)

        ax.set_ylim(0, 3.6)
        ax.set_yticks([])  # hide radial tick labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # hide axis labels (we'll add short labels only on top row)
        ax.spines['polar'].set_visible(False)

        # draw rings and radial lines lightly
        for r in [1, 2, 3]:
            theta_ring = np.linspace(0, 2*np.pi, 200)
            ax.plot(theta_ring, [r]*200, color='#e6edf3', linewidth=0.8, zorder=1)
        for a in angles[:-1]:
            ax.plot([a, a], [0, 3], color='#f1f5f9', linewidth=0.8, zorder=1)

        # Values
        vals = list(vals[0:5]) if len(vals) > 0 and isinstance(vals[0], (list, tuple)) else vals
        vals = [int(v) for v in vals]
        data = vals + vals[:1]

        color = tier_colors.get(verdict, '#64748b')
        ax.plot(angles, data, color=color, linewidth=2.2, zorder=3)
        ax.fill(angles, data, alpha=0.12, color=color, zorder=2)

        # Title on top of subplot, short
        ax.set_title(f"{name}\n({sum(vals)/len(vals):.1f})", fontsize=9.5,
                     fontweight='700', color='#1f2937', pad=8)

        # Draw a thin guide polygon for full scale to aid visual alignment
        max_vals = [3]*N + [3]
        ax.plot(angles, max_vals, color='#f8fafc', linewidth=1.0, zorder=0)

        # Optionally add small tick marks at axes for readability (no numbers)
        for a in angles[:-1]:
            ax.plot([a, a], [3.2, 3.35], color='#e2e8f0', linewidth=1.0, zorder=4)

        # Remove grid/frame clutter
        ax.set_frame_on(False)

    # Shared legend (tier labels) on the left
    handles = []
    for t in tier_order:
        handles.append(mpatches.Patch(facecolor=tier_colors[t], edgecolor=tier_colors[t], label=t))
    fig.legend(handles=handles, loc='lower center', ncol=len(tier_order), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(hspace=0.8, wspace=0.6, left=0.03, right=0.98, top=0.95, bottom=0.08)
    fig.savefig(f'{outdir}/fig_radar_by_tier_v2_new.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("6b: Radar grid by tier (improved)")


def fig7_family_matrix():
    families = ["Causal", "Structural", "Info-theoretic", "Behavioral", "Representational"]
    validity = dim_names

    # Which families contribute to which validity types (strength 0-3)
    matrix = np.array([
        [1, 3, 1, 2, 1],  # Causal → C,I,M,E,V
        [2, 1, 1, 1, 1],  # Structural
        [1, 2, 2, 1, 0],  # Info-theoretic
        [1, 2, 1, 3, 1],  # Behavioral
        [2, 1, 2, 1, 2],  # Representational
    ])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.YlGnBu
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(5))
    ax.set_xticklabels(validity, fontsize=10, fontweight='600', fontfamily='sans-serif')
    ax.set_yticks(range(5))
    ax.set_yticklabels(families, fontsize=10, fontfamily='sans-serif')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Validity Type', fontsize=11, fontweight='600', fontfamily='sans-serif')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Evidence Family', fontsize=11, fontweight='600', fontfamily='sans-serif')

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            label = ['—', '○', '●', '★'][val]
            color = 'white' if val >= 2 else '#1e293b'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=16, color=color, fontfamily='sans-serif')

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['None', 'Weak', 'Moderate', 'Strong'])

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig7_family_matrix.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("7: Evidence family × validity type")


# ============================================================
# Figure 8: Verdict tier list with circuits placed
# ============================================================
def fig8_tier_list():
    tiers = [
        ("Validated", '#7c3aed', '#ede9fe', []),
        ("Triangulated", '#3b82f6', '#eff6ff', ["Induction Heads"]),
        ("Mech. Supported", '#10b981', '#ecfdf5', ["IOI Circuit", "Greater-Than", "Copy Suppression"]),
        ("Caus. Suggestive", '#f59e0b', '#fffbeb', ["Grokking", "Successor Heads", "Docstring",
                                                      "SAE Features", "Othello"]),
        ("Proposed", '#ef4444', '#fef2f2', ["Knowledge Neurons", "Superposition",
                                             "Probing", "Gender Bias"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(tiers) * 1.4 + 0.5)
    ax.axis('off')

    for i, (tier_name, color, bg, circuits) in enumerate(reversed(tiers)):
        y = i * 1.4 + 0.3
        rect = mpatches.FancyBboxPatch((0.3, y), 2.8, 1.0, boxstyle="round,pad=0.1",
                                        facecolor=bg, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(1.7, y + 0.5, tier_name, ha='center', va='center',
                fontsize=12, fontweight='700', color=color, fontfamily='sans-serif')

        for j, circuit in enumerate(circuits):
            cx = 4.0 + j * 1.8
            pill = mpatches.FancyBboxPatch((cx, y + 0.15), 1.6, 0.7,
                                            boxstyle="round,pad=0.08",
                                            facecolor='white', edgecolor='#cbd5e1', linewidth=1)
            ax.add_patch(pill)
            ax.text(cx + 0.8, y + 0.5, circuit, ha='center', va='center',
                    fontsize=8.5, color='#1e293b', fontfamily='sans-serif')

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig8_tier_list.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("8: Verdict tier list")


# ============================================================
# Generate all
# ============================================================
fig1_heatmap()
fig2_criteria_bars()
fig3_slope()
fig4_pipeline_arch()
fig5_confusion()
fig6_radar()
fig6_radar_by_tier()
fig7_family_matrix()
fig8_tier_list()
print("\nAll figures saved!")
