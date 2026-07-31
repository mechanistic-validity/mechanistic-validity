import os
import numpy as np
import matplotlib
# Use Agg backend for environments without display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

outdir = os.path.dirname(__file__)

# Build case studies from authoritative harvey data copy
import sys
sys.path.insert(0, outdir)
try:
    import gen_harvey_4level_v19_v2 as harvey
except Exception:
    # Fallback: keep internal sample data if import fails
    harvey = None

if harvey is None:
    # fallback static list
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
else:
    # Map harvey status values to 0-3 scale used by the radar script
    status_map = {1.0: 3, 0.7: 2, 0.3: 1, -1.0: 0, 0.0: 0}
    case_studies = []
    # preserve ordering by cvs_scores
    order = sorted(harvey.criteria_4level.keys(), key=lambda c: -harvey.cvs_scores.get(c, 0))
    for name in order:
        cdata = harvey.criteria_4level[name]
        dims = []
        for dim_keys in harvey.dim_criteria:
            vals = [status_map.get(cdata[k], 0) for k in dim_keys]
            avg = float(np.mean(vals)) if vals else 0.0
            dims.append(int(round(avg)))
        verdict = harvey.tier_map.get(name, 'Proposed')
        case_studies.append((name, verdict, dims[0], dims[1], dims[2], dims[3], dims[4]))
    # adopt harvey tier colors if present
    try:
        tier_colors.update(harvey.tier_colors)
    except Exception:
        pass

# Tier colors for legend
tier_colors = {
    "Proposed": "#ef4444",
    "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981",
    "Triangulated": "#3b82f6",
    "Validated": "#7c3aed",
}

dim_names = ["Construct", "Internal", "Measurement", "External", "Interpretive"]


def make_radar_grid(case_studies, cols=4, filename='fig_radar_by_tier_v5.png'):
    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    rows = int(np.ceil(len(case_studies) / cols))
    fig_w = 3.2 * cols
    fig_h = 2.6 * rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Center vertically (theta_offset = pi/2) for symmetry
    theta_offset = np.pi/2

    for idx, (name, verdict, c, i, m, e, v) in enumerate(case_studies):
        row = idx // cols
        col = idx % cols
        ax = plt.subplot(rows, cols, idx + 1, polar=True)
        ax.set_theta_offset(theta_offset)
        ax.set_theta_direction(-1)

        ax.set_ylim(0, 3.2)
        ax.set_yticks([])  # hide radial tick labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # hide axis labels (we'll add custom labels)
        ax.spines['polar'].set_visible(False)

        # subtle guide rings and spokes using a neutral gray
        ring_color = '#d1d5db'
        spoke_color = '#d1d5db'
        for r in [1, 2, 3]:
            theta_ring = np.linspace(0, 2*np.pi, 200)
            ax.plot(theta_ring, [r]*200, color=ring_color, linewidth=0.9, zorder=1)
        for a in angles[:-1]:
            ax.plot([a, a], [0, 3], color=spoke_color, linewidth=0.9, zorder=1)

        vals = [c, i, m, e, v]
        data = vals + vals[:1]

        color = tier_colors.get(verdict, '#64748b')
        ax.plot(angles, data, color=color, linewidth=2.4, zorder=3)
        ax.fill(angles, data, alpha=0.14, color=color, zorder=2)

        # move title slightly upwards for readability
        ax.set_title(f"{name}\n({sum(vals)/len(vals):.1f})", fontsize=9.5,
                     fontweight='700', color='#0f172a', pad=12)

        # labels for each dimension placed outside the plot with consistent spacing
        label_radius = 3.35
        for i_dim, dname in enumerate(dim_names):
            angle = angles[i_dim]
            x_cart = np.cos(np.pi/2 - angle)
            ha = 'center'
            if x_cart > 0.25:
                ha = 'left'
            elif x_cart < -0.25:
                ha = 'right'
            ax.text(angle, label_radius, dname, ha=ha, va='center', fontsize=8.5,
                    fontweight='600', color='#111827')

        # small outer tick markers for alignment using neutral gray
        for a in angles[:-1]:
            ax.plot([a, a], [3.05, 3.25], color='#e6eef8', linewidth=1.0, zorder=4)

    # shared legend
    handles = [mpatches.Patch(facecolor=tier_colors[t], edgecolor=tier_colors[t], label=t) for t in tier_colors]
    fig.legend(handles=handles, loc='lower center', ncol=len(tier_colors), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(hspace=0.9, wspace=0.6, left=0.03, right=0.98, top=0.95, bottom=0.08)
    outpath = os.path.join(outdir, filename)
    fig.savefig(outpath, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', outpath)


if __name__ == '__main__':
    make_radar_grid(case_studies)
