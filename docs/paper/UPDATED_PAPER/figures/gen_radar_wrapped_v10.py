import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path(__file__).parent
sys.path.insert(0, str(OUTDIR))
try:
    import gen_harvey_4level_v19_v2 as harvey
except Exception:
    harvey = None

# Tier ordering and colors
TIER_ORDER = ["Triangulated", "Mech. Supported", "Caus. Suggestive", "Proposed"]
TIER_COLORS = {
    "Proposed": "#ef4444",
    "Caus. Suggestive": "#f59e0b",
    "Mech. Supported": "#10b981",
    "Triangulated": "#3b82f6",
}

DIM_NAMES = ["Construct", "Internal", "Measurement", "External", "Interpretive"]
N_DIM = len(DIM_NAMES)
ANGLES = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False).tolist()
ANGLES += ANGLES[:1]
STATUS_MAP = {1.0: 3, 0.7: 2, 0.3: 1, -1.0: 0, 0.0: 0}


def build_case_studies():
    if harvey is None:
        # fallback list
        return [
            ("Induction Heads", "Triangulated", 3, 3, 2, 3, 2),
            ("Grokking", "Triangulated", 3, 3, 3, 2, 3),
            ("Superposition", "Triangulated", 2, 2, 2, 1, 2),
            ("IOI Circuit", "Mech. Supported", 2, 2, 2, 2, 2),
            ("Greater-Than", "Mech. Supported", 3, 2, 2, 1, 2),
            ("Copy Suppression", "Mech. Supported", 2, 3, 2, 1, 2),
            ("Successor Heads", "Caus. Suggestive", 2, 2, 1, 2, 1),
            ("Docstring Circuit", "Caus. Suggestive", 1, 2, 1, 1, 1),
            ("Probing Classifiers", "Proposed", 1, 0, 1, 0, 1),
            ("Knowledge Neurons", "Caus. Suggestive", 1,2,1,1,1),
            ("Othello World Model", "Caus. Suggestive", 2,2,1,1,0),
            ("SAE Features", "Caus. Suggestive",1,1,1,1,1),
            ("Gender Bias Circuits", "Proposed", 0,1,1,0,0),
        ]
    # build from harvey ordering by tier then cvs score
    items = []
    # gather names grouped by tier in TIER_ORDER
    names = []
    for tier in TIER_ORDER:
        for name, cdata in harvey.criteria_4level.items():
            if harvey.tier_map.get(name) == tier:
                names.append((tier, name))
    # stable sort by tier then descending cvs
    names_sorted = sorted(names, key=lambda tn: (TIER_ORDER.index(tn[0]), -harvey.cvs_scores.get(tn[1], 0)))
    for tier, name in names_sorted:
        cdata = harvey.criteria_4level[name]
        dims = []
        for dim_keys in harvey.dim_criteria:
            vals = [STATUS_MAP.get(cdata.get(k, 0.0), 0) for k in dim_keys]
            avg = float(np.mean(vals)) if vals else 0.0
            dims.append(int(round(avg)))
        verdict = harvey.tier_map.get(name, 'Proposed')
        items.append((name, verdict, dims[0], dims[1], dims[2], dims[3], dims[4]))
    return items


def draw_radar(ax, vals, color, show_labels=False, title=None):
    data = vals + vals[:1]
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 3.2)
    ax.set_yticks([])
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ring_color = '#e6eef8'
    for r in [1,2,3]:
        theta_ring = np.linspace(0,2*np.pi,200)
        ax.plot(theta_ring, [r]*200, color=ring_color, linewidth=0.8, zorder=1)
    for a in ANGLES[:-1]:
        ax.plot([a,a],[0,3], color=ring_color, linewidth=0.8, zorder=1)
    ax.plot(ANGLES, data, color=color, linewidth=2.0, zorder=3)
    ax.fill(ANGLES, data, alpha=0.14, color=color, zorder=2)
    if show_labels:
        label_radius = 3.35
        for i_dim, dname in enumerate(DIM_NAMES):
            angle = ANGLES[i_dim]
            x_cart = math.cos(np.pi/2 - angle)
            ha='center'
            if x_cart>0.25: ha='left'
            if x_cart<-0.25: ha='right'
            ax.text(angle, label_radius, dname, ha=ha, va='center', fontsize=7, color='#111827')
    if title:
        ax.set_title(title, fontsize=9, fontweight='700', pad=8)


def make_wrapped(case_studies, cols=3, filename='fig_radar_wrapped_v10.png'):
    total = len(case_studies)
    rows = int(math.ceil(total / float(cols)))
    fig_w = 3.0 * cols
    fig_h = 2.6 * rows
    fig = plt.figure(figsize=(fig_w, fig_h))
    for idx, (name, verdict, c, i, m, e, v) in enumerate(case_studies):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(rows, cols, idx+1, polar=True)
        color = (harvey.tier_colors.get(verdict) if harvey and hasattr(harvey, 'tier_colors') else TIER_COLORS.get(verdict, '#64748b'))
        # title with score if available
        if harvey and hasattr(harvey, 'cvs_scores'):
            score = harvey.cvs_scores.get(name, None)
            title = f"{name}\n{score:.1f}/10" if score is not None else name
        else:
            title = name
        draw_radar(ax, [c,i,m,e,v], color, show_labels=False, title=title)
    # legend boxed on bottom
    handles = [mpatches.Patch(facecolor=(harvey.tier_colors.get(t) if harvey and hasattr(harvey, 'tier_colors') else TIER_COLORS[t]), edgecolor='none', label=t) for t in TIER_ORDER if any(cs[1]==t for cs in case_studies)]
    # place boxed legend axis
    ax_legend = fig.add_axes([0.02, 0.02, 0.96, 0.08])
    ax_legend.axis('off')
    box = mpatches.FancyBboxPatch((0.02,0.05),0.96,0.9, transform=ax_legend.transAxes, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.0)
    ax_legend.add_patch(box)
    ax_legend.legend(handles=handles, loc='center', ncol=len(handles), frameon=False)
    fig.subplots_adjust(hspace=0.6, wspace=0.5, left=0.03, right=0.98, top=0.96, bottom=0.16)
    outpath = OUTDIR / filename
    fig.savefig(str(outpath), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', outpath)

if __name__ == '__main__':
    cs = build_case_studies()
    make_wrapped(cs, cols=3)
