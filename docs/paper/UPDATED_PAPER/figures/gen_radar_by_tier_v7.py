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

# Attempt to import authoritative harvey data
sys.path.insert(0, str(OUTDIR))
try:
    import gen_harvey_4level_v19_v2 as harvey
except Exception:
    harvey = None

# Default tier order and colors (will adopt harvey's if available)
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

# Convert harvey criteria values into 0-3 integer scale used in radars
STATUS_MAP = {1.0: 3, 0.7: 2, 0.3: 1, -1.0: 0, 0.0: 0}


def build_case_studies_from_harvey():
    if harvey is None:
        return None
    order = sorted(harvey.criteria_4level.keys(), key=lambda c: -harvey.cvs_scores.get(c, 0))
    cs = []
    for name in order:
        cdata = harvey.criteria_4level[name]
        dims = []
        for dim_keys in harvey.dim_criteria:
            vals = [STATUS_MAP.get(cdata.get(k, 0.0), 0) for k in dim_keys]
            avg = float(np.mean(vals)) if vals else 0.0
            dims.append(int(round(avg)))
        verdict = harvey.tier_map.get(name, 'Proposed')
        cs.append((name, verdict, dims[0], dims[1], dims[2], dims[3], dims[4]))
    return cs


def draw_single_radar(ax, vals, color='#64748b', show_dim_labels=True, title=None, title_pad=12,
                      label_fontsize=8.0, label_fontweight='normal', title_fontsize=9.5):
    """Draw one small radar on a polar axes.
    label_fontsize and title_fontsize allow per-call control (used for example inset).
    Title may include a newline to show a score on the second line (e.g. 'Name\n5.8/10').
    """
    data = vals + vals[:1]
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 3.2)
    ax.set_yticks([])
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)

    # subtle rings & spokes
    ring_color = '#d1d5db'
    for r in [1, 2, 3]:
        theta_ring = np.linspace(0, 2*np.pi, 200)
        ax.plot(theta_ring, [r]*200, color=ring_color, linewidth=0.8, zorder=1)
    for a in ANGLES[:-1]:
        ax.plot([a, a], [0, 3], color=ring_color, linewidth=0.8, zorder=1)

    ax.plot(ANGLES, data, color=color, linewidth=2.2, zorder=3)
    ax.fill(ANGLES, data, alpha=0.14, color=color, zorder=2)

    if show_dim_labels:
        label_radius = 3.35
        for i_dim, dname in enumerate(DIM_NAMES):
            angle = ANGLES[i_dim]
            x_cart = math.cos(np.pi/2 - angle)
            ha = 'center'
            if x_cart > 0.25:
                ha = 'left'
            elif x_cart < -0.25:
                ha = 'right'
            ax.text(angle, label_radius, dname, ha=ha, va='center', fontsize=label_fontsize,
                    fontweight=label_fontweight, color='#111827')

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='700', color='#0f172a', pad=title_pad)


def make_vertical_grouped_radars(case_studies, filename, show_dim_labels=True, title_pad_base=18):
    # group by tier
    groups = {t: [] for t in TIER_ORDER}
    for cs in case_studies:
        groups.setdefault(cs[1], []).append(cs)
    # preserve tier order
    tier_groups = [(t, groups.get(t, [])) for t in TIER_ORDER if groups.get(t)]

    # compute figure size: each tier is its own row
    max_cols = max(len(g) for _, g in tier_groups)
    n_rows = len(tier_groups)
    fig_w = max(3.2 * max_cols, 6)
    # increase vertical space per your request
    fig_h = 3.0 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # For each tier row, make subplots in that row
    for tier_idx, (tier_name, items) in enumerate(tier_groups):
        ncols = len(items)
        for j, (name, verdict, c, i, m, e, v) in enumerate(items):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + j + 1, polar=True)
            color = (harvey.tier_colors.get(tier_name) if harvey and hasattr(harvey, 'tier_colors')
                     else TIER_COLORS.get(tier_name, '#64748b'))
            # nudge title pad per tier-row
            title_pad = title_pad_base + tier_idx * 6
            # include cvs score under the name if available
            if harvey and hasattr(harvey, 'cvs_scores'):
                score = harvey.cvs_scores.get(name, None)
                title_text = f"{name}\n{score:.1f}/10" if score is not None else name
            else:
                title_text = name
            draw_single_radar(ax, [c, i, m, e, v], color=color, show_dim_labels=show_dim_labels,
                              title=title_text, title_pad=title_pad, label_fontweight='normal')

        # fill empty slots in the row with invisible axes to keep grid aligned
        for empty_col in range(ncols, max_cols):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + empty_col + 1, polar=True)
            ax.axis('off')

    # legend (tier colors) at bottom
    handles = [mpatches.Patch(facecolor=(harvey.tier_colors.get(t) if harvey and hasattr(harvey, 'tier_colors') else TIER_COLORS[t]),
                               edgecolor='none', label=t) for t in TIER_ORDER if groups.get(t)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    # increase top margin so top row titles don't get cut off
    fig.subplots_adjust(hspace=1.0, wspace=0.6, left=0.03, right=0.98, top=0.985, bottom=0.12)
    outpath = OUTDIR / filename
    fig.savefig(str(outpath), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', outpath)


def make_no_labels_with_example_legend(case_studies, filename):
    # same grouping as vertical version but hide dim labels on all individual plots
    # and add a small example radar with labels in the legend margin
    groups = {t: [] for t in TIER_ORDER}
    for cs in case_studies:
        groups.setdefault(cs[1], []).append(cs)
    tier_groups = [(t, groups.get(t, [])) for t in TIER_ORDER if groups.get(t)]

    max_cols = max(len(g) for _, g in tier_groups)
    n_rows = len(tier_groups)
    fig_w = max(3.2 * max_cols, 6)
    fig_h = 3.0 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # draw grouped radars without dim labels
    for tier_idx, (tier_name, items) in enumerate(tier_groups):
        ncols = len(items)
        for j, (name, verdict, c, i, m, e, v) in enumerate(items):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + j + 1, polar=True)
            color = (harvey.tier_colors.get(tier_name) if harvey and hasattr(harvey, 'tier_colors')
                     else TIER_COLORS.get(tier_name, '#64748b'))
            draw_single_radar(ax, [c, i, m, e, v], color=color, show_dim_labels=False, title=name, title_pad=14,
                              label_fontweight='normal')
        for empty_col in range(ncols, max_cols):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + empty_col + 1, polar=True)
            ax.axis('off')

    # add an example radar (with labels) in bottom-right scaled to main axes size
    inset_size = min(0.9 / max_cols, 0.22)
    inset_x = 0.98 - inset_size - 0.02
    inset_y = 0.06
    ax_inset = fig.add_axes([inset_x, inset_y, inset_size, inset_size], polar=True)
    # use the first case study (highest cvs) as the example if possible
    ex_name, ex_verdict, ex_c, ex_i, ex_m, ex_e, ex_v = case_studies[0]
    example_vals = [ex_c, ex_i, ex_m, ex_e, ex_v]
    # get score from harvey if present, else compute from values
    if harvey and hasattr(harvey, 'cvs_scores') and ex_name in harvey.cvs_scores:
        example_score = harvey.cvs_scores[ex_name]
    else:
        example_score = sum(example_vals) / (3.0 * len(example_vals)) * 10.0
    draw_single_radar(ax_inset, example_vals, color='#64748b', show_dim_labels=True,
                      title=f'Example Circuit\n{example_score:.1f}/10', title_pad=6,
                      label_fontsize=9.0, label_fontweight='normal', title_fontsize=9.0)

    # tier legend at bottom-right (no box variant)
    handles = [mpatches.Patch(facecolor=(harvey.tier_colors.get(t) if harvey and hasattr(harvey, 'tier_colors') else TIER_COLORS[t]),
                               edgecolor='none', label=t) for t in TIER_ORDER if groups.get(t)]
    fig.legend(handles=handles, loc='lower right', ncol=len(handles), fontsize=10,
               frameon=False, bbox_to_anchor=(0.98, -0.02))

    fig.subplots_adjust(hspace=1.0, wspace=0.6, left=0.03, right=0.98, top=0.985, bottom=0.12)
    outpath = OUTDIR / filename
    fig.savefig(str(outpath), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', outpath)

    # --- Now save a boxed-legend variant with the legend in a boxed axes to the left of the inset ---
    fig = plt.figure(figsize=(fig_w, fig_h))
    # redraw radars for boxed variant
    for tier_idx, (tier_name, items) in enumerate(tier_groups):
        ncols = len(items)
        for j, (name, verdict, c, i, m, e, v) in enumerate(items):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + j + 1, polar=True)
            color = (harvey.tier_colors.get(tier_name) if harvey and hasattr(harvey, 'tier_colors')
                     else TIER_COLORS.get(tier_name, '#64748b'))
            draw_single_radar(ax, [c, i, m, e, v], color=color, show_dim_labels=False, title=name, title_pad=14,
                              label_fontweight='normal')
        for empty_col in range(ncols, max_cols):
            ax = fig.add_subplot(n_rows, max_cols, tier_idx * max_cols + empty_col + 1, polar=True)
            ax.axis('off')

    # boxed legend axes
    legend_w = 0.18
    legend_h = 0.14
    legend_x = inset_x - legend_w - 0.02
    legend_y = 0.02
    ax_legend = fig.add_axes([legend_x, legend_y, legend_w, legend_h])
    ax_legend.axis('off')
    # background box
    box = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.12', transform=ax_legend.transAxes,
                                  facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2)
    ax_legend.add_patch(box)
    legend_handles = handles
    ax_legend.legend(handles=legend_handles, loc='center', ncol=len(legend_handles), frameon=False)

    # example radar same size as the main axes placed to the right of the legend box
    ax_ex = fig.add_axes([inset_x, inset_y, inset_size, inset_size], polar=True)
    draw_single_radar(ax_ex, example_vals, color='#64748b', show_dim_labels=True,
                      title=f'Example Circuit\n{example_score:.1f}/10', title_pad=6,
                      label_fontsize=9.0, label_fontweight='normal', title_fontsize=9.0)

    fig.subplots_adjust(hspace=1.0, wspace=0.6, left=0.03, right=0.98, top=0.985, bottom=0.12)
    outpath_box = OUTDIR / filename.replace('.png', '_box.png')
    fig.savefig(str(outpath_box), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', outpath_box)


if __name__ == '__main__':
    cs = build_case_studies_from_harvey() or [
        ("Induction Heads", "Triangulated", 3, 3, 2, 3, 2),
        ("Grokking", "Triangulated", 3, 3, 3, 2, 3),
        ("Superposition", "Triangulated", 2, 2, 2, 1, 2),
        ("IOI Circuit", "Mech. Supported", 2, 2, 2, 2, 2),
        ("Greater-Than", "Mech. Supported", 3, 2, 2, 1, 2),
        ("Copy Suppression", "Mech. Supported", 2, 3, 2, 1, 2),
        ("Successor Heads", "Caus. Suggestive", 2, 2, 1, 2, 1),
        ("Docstring Circuit", "Caus. Suggestive", 1, 2, 1, 1, 1),
        ("Probing Classifiers", "Proposed", 1, 0, 1, 0, 1),
    ]

    make_no_labels_with_example_legend(cs, 'fig_radar_no_labels_v7.png')
    make_vertical_grouped_radars(cs, 'fig_radar_vertical_v7.png', show_dim_labels=True, title_pad_base=18)
