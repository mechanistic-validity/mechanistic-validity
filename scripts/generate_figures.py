"""Generate candidate figures for the mechanistic validity docs site.

Produces multiple style variants for each figure so the user can compare.
Output: docs/public/figures/<figure_name>_<style>.svg
"""

import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "public" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STYLES = {
    "academic": {
        "bg": "#ffffff", "fg": "#1a1a1a", "grid": "#e0e0e0",
        "accent": ["#2563eb", "#dc2626", "#16a34a", "#d97706"],
        "font": "serif",
    },
    "dark": {
        "bg": "#1a1a2e", "fg": "#e0e0e0", "grid": "#2a2a4a",
        "accent": ["#60a5fa", "#f87171", "#4ade80", "#fbbf24"],
        "font": "sans-serif",
    },
    "minimal": {
        "bg": "#fafafa", "fg": "#333333", "grid": "#eeeeee",
        "accent": ["#3b82f6", "#ef4444", "#22a352", "#f59e0b"],
        "font": "sans-serif",
    },
    "nature": {
        "bg": "#ffffff", "fg": "#2d2d2d", "grid": "#d4d4d4",
        "accent": ["#0077b6", "#e63946", "#2a9d8f", "#e9c46a"],
        "font": "serif",
    },
}


def fig1_selectivity_index(style_name, s):
    """Bar chart: selectivity indices for four circuit claims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    circuits = ["IOI\n(Wang 2022)", "Greater-Than\n(Hanna 2023)", "Induction\n(Olsson 2022)", "Hypothetical\nnon-selective"]
    si_values = [18.5, 14.2, 9.8, 3.1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])

    colors = [s["accent"][0], s["accent"][3], s["accent"][2], s["accent"][1]]
    bars = ax.bar(circuits, si_values, color=colors, width=0.6, edgecolor=s["fg"], linewidth=0.5, alpha=0.85)

    ax.axhline(y=10, color=s["accent"][1], linestyle="--", linewidth=1.5, alpha=0.7, label="SI = 10 threshold")
    ax.set_ylabel("Selectivity Index (SI)", fontsize=12, color=s["fg"], fontfamily=s["font"])
    ax.set_title("Selectivity Index Across Circuit Claims", fontsize=14, color=s["fg"], fontfamily=s["font"], pad=12)
    ax.tick_params(colors=s["fg"], labelsize=10)
    ax.set_ylim(0, 25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(s["fg"])
    ax.yaxis.grid(True, color=s["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, labelcolor=s["fg"])

    for bar, val in zip(bars, si_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val}",
                ha="center", va="bottom", fontsize=10, color=s["fg"], fontfamily=s["font"])

    plt.tight_layout()
    fig.savefig(OUT / f"selectivity_index_{style_name}.svg", facecolor=s["bg"], bbox_inches="tight")
    fig.savefig(OUT / f"selectivity_index_{style_name}.png", facecolor=s["bg"], bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig2_signal_detection(style_name, s):
    """Two-panel: overlapping normal distributions for d' framework."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.patch.set_facecolor(s["bg"])

    x = np.linspace(-3, 7, 500)

    for ax, title, mu_signal, subtitle in [
        (ax1, "Standard baselines", 3.0, "d′ = 2.0"),
        (ax2, "High random baseline", 1.8, "d′ = 0.8"),
    ]:
        ax.set_facecolor(s["bg"])
        mu_noise = 1.0

        noise = norm.pdf(x, mu_noise, 1.0)
        signal = norm.pdf(x, mu_signal, 1.0)

        ax.fill_between(x, noise, alpha=0.3, color=s["accent"][1], label="Non-circuit")
        ax.fill_between(x, signal, alpha=0.3, color=s["accent"][0], label="Circuit")
        ax.plot(x, noise, color=s["accent"][1], linewidth=1.5)
        ax.plot(x, signal, color=s["accent"][0], linewidth=1.5)

        criterion = (mu_noise + mu_signal) / 2
        ax.axvline(criterion, color=s["fg"], linestyle="--", linewidth=1, alpha=0.6)
        ax.annotate("criterion", xy=(criterion, 0.42), fontsize=9, color=s["fg"],
                     ha="center", fontfamily=s["font"])

        mid_y = 0.05
        ax.annotate("", xy=(mu_signal, mid_y), xytext=(mu_noise, mid_y),
                     arrowprops=dict(arrowstyle="<->", color=s["accent"][2], lw=2))
        ax.text((mu_noise + mu_signal) / 2, mid_y + 0.02, subtitle,
                ha="center", fontsize=11, color=s["accent"][2], fontweight="bold", fontfamily=s["font"])

        ax.set_title(title, fontsize=12, color=s["fg"], fontfamily=s["font"], pad=10)
        ax.set_xlabel("Instrument score", fontsize=11, color=s["fg"], fontfamily=s["font"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ax.spines.values():
            sp.set_color(s["fg"])
        ax.tick_params(colors=s["fg"], labelsize=9)
        ax.set_xlim(-2.5, 6.5)
        ax.set_ylim(0, 0.48)

    ax1.set_ylabel("Density", fontsize=11, color=s["fg"], fontfamily=s["font"])
    ax1.legend(fontsize=9, frameon=False, labelcolor=s["fg"], loc="upper left")

    fig.suptitle("Signal Detection Framework for Circuit Identification",
                 fontsize=14, color=s["fg"], fontfamily=s["font"], y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / f"signal_detection_{style_name}.svg", facecolor=s["bg"], bbox_inches="tight")
    fig.savefig(OUT / f"signal_detection_{style_name}.png", facecolor=s["bg"], bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig3_marr_levels(style_name, s):
    """Marr's three levels with implementational sub-modes and drift arrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.3", linewidth=1.5)

    levels = [
        (5, 8.5, "Computational", s["accent"][0], "What is computed and why"),
        (5, 6.0, "Algorithmic", s["accent"][2], "What procedure is executed"),
    ]

    submodes = [
        (2.5, 2.8, "Topographic"),
        (4.5, 2.8, "Connectomic"),
        (6.5, 2.8, "Activation-\nstatistical"),
        (8.5, 2.8, "Functional"),
    ]

    for cx, cy, label, color, desc in levels:
        box = FancyBboxPatch((cx - 1.8, cy - 0.45), 3.6, 0.9,
                              facecolor=color, alpha=0.15, edgecolor=color, **box_kw)
        ax.add_patch(box)
        ax.text(cx, cy + 0.05, label, ha="center", va="center", fontsize=13,
                fontweight="bold", color=color, fontfamily=s["font"])
        ax.text(cx + 2.0, cy, desc, ha="left", va="center", fontsize=9,
                color=s["fg"], alpha=0.7, fontfamily=s["font"])

    impl_box = FancyBboxPatch((1.0, 1.8), 8.0, 2.2,
                               facecolor=s["accent"][3], alpha=0.08,
                               edgecolor=s["accent"][3], **box_kw)
    ax.add_patch(impl_box)
    ax.text(5, 3.7, "Implementational", ha="center", va="center", fontsize=13,
            fontweight="bold", color=s["accent"][3], fontfamily=s["font"])

    for cx, cy, label in submodes:
        box = FancyBboxPatch((cx - 0.85, cy - 0.35), 1.7, 0.7,
                              facecolor=s["accent"][3], alpha=0.12,
                              edgecolor=s["accent"][3], linewidth=1,
                              boxstyle="round,pad=0.2")
        ax.add_patch(box)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.5,
                color=s["fg"], fontfamily=s["font"])

    ax.annotate("", xy=(5, 5.5), xytext=(5, 4.1),
                arrowprops=dict(arrowstyle="-|>", color=s["accent"][2], lw=1.5))
    ax.text(5.15, 4.8, "I/O function\ndemonstration", fontsize=8, color=s["fg"],
            ha="left", alpha=0.7, fontfamily=s["font"])

    ax.annotate("", xy=(5, 8.0), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color=s["accent"][0], lw=1.5))
    ax.text(5.15, 7.2, "Generalization +\nedge-case coverage", fontsize=8,
            color=s["fg"], ha="left", alpha=0.7, fontfamily=s["font"])

    ax.annotate("", xy=(1.5, 8.2), xytext=(1.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=s["accent"][1], lw=2.5,
                               linestyle="--", connectionstyle="arc3,rad=0.3"))
    ax.text(0.4, 6.0, "common\ndrift", fontsize=9, color=s["accent"][1],
            fontweight="bold", ha="center", rotation=90, fontfamily=s["font"])

    ax.set_title("Description Levels and Evidence Requirements",
                 fontsize=14, color=s["fg"], fontfamily=s["font"], pad=15)

    fig.savefig(OUT / f"marr_levels_{style_name}.svg", facecolor=s["bg"], bbox_inches="tight")
    fig.savefig(OUT / f"marr_levels_{style_name}.png", facecolor=s["bg"], bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for style_name, s in STYLES.items():
        print(f"Generating {style_name} style...")
        fig1_selectivity_index(style_name, s)
        fig2_signal_detection(style_name, s)
        fig3_marr_levels(style_name, s)

    print(f"\nDone. {len(STYLES) * 3 * 2} files written to {OUT}")
    for f in sorted(OUT.glob("*.svg")):
        print(f"  {f.name}")
