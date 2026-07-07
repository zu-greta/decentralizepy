"""Shared plotting style for every figure in this project.

Rules (apply to ALL current and future plots):
  * NEVER use a dual / twin y-axis. If two quantities have different units, use
    stacked panels (share the x-axis) instead — see `stacked_panels`.
  * Colour-blind-safe palette only (Okabe-Ito). Also vary line style + marker so
    figures survive greyscale printing.
  * Always: descriptive title, axis labels with units, a legend, light grid,
    despined axes. No chartjunk.

Import this from any plot script:
    import os, sys; sys.path.insert(0, os.path.dirname(__file__))
    import plotstyle as ps
    ps.apply()
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette (8 colours) + semantic aliases
OKABE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#7F7F7F",
}
CYCLE = [OKABE[k] for k in ["blue", "orange", "green", "red", "purple", "sky", "black"]]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# semantic colours (kept consistent across every figure)
C_HONEST = OKABE["blue"]
C_FR     = OKABE["red"]
C_ACC    = OKABE["green"]
C_EFFORT = OKABE["blue"]
C_ETA    = OKABE["black"]
C_GOOD   = OKABE["green"]
C_BAD    = OKABE["red"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def style(i):
    """Consistent (color, linestyle, marker) for the i-th series."""
    return dict(color=CYCLE[i % len(CYCLE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)])


def stacked_panels(n, figsize=None, height_ratios=None):
    """n vertically-stacked panels sharing the x-axis — the approved replacement
    for a dual y-axis. Returns (fig, axes)."""
    fig, axes = plt.subplots(n, 1, sharex=True,
                             figsize=figsize or (9, 2.6 * n),
                             gridspec_kw={"height_ratios": height_ratios})
    if n == 1:
        axes = [axes]
    return fig, axes


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def cbar_label(mappable, ax, label):
    cb = ax.figure.colorbar(mappable, ax=ax)
    cb.set_label(label)
    return cb
