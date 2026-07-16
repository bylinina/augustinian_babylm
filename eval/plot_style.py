"""Shared minimal plotting style for project figures."""
import matplotlib as mpl

INK = "#2b2b2b"
BLUE = "#4477AA"     # vision-init / primary accent
GRAY = "#777777"     # baseline / neutral
RED = "#CC6677"      # negative
GREEN = "#228833"    # all-9-positive highlight
LIGHT = "#DDDDDD"

def setup():
    mpl.rcParams.update({
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    })

def style_ax(ax, ygrid_only=True):
    if ygrid_only:
        ax.grid(axis="x", visible=False)
