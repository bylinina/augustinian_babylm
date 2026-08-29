#!/usr/bin/env python3
"""Redraw held-out mask-prediction loss by token class.

Reads eval/mlm_class_results.json; writes eval/plots/abstract_mlm.{pdf,png}.
Run from the repo root:  python eval/plot_abstract_mlm.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#2b2b2b"; BLUE = "#4477AA"; RED = "#CC6677"; LIGHT = "#DDDDDD"

res = json.load(open("eval/mlm_class_results.json"))

LABELS = {
    "function":       "function words\n(the, not, every, three)",
    "concrete(>=4)":  "concrete nouns\n(seeded, conc. \u22654)",
    "mid(2.5-4)":     "mid-concreteness\n(seeded, 2.5\u20134)",
    "abstract(<2.5)": "abstract words\n(seeded, conc. <2.5)",
    "seeded-other":   "other seeded\n(no norm available)",
    "unseeded":       "unseeded\n(placebo)",
}
order = ["function", "concrete(>=4)", "mid(2.5-4)",
         "abstract(<2.5)", "seeded-other", "unseeded"]

names = [LABELS[k] for k in order]
seeds = [res[k]["deltas"] for k in order]
means = [float(np.mean(s)) for s in seeds]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.axisbelow": True,
    "savefig.dpi": 200, "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(7.0, 3.4))
y = np.arange(len(names))[::-1]
colors = [BLUE if m < 0 else RED for m in means]
ax.barh(y, means, color=colors, height=0.6, zorder=3)
ax.axvline(0, color=INK, lw=1.0, zorder=2)
ax.grid(axis="x", color=LIGHT, lw=0.7, zorder=0)
for yi, sv in zip(y, seeds):
    ax.scatter(sv, [yi] * len(sv), color=INK, s=9, zorder=4, alpha=0.7)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8.8)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3, color=INK)

# Two short centered lines. Each is well inside the axes width, so neither
# can run off the canvas the way the old single long line did.
ax.set_xlabel("change in mask-prediction loss (vision-init $-$ baseline)\n"
              "$\\leftarrow$ seeded model predicts the held-out word better",
              fontsize=9, linespacing=1.5)

ax.set_xlim(-0.22, 0.10)
fig.tight_layout(pad=0.4)

out = Path("eval/plots")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "abstract_mlm.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "abstract_mlm.png", dpi=200,
            bbox_inches="tight", pad_inches=0.02)
print("wrote eval/plots/abstract_mlm.pdf and .png")
