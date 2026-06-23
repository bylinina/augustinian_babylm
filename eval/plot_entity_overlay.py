#!/usr/bin/env python
"""Single-panel overlay: entity-tracking (encoder - baseline) vs step, all three
vocabs, to show the transient spike is a 50k phenomenon that vanishes at higher
vocab. Reads collect.py's fast-eval CSV. Usage:
  python plot_entity_overlay.py --csv results_dynamics.csv --out plots/entity_overlay.png
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", type=Path, default=Path("plots/entity_overlay.png"))
ap.add_argument("--task", default="entity_tracking_fast")
args = ap.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.csv)
d = df[(df.task == args.task) & (df.section == "average") & (df.step >= 1)]

VOCABS = ["50k", "75k", "100k"]
ENCODERS = ["dinov3", "sam", "ibot"]
VCOLOR = {"50k": "tab:red", "75k": "tab:blue", "100k": "tab:green"}

fig, ax = plt.subplots(figsize=(9, 5.5))
for v in VOCABS:
    base = d[(d.vocab == v) & (d.init == "baseline")].set_index("step")["value"]
    if base.empty:
        continue
    # mean delta across the 3 encoders at each step
    deltas = []
    steps = sorted(base.index)
    for enc in ENCODERS:
        e = d[(d.vocab == v) & (d.init == enc)].set_index("step")["value"]
        common = [s for s in steps if s in e.index]
        deltas.append(pd.Series({s: e[s] - base[s] for s in common}))
    mean_delta = pd.concat(deltas, axis=1).mean(axis=1).sort_index()
    ax.plot(mean_delta.index, mean_delta.values, marker="o", ms=3,
            color=VCOLOR[v], label=f"{v} (mean of 3 encoders)")

ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xscale("symlog")
ax.set_xlabel("training step (symlog)")
ax.set_ylabel("entity-tracking accuracy: encoder - baseline (pts)")
ax.set_title("Vision-init effect on entity tracking, by vocabulary size")
ax.legend()
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print("wrote", args.out)
