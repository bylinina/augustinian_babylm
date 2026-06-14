#!/usr/bin/env python
"""
Plot training-dynamics curves from collect.py's CSV (with vocab + init columns).
Produces:
  overview.png            : per-vocab panels, baseline only, tasks overlaid
  visioninit_<task>.png   : per task, 50k baseline vs dinov3/sam/ibot (the
                            vision-init comparison)
  visioninit_overview.png : all tasks, 4 init lines each, small-multiples
Usage:  python plot_dynamics.py --csv results_dynamics.csv --outdir plots
Needs:  pip install --user matplotlib pandas
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INIT_ORDER = ["baseline", "dinov3", "sam", "ibot"]
INIT_COLORS = {"baseline": "black", "dinov3": "tab:blue",
               "sam": "tab:orange", "ibot": "tab:green"}
INIT_STYLE = {"baseline": "--", "dinov3": "-", "sam": "-", "ibot": "-"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots"))
    ap.add_argument("--init_vocab", default="50k",
                    help="which vocab has the vision-init runs to compare")
    args = ap.parse_args()
    args.outdir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    avg = df[(df.section == "average") & (df.step >= 1)]
    tasks = sorted(avg.task.unique())

    # ---- original: per-vocab overview, BASELINE only ----
    base = avg[avg.init == "baseline"]
    vocabs = [v for v in ["50k", "75k", "100k"] if v in set(base.vocab)]
    fig, axes = plt.subplots(1, max(len(vocabs), 1),
                             figsize=(5 * max(len(vocabs), 1), 4), sharey=True)
    if len(vocabs) == 1:
        axes = [axes]
    for ax, v in zip(axes, vocabs):
        sub = base[base.vocab == v]
        for t in tasks:
            ts = sub[sub.task == t].sort_values("step")
            ax.plot(ts.step, ts.value, marker="o", ms=3, label=t)
        ax.set_xscale("log"); ax.set_title(f"vocab {v} (baseline)")
        ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.outdir / "overview.png", dpi=150)
    plt.close(fig)

    # ---- vision-init comparison at init_vocab: 4 lines per task ----
    vi = avg[avg.vocab == args.init_vocab]
    inits = [i for i in INIT_ORDER if i in set(vi.init)]

    # per-task figures
    for t in tasks:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for ini in inits:
            ts = vi[(vi.task == t) & (vi.init == ini)].sort_values("step")
            if len(ts):
                ax.plot(ts.step, ts.value, INIT_STYLE.get(ini, "-"),
                        color=INIT_COLORS.get(ini), marker="o", ms=3,
                        label=ini, alpha=0.9)
        ax.set_xscale("log"); ax.set_title(f"{t}  (vocab {args.init_vocab})")
        ax.set_xlabel("training step"); ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.outdir / f"visioninit_{t}.png", dpi=150)
        plt.close(fig)

    # small-multiples overview: all tasks, 4 init lines each
    n = len(tasks)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows),
                             squeeze=False)
    for idx, t in enumerate(tasks):
        ax = axes[idx // cols][idx % cols]
        for ini in inits:
            ts = vi[(vi.task == t) & (vi.init == ini)].sort_values("step")
            if len(ts):
                ax.plot(ts.step, ts.value, INIT_STYLE.get(ini, "-"),
                        color=INIT_COLORS.get(ini), marker="o", ms=2.5,
                        label=ini, alpha=0.9)
        ax.set_xscale("log"); ax.set_title(t, fontsize=10)
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
        if idx % cols == 0:
            ax.set_ylabel("accuracy (%)")
    # hide any empty panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Vision-init vs baseline (vocab {args.init_vocab})", y=1.0)
    fig.tight_layout()
    fig.savefig(args.outdir / "visioninit_overview.png", dpi=150)
    plt.close(fig)

    print(f"plots -> {args.outdir}/ (overview.png, visioninit_overview.png, "
          f"visioninit_<task>.png)")


if __name__ == "__main__":
    main()
