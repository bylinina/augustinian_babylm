#!/usr/bin/env python
"""
Plot training-dynamics curves from collect.py's CSV.
Usage:  python plot_dynamics.py --csv results_dynamics.csv --outdir plots
Needs:  pip install --user matplotlib pandas   (no GPU needed)
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots"))
    args = ap.parse_args()
    args.outdir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    avg = df[(df.section == "average") & (df.step >= 1)]
    vocabs = [v for v in ["50k", "75k", "100k"] if v in set(avg.vocab)]
    tasks = sorted(avg.task.unique())

    fig, axes = plt.subplots(1, max(len(vocabs), 1),
                             figsize=(5 * max(len(vocabs), 1), 4), sharey=True)
    if len(vocabs) == 1:
        axes = [axes]
    for ax, v in zip(axes, vocabs):
        sub = avg[avg.vocab == v]
        for t in tasks:
            ts = sub[sub.task == t].sort_values("step")
            ax.plot(ts.step, ts.value, marker="o", ms=3, label=t)
        ax.set_xscale("log"); ax.set_title(f"vocab {v}")
        ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "overview.png", dpi=150)

    for t in tasks:
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in vocabs:
            ts = avg[(avg.task == t) & (avg.vocab == v)].sort_values("step")
            ax.plot(ts.step, ts.value, marker="o", ms=3, label=v)
        ax.set_xscale("log"); ax.set_title(t)
        ax.set_xlabel("training step"); ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"{t}.png", dpi=150)
    print(f"plots -> {args.outdir}/")


if __name__ == "__main__":
    main()
