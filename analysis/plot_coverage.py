#!/usr/bin/env python
"""
Stage 1.5 plots: coverage of text-only corpus by image-grounded text.
Reads analysis/out/{word_coverage.csv, token_coverage_*.csv, *_summary.json}.
Needs: pip install --user matplotlib pandas   (no torch).

Usage: python analysis/plot_coverage.py --out_dir analysis/out
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VOCABS = ["50k", "75k", "100k"]


def plot_coverage_bars(out_dir):
    tok = json.load(open(out_dir / "token_coverage_summary.json"))
    word = json.load(open(out_dir / "word_coverage_summary.json"))
    thresholds = ["image_freq_ge_1", "image_freq_ge_5", "image_freq_ge_20"]
    labels = ["≥1", "≥5", "≥20"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(thresholds))
    w = 0.2
    # type coverage
    for i, v in enumerate(VOCABS):
        vals = [tok[v][t]["type_coverage"] * 100 for t in thresholds]
        axes[0].bar(x + (i - 1) * w, vals, w, label=f"{v} tokens")
    axes[0].plot(x, [word["coverage"][t]["type_coverage"] * 100 for t in thresholds],
                 "ko--", label="words", ms=6)
    axes[0].set_title("Type coverage\n(% of distinct units that are vision-seedable)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("min image-text frequency"); axes[0].set_ylabel("%")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # token coverage
    for i, v in enumerate(VOCABS):
        vals = [tok[v][t]["token_coverage"] * 100 for t in thresholds]
        axes[1].bar(x + (i - 1) * w, vals, w, label=f"{v} tokens")
    axes[1].plot(x, [word["coverage"][t]["token_coverage"] * 100 for t in thresholds],
                 "ko--", label="words", ms=6)
    axes[1].set_title("Token coverage\n(% of running-text occurrences seedable)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("min image-text frequency"); axes[1].set_ylabel("%")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_bars.png", dpi=150)
    plt.close(fig)


def plot_freq_scatter(out_dir, which="word_coverage.csv", title="words"):
    df = pd.read_csv(out_dir / which)
    df = df[(df.text_only_count > 0)]
    # rates per million for comparability (sources differ in size)
    t_total = df.text_only_count.sum()
    i_total = df.image_count.sum()
    df["t_rate"] = df.text_only_count / t_total * 1e6
    df["i_rate"] = df.image_count / i_total * 1e6
    fig, ax = plt.subplots(figsize=(7, 7))
    covered = df[df.image_count > 0]
    uncov = df[df.image_count == 0]
    ax.scatter(uncov.t_rate, [0.5] * len(uncov), s=4, alpha=0.2, c="red",
               label="no image support")
    ax.scatter(covered.t_rate, covered.i_rate, s=4, alpha=0.3, c="steelblue",
               label="vision-supported")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("text-only rate (per M)"); ax.set_ylabel("image-text rate (per M)")
    ax.set_title(f"Frequency: text-only vs image-text ({title})")
    # label a few extreme divergent points (high text-only, covered but low image)
    covered2 = covered.copy()
    covered2["ratio"] = covered2.t_rate / covered2.i_rate
    extreme = covered2.nlargest(12, "t_rate")
    for _, r in extreme.iterrows():
        lbl = r["word"] if "word" in r else r.get("token_str", "")
        ax.annotate(str(lbl), (r.t_rate, r.i_rate), fontsize=7, alpha=0.8)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"freq_scatter_{title}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/out"))
    args = ap.parse_args()
    plot_coverage_bars(args.out_dir)
    plot_freq_scatter(args.out_dir, "word_coverage.csv", "words")
    plot_freq_scatter(args.out_dir, "token_coverage_50k.csv", "tokens_50k")
    print(f"plots -> {args.out_dir}/ (coverage_bars.png, freq_scatter_*.png)")


if __name__ == "__main__":
    main()
