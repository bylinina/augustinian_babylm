#!/usr/bin/env python
"""
Analyze VP-Swap results: tables + figures + markdown summary.

Reads:  results/vpswap/<model>/<rev>.jsonl   (from vpswap_score.py)
        eval/vpswap_bb24/vp_swap_<prop>_pairs.meta.jsonl
Writes: eval/plots/vpswap_trajectory.png
        eval/plots/vpswap_did_by_bin.png
        eval/plots/vpswap_2x2.png
        eval/plots/vpswap_property_frame.png
        eval/vpswap_results.md

Usage:  python eval/analyze_vpswap.py \
          --vision deberta-base-75k-sam --baseline deberta-base-75k
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup, style_ax, INK, BLUE, GRAY, RED, GREEN
setup()

PROPS = ("color", "material", "relative_size", "shape")
REVS = (["step0"] + [f"chck_{i}M" for i in range(1, 10)]
        + [f"chck_{i}M" for i in range(10, 101, 10)])
WORDS = {r: (0 if r == "step0" else int(r[5:-1])) for r in REVS}


def load_meta(pairs_dir: Path) -> dict:
    meta = {}
    for prop in PROPS:
        f = pairs_dir / f"vp_swap_{prop}_pairs.meta.jsonl"
        for i, line in enumerate(f.open()):
            meta[(prop, i)] = json.loads(line)
    return meta


def load_items(results_dir: Path, model: str, rev: str) -> dict | None:
    p = results_dir / model / f"{rev}.jsonl"
    if not p.exists():
        return None
    return {(r["property"], r["line"], r["which"]): r
            for r in map(json.loads, p.open())}


def acc(items: dict, keys) -> float:
    return sum(items[k]["correct"] for k in keys) / len(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", default="deberta-base-75k-sam")
    ap.add_argument("--baseline", default="deberta-base-75k")
    ap.add_argument("--results_dir", type=Path, default=Path("results/vpswap"))
    ap.add_argument("--pairs_dir", type=Path, default=Path("eval/vpswap_bb24"))
    ap.add_argument("--plots_dir", type=Path, default=Path("eval/plots"))
    ap.add_argument("--out_md", type=Path, default=Path("eval/vpswap_results.md"))
    args = ap.parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.pairs_dir)
    md = ["# VP-Swap results", "",
          f"vision-init: `{args.vision}` | baseline: `{args.baseline}`", ""]

    def seeded_orig(k):
        m = meta[(k[0], k[1])]
        return m["seeded_w1"] if k[2] == 1 else m["seeded_w2"]

    def seeded_swap(k):
        m = meta[(k[0], k[1])]
        return m["seeded_w2"] if k[2] == 1 else m["seeded_w1"]

    # ---------- trajectory ----------
    traj = []
    for rev in REVS:
        a = load_items(args.results_dir, args.vision, rev)
        b = load_items(args.results_dir, args.baseline, rev)
        if not a or not b:
            continue
        keys = sorted(set(a) & set(b))
        traj.append((WORDS[rev], acc(a, keys), acc(b, keys)))
    md += ["## Accuracy trajectory", "",
           "| words (M) | vision | baseline | delta |", "|--:|--:|--:|--:|"]
    print("=== trajectory ===")
    for w, va, ba in traj:
        md.append(f"| {w} | {va:.3f} | {ba:.3f} | {va - ba:+.3f} |")
        print(f"  {w:>4}M  vision {va:.3f}  base {ba:.3f}  d {va-ba:+.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ws = [t[0] for t in traj]
    ax1.plot(ws, [t[1] for t in traj], "o-", ms=4, lw=1.8,
             label="vision-init", color=BLUE)
    ax1.plot(ws, [t[2] for t in traj], "o--", ms=4, lw=1.5,
             label="baseline", color=GRAY)
    ax1.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax1.text(ws[0], 0.502, "chance", ha="left", va="bottom",
             fontsize=8.5, color=GRAY)
    ax1.set_xlabel("words seen (M)"); ax1.set_ylabel("VP-Swap accuracy")
    ax1.legend(loc="lower right")
    ax1.set_title("Accuracy over training", loc="left")
    style_ax(ax1)
    deltas = [t[1] - t[2] for t in traj]
    ax2.fill_between(ws, deltas, 0, color=BLUE, alpha=0.15)
    ax2.plot(ws, deltas, "o-", ms=4, lw=1.8, color=BLUE)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xlabel("words seen (M)")
    ax2.set_ylabel("vision − baseline")
    ax2.set_title("Advantage over training", loc="left")
    style_ax(ax2)
    fig.tight_layout()
    fig.savefig(args.plots_dir / "vpswap_trajectory.png")
    plt.close(fig)

    # ---------- final-checkpoint cuts ----------
    a = load_items(args.results_dir, args.vision, "chck_100M")
    b = load_items(args.results_dir, args.baseline, "chck_100M")
    keys = sorted(set(a) & set(b))

    b01 = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    b10 = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    z = (b01 - b10) / sqrt(b01 + b10)
    md += ["", f"## Final checkpoint (100M), {len(keys)} paired items", "",
           f"McNemar: vision-only-correct {b01}, baseline-only-correct {b10}, "
           f"**z = {z:.2f}**", ""]
    print(f"\nMcNemar z = {z:.2f}  ({b01} vs {b10})")

    def cut_table(title, grouper, fig_ax=None, sort_key=None):
        groups = defaultdict(list)
        for k in keys:
            groups[grouper(k)].append(k)
        rows = []
        for g in sorted(groups, key=sort_key):
            ks = groups[g]
            va, ba = acc(a, ks), acc(b, ks)
            rows.append((g, len(ks), va, ba, va - ba))
        md.extend(["", f"### by {title}", "",
                   "| group | n | vision | baseline | delta |",
                   "|--|--:|--:|--:|--:|"])
        print(f"\n=== by {title} ===")
        for g, n, va, ba, d in rows:
            md.append(f"| {g} | {n} | {va:.3f} | {ba:.3f} | {d:+.3f} |")
            print(f"  {str(g)[:20]:20s} n={n:5d}  v {va:.3f}  b {ba:.3f}  d {d:+.3f}")
        return rows

    prop_rows = cut_table("property", lambda k: a[k]["property"])
    frame_rows = cut_table(
        "frame", lambda k: meta[(k[0], k[1])].get("frame", "?"))
    cut_table("seeded (orig word)",
              lambda k: "seeded" if seeded_orig(k) else "unseeded")
    bin_rows = cut_table("frequency bin", lambda k: a[k]["bin"],
                         sort_key=lambda g: g)

    # property + frame deltas figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, rows, title in ((ax1, prop_rows, "By property"),
                            (ax2, frame_rows, "By frame")):
        names = [str(r[0]).replace("relative_size", "size") for r in rows]
        vals = [r[4] for r in rows]
        cols = [BLUE if v >= 0 else RED for v in vals]
        ax.barh(names[::-1], vals[::-1],
                color=cols[::-1], height=0.6)
        ax.axvline(0, color=INK, lw=0.8)
        for i, v in enumerate(vals[::-1]):
            ax.text(v + (0.0015 if v >= 0 else -0.0015), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK)
        ax.set_xlabel("vision − baseline")
        ax.set_title(title, loc="left")
        ax.margins(x=0.25)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(args.plots_dir / "vpswap_property_frame.png")
    plt.close(fig)

    # ---------- seeded x bin diff-in-diff ----------
    md += ["", "### Seeded-vs-unseeded delta within frequency bins "
           "(diff-in-differences)", "",
           "| bin | n seeded | d seeded | n unseeded | d unseeded | DiD |",
           "|--:|--:|--:|--:|--:|--:|"]
    print("\n=== DiD within bins ===")
    did_bins, did_vals = [], []
    for bn in range(10):
        ks_s = [k for k in keys if a[k]["bin"] == bn and seeded_orig(k)]
        ks_u = [k for k in keys if a[k]["bin"] == bn and not seeded_orig(k)]
        def d(ks):
            return (acc(a, ks) - acc(b, ks)) if len(ks) >= 30 else None
        ds, du = d(ks_s), d(ks_u)
        did = (ds - du) if (ds is not None and du is not None) else None
        f = lambda x: f"{x:+.3f}" if x is not None else "--"
        md.append(f"| {bn} | {len(ks_s)} | {f(ds)} | {len(ks_u)} | "
                  f"{f(du)} | {f(did)} |")
        print(f"  bin {bn}: d_seed {f(ds)} (n={len(ks_s)})  "
              f"d_unseed {f(du)} (n={len(ks_u)})  DiD {f(did)}")
        if did is not None:
            did_bins.append(bn); did_vals.append(did)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(did_bins, did_vals, color=BLUE, width=0.62)
    for x, v in zip(did_bins, did_vals):
        ax.text(x, v + 0.005, f"{v:+.2f}", ha="center", fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("DiD (seeded − unseeded delta)")
    ax.set_title("Word-specific effect at matched frequency", loc="left")
    ax.set_xticks(did_bins)
    ax.margins(y=0.18)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(args.plots_dir / "vpswap_did_by_bin.png")
    plt.close(fig)

    # ---------- 2x2 (orig seeded x swap seeded), 50M + 100M ----------
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6),
                             constrained_layout=True)
    for ax, rev in zip(axes, ("chck_50M", "chck_100M")):
        ar = load_items(args.results_dir, args.vision, rev)
        br = load_items(args.results_dir, args.baseline, rev)
        ks = sorted(set(ar) & set(br))
        grid = [[0.0, 0.0], [0.0, 0.0]]
        md += ["", f"### 2x2 orig-seeded x swap-seeded @ {rev}", "",
               "| orig | swap | n | vision | baseline | delta |",
               "|--|--|--:|--:|--:|--:|"]
        print(f"\n=== 2x2 @ {rev} ===")
        cells = defaultdict(list)
        for k in ks:
            cells[(seeded_orig(k), seeded_swap(k))].append(k)
        for (so, ss), kk in sorted(cells.items(), reverse=True):
            va, ba = acc(ar, kk), acc(br, kk)
            grid[0 if so else 1][0 if ss else 1] = va - ba
            md.append(f"| {'S' if so else 'U'} | {'S' if ss else 'U'} | "
                      f"{len(kk)} | {va:.3f} | {ba:.3f} | {va - ba:+.3f} |")
            print(f"  {'S' if so else 'U'}{'S' if ss else 'U'} n={len(kk):5d} "
                  f"d {va-ba:+.3f}")
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.08, vmax=0.08)
        ax.set_xticks([0, 1], ["swap word\nseeded", "swap word\nunseeded"])
        if ax is axes[0]:
            ax.set_yticks([0, 1], ["orig word\nseeded", "orig word\nunseeded"])
        else:
            ax.set_yticks([])
        for i in range(2):
            for j in range(2):
                dark = abs(grid[i][j]) > 0.045
                ax.text(j, i, f"{grid[i][j]:+.3f}", ha="center", va="center",
                        fontsize=11.5,
                        color="white" if dark else INK)
        ax.set_title(rev.replace("chck_", "") + " words", loc="left")
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03)
    cb.set_label("vision − baseline")
    cb.outline.set_visible(False)
    fig.savefig(args.plots_dir / "vpswap_2x2.png")
    plt.close(fig)

    args.out_md.write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out_md} and 4 figures to {args.plots_dir}/")


if __name__ == "__main__":
    main()
