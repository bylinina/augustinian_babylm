#!/usr/bin/env python
"""
Synthetic coverage extension: three-way VP-Swap analysis, seed-averaged.

Word groups (by the ORIGINAL seeding mask + the synthetic detection set):
  real      = seeded in the original grounding data
  synthetic = zero-support originally, but detected in >=1 synthetic image
  unseeded  = still no visual support in either
Pre-registered prediction (restated after the seed replication): the
synthetic group should show a seeded-like ext-vs-baseline advantage that
the still-unseeded group does not; the real group should show ext ~ sam.

Writes eval/ext_results.md + eval/plots/vpswap_ext_groups.png.
"""
from __future__ import annotations
import argparse, json, os, re
from collections import defaultdict
from math import sqrt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup, style_ax, INK, BLUE, GRAY, GREEN
setup()

SEEDS = {
    "s1": ("deberta-base-75k-sam_ext-s1", "deberta-base-75k-sam", "deberta-base-75k"),
    "s2": ("deberta-base-75k-sam_ext-s2", "deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
    "s3": ("deberta-base-75k-sam_ext-s3", "deberta-base-75k-sam-s3", "deberta-base-75k-s3"),
}
REVS = (["step0"] + [f"chck_{i}M" for i in range(1, 10)]
        + [f"chck_{i}M" for i in range(10, 101, 10)])
WORDS_M = {r: (0 if r == "step0" else int(r[5:-1])) for r in REVS}
VP = Path("results/vpswap")

ap = argparse.ArgumentParser()
ap.add_argument("--synth_parquet", default=str(Path("/scratch-shared")
                / os.environ.get("USER", "") / "region_synth/region_embeddings.parquet"))
args = ap.parse_args()

import pandas as pd
synth_words = set(pd.read_parquet(args.synth_parquet)["text"].str.lower())
print(f"synthetic-detected words: {len(synth_words)}")

meta = {}
for prop in ["color", "material", "relative_size", "shape"]:
    for i, l in enumerate(open(f"eval/vpswap_bb24/vp_swap_{prop}_pairs.meta.jsonl")):
        meta[(prop, i)] = json.loads(l)

def group_of(k):
    m = meta[(k[0], k[1])]
    w = (m["w1"] if k[2] == 1 else m["w2"]).lower()
    sd = m["seeded_w1"] if k[2] == 1 else m["seeded_w2"]
    if sd: return "real"
    return "synthetic" if w in synth_words else "unseeded"

def items(model, rev):
    p = VP / model / f"{rev}.jsonl"
    if not p.exists(): return None
    return {(r["property"], r["line"], r["which"]): r
            for r in map(json.loads, p.open())}

def acc(d, ks): return sum(d[k]["correct"] for k in ks) / len(ks)

md = ["# Synthetic coverage extension: results", "",
      f"ext = 75k-sam_ext (sam + {len(synth_words)} synthetically grounded "
      "words), 3 seeds each.", ""]

# ---- step0 gate + final-checkpoint three-way table, per seed ----
md += ["## Final checkpoint (chck_100M): three-way split", "",
       "| seed | group | n | ext | sam | base | ext-base | sam-base | ext-sam |",
       "|--|--|--:|--:|--:|--:|--:|--:|--:|"]
print("=== step0 gate ===")
group_deltas = defaultdict(lambda: defaultdict(list))  # group -> series -> [per-seed]
for sd, (em, sm, bm) in SEEDS.items():
    e0 = items(em, "step0")
    if e0:
        ks = sorted(e0)
        print(f"  ext-{sd} step0: {acc(e0, ks):.3f}")
print("\n=== chck_100M three-way ===")
for sd, (em, sm, bm) in SEEDS.items():
    e, s_, b = items(em, "chck_100M"), items(sm, "chck_100M"), items(bm, "chck_100M")
    keys = sorted(set(e) & set(s_) & set(b))
    by_g = defaultdict(list)
    for k in keys: by_g[group_of(k)].append(k)
    for g in ["real", "synthetic", "unseeded"]:
        ks = by_g[g]
        ae, as_, ab = acc(e, ks), acc(s_, ks), acc(b, ks)
        md.append(f"| {sd} | {g} | {len(ks)} | {ae:.3f} | {as_:.3f} | {ab:.3f} "
                  f"| {ae-ab:+.3f} | {as_-ab:+.3f} | {ae-as_:+.3f} |")
        print(f"  {sd} {g:10s} n={len(ks):5d} ext-base {ae-ab:+.3f}  "
              f"sam-base {as_-ab:+.3f}  ext-sam {ae-as_:+.3f}")
        group_deltas[g]["ext-base"].append(ae - ab)
        group_deltas[g]["sam-base"].append(as_ - ab)
        group_deltas[g]["ext-sam"].append(ae - as_)

md += ["", "### Seed means", "",
       "| group | ext-base | sam-base | ext-sam |", "|--|--:|--:|--:|"]
print("\n=== seed means ===")
for g in ["real", "synthetic", "unseeded"]:
    row = [sum(group_deltas[g][c]) / 3 for c in ["ext-base", "sam-base", "ext-sam"]]
    md.append(f"| {g} | {row[0]:+.3f} | {row[1]:+.3f} | {row[2]:+.3f} |")
    print(f"  {g:10s} ext-base {row[0]:+.3f}  sam-base {row[1]:+.3f}  "
          f"ext-sam {row[2]:+.3f}")

# ---- figure: grouped deltas with per-seed markers ----
fig, ax = plt.subplots(figsize=(7.0, 3.8))
groups = ["real", "synthetic", "unseeded"]
width = 0.35
for off, (series, color, label) in enumerate(
        [("sam-base", GRAY, "sam \u2212 baseline"),
         ("ext-base", BLUE, "ext \u2212 baseline")]):
    xs = [i + (off - 0.5) * width for i in range(len(groups))]
    means = [sum(group_deltas[g][series]) / 3 for g in groups]
    ax.bar(xs, means, width * 0.9, color=color, label=label)
    for x, g in zip(xs, groups):
        ax.scatter([x] * 3, group_deltas[g][series], color=INK, s=12, zorder=3)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(len(groups)),
              [f"real-seeded", "synthetic-seeded", "still-unseeded"])
ax.set_ylabel("accuracy delta vs baseline")
ax.set_title("VP-Swap @ 100M by word group (bars = seed mean, dots = seeds)",
             loc="left")
ax.legend()
style_ax(ax)
fig.tight_layout()
fig.savefig("eval/plots/vpswap_ext_groups.png")
plt.close(fig)

# ---- synthetic-group delta trajectory (mean across seeds) ----
md += ["", "## Synthetic-group ext-vs-baseline delta over training", "",
       "| words (M) | ext-base (synthetic group) | sam-base (synthetic group) |",
       "|--:|--:|--:|"]
print("\n=== synthetic-group trajectory (seed mean) ===")
for rev in REVS:
    de, ds = [], []
    for sd, (em, sm, bm) in SEEDS.items():
        e, s_, b = items(em, rev), items(sm, rev), items(bm, rev)
        if not (e and s_ and b): continue
        keys = [k for k in (set(e) & set(s_) & set(b)) if group_of(k) == "synthetic"]
        de.append(acc(e, keys) - acc(b, keys))
        ds.append(acc(s_, keys) - acc(b, keys))
    if de:
        md.append(f"| {WORDS_M[rev]} | {sum(de)/len(de):+.3f} | {sum(ds)/len(ds):+.3f} |")
        print(f"  {WORDS_M[rev]:>4}M  ext-base {sum(de)/len(de):+.3f}   "
              f"sam-base {sum(ds)/len(ds):+.3f}")

# ---- zero-shot: did the extension cost anything overall? ----
ZS = Path.home() / "babylm-eval/strict/results"
ZS_TASKS = {"blimp": "blimp/blimp_filtered", "comps": "comps/comps",
            "ewok": "ewok/ewok_filtered"}
def zsavg(model, glob):
    reps = list((ZS / model / "main/zero_shot/mlm").glob(glob + "/best_temperature_report.txt"))
    if not reps: return None
    val, in_avg = None, False
    for line in reps[0].read_text().splitlines():
        t = line.strip()
        if t.startswith("### "): in_avg = "AVERAGE" in t; continue
        if in_avg and re.fullmatch(r"[-\d.]+", t): val = float(t)
    return val
md += ["", "## Zero-shot (ext - baseline, per seed)", "",
       "| task | s1 | s2 | s3 |", "|--|--:|--:|--:|"]
print("\n=== zero-shot ext-base ===")
for task, glob in ZS_TASKS.items():
    ds = []
    for sd, (em, _sm, bm) in SEEDS.items():
        v, b = zsavg(em, glob), zsavg(bm, glob)
        ds.append(None if (v is None or b is None) else v - b)
    fmt = lambda d: f"{d:+.2f}" if d is not None else "--"
    md.append(f"| {task} | " + " | ".join(fmt(d) for d in ds) + " |")
    print(f"  {task:8s} " + " ".join(fmt(d) for d in ds))

Path("eval/ext_results.md").write_text("\n".join(md) + "\n")
print("\nwrote eval/ext_results.md + eval/plots/vpswap_ext_groups.png")
