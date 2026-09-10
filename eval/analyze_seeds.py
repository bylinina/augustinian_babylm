#!/usr/bin/env python
"""Multi-seed aggregation: VP-Swap + zero-shot, 75k-sam vs 75k, 3 seeds.
Writes eval/seed_results.md, eval/plots/vpswap_trajectory_seeds.png,
eval/plots/vpswap_2x2_seeds.png."""
from __future__ import annotations
import json, re
from collections import defaultdict
from math import sqrt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup, style_ax, INK, BLUE, GRAY
setup()

SEEDS = {"s1": ("deberta-base-75k-sam", "deberta-base-75k"),
         "s2": ("deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
         "s3": ("deberta-base-75k-sam-s3", "deberta-base-75k-s3")}
REVS = (["step0"] + [f"chck_{i}M" for i in range(1, 10)]
        + [f"chck_{i}M" for i in range(10, 101, 10)])
WORDS = {r: (0 if r == "step0" else int(r[5:-1])) for r in REVS}
VP = Path("results/vpswap")
ZS = Path.home() / "babylm-eval/strict/results"
ZS_TASKS = {"blimp": "blimp/blimp_filtered", "supplement": "blimp/supplement_filtered",
            "ewok": "ewok/ewok_filtered", "entity_tracking": "entity_tracking/entity_tracking",
            "comps": "comps/comps"}

meta = {}
for prop in ["color", "material", "relative_size", "shape"]:
    for i, l in enumerate(open(f"eval/vpswap_bb24/vp_swap_{prop}_pairs.meta.jsonl")):
        meta[(prop, i)] = json.loads(l)

def items(model, rev):
    p = VP / model / f"{rev}.jsonl"
    if not p.exists(): return None
    return {(r["property"], r["line"], r["which"]): r
            for r in map(json.loads, p.open())}

def acc(d, ks): return sum(d[k]["correct"] for k in ks) / len(ks)
def s_orig(k):
    m = meta[(k[0], k[1])]
    return m["seeded_w1"] if k[2] == 1 else m["seeded_w2"]
def s_swap(k):
    m = meta[(k[0], k[1])]
    return m["seeded_w2"] if k[2] == 1 else m["seeded_w1"]

md = ["# Multi-seed results (3 seeds, 75k-SAM vs 75k)", ""]

# ---- trajectories per seed ----
traj = {}   # seed -> [(words, v, b)]
for sd, (vm, bm) in SEEDS.items():
    rows = []
    for rev in REVS:
        a, b = items(vm, rev), items(bm, rev)
        if a and b:
            ks = sorted(set(a) & set(b))
            rows.append((WORDS[rev], acc(a, ks), acc(b, ks)))
    traj[sd] = rows
    print(f"{sd}: {len(rows)} revisions loaded")

common = sorted(set.intersection(*(set(w for w, _, _ in traj[s]) for s in traj)))
md += ["## VP-Swap delta trajectory (vision - baseline) per seed", "",
       "| words (M) | " + " | ".join(traj.keys()) + " | mean |",
       "|--:|" + "--:|" * (len(traj) + 1)]
mean_d, lo_d, hi_d = [], [], []
mean_v, mean_b = [], []
for w in common:
    ds, vs, bs = [], [], []
    for sd in traj:
        row = next(r for r in traj[sd] if r[0] == w)
        ds.append(row[1] - row[2]); vs.append(row[1]); bs.append(row[2])
    md.append(f"| {w} | " + " | ".join(f"{d:+.3f}" for d in ds)
              + f" | {sum(ds)/len(ds):+.3f} |")
    mean_d.append(sum(ds)/len(ds)); lo_d.append(min(ds)); hi_d.append(max(ds))
    mean_v.append(sum(vs)/len(vs)); mean_b.append(sum(bs)/len(bs))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax1.plot(common, mean_v, "o-", ms=4, lw=1.8, color=BLUE, label="vision-init (mean)")
ax1.plot(common, mean_b, "o--", ms=4, lw=1.5, color=GRAY, label="baseline (mean)")
ax1.axhline(0.5, color=GRAY, lw=0.8, ls=":")
ax1.set_xlabel("words seen (M)"); ax1.set_ylabel("VP-Swap accuracy")
ax1.legend(loc="lower right"); ax1.set_title("Accuracy (mean of 3 seeds)", loc="left")
style_ax(ax1)
ax2.fill_between(common, lo_d, hi_d, color=BLUE, alpha=0.18, label="seed range")
ax2.plot(common, mean_d, "o-", ms=4, lw=1.8, color=BLUE, label="mean delta")
ax2.axhline(0, color=INK, lw=0.8)
ax2.set_xlabel("words seen (M)"); ax2.set_ylabel("vision \u2212 baseline")
ax2.legend(loc="lower right")
ax2.set_title("Advantage, mean \u00b1 seed range", loc="left")
style_ax(ax2)
fig.tight_layout()
fig.savefig("eval/plots/vpswap_trajectory_seeds.png")
fig.savefig("eval/plots/vpswap_trajectory_seeds.pdf"); plt.close(fig)

# ---- final checkpoint: McNemar + seeded/DiD + 2x2, per seed ----
md += ["", "## Final checkpoint (chck_100M) per seed", ""]
grids = []
for sd, (vm, bm) in SEEDS.items():
    a, b = items(vm, "chck_100M"), items(bm, "chck_100M")
    if not a or not b: continue
    ks = sorted(set(a) & set(b))
    b01 = sum(1 for k in ks if a[k]["correct"] and not b[k]["correct"])
    b10 = sum(1 for k in ks if b[k]["correct"] and not a[k]["correct"])
    z = (b01 - b10) / sqrt(b01 + b10)
    dsee = acc(a, [k for k in ks if s_orig(k)]) - acc(b, [k for k in ks if s_orig(k)])
    duns = acc(a, [k for k in ks if not s_orig(k)]) - acc(b, [k for k in ks if not s_orig(k)])
    md.append(f"- **{sd}**: overall {acc(a,ks):.3f} vs {acc(b,ks):.3f} "
              f"(delta {acc(a,ks)-acc(b,ks):+.3f}), McNemar z = {z:.2f}; "
              f"seeded delta {dsee:+.3f}, unseeded {duns:+.3f}")
    print(f"{sd}: z={z:.2f} d={acc(a,ks)-acc(b,ks):+.3f} seeded {dsee:+.3f} unseeded {duns:+.3f}")
    g = [[0., 0.], [0., 0.]]; cnt = defaultdict(list)
    for k in ks: cnt[(s_orig(k), s_swap(k))].append(k)
    for (so, ss), kk in cnt.items():
        g[0 if so else 1][0 if ss else 1] = acc(a, kk) - acc(b, kk)
    grids.append((sd, g))

fig, axes = plt.subplots(1, len(grids) + 1, figsize=(3.1 * (len(grids) + 1), 3.2),
                         constrained_layout=True)
mean_g = [[sum(g[i][j] for _, g in grids) / len(grids) for j in range(2)]
          for i in range(2)]
for ax, (title, g) in zip(axes, grids + [("mean of seeds", mean_g)]):
    im = ax.imshow(g, cmap="RdBu_r", vmin=-0.08, vmax=0.08)
    ax.set_xticks([0, 1], ["swap\nseeded", "swap\nunseeded"], fontsize=8.5)
    if ax is axes[0]:
        ax.set_yticks([0, 1], ["orig\nseeded", "orig\nunseeded"], fontsize=8.5)
    else:
        ax.set_yticks([])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{g[i][j]:+.3f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(g[i][j]) > 0.045 else INK)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(visible=False); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cb.set_label("vision \u2212 baseline"); cb.outline.set_visible(False)
fig.savefig("eval/plots/vpswap_2x2_seeds.png")
fig.savefig("eval/plots/vpswap_2x2_seeds.pdf"); plt.close(fig)

# ---- zero-shot task deltas per seed ----
def zsavg(model, glob):
    reps = list((ZS / model / "main/zero_shot/mlm").glob(glob + "/best_temperature_report.txt"))
    if not reps: return None
    val, in_avg = None, False
    for line in reps[0].read_text().splitlines():
        s = line.strip()
        if s.startswith("### "): in_avg = "AVERAGE" in s; continue
        if in_avg and re.fullmatch(r"[-\d.]+", s): val = float(s)
    return val

md += ["", "## Zero-shot task deltas per seed (main revision)", "",
       "| task | " + " | ".join(SEEDS) + " | mean |", "|--|" + "--:|" * (len(SEEDS) + 1)]
print("\nzero-shot deltas:")
for task, glob in ZS_TASKS.items():
    ds = []
    for sd, (vm, bm) in SEEDS.items():
        v, b = zsavg(vm, glob), zsavg(bm, glob)
        ds.append(None if (v is None or b is None) else v - b)
    fmt = lambda d: f"{d:+.2f}" if d is not None else "--"
    good = [d for d in ds if d is not None]
    md.append(f"| {task} | " + " | ".join(fmt(d) for d in ds)
              + f" | {fmt(sum(good)/len(good)) if good else '--'} |")
    print(f"  {task:16s} " + " ".join(fmt(d) for d in ds))

Path("eval/seed_results.md").write_text("\n".join(md) + "\n")
print("\nwrote eval/seed_results.md + 2 figures")
