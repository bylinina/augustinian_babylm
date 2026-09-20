#!/usr/bin/env python
"""VP-Swap re-analysis at the pair level (75k-SAM vs 75k, 3 seeds; + 75k-SAM-ext).

Each pair-file line yields two items with the nouns swapped, so a preference for
one noun over the other cancels only when both items of a line are counted
together. Everything here is computed per line (pair):

  1. pair composition: both / one / neither noun seeded (word-level, as in the
     paper), plus the token-level count of pairs with no seeded token at all
  2. matched frequency: both-seeded vs other pairs within frequency bins,
     reweighted to the bins where the other pairs live
  3. mixed pairs item by item: does the vision model pick the seeded noun
     even when it is the wrong one?
  4. synthetic extension, token-level treatment: pairs whose in-sentence noun
     tokens were rebuilt mostly from synthetic regions vs pairs untouched

Run from the repo root:  python eval/vpswap_pairlevel.py
Writes eval/vpswap_pairlevel.md and eval/plots/vpswap_pairlevel_*.{png,pdf}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup, INK, BLUE, GRAY
setup()

REV, VP, B = "chck_100M", Path("results/vpswap"), 2000
SEEDS = {"s1": ("deberta-base-75k-sam", "deberta-base-75k", "deberta-base-75k-sam_ext-s1"),
         "s2": ("deberta-base-75k-sam-s2", "deberta-base-75k-s2", "deberta-base-75k-sam_ext-s2"),
         "s3": ("deberta-base-75k-sam-s3", "deberta-base-75k-s3", "deberta-base-75k-sam_ext-s3")}
PROPS = ("color", "material", "relative_size", "shape")
rng = np.random.default_rng(0)

# ---- items, token-level seeding (sam/75k) and extension change (sam_ext/75k) ----
tok = Tokenizer.from_file(hf_hub_download("augustinian-babylm/babylm-bpe-75k", "tokenizer.json"))
cov = {t: pd.read_parquet(hf_hub_download("augustinian-babylm/token-embeddings",
                                          f"{t}/75k/coverage.parquet", repo_type="dataset")
                          ).set_index("token_id") for t in ("sam", "sam_ext")}

def noun_tokens(sent, i, w):
    enc = tok.encode(sent, add_special_tokens=False)
    return [t for t, (a, b) in zip(enc.ids, enc.offsets) if b > i and a < i + len(w)]

meta = {}
for p in PROPS:
    txt = open(f"eval/vpswap_bb24/vp_swap_{p}_pairs.txt").read().splitlines()
    for i, (line, m) in enumerate(zip(txt, open(f"eval/vpswap_bb24/vp_swap_{p}_pairs.meta.jsonl"))):
        m = json.loads(m)
        _, _, w1, s1, i1, w2, s2, i2 = line.split("|")[:8]
        m["n_seeded"] = int(m["seeded_w1"]) + int(m["seeded_w2"])
        m["n_notok"], m["n_treated"], m["n_touched"] = 0, 0, 0
        for w, s, ix in ((w1, s1, int(i1)), (w2, s2, int(i2))):
            ts = noun_tokens(s, ix, w)
            n0, n1 = cov["sam"].n_rows[ts].values, cov["sam_ext"].n_rows[ts].values
            share = np.where(n1 > 0, (n1 - n0) / np.maximum(n1, 1), 0).max()
            m["n_notok"] += int(not cov["sam"].seeded[ts].any())
            m["n_treated"] += int(share >= 0.5)
            m["n_touched"] += int(share > 0)
        meta[(p, i)] = m

_cache = {}
def load(model):
    if model not in _cache:
        f = VP / model / f"{REV}.jsonl"
        _cache[model] = ({(r["property"], r["line"], r["which"]): r
                          for r in map(json.loads, f.open())} if f.exists() else None)
    return _cache[model]

def pairs_in(*models):
    ds = [load(m) for m in models]
    if any(d is None for d in ds):
        return None
    return [k for k in meta if all((*k, w) in d for d in ds for w in (1, 2))]

def pdelta(a, b, ks):  # per pair: accuracy difference over the pair's two items
    return np.array([(a[(*k, 1)]["correct"] + a[(*k, 2)]["correct"]
                      - b[(*k, 1)]["correct"] - b[(*k, 2)]["correct"]) / 2 for k in ks], float)

def boot(x):  # mean and pair-bootstrap 95% CI
    if len(x) == 0:
        return (np.nan,) * 3
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return (x.mean(), *np.percentile(m, [2.5, 97.5]))

f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
md = ["# VP-Swap: pair-level re-analysis (chck_100M)", "",
      "Deltas: accuracy over both items of each pair; [95% CI] from resampling pairs.", ""]

def table(title, note, groups, fn, rows_out):
    """groups: [(label, pair filter)]; fn(seed) -> (a, b, pairs) or None"""
    global md
    md += [f"## {title}", "", note, "", "| pairs | n | s1 | s2 | s3 | mean |", "|--|--:|--:|--:|--:|--:|"]
    for label, keep in groups:
        vals, n = {}, 0
        for sd in SEEDS:
            got = fn(sd)
            if got is None:
                continue
            a, b, ks = got
            kk = [k for k in ks if keep(meta[k])]
            n, vals[sd] = len(kk), boot(pdelta(a, b, kk))
        if vals:
            md.append(f"| {label} | {n} | " + " | ".join(f3(vals[s]) if s in vals else "--" for s in SEEDS)
                      + f" | {np.mean([v[0] for v in vals.values()]):+.3f} |")
            rows_out.append((label, n, [v[0] for v in vals.values()]))
    md.append("")

def sam_base(sd):
    vm, bm, _ = SEEDS[sd]
    ks = pairs_in(vm, bm)
    if ks is None:
        sys.exit(f"missing {REV} results for {vm} / {bm} under {VP}")
    return load(vm), load(bm), ks

# ---- 1. pair composition ----
comp_rows = []
table("1. By pair composition (vision − baseline)",
      "Word-level seeded status, as in the paper. Last row: neither noun has any seeded "
      "in-sentence token (the only token-clean placebo; tiny).",
      [("both seeded", lambda m: m["n_seeded"] == 2), ("one seeded", lambda m: m["n_seeded"] == 1),
       ("neither seeded", lambda m: m["n_seeded"] == 0),
       ("no seeded token in either noun", lambda m: m["n_notok"] == 2)],
      sam_base, comp_rows)

# ---- 2. matched frequency ----
md += ["## 2. Matched frequency: both-seeded vs other pairs within bins", "",
       "Both-seeded pairs reweighted to the bin distribution of the other (one/neither) pairs; "
       "bins with >= 10 pairs of each kind.", "",
       "| seed | both seeded | one/neither seeded | difference |", "|--|--:|--:|--:|"]
match_rows = {"both seeded, matched bins": [], "one/neither seeded": []}
for sd in SEEDS:
    a, b, ks = sam_base(sd)
    cells = []
    for bn in range(10):
        kb = [k for k in ks if meta[k]["bin"] == bn]
        x = pdelta(a, b, [k for k in kb if meta[k]["n_seeded"] == 2])
        y = pdelta(a, b, [k for k in kb if meta[k]["n_seeded"] < 2])
        if len(x) >= 10 and len(y) >= 10:
            cells.append((bn, x, y))
    w = np.array([len(y) for _, _, y in cells], float)
    w /= w.sum()
    rs = lambda v: v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    bx = sum(wi * rs(x) for wi, (_, x, _) in zip(w, cells))
    by = sum(wi * rs(y) for wi, (_, _, y) in zip(w, cells))
    px = sum(wi * x.mean() for wi, (_, x, _) in zip(w, cells))
    py = sum(wi * y.mean() for wi, (_, _, y) in zip(w, cells))
    ci = lambda arr, pt: (pt, *np.percentile(arr, [2.5, 97.5]))
    md.append(f"| {sd} | {f3(ci(bx, px))} | {f3(ci(by, py))} | {f3(ci(bx - by, px - py))} |")
    match_rows["both seeded, matched bins"].append(px)
    match_rows["one/neither seeded"].append(py)
    match_n = (sum(len(x) for _, x, _ in cells), sum(len(y) for _, _, y in cells))
md += ["", f"bins used: {[c[0] for c in cells]}", ""]

# ---- 3. mixed pairs: the two items separately ----
md += ["## 3. Mixed pairs, item by item (vision − baseline)", "",
       "Seeded noun correct = top-right cell of Fig. 5; seeded noun swapped in = bottom-left. "
       "Knowledge of the seeded noun can only help or do nothing in either item, so a reliably "
       "negative bottom-left means the vision model picks the seeded noun even when it is the "
       "wrong one. PLL column: the seeded noun's advantage averaged over both contexts; "
       "it mixes that preference with knowledge that helps only when the seeded noun fits.", "",
       "| seed | n pairs | seeded noun correct | seeded noun swapped in | Δ seeded-noun advantage, PLL |",
       "|--|--:|--:|--:|--:|"]
for sd in SEEDS:
    a, b, ks = sam_base(sd)
    su, us, dp = [], [], []
    for k in [k for k in ks if meta[k]["n_seeded"] == 1]:
        s = 1 if meta[k]["seeded_w1"] else 2          # item whose original noun is the seeded one
        mg = lambda d, w: d[(*k, w)]["pll_orig"] - d[(*k, w)]["pll_swap"]
        c = lambda d, w: float(d[(*k, w)]["correct"])
        su.append(c(a, s) - c(b, s))
        us.append(c(a, 3 - s) - c(b, 3 - s))
        dp.append(((mg(a, s) - mg(a, 3 - s)) - (mg(b, s) - mg(b, 3 - s))) / 2)
    md.append(f"| {sd} | {len(su)} | {f3(boot(np.array(su)))} | {f3(boot(np.array(us)))} "
              f"| {f3(boot(np.array(dp)))} |")
md.append("")

# ---- 4. synthetic extension, token-level treatment ----
ext_rows = []
def ext_sam(sd):
    sm, _, em = SEEDS[sd]
    ks = pairs_in(em, sm)
    return None if ks is None else (load(em), load(sm), ks)
if any(ext_sam(sd) for sd in SEEDS):
    table("4. Synthetic extension at the pair level (ext − sam)",
          "Treated noun: an in-sentence token whose seed is now >= 50% synthetic regions "
          "(from the sam vs sam_ext coverage tables). Clean control: no token of either noun changed.",
          [(">= 1 treated noun", lambda m: m["n_treated"] >= 1),
           ("touched, < 50% synthetic", lambda m: m["n_treated"] == 0 and m["n_touched"] >= 1),
           ("clean control", lambda m: m["n_touched"] == 0)],
          ext_sam, ext_rows)
else:
    md += ["## 4. Synthetic extension", "", "ext results not found; skipped.", ""]

# ---- figures (slide/poster) ----
Path("eval/plots").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 11.5})
MINUS = lambda m: f"{m:+.3f}".replace("-", "\u2212")

def panel(ax, rows, title, slots=3):
    """rows: (label, per-seed values, colour); bar = seed mean, dots = seeds"""
    off = (slots - len(rows)) / 2
    for j, (label, vals, col) in enumerate(rows[::-1]):
        y, m = j + off, float(np.mean(vals))
        ax.barh(y, m, color=col, height=0.6)
        ax.scatter(vals, [y] * len(vals), color=INK, s=18, zorder=3)
        ax.text(max(max(vals), m) + 0.003, y, MINUS(m), va="center", fontsize=11)
    ax.set_yticks([j + off for j in range(len(rows))], [r[0] for r in rows[::-1]])
    ax.set_ylim(-0.5, slots - 0.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)

def xlim(*groups):
    v = [x for rows in groups for r in rows for x in r[1]]
    return min(min(v), 0) - 0.008, max(v) + 0.018

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"eval/plots/{name}.png"); fig.savefig(f"eval/plots/{name}.pdf"); plt.close(fig)

# pair composition, all pairs vs matched for frequency
C = {r[0]: r for r in comp_rows}
nice = {"both seeded": "both nouns seeded", "one seeded": "one noun seeded",
        "neither seeded": "neither noun seeded"}
left = [(f"{nice[k]}\n({C[k][1]:,} pairs)", C[k][2], BLUE if k == "both seeded" else GRAY) for k in nice]
right = [(f"both nouns seeded\n({match_n[0]:,} pairs, reweighted)", match_rows["both seeded, matched bins"], BLUE),
         (f"one or neither noun\nseeded ({match_n[1]:,} pairs)", match_rows["one/neither seeded"], GRAY)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 3.3), sharex=True)
panel(a1, left, "All pairs")
panel(a2, right, f"Matched for frequency (bins {cells[0][0]}\u2013{cells[-1][0]})")
a1.set_xlim(*xlim(left, right))
fig.supxlabel("VP-Swap accuracy per pair, vision-seeded minus baseline "
              "(bar: mean of 3 seeds; dots: individual seeds)", fontsize=11.5)
save(fig, "vpswap_pairlevel")

# synthetic extension
if ext_rows:
    E = {r[0]: r for r in ext_rows}
    nice = {">= 1 treated noun": "a noun's seed now\nmostly synthetic",
            "touched, < 50% synthetic": "a noun's seed\nchanged a little",
            "clean control": "no noun's seed\nchanged"}
    rows = [(f"{nice[k]} ({E[k][1]:,} pairs)", E[k][2], BLUE if k.startswith(">=") else GRAY)
            for k in nice if k in E]
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    panel(ax, rows, "Synthetic grounding")
    ax.set_xlim(*xlim(rows))
    ax.set_xlabel("VP-Swap accuracy per pair, extended minus original seeded model\n"
                  "(bar: mean of 3 seeds; dots: individual seeds)")
    save(fig, "vpswap_pairlevel_ext")

Path("eval/vpswap_pairlevel.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
