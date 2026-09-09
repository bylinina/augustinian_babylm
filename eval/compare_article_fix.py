#!/usr/bin/env python3
"""Before/after comparison of the article fix, on the numbers the paper reports.

Reads results/vpswap/<model>/chck_100M.jsonl and its .prearticle backup, joins
the per-line metadata for seeded status, and prints old vs new for:
  overall accuracy and delta, by syntactic frame, by property, and the
  seeded/unseeded split.

Usage:  python eval/compare_article_fix.py
"""
import json
from pathlib import Path

RESULTS = Path("results/vpswap")
PAIRS = Path("eval/vpswap_bb24")
REV = "chck_100M"
PROPS = ["color", "material", "relative_size", "shape"]

SEEDS = [("deberta-base-75k",            "deberta-base-75k-sam",         "s1"),
         ("deberta-base-75k-s2",         "deberta-base-75k-sam-s2",      "s2"),
         ("deberta-base-75k-s3",         "deberta-base-75k-sam-s3",      "s3")]


def load_meta():
    """(property, line) -> metadata dict"""
    meta = {}
    for prop in PROPS:
        f = PAIRS / f"vp_swap_{prop}_pairs.meta.jsonl"
        if not f.exists():
            continue
        for i, line in enumerate(l for l in f.open() if l.strip()):
            meta[(prop, i)] = json.loads(line)
    return meta


def load(model, which_file):
    f = RESULTS / model / (REV + which_file)
    if not f.exists():
        return None
    return {(r["property"], r["line"], r["which"]): r
            for r in (json.loads(l) for l in f.open() if l.strip())}


def acc(rows, keys=None):
    sel = [r for k, r in rows.items() if keys is None or k in keys]
    return (sum(r["correct"] for r in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)


def seeded_orig(meta, key):
    prop, line, which = key
    m = meta.get((prop, line))
    if not m:
        return None
    return m.get("seeded_w1") if which == 1 else m.get("seeded_w2")


meta = load_meta()
print(f"metadata lines loaded: {len(meta):,}\n")

for base, sam, tag in SEEDS:
    for suffix, label in ((".jsonl.prearticle", "OLD"), (".jsonl", "NEW")):
        b, s = load(base, suffix), load(sam, suffix)
        if b is None or s is None:
            print(f"[{tag}] {label}: missing files, skipped")
            continue
        keys = set(b) & set(s)
        ab, _ = acc(b, keys)
        asam, _ = acc(s, keys)
        print(f"[{tag}] {label:3}  baseline {ab:.4f}   vision {asam:.4f}   "
              f"delta {100*(asam-ab):+.2f} pts")

        by_frame = {}
        for k in keys:
            m = meta.get((k[0], k[1]))
            fr = (m or {}).get("frame", "?")
            by_frame.setdefault(fr, []).append(k)
        line = "        "
        for fr in sorted(by_frame):
            kk = set(by_frame[fr])
            d = acc(s, kk)[0] - acc(b, kk)[0]
            line += f"{fr}: {d:+.3f}   "
        print(line)

        for name, want in (("seeded", True), ("unseeded", False)):
            kk = {k for k in keys if seeded_orig(meta, k) is want}
            if kk:
                d = acc(s, kk)[0] - acc(b, kk)[0]
                print(f"        orig-{name:8} n={len(kk):5}  delta {d:+.4f}")
    print()

print("--- synthetic extension (ext vs sam), NEW only ---")
for i, (sam, ext) in enumerate(
        [("deberta-base-75k-sam",    "deberta-base-75k-sam_ext-s1"),
         ("deberta-base-75k-sam-s2", "deberta-base-75k-sam_ext-s2"),
         ("deberta-base-75k-sam-s3", "deberta-base-75k-sam_ext-s3")], start=1):
    s, e = load(sam, ".jsonl"), load(ext, ".jsonl")
    if s is None or e is None:
        print(f"  s{i}: missing, skipped")
        continue
    keys = set(s) & set(e)
    print(f"  s{i}: ext-sam {acc(e, keys)[0] - acc(s, keys)[0]:+.4f}  (n={len(keys)})")
