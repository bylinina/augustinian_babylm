#!/usr/bin/env python
"""Patch existing VP-Swap per-item JSONL results for the article-agreement fix.

Only items whose swapped sentence actually changes are rescored, and only their
`pll_swap`: the original sentence is untouched, so `pll_orig` carries over
unchanged. That is 278 of 7,416 items per checkpoint.

Place next to vpswap_score.py (it imports the scoring functions from there so
the PLL protocol is guaranteed identical to the original run).

Usage:
  # see what would change, no model loading
  python eval/patch_vpswap_articles.py --pairs_dir eval/vpswap_bb24 \
      --results_root results/vpswap --dry_run

  # patch in place (originals saved as *.jsonl.prearticle)
  python eval/patch_vpswap_articles.py --pairs_dir eval/vpswap_bb24 \
      --results_root results/vpswap
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from vpswap_score import (PROPERTIES, build_article_lexicon, load_tok, pll,
                          swap_at)
from transformers import AutoModelForMaskedLM


def old_swap_at(s: str, i: int, w_old: str, w_new: str) -> str:
    """The pre-fix behaviour, for identifying which items changed."""
    return s[:i] + w_new + s[i + len(w_old):]


def affected_items(pairs_dir: str, art_lex: dict) -> dict:
    """Map (property, line, which) -> new swapped sentence, for changed items."""
    out = {}
    for prop in PROPERTIES:
        pf = Path(pairs_dir) / f"vp_swap_{prop}_pairs.txt"
        if not pf.exists():
            continue
        for line_idx, raw in enumerate(pf.open()):
            parts = raw.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            _b, _r, w1, s1, i1, w2, s2, i2 = parts[:8]
            i1, i2 = int(i1), int(i2)
            for which, (s, i, wo, wn) in enumerate(
                    [(s1, i1, w1, w2), (s2, i2, w2, w1)], start=1):
                old = old_swap_at(s, i, wo, wn)
                new = swap_at(s, i, wo, wn, art_lex)
                if old != new:
                    out[(prop, line_idx, which)] = new
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--models_root", default=None,
                    help="local checkpoint root; falls back to the HF Hub")
    ap.add_argument("--org", default="augustinian-babylm")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the HF snapshot after each revision (disk quota)")
    args = ap.parse_args()

    art_lex = build_article_lexicon(args.pairs_dir)
    changed = affected_items(args.pairs_dir, art_lex)
    print(f"article lexicon: {len(art_lex)} nouns")
    print(f"items whose swapped sentence changes: {len(changed)}")
    by_prop: dict = {}
    for (prop, _l, _w) in changed:
        by_prop[prop] = by_prop.get(prop, 0) + 1
    for prop in PROPERTIES:
        print(f"   {prop:<15} {by_prop.get(prop, 0):>5}")

    files = sorted(Path(args.results_root).glob("*/*.jsonl"))
    print(f"\nresult files found: {len(files)}")
    if args.dry_run:
        for f in files[:5]:
            print("  would patch", f)
        print("  ...")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for n, f in enumerate(files, 1):
        model_name = f.parent.name
        revision = f.stem
        src = (str(Path(args.models_root) / model_name / revision)
               if args.models_root else f"{args.org}/{model_name}")
        rev = "main" if args.models_root else revision

        rows = [json.loads(l) for l in f.open() if l.strip()]
        hits = [r for r in rows
                if (r["property"], r["line"], r["which"]) in changed]
        if not hits:
            print(f"[{n}/{len(files)}] {model_name}/{revision}: nothing to patch")
            continue

        tok = load_tok(src, rev)
        model = AutoModelForMaskedLM.from_pretrained(
            src, trust_remote_code=True, revision=rev)
        model.to(device).eval()

        flips = 0
        for r in hits:
            new_sent = changed[(r["property"], r["line"], r["which"])]
            p_s = pll(new_sent, model, tok, device)
            was = r["correct"]
            r["pll_swap"] = round(p_s, 4)
            r["correct"] = bool(r["pll_orig"] > p_s)
            flips += (was != r["correct"])

        backup = f.with_suffix(".jsonl.prearticle")
        if not backup.exists():
            shutil.copy2(f, backup)
        with f.open("w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")

        acc = 100 * sum(r["correct"] for r in rows) / len(rows)
        print(f"[{n}/{len(files)}] {model_name}/{revision}: "
              f"{len(hits)} rescored, {flips} flipped, acc {acc:.2f}%")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.cleanup and not args.models_root:
            from huggingface_hub import scan_cache_dir
            try:
                cache = scan_cache_dir()
                for repo in cache.repos:
                    if repo.repo_id == src:
                        revs = [r.commit_hash for r in repo.revisions]
                        cache.delete_revisions(*revs).execute()
            except Exception as e:
                print("   (cache cleanup skipped:", e, ")")

    print("\ndone. originals saved as *.jsonl.prearticle")


if __name__ == "__main__":
    main()
