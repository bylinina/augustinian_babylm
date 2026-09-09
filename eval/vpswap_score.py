#!/usr/bin/env python
"""
Score a DeBERTa MLM checkpoint on VP-Swap pair files (EgoBabyVLM
LT-Swap "visualswap" format, + our optional 9th frame column).

Protocol: mask-each-token pseudo-log-likelihood (identical to the
BLiMP-style scoring in egobabyvlm's evaluation/text/ltswap.py). Each
pair-file line yields TWO minimal-pair items:
  item 1: s1 (w1 at i1)  vs  s1 with w2 substituted at i1
  item 2: s2 (w2 at i2)  vs  s2 with w1 substituted at i2
Correct iff PLL(original) > PLL(swapped).

Emits per-item JSONL (one row per item) for downstream joins against
the .meta.jsonl sidecars (seeded flags, frequencies, frames).

Usage:
  python vpswap_score.py --pairs_dir eval/vpswap_bb24 \
      --model augustinian-babylm/deberta-base-75k-sam --revision chck_10M \
      --out results/vpswap/deberta-base-75k-sam/chck_10M.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROPERTIES = ("color", "material", "relative_size", "shape")


def load_tok(model: str, revision: str):
    try:
        return AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, revision=revision)
    except Exception:
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast.from_pretrained(
            model, trust_remote_code=True, revision=revision)


@torch.no_grad()
def pll(sentence: str, model, tok, device, max_rows: int = 256) -> float:
    """Sum of log P(token_i | rest), masking one position at a time."""
    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"][0]
    special = set(tok.all_special_ids)
    positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
    if not positions:
        return float("-inf")
    total = 0.0
    for start in range(0, len(positions), max_rows):
        chunk = positions[start:start + max_rows]
        batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for r, p in enumerate(chunk):
            batch[r, p] = tok.mask_token_id
        attn = torch.ones_like(batch)
        out = model(input_ids=batch.to(device),
                    attention_mask=attn.to(device)).logits
        logprobs = torch.log_softmax(out, dim=-1)
        for r, p in enumerate(chunk):
            total += logprobs[r, p, ids[p]].item()
    return total


_ART_RE = re.compile(r"\b([Aa]n?)(\s+)$")


def _orthographic_article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def build_article_lexicon(pairs_dir) -> dict:
    """Learn each noun's article from the generated originals.

    The source sentences were written with correct agreement, so they are a
    better authority than a spelling rule ('a unicycle', 'a uterus').
    """
    lex: dict = {}
    for prop in PROPERTIES:
        pf = Path(pairs_dir) / f"vp_swap_{prop}_pairs.txt"
        if not pf.exists():
            continue
        for raw in pf.open():
            parts = raw.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            _b, _r, w1, s1, i1, w2, s2, i2 = parts[:8]
            for w, s, i in ((w1, s1, int(i1)), (w2, s2, int(i2))):
                m = _ART_RE.search(s[:i])
                if m:
                    lex.setdefault(w.lower(), m.group(1).lower())
    return lex


def swap_at(s: str, i: int, w_old: str, w_new: str,
            art_lex: dict | None = None) -> str:
    """Replace the noun at offset i, fixing a preceding a/an to agree.

    Without this, swapping a vowel-initial noun under 'a' yields an
    ungrammatical distractor that the model can reject on agreement alone,
    independently of any property knowledge.
    """
    head, tail = s[:i], s[i + len(w_old):]
    m = _ART_RE.search(head)
    if m:
        want = (art_lex or {}).get(w_new.lower()) or _orthographic_article(w_new)
        if m.group(1)[0].isupper():
            want = want.capitalize()
        head = head[:m.start(1)] + want + m.group(2)
    return head + w_new + tail

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--properties", nargs="+", default=list(PROPERTIES))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tok(args.model, args.revision)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, revision=args.revision)
    model.to(device).eval()

    art_lex = build_article_lexicon(args.pairs_dir)
    print(f"article lexicon: {len(art_lex)} nouns", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_items = n_correct = 0
    with out_path.open("w") as out:
        for prop in args.properties:
            pf = Path(args.pairs_dir) / f"vp_swap_{prop}_pairs.txt"
            for line_idx, raw in enumerate(pf.open()):
                parts = raw.rstrip("\n").split("|")
                b, _rule, w1, s1, i1, w2, s2, i2 = parts[:8]
                frame = parts[8] if len(parts) > 8 else ""
                i1, i2 = int(i1), int(i2)
                for which, (s, i, wo, wn) in enumerate(
                        [(s1, i1, w1, w2), (s2, i2, w2, w1)], start=1):
                    orig, swap = s, swap_at(s, i, wo, wn, art_lex)
                    p_o, p_s = pll(orig, model, tok, device), \
                               pll(swap, model, tok, device)
                    correct = p_o > p_s
                    n_items += 1
                    n_correct += correct
                    out.write(json.dumps({
                        "property": prop, "line": line_idx, "which": which,
                        "bin": int(b), "frame": frame,
                        "w_orig": wo, "w_swap": wn,
                        "pll_orig": round(p_o, 4), "pll_swap": round(p_s, 4),
                        "correct": bool(correct),
                    }) + "\n")
            print(f"[{args.revision}] {prop}: done", flush=True)
    print(f"[{args.revision}] accuracy: {n_correct}/{n_items} "
          f"= {100 * n_correct / max(n_items, 1):.2f}%")


if __name__ == "__main__":
    main()
