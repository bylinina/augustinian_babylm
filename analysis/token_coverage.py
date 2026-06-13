#!/usr/bin/env python
"""
Stage 1.5 (per-tokenizer): coverage of text-only corpus tokens by image-text,
for each BabyLM tokenizer. Directly predicts how much of each tokenizer's
embedding table can be vision-initialized (image_count > 0 == seedable).

CPU only. Usage:
    python analysis/token_coverage.py --out_dir analysis/out
"""
import argparse
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

CORPUS_URL = "https://raw.githubusercontent.com/Leukas/babylm25/main/data/bb24.train"
VOCABS = ["50k", "75k", "100k"]


def iter_corpus_lines(cache_path):
    if not cache_path.exists():
        urllib.request.urlretrieve(CORPUS_URL, cache_path)
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def count_tokens(texts, tokenizer, batch=2000):
    c = Counter()
    buf = []
    def flush():
        if not buf:
            return
        for enc in tokenizer.encode_batch(buf):
            c.update(enc.ids)
        buf.clear()
    for t in texts:
        buf.append(t)
        if len(buf) >= batch:
            flush()
    flush()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_repo", default="augustinian-babylm/augustinian_babylm")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/out"))
    ap.add_argument("--corpus_cache", type=Path, default=Path("analysis/bb24.train"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer  # pure Rust/Python, no torch

    print("loading image-text captions...")
    ann = pd.read_parquet(hf_hub_download(args.dataset_repo, "annotations.parquet",
                          repo_type="dataset"), columns=["text"])
    img_texts = ann["text"].fillna("").tolist()

    summary = {}
    for v in VOCABS:
        print(f"\n=== tokenizer {v} ===")
        tok_path = hf_hub_download(f"augustinian-babylm/babylm-bpe-{v}",
                                   "tokenizer.json", repo_type="model")
        tok = Tokenizer.from_file(tok_path)

        print("  tokenizing corpus...")
        text_only = count_tokens(iter_corpus_lines(args.corpus_cache), tok)
        print("  tokenizing image-text...")
        image = count_tokens(img_texts, tok)

        vocab_size = tok.get_vocab_size()
        text_tokens_total = sum(text_only.values())
        # seedable = appears in image-text at least once
        for thr in (1, 5, 20):
            seedable = {t for t in text_only if image.get(t, 0) >= thr}
            # vocab-level: of all vocab ids that occur in corpus, frac seedable
            corpus_vocab = set(text_only)
            type_cov = len(seedable) / len(corpus_vocab)
            token_cov = sum(text_only[t] for t in seedable) / text_tokens_total
            summary.setdefault(v, {})[f"image_freq_ge_{thr}"] = {
                "type_coverage": round(type_cov, 4),
                "token_coverage": round(token_cov, 4)}
            print(f"  image_freq>={thr}: vocab-in-corpus {len(corpus_vocab):,} | "
                  f"type {type_cov:.1%} | token {token_cov:.1%}")
        summary[v]["vocab_size"] = vocab_size
        summary[v]["corpus_vocab_used"] = len(text_only)

        # per-token table
        allt = set(text_only) | set(image)
        rows = []
        for tid in allt:
            s = tok.id_to_token(tid) or ""
            rows.append((tid, s, text_only.get(tid, 0), image.get(tid, 0)))
        rows.sort(key=lambda r: -r[2])
        with open(args.out_dir / f"token_coverage_{v}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["token_id", "token_str", "text_only_count", "image_count"])
            wr.writerows(rows)

    with open(args.out_dir / "token_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote -> {args.out_dir}/ (token_coverage_<vocab>.csv, summary.json)")


if __name__ == "__main__":
    main()
