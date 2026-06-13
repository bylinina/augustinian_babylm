#!/usr/bin/env python
"""
Stage 1.5: word-level coverage of the text-only BabyLM corpus by the
image-grounded text (annotation captions).

For every word: how often it appears in the text-only training corpus vs. in
the image-text. "Covered" = appears at least once in image-text. Reports type
coverage (unique words) and token coverage (occurrences), at several image-
frequency thresholds, plus the most divergent words on each side.

Normalization: lowercase, strip leading/trailing punctuation, whitespace split.
No lemmatization. Region + whole-image captions pooled together.

CPU only. Usage:
    python analysis/word_coverage.py --out_dir analysis/out
"""
import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

CORPUS_URL = "https://raw.githubusercontent.com/Leukas/babylm25/main/data/bb24.train"
_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def norm_words(text):
    out = []
    for tok in text.split():
        w = _PUNCT.sub("", tok.lower())
        if w:
            out.append(w)
    return out


def count_corpus(cache_path):
    if not cache_path.exists():
        print("downloading corpus...")
        urllib.request.urlretrieve(CORPUS_URL, cache_path)
    c = Counter()
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            c.update(norm_words(line))
    return c


def count_image_text(dataset_repo):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    ann = pd.read_parquet(hf_hub_download(dataset_repo, "annotations.parquet",
                          repo_type="dataset"), columns=["text"])
    c = Counter()
    for t in ann["text"].fillna(""):
        c.update(norm_words(t))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_repo", default="augustinian-babylm/augustinian_babylm")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/out"))
    ap.add_argument("--corpus_cache", type=Path, default=Path("analysis/bb24.train"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("counting text-only corpus...")
    text_only = count_corpus(args.corpus_cache)
    print("counting image-text...")
    image = count_image_text(args.dataset_repo)

    total_text_types = len(text_only)
    total_text_tokens = sum(text_only.values())
    print(f"corpus: {total_text_types:,} word types, {total_text_tokens:,} tokens")
    print(f"image-text: {len(image):,} word types, {sum(image.values()):,} tokens")

    # coverage at several image-frequency thresholds
    summary = {"corpus_types": total_text_types, "corpus_tokens": total_text_tokens,
               "image_types": len(image), "image_tokens": sum(image.values()),
               "coverage": {}}
    for thr in (1, 5, 20):
        covered = {w for w in text_only if image.get(w, 0) >= thr}
        type_cov = len(covered) / total_text_types
        token_cov = sum(text_only[w] for w in covered) / total_text_tokens
        summary["coverage"][f"image_freq_ge_{thr}"] = {
            "type_coverage": round(type_cov, 4),
            "token_coverage": round(token_cov, 4)}
        print(f"  image_freq>={thr}: type {type_cov:.1%} | token {token_cov:.1%}")

    # per-word table (union of both vocabularies)
    import csv
    allw = set(text_only) | set(image)
    rows = [(w, text_only.get(w, 0), image.get(w, 0)) for w in allw]
    rows.sort(key=lambda r: -r[1])
    with open(args.out_dir / "word_coverage.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["word", "text_only_count", "image_count"])
        wr.writerows(rows)

    # divergence: frequent in corpus, ABSENT from image-text
    no_img = [(w, n) for w, n in text_only.most_common() if image.get(w, 0) == 0]
    # frequent in image-text, rare/absent in corpus
    img_heavy = sorted(((w, image[w], text_only.get(w, 0)) for w in image),
                       key=lambda r: -r[1])[:50]
    with open(args.out_dir / "divergence_report.txt", "w") as f:
        f.write("=== Top 50 corpus words with ZERO image-text support ===\n")
        f.write("(frequent in training text, no visual grounding)\n")
        for w, n in no_img[:50]:
            f.write(f"  {w:<25} corpus={n:,}\n")
        f.write("\n=== Top 50 image-text words by image frequency ===\n")
        f.write("(image_count, corpus_count)\n")
        for w, ic, tc in img_heavy:
            f.write(f"  {w:<25} image={ic:,}  corpus={tc:,}\n")

    with open(args.out_dir / "word_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote -> {args.out_dir}/ (word_coverage.csv, divergence_report.txt, "
          f"word_coverage_summary.json)")


if __name__ == "__main__":
    main()
