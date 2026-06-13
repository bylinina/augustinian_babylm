#!/usr/bin/env python
"""
Stage 1.5 deep dive: characterize which corpus vocabulary lacks visual support,
by POS (spaCy), concreteness (Brysbaert 2014), and frequency band.

Inputs:
  analysis/out/word_coverage.csv   (from word_coverage.py)
  analysis/bb24.train              (cached corpus, for spaCy POS tagging)
Outputs (analysis/out/):
  coverage_by_pos.csv, coverage_by_concreteness.csv, coverage_by_freqband.csv,
  deep_summary.json, and plots coverage_by_pos.png / _concreteness.png / _freqband.png

CPU only. spaCy en_core_web_sm + Brysbaert norms (downloaded once).
"""
import argparse
import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONC_URL = ("https://raw.githubusercontent.com/ArtsEngine/concreteness/master/"
            "Concreteness_ratings_Brysbaert_et_al_BRM.txt")


def get_pos_for_vocab(corpus_path, vocab, max_chars=None):
    """Tag the corpus with spaCy; assign each word type its MOST COMMON POS
    (context-aware tagging, then majority-vote per type). Only tags words in
    `vocab` to keep it fast."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    pos_counts = defaultdict(Counter)
    text = open(corpus_path, encoding="utf-8").read()
    if max_chars:
        text = text[:max_chars]
    # process in pipe batches over lines for memory
    lines = text.splitlines()
    for doc in nlp.pipe(lines, batch_size=200):
        for t in doc:
            w = t.text.lower().strip(".,!?;:\"'()[]{}")
            if w in vocab:
                pos_counts[w][t.pos_] += 1
    return {w: c.most_common(1)[0][0] for w, c in pos_counts.items() if c}


def get_concreteness(cache_path):
    if not cache_path.exists():
        print("downloading Brysbaert concreteness norms...")
        urllib.request.urlretrieve(CONC_URL, cache_path)
    df = pd.read_csv(cache_path, sep="\t")
    # columns: Word, Conc.M, ...
    return dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/out"))
    ap.add_argument("--corpus_cache", type=Path, default=Path("analysis/bb24.train"))
    ap.add_argument("--conc_cache", type=Path,
                    default=Path("analysis/concreteness.txt"))
    ap.add_argument("--pos_max_chars", type=int, default=0,
                    help="cap corpus chars for POS tagging (0 = full; set e.g. "
                         "20000000 to subsample for speed)")
    args = ap.parse_args()

    cov = pd.read_csv(args.out_dir / "word_coverage.csv")
    cov = cov[cov.text_only_count > 0].copy()
    cov["covered"] = cov.image_count > 0
    corpus_vocab = set(cov.word.astype(str))

    # ---- POS ----
    print("tagging POS with spaCy (this is the slow part)...")
    pos_map = get_pos_for_vocab(args.corpus_cache, corpus_vocab,
                                args.pos_max_chars or None)
    cov["pos"] = cov.word.astype(str).map(pos_map)
    pos_rows = []
    for pos, g in cov.dropna(subset=["pos"]).groupby("pos"):
        # token-weighted coverage within this POS
        tok = g.text_only_count.sum()
        tok_cov = g.loc[g.covered, "text_only_count"].sum() / tok if tok else 0
        type_cov = g.covered.mean()
        pos_rows.append((pos, len(g), int(tok), round(type_cov, 4), round(tok_cov, 4)))
    pos_df = pd.DataFrame(pos_rows, columns=["pos", "n_types", "n_tokens",
                                             "type_coverage", "token_coverage"])
    pos_df = pos_df.sort_values("n_tokens", ascending=False)
    pos_df.to_csv(args.out_dir / "coverage_by_pos.csv", index=False)

    # ---- concreteness ----
    print("joining concreteness norms...")
    conc = get_concreteness(args.conc_cache)
    cov["conc"] = cov.word.astype(str).map(conc)
    have = cov.dropna(subset=["conc"])
    bins = [1, 2, 3, 4, 5.01]
    labels = ["1-2 (abstract)", "2-3", "3-4", "4-5 (concrete)"]
    have = have.assign(conc_bin=pd.cut(have.conc, bins=bins, labels=labels,
                                       right=False))
    conc_rows = []
    for b, g in have.groupby("conc_bin", observed=True):
        tok = g.text_only_count.sum()
        conc_rows.append((str(b), len(g), round(g.covered.mean(), 4),
                          round(g.loc[g.covered, "text_only_count"].sum() / tok, 4)
                          if tok else 0))
    conc_df = pd.DataFrame(conc_rows, columns=["concreteness_bin", "n_types",
                                               "type_coverage", "token_coverage"])
    conc_df.to_csv(args.out_dir / "coverage_by_concreteness.csv", index=False)
    # correlation: per-word concreteness vs covered (point-biserial-ish)
    conc_corr = float(np.corrcoef(have.conc, have.covered.astype(int))[0, 1])

    # ---- frequency bands ----
    print("frequency bands...")
    fb_bins = [1, 5, 20, 100, 1000, 1e9]
    fb_labels = ["1-4", "5-19", "20-99", "100-999", "1000+"]
    cov = cov.assign(freq_bin=pd.cut(cov.text_only_count, bins=fb_bins,
                                     labels=fb_labels, right=False))
    fb_rows = []
    for b, g in cov.groupby("freq_bin", observed=True):
        fb_rows.append((str(b), len(g), round(g.covered.mean(), 4)))
    fb_df = pd.DataFrame(fb_rows, columns=["freq_band", "n_types", "type_coverage"])
    fb_df.to_csv(args.out_dir / "coverage_by_freqband.csv", index=False)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    p = pos_df.head(12)
    x = np.arange(len(p))
    ax.bar(x - 0.2, p.type_coverage * 100, 0.4, label="type coverage")
    ax.bar(x + 0.2, p.token_coverage * 100, 0.4, label="token coverage")
    ax.set_xticks(x); ax.set_xticklabels(p.pos, rotation=45, ha="right")
    ax.set_ylabel("% vision-supported"); ax.set_title("Coverage by part of speech")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "coverage_by_pos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(conc_df))
    ax.bar(x - 0.2, conc_df.type_coverage * 100, 0.4, label="type coverage")
    ax.bar(x + 0.2, conc_df.token_coverage * 100, 0.4, label="token coverage")
    ax.set_xticks(x); ax.set_xticklabels(conc_df.concreteness_bin, rotation=20)
    ax.set_ylabel("% vision-supported")
    ax.set_title(f"Coverage by concreteness (corr={conc_corr:.2f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "coverage_by_concreteness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(fb_df.freq_band, fb_df.type_coverage * 100, color="steelblue")
    ax.set_ylabel("% types vision-supported"); ax.set_xlabel("corpus frequency band")
    ax.set_title("Coverage by frequency band")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "coverage_by_freqband.png", dpi=150)
    plt.close(fig)

    summary = {"concreteness_correlation_covered": round(conc_corr, 4),
               "pos_table": pos_df.to_dict("records"),
               "concreteness_table": conc_df.to_dict("records"),
               "freqband_table": fb_df.to_dict("records")}
    json.dump(summary, open(args.out_dir / "deep_summary.json", "w"), indent=2)
    print("\n=== POS ==="); print(pos_df.to_string(index=False))
    print("\n=== concreteness ==="); print(conc_df.to_string(index=False))
    print(f"  (corr concreteness vs covered: {conc_corr:.3f})")
    print("\n=== freq bands ==="); print(fb_df.to_string(index=False))
    print(f"\nwrote tables + plots -> {args.out_dir}/")


if __name__ == "__main__":
    main()
