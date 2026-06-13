#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_token_embeddings.py  —  STAGE 2 (tokenizer + aggregation)
===============================================================

Take the per-ROW visual embeddings from Stage 1 (extract_region_embeddings.py)
and turn them into a per-TOKEN embedding table for a given tokenizer. This is
the stage we want to experiment with freely — how to get from a
sentence/description-level visual vector to token-level vectors — so it is
deliberately separate from (and much cheaper than) the GPU encoding stage.

Input  (Stage 1 output, per encoder):
  region_embeddings.parquet : row_id, image_uid, bbox, text, source, is_region
  region_embeddings.npy     : [N, 768], row i aligned with parquet row i

Mapping strategy (pluggable; default = average):
  AGGREGATORS["average"]  : for each token, average the row-embeddings of every
                            row whose text contains that token. Token attribution
                            uses the tokenizer; --seed_last_subword attributes a
                            row only to the LAST subword of its text (default),
                            else to all subwords.
  Add new strategies here later (e.g. weighting by region size, tf-idf over
  tokens, nearest-neighbour, learned maps) without touching Stage 1.

Postprocessing (optional, on by default; matches the project spec):
  mean-center -> L2-normalize -> x scale (0.55 -> std ~= 0.02).

Output (per encoder x tokenizer):
  E_raw.safetensors   [V, 768]  pre-postprocessing per-token aggregate
  E_init.safetensors  [V, 768] (+ seeded_mask) ready for the DeBERTa table
  coverage.parquet    per token: n_rows, seeded
  config.json
"""

import os
import sys
import json
import argparse
import subprocess


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def _ensure_deps():
    try:
        import numpy, pandas, transformers, safetensors  # noqa
    except Exception:
        _pip("-U", "numpy", "pandas", "transformers", "huggingface_hub",
             "safetensors", "tqdm")


# ==========================================================================
# Aggregators: row embeddings + texts + tokenizer -> (sum[V,H], count[V])
# ==========================================================================
def aggregate_average(texts, row_emb, tokenizer, seed_last_subword, H):
    """Each row's embedding is added to the token(s) of its text; per token we
    later divide by count = a plain average over contributing rows."""
    import numpy as np
    from tqdm.auto import tqdm
    V = tokenizer.vocab_size
    sums = np.zeros((V, H), np.float64)
    counts = np.zeros((V,), np.int64)
    for text, vec in tqdm(zip(texts, row_emb), total=len(texts), desc="aggregating"):
        ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        targets = [ids[-1]] if seed_last_subword else ids
        for tid in targets:
            sums[tid] += vec
            counts[tid] += 1
    return sums, counts


AGGREGATORS = {"average": aggregate_average}


def postprocess(E_raw, seeded, scale):
    """mean-center -> L2-normalize -> scale. Operates on seeded rows only."""
    import numpy as np
    X = E_raw[seeded].astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    X = scale * (X / n)
    E = np.zeros_like(E_raw)
    E[seeded] = X.astype(np.float32)
    return E


# ==========================================================================
# Build
# ==========================================================================
def build(args):
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    from safetensors.numpy import save_file

    # ---- load Stage 1 output (local dir or HF repo) ----
    src = args.region_dir
    if args.region_repo:
        from huggingface_hub import snapshot_download
        src = snapshot_download(args.region_repo, repo_type="dataset",
                                allow_patterns=[f"{args.encoder_name}/*"])
        src = os.path.join(src, args.encoder_name)
    meta = pd.read_parquet(os.path.join(src, "region_embeddings.parquet"))
    row_emb = np.load(os.path.join(src, "region_embeddings.npy")).astype(np.float32)
    assert len(meta) == len(row_emb), "parquet/npy row mismatch"
    H = row_emb.shape[1]
    print(f"Loaded {len(meta):,} region embeddings (dim {H}) for encoder={args.encoder_name}")
    if args.region_kind == "region":
        m = meta["is_region"].values
        meta, row_emb = meta[m].reset_index(drop=True), row_emb[m]
    elif args.region_kind == "whole":
        m = ~meta["is_region"].values
        meta, row_emb = meta[m].reset_index(drop=True), row_emb[m]
    # "both" (default): use all rows
    print(f"Using {len(meta):,} rows (region_kind={args.region_kind})")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    V = tokenizer.vocab_size
    print(f"Tokenizer {args.tokenizer}: vocab {V:,} | aggregator={args.aggregator}")

    agg = AGGREGATORS[args.aggregator]
    sums, counts = agg(meta["text"].tolist(), row_emb, tokenizer,
                       args.seed_last_subword, H)

    seeded = counts > 0
    n_seeded = int(seeded.sum())
    E_raw = np.zeros((V, H), np.float32)
    E_raw[seeded] = (sums[seeded] / counts[seeded, None]).astype(np.float32)
    E_init = postprocess(E_raw, seeded, args.scale) if args.postprocess else E_raw.copy()
    std = float(E_init[seeded].std()) if n_seeded else 0.0
    print(f"Seeded {n_seeded:,}/{V:,} ({100*n_seeded/V:.1f}%) | "
          f"post-proc std {std:.4f}" + ("" if args.postprocess else " (postproc OFF)"))

    out = os.path.join(args.out_dir, args.encoder_name, args.tokenizer_tag)
    os.makedirs(out, exist_ok=True)
    save_file({"embeddings": E_raw}, os.path.join(out, "E_raw.safetensors"))
    save_file({"embeddings": E_init, "seeded_mask": seeded.astype(np.int8)},
              os.path.join(out, "E_init.safetensors"))
    pd.DataFrame({"token_id": np.arange(V), "n_rows": counts,
                  "seeded": seeded}).to_parquet(os.path.join(out, "coverage.parquet"))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"encoder": args.encoder_name, "tokenizer": args.tokenizer,
                   "vocab_size": V, "hidden": H, "aggregator": args.aggregator,
                   "seed_last_subword": args.seed_last_subword,
                   "region_kind": args.region_kind, "postprocess": args.postprocess,
                   "scale": args.scale, "n_seeded": n_seeded,
                   "realised_std": std}, f, indent=2)
    print(f"Saved -> {out}")

    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
        HfApi().upload_folder(folder_path=out, repo_id=args.push_to_hub,
                              repo_type="dataset",
                              path_in_repo=f"{args.encoder_name}/{args.tokenizer_tag}")
        print(f"Pushed -> {args.push_to_hub}/{args.encoder_name}/{args.tokenizer_tag}")


def build_parser():
    p = argparse.ArgumentParser(description="Stage 2: region embeddings -> per-token table")
    # where Stage 1 output lives: a local dir OR an HF repo
    p.add_argument("--region_dir", type=str, default="./region_embeddings",
                   help="Local dir containing <encoder_name>/region_embeddings.*")
    p.add_argument("--region_repo", type=str, default=None,
                   help="Alternatively, the HF dataset repo Stage 1 pushed to.")
    p.add_argument("--encoder_name", type=str, default="dinov3")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="e.g. augustinian-babylm/babylm-bpe-75k")
    p.add_argument("--tokenizer_tag", type=str, required=True, help="e.g. 75k")
    p.add_argument("--out_dir", type=str, default="./token_embeddings")
    p.add_argument("--aggregator", type=str, default="average", choices=list(AGGREGATORS))
    p.add_argument("--region_kind", type=str, default="both",
                   choices=["both", "region", "whole"],
                   help="Use bbox-region rows, whole-image rows, or both.")
    p.add_argument("--seed_last_subword", action="store_true", default=True,
                   help="Attribute each row's vector only to the LAST subword of "
                        "its text (head-noun heuristic). Default True.")
    p.add_argument("--no-seed_last_subword", dest="seed_last_subword",
                   action="store_false",
                   help="Attribute each row's vector to ALL subwords of its text.")
    p.add_argument("--postprocess", action="store_true", default=True,
                   help="mean-center/L2/scale. Use --no-postprocess to skip.")
    p.add_argument("--no-postprocess", dest="postprocess", action="store_false")
    p.add_argument("--scale", type=float, default=0.55)
    p.add_argument("--push_to_hub", type=str, default=None)
    return p


def main(argv=None):
    _ensure_deps()
    build(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
