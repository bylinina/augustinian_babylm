#!/usr/bin/env python
"""Do seeded embeddings drift from their visual anchors, and does drift
explain the late-training decay of the VP-Swap advantage?"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoTokenizer

MODEL = "augustinian-babylm/deberta-base-75k-sam"
REVS = ["chck_1M", "chck_5M", "chck_10M", "chck_30M", "chck_50M",
        "chck_70M", "chck_100M"]

init = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_init.safetensors", repo_type="dataset"))
A, seeded = init["embeddings"], init["seeded_mask"].astype(bool)
An = A[seeded] / (np.linalg.norm(A[seeded], axis=1, keepdims=True) + 1e-9)
print(f"anchor: {seeded.sum()} seeded rows")

import pandas as pd
cov = pd.read_parquet(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/coverage.parquet", repo_type="dataset"))
n_rows = cov["n_rows"].values[seeded]   # image-support per seeded token

def emb(rev):
    f = hf_hub_download(MODEL, "model.safetensors", revision=rev)
    d = load_file(f)
    key = [k for k in d if "word_embeddings.weight" in k][0]
    return d[key][:75000]

print(f"\n{'rev':>10s}  {'mean cos':>8s}  {'q25':>6s}  {'q75':>6s}")
drift_final = None
for rev in REVS:
    E = emb(rev)[seeded]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    cos = (En * An).sum(1)
    print(f"{rev:>10s}  {cos.mean():8.3f}  {np.quantile(cos,.25):6.3f}  "
          f"{np.quantile(cos,.75):6.3f}")
    if rev == "chck_100M":
        drift_final = 1 - cos

# --- join drift to VP-Swap advantage change (peak=10M vs final=100M) ---
tok = AutoTokenizer.from_pretrained("augustinian-babylm/babylm-bpe-75k")
seeded_ids = {int(i): j for j, i in enumerate(np.where(seeded)[0])}

def word_drift(w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    ds = [drift_final[seeded_ids[t]] for t in ids if t in seeded_ids]
    return float(np.mean(ds)) if ds else None

def load_items(model, rev):
    return {(r["property"], r["line"], r["which"]): r for r in
            map(json.loads, open(f"results/vpswap/{model}/{rev}.jsonl"))}

meta = {}
for prop in ["color", "material", "relative_size", "shape"]:
    for i, l in enumerate(open(f"eval/vpswap_bb24/vp_swap_{prop}_pairs.meta.jsonl")):
        meta[(prop, i)] = json.loads(l)

word_delta = defaultdict(lambda: [0, 0, 0, 0])  # peak_v, peak_b, fin_v, fin_b sums
word_n = defaultdict(int)
for j, (rev, off) in enumerate([("chck_10M", 0), ("chck_100M", 2)]):
    a = load_items("deberta-base-75k-sam", rev)
    b = load_items("deberta-base-75k", rev)
    for k in a:
        m = meta[(k[0], k[1])]
        w = m["w1"] if k[2] == 1 else m["w2"]
        sd = m["seeded_w1"] if k[2] == 1 else m["seeded_w2"]
        if not sd: continue
        word_delta[w][off] += a[k]["correct"]
        word_delta[w][off + 1] += b[k]["correct"]
        if j == 0: word_n[w] += 1

pairs = []
for w, (pv, pb, fv, fb) in word_delta.items():
    n = word_n[w]
    if n < 3: continue
    d = word_drift(w)
    if d is None: continue
    adv_change = (fv - fb) / n - (pv - pb) / n   # negative = advantage shrank
    pairs.append((d, adv_change))
dr = np.array([p[0] for p in pairs]); ac = np.array([p[1] for p in pairs])
r = np.corrcoef(dr, ac)[0, 1]
print(f"\n{len(pairs)} seeded words (>=3 items): corr(drift, advantage change) "
      f"= {r:+.3f}")
lo, hi = dr < np.median(dr), dr >= np.median(dr)
print(f"low-drift words: mean advantage change {ac[lo].mean():+.4f}")
print(f"high-drift words: mean advantage change {ac[hi].mean():+.4f}")

# ======================================================================
# RSA: does the RELATIONAL structure among seeded words survive, even
# though absolute positions do not?  (rotation -> cos ~ 0 but RSA ~ 1)
# ======================================================================
def last_seeded_token(w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    for t in reversed(ids):
        if seeded[t]:
            return t
    return None

word_list = sorted(word_n)                      # seeded VP-Swap words
tid = {}
for w in word_list:
    t = last_seeded_token(w)
    if t is not None and t not in tid.values():
        tid[w] = t
ids = np.array(sorted(set(tid.values())))
print(f"\nRSA over {len(ids)} seeded VP-Swap word tokens")

def sim_upper(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = Mn @ Mn.T
    iu = np.triu_indices(len(M), k=1)
    return S[iu]

anchor_sims = sim_upper(A[ids].astype(np.float64))

print(f"{'model @ rev':>28s}  {'RSA to visual anchor':>20s}")
for model, revs in [("augustinian-babylm/deberta-base-75k-sam",
                     ["chck_1M", "chck_10M", "chck_50M", "chck_100M"]),
                    ("augustinian-babylm/deberta-base-75k",
                     ["chck_100M"])]:
    for rev in revs:
        f = hf_hub_download(model, "model.safetensors", revision=rev)
        d = load_file(f)
        key = [k for k in d if "word_embeddings.weight" in k][0]
        E = d[key][:75000][ids].astype(np.float64)
        r = np.corrcoef(anchor_sims, sim_upper(E))[0, 1]
        tag = model.split("-")[-1] + " @ " + rev
        print(f"{tag:>28s}  {r:20.3f}")

