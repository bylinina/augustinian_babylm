#!/usr/bin/env python
"""Does grounding of ABSTRACT words matter?  Four zero-training probes:
A. composition: seeded tokens across concreteness bands / POS
B. seed informativeness: pre-normalization centered norm vs concreteness
   (abstract seeds averaged over diverse images should sit near the mean)
C. anchor retention (cos + RSA) by concreteness band at 100M
D. behavior: BLiMP phenomenon deltas (3 seeds), function-word phenomena
   vs the rest."""
import csv, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoTokenizer

# ---- norms + tokenizer + seeded mask ----
conc, pos = {}, {}
with open("analysis/concreteness.txt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try:
            conc[row["Word"].lower()] = float(row["Conc.M"])
            pos[row["Word"].lower()] = row.get("Dom_Pos", "")
        except (ValueError, KeyError): pass

tok = AutoTokenizer.from_pretrained("augustinian-babylm/babylm-bpe-75k")
init = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_init.safetensors", repo_type="dataset"))
raw = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_raw.safetensors", repo_type="dataset"))["embeddings"]
seeded = init["seeded_mask"].astype(bool)

# map whole-word tokens -> concreteness
tok_word = {}
for tid in np.where(seeded)[0]:
    t = tok.convert_ids_to_tokens(int(tid))
    w = t.lstrip("\u0120").lower()
    if w.isalpha() and w in conc:
        tok_word[int(tid)] = w

BANDS = [(1.0, 2.0, "1-2 abstract"), (2.0, 3.0, "2-3"),
         (3.0, 4.0, "3-4"), (4.0, 5.01, "4-5 concrete")]
def band(w):
    c = conc[w]
    for lo, hi, name in BANDS:
        if lo <= c < hi: return name

by_band = defaultdict(list)
for tid, w in tok_word.items():
    by_band[band(w)].append(tid)

print("=== A. seeded tokens with norms, by concreteness band ===")
for _, _, name in BANDS:
    tids = by_band[name]
    ex_pos = defaultdict(int)
    for t in tids: ex_pos[pos.get(tok_word[t], "")] += 1
    top = sorted(ex_pos.items(), key=lambda kv: -kv[1])[:3]
    print(f"  {name:14s} n={len(tids):5d}  top POS: "
          + ", ".join(f"{p or '?'}:{n}" for p, n in top))

# ---- B. informativeness: centered norm of the raw seed ----
print("\n=== B. seed informativeness (centered E_raw norm) by band ===")
mu = raw[seeded].mean(0)
for _, _, name in BANDS:
    tids = by_band[name]
    d = np.linalg.norm(raw[tids] - mu, axis=1)
    print(f"  {name:14s} mean {d.mean():7.3f}  median {np.median(d):7.3f}")

# ---- C. anchor retention at 100M by band (cos + RSA) ----
print("\n=== C. anchor retention at chck_100M by band ===")
A = init["embeddings"]
def embeddings(model, rev):
    f = hf_hub_download(model, "model.safetensors", revision=rev)
    d = load_file(f)
    key = [k for k in d if "word_embeddings.weight" in k][0]
    return d[key][:75000]
E = embeddings("augustinian-babylm/deberta-base-75k-sam", "chck_100M")
B_ctl = embeddings("augustinian-babylm/deberta-base-75k", "chck_100M")
def rsa(X, Y):
    Xn = X / (np.linalg.norm(X, 1e-9 + X.std()*0 + np.ones(1), keepdims=True) if False else np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    Sx, Sy = Xn @ Xn.T, Yn @ Yn.T
    iu = np.triu_indices(len(X), k=1)
    return float(np.corrcoef(Sx[iu], Sy[iu])[0, 1])
for _, _, name in BANDS:
    tids = by_band[name]
    if len(tids) < 20: continue
    An = A[tids] / (np.linalg.norm(A[tids], axis=1, keepdims=True) + 1e-9)
    En = E[tids] / (np.linalg.norm(E[tids], axis=1, keepdims=True) + 1e-9)
    cos = float((An * En).sum(1).mean())
    r_sam = rsa(A[tids].astype(np.float64), E[tids].astype(np.float64))
    r_ctl = rsa(A[tids].astype(np.float64), B_ctl[tids].astype(np.float64))
    print(f"  {name:14s} n={len(tids):5d}  cos {cos:+.3f}  "
          f"RSA sam {r_sam:.3f} | baseline floor {r_ctl:.3f}")

# ---- D. BLiMP phenomenon deltas, 3 seeds, function-word phenomena ----
print("\n=== D. BLiMP phenomenon deltas (mean of 3 seeds) ===")
ZS = Path.home() / "babylm-eval/strict/results"
PAIRS = [("deberta-base-75k-sam", "deberta-base-75k"),
         ("deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
         ("deberta-base-75k-sam-s3", "deberta-base-75k-s3")]
FUNC = ("quantifier", "determiner", "npi", "existential")
def subtasks(model):
    rep = list((ZS / model / "main/zero_shot/mlm").glob(
        "blimp/blimp_filtered/best_temperature_report.txt"))
    out, sec = {}, None
    if not rep: return out
    for line in rep[0].read_text().splitlines():
        t = line.strip()
        if t.startswith("### "): sec = t; continue
        m = re.fullmatch(r"([\w\-./ ]+):\s*([\d.]+)", t)
        if m and "AVERAGE" not in (sec or "") and m.group(1) != "TEMPERATURE":
            out[m.group(1)] = float(m.group(2))
    return out
agg = defaultdict(list)
for vm, bm in PAIRS:
    v, b = subtasks(vm), subtasks(bm)
    for k in v:
        if k in b: agg[k].append(v[k] - b[k])
fw = [(k, sum(d)/len(d)) for k, d in agg.items()
      if any(t in k.lower() for t in FUNC)]
other = [(k, sum(d)/len(d)) for k, d in agg.items()
         if not any(t in k.lower() for t in FUNC)]
print(f"  function-word phenomena (n={len(fw)}): "
      f"mean delta {np.mean([d for _, d in fw]):+.2f}")
for k, d in sorted(fw, key=lambda kv: -abs(kv[1]))[:6]:
    print(f"    {k[:50]:50s} {d:+.2f}")
print(f"  all other phenomena (n={len(other)}): "
      f"mean delta {np.mean([d for _, d in other]):+.2f}")
