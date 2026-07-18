#!/usr/bin/env python
"""Is the retained visual structure USED for mask prediction?
Per-token-class validation MLM loss, sam vs baseline, 3 seed pairs,
identical masks across all models (paired comparison)."""
import argparse, csv
import numpy as np
import torch
from collections import defaultdict
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.numpy import load_file
from transformers import AutoTokenizer, AutoModelForMaskedLM

PAIRS = [("deberta-base-75k-sam", "deberta-base-75k"),
         ("deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
         ("deberta-base-75k-sam-s3", "deberta-base-75k-s3")]
SYNCAT = set("""not no never nothing nobody none neither nor every each all some
any few many most several both either or and but if unless because although
whether than as only even also too very quite must might may could should would
will can one two three four five six seven eight nine ten first second third
the a an this that these those there it""".split())

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_repo", default="Leukas/babylm25")
ap.add_argument("--text_file", default=None, help="override: local valid file")
ap.add_argument("--n_lines", type=int, default=3000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--mask_prob", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

# ---- held-out text ----
if args.text_file:
    path = args.text_file
else:
    files = list_repo_files(args.corpus_repo, repo_type="dataset")
    cand = sorted([f for f in files if "valid" in f.lower() or "dev" in f.lower()],
                  key=lambda f: ("bb24" not in f, f))
    assert cand, f"no valid/dev file in {args.corpus_repo}: pass --text_file"
    print("validation file:", cand[0])
    path = hf_hub_download(args.corpus_repo, cand[0], repo_type="dataset")
lines = [l.strip() for l in open(path, errors="ignore") if len(l.split()) >= 5]
rng = np.random.default_rng(args.seed)
lines = [lines[i] for i in rng.choice(len(lines), min(args.n_lines, len(lines)),
                                      replace=False)]
print(f"{len(lines)} held-out lines")

try:
    tok = AutoTokenizer.from_pretrained("augustinian-babylm/babylm-bpe-75k")
except Exception:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(
        "augustinian-babylm/babylm-bpe-75k")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- token classes ----
conc = {}
with open("analysis/concreteness.txt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try: conc[row["Word"].lower()] = float(row["Conc.M"])
        except (ValueError, KeyError): pass
seeded = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_init.safetensors", repo_type="dataset"))["seeded_mask"].astype(bool)

def tclass(tid):
    t = tok.convert_ids_to_tokens(int(tid))
    w = t.lstrip("\u0120").lower()
    if w in SYNCAT: return "function"
    c = conc.get(w)
    if c is not None and seeded[tid]:
        if c < 2.5: return "abstract(<2.5)"
        if c < 4.0: return "mid(2.5-4)"
        return "concrete(>=4)"
    return "seeded-other" if seeded[tid] else "unseeded"

# ---- build batches + fixed masks ONCE ----
enc = tok(lines, truncation=True, max_length=args.max_len, padding=True,
          return_tensors="np")
ids, attn = enc["input_ids"], enc["attention_mask"]
special = np.isin(ids, list(tok.all_special_ids))
mask = (rng.random(ids.shape) < args.mask_prob) & (attn == 1) & ~special
print(f"masked positions: {mask.sum()}")
masked_ids = ids.copy()
masked_ids[mask] = tok.mask_token_id
pos = np.argwhere(mask)                      # (n_masked, 2)
targets = ids[mask]                          # original token ids
classes = np.array([tclass(t) for t in targets])

def model_losses(name):
    model = AutoModelForMaskedLM.from_pretrained(
        f"augustinian-babylm/{name}", trust_remote_code=True).to(device).eval()
    ce = np.zeros(len(pos), np.float64)
    with torch.inference_mode():
        for s in range(0, len(ids), args.batch):
            sl = slice(s, s + args.batch)
            out = model(input_ids=torch.tensor(masked_ids[sl]).to(device),
                        attention_mask=torch.tensor(attn[sl]).to(device)).logits
            lp = torch.log_softmax(out.float(), -1).cpu().numpy()
            sel = (pos[:, 0] >= s) & (pos[:, 0] < min(s + args.batch, len(ids)))
            for j in np.where(sel)[0]:
                r, c = pos[j]
                ce[j] = -lp[r - s, c, targets[j]]
    del model
    return ce

results = {}
for vm, bm in PAIRS:
    for name in (vm, bm):
        print("scoring", name, flush=True)
        results[name] = model_losses(name)

print(f"\n{'class':>15s} {'n':>7s} | " +
      " | ".join(f"{'d(s'+str(i+1)+')':>7s}" for i in range(3)) +
      f" | {'mean d':>7s}")
order = ["function", "abstract(<2.5)", "mid(2.5-4)", "concrete(>=4)",
         "seeded-other", "unseeded"]
for cls in order:
    m = classes == cls
    if m.sum() < 30: continue
    ds = []
    for vm, bm in PAIRS:
        ds.append(float(results[vm][m].mean() - results[bm][m].mean()))
    print(f"{cls:>15s} {int(m.sum()):7d} | " +
          " | ".join(f"{d:+7.4f}" for d in ds) +
          f" | {np.mean(ds):+7.4f}")
import json as _json
out = {}
for cls in order:
    m = classes == cls
    if m.sum() < 30: continue
    out[cls] = {"n": int(m.sum()),
                "deltas": [float(results[vm][m].mean() - results[bm][m].mean())
                           for vm, bm in PAIRS]}
_json.dump(out, open("eval/mlm_class_results.json", "w"), indent=1)
print("wrote eval/mlm_class_results.json")
print("\nnegative delta = vision-init predicts these tokens BETTER "
      "(lower CE). Same masks for all models; paired per position.")
