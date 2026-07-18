#!/usr/bin/env python
"""Part 2 of the abstract-grounding diagnostic:
D2. BLiMP: per-phenomenon mean concreteness of varying words vs 3-seed delta
D3. supplement + EWoK 3-seed subtask deltas
E.  A-B-C for curated syncategorematic vs classic concrete words."""
import csv, json, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoTokenizer

conc = {}
with open("analysis/concreteness.txt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try: conc[row["Word"].lower()] = float(row["Conc.M"])
        except (ValueError, KeyError): pass

tok = AutoTokenizer.from_pretrained("augustinian-babylm/babylm-bpe-75k")
init = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_init.safetensors", repo_type="dataset"))
raw = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_raw.safetensors", repo_type="dataset"))["embeddings"]
seeded = init["seeded_mask"].astype(bool)
A = init["embeddings"]
ZS = Path.home() / "babylm-eval/strict/results"
PAIRS = [("deberta-base-75k-sam", "deberta-base-75k"),
         ("deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
         ("deberta-base-75k-sam-s3", "deberta-base-75k-s3")]

def report_subtasks(model, glob):
    rep = list((ZS / model / "main/zero_shot/mlm").glob(glob + "/best_temperature_report.txt"))
    out, sec = {}, None
    if not rep: return out
    for line in rep[0].read_text().splitlines():
        t = line.strip()
        if t.startswith("### "): sec = t; continue
        m = re.fullmatch(r"([\w\-./ ]+):\s*([\d.]+)", t)
        if m and "AVERAGE" not in (sec or "") and m.group(1) != "TEMPERATURE":
            out[m.group(1)] = float(m.group(2))
    return out

def seed_mean_deltas(glob):
    agg = defaultdict(list)
    for vm, bm in PAIRS:
        v, b = report_subtasks(vm, glob), report_subtasks(bm, glob)
        for k in v:
            if k in b: agg[k].append(v[k] - b[k])
    return {k: sum(d)/len(d) for k, d in agg.items() if len(d) == 3}

# ---------------- D2: BLiMP concreteness-of-varying-words ----------------
print("=== D2. BLiMP: per-phenomenon concreteness of varying words vs delta ===")
deltas = seed_mean_deltas("blimp/blimp_filtered")
base = Path.home() / "babylm-eval/strict/evaluation_data"
cands = [d for d in base.rglob("*") if d.is_dir() and "blimp" in d.name.lower()
         and "suppl" not in d.name.lower() and list(d.glob("*.jsonl"))]
data_dir = cands[0] if cands else base / "MISSING"
print(f"  using data dir: {data_dir}")
rows = []
for f in sorted(data_dir.glob("*.jsonl")):
    name = f.stem
    if name not in deltas: continue
    cs = []
    for line in f.open():
        d = json.loads(line)
        g = d.get("sentence_good") or d.get("sentence_grammatical") or ""
        b = d.get("sentence_bad") or d.get("sentence_ungrammatical") or ""
        gw = set(re.findall(r"[a-z']+", g.lower()))
        bw = set(re.findall(r"[a-z']+", b.lower()))
        for w in gw ^ bw:
            if w in conc: cs.append(conc[w])
    if cs:
        rows.append((name, float(np.mean(cs)), deltas[name]))
if rows:
    cvals = np.array([r[1] for r in rows]); dvals = np.array([r[2] for r in rows])
    r = np.corrcoef(cvals, dvals)[0, 1]
    print(f"  {len(rows)} phenomena | corr(mean concreteness of varying words, "
          f"3-seed delta) = {r:+.3f}")
    lo = cvals < np.median(cvals)
    print(f"  abstract-varying half: mean delta {dvals[lo].mean():+.2f} | "
          f"concrete-varying half: {dvals[~lo].mean():+.2f}")
    rows.sort(key=lambda x: x[1])
    print("  most abstract-varying:", ", ".join(f"{n}({d:+.1f})" for n, _, d in rows[:4]))
    print("  most concrete-varying:", ", ".join(f"{n}({d:+.1f})" for n, _, d in rows[-4:]))
else:
    print("  no jsonl matched -- check data path:", data_dir)

# ---------------- D3: supplement + ewok 3-seed subtask deltas ----------------
for label, glob in [("supplement", "blimp/supplement_filtered"),
                    ("ewok", "ewok/ewok_filtered")]:
    print(f"\n=== D3. {label}: 3-seed subtask deltas ===")
    for k, d in sorted(seed_mean_deltas(glob).items(), key=lambda kv: -kv[1]):
        print(f"  {k[:44]:44s} {d:+.2f}")

# ---------------- E: curated syncategorematic vs concrete ----------------
SYNCAT = """not no never nothing nobody none neither nor every each all some any
few many most several both either or and but if unless because although whether
than as only even also too very quite must might may could should would will can
one two three four five six seven eight nine ten first second third the a an
this that these those there it""".split()
CONCRETE = """dog cat table chair apple banana car tree house ball cup door book
shoe water bread bird fish horse bed spoon window bottle hat box stone flower
truck boat chicken""".split()

def whole_token(w):
    for form in ("\u0120" + w, w):
        tid = tok.convert_tokens_to_ids(form)
        if tid is not None and tid != tok.unk_token_id and tid < 75000:
            return tid
    return None

def abc(words, label, mu):
    tids = [(w, whole_token(w)) for w in words]
    found = [(w, t) for w, t in tids if t is not None and seeded[t]]
    missing = [w for w, t in tids if t is None or not seeded[t]]
    ids = np.array([t for _, t in found])
    print(f"\n--- {label}: {len(found)}/{len(words)} seeded "
          f"(unseeded/no-token: {', '.join(missing[:8])}{'...' if len(missing)>8 else ''}) ---")
    d = np.linalg.norm(raw[ids] - mu, axis=1)
    print(f"  B centered-norm: mean {d.mean():.2f}  median {np.median(d):.2f}")
    return ids, [w for w, _ in found]

mu = raw[seeded].mean(0)
ids_s, words_s = abc(SYNCAT, "syncategorematic", mu)
ids_c, words_c = abc(CONCRETE, "concrete control", mu)

def embeddings(model):
    f = hf_hub_download(model, "model.safetensors", revision="chck_100M")
    d = load_file(f)
    key = [k for k in d if "word_embeddings.weight" in k][0]
    return d[key][:75000]
E = embeddings("augustinian-babylm/deberta-base-75k-sam")
B_ctl = embeddings("augustinian-babylm/deberta-base-75k")

def rsa(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(len(X), k=1)
    return float(np.corrcoef((Xn @ Xn.T)[iu], (Yn @ Yn.T)[iu])[0, 1])

print("\n=== C at chck_100M ===")
for label, ids in [("syncategorematic", ids_s), ("concrete", ids_c)]:
    An = A[ids] / (np.linalg.norm(A[ids], axis=1, keepdims=True) + 1e-9)
    En = E[ids] / (np.linalg.norm(E[ids], axis=1, keepdims=True) + 1e-9)
    print(f"  {label:16s} n={len(ids):3d}  cos {float((An*En).sum(1).mean()):+.3f}  "
          f"RSA sam {rsa(A[ids].astype(np.float64), E[ids].astype(np.float64)):.3f} | "
          f"floor {rsa(A[ids].astype(np.float64), B_ctl[ids].astype(np.float64)):.3f}")
