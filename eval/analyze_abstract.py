#!/usr/bin/env python
"""Abstract-word grounding: consolidated figures, tables, word lists.
Writes eval/plots/abstract_{retention,gradient,mlm}.png,
eval/abstract_results.md, eval/abstract_words/*.csv.
Requires eval/mlm_class_results.json (from mlm_loss_by_class.py)."""
import csv, json, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup, style_ax, INK, BLUE, GRAY, RED, GREEN
setup()

conc = {}
with open("analysis/concreteness.txt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try: conc[row["Word"].lower()] = float(row["Conc.M"])
        except (ValueError, KeyError): pass
freq = {}
with open("analysis/out/word_coverage.csv") as f:
    for row in csv.DictReader(f):
        freq[row["word"].lower()] = int(row["text_only_count"])

from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("augustinian-babylm/babylm-bpe-75k")
init = load_file(hf_hub_download("augustinian-babylm/token-embeddings",
        "sam/75k/E_init.safetensors", repo_type="dataset"))
A, seeded = init["embeddings"], init["seeded_mask"].astype(bool)

def emb(model):
    f = hf_hub_download(f"augustinian-babylm/{model}", "model.safetensors",
                        revision="chck_100M")
    d = load_file(f)
    key = [k for k in d if "word_embeddings.weight" in k][0]
    return d[key][:75000]
E, B = emb("deberta-base-75k-sam"), emb("deberta-base-75k")

def rsa(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(len(X), k=1)
    return float(np.corrcoef((Xn @ Xn.T)[iu], (Yn @ Yn.T)[iu])[0, 1])

def cosm(ids):
    An = A[ids] / (np.linalg.norm(A[ids], axis=1, keepdims=True) + 1e-9)
    En = E[ids] / (np.linalg.norm(E[ids], axis=1, keepdims=True) + 1e-9)
    return float((An * En).sum(1).mean())

# ---- groups ----
tok_word = {}
for tid in np.where(seeded)[0]:
    w = tok.convert_ids_to_tokens(int(tid)).lstrip("\u0120").lower()
    if w.isalpha() and len(w) >= 2:
        tok_word[int(tid)] = w
BANDS = [("1-2\n(abstract)", 1.0, 2.0), ("2-3", 2.0, 3.0),
         ("3-4", 3.0, 4.0), ("4-5\n(concrete)", 4.0, 5.01)]
band_ids = {n: [t for t, w in tok_word.items()
                if w in conc and lo <= conc[w] < hi]
            for n, lo, hi in BANDS}
SYNCAT = """not no never nothing nobody none neither nor every each all some any
few many most several both either or and but if unless because whether than as
only even also too very quite must might may could should would will can one
two three four five six seven eight nine ten first second third the a an this
that these those there it""".split()
def whole(w):
    for form in ("\u0120" + w, w):
        t = tok.convert_tokens_to_ids(form)
        if t is not None and t != getattr(tok, "unk_token_id", None) and t and t < 75000:
            return t
CONCRETE = """dog cat table chair apple banana car tree house ball cup door book
shoe water bread bird fish horse bed spoon window bottle hat box stone flower
truck boat chicken""".split()
sync_pairs = [(w, whole(w)) for w in SYNCAT]
sync_pairs = [(w, t) for w, t in sync_pairs if t is not None and seeded[t]]
conc_pairs = [(w, whole(w)) for w in CONCRETE]
conc_pairs = [(w, t) for w, t in conc_pairs if t is not None and seeded[t]]
sync_ids = np.array([t for _, t in sync_pairs])
conc_ids = np.array([t for _, t in conc_pairs])

md = ["# Abstract-word grounding: results", ""]

# ---- Figure 1: retention by band + syncat/concrete ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))
names = [n for n, _, _ in BANDS]
r_sam = [rsa(A[band_ids[n]].astype(np.float64), E[band_ids[n]].astype(np.float64))
         for n in names]
r_flo = [rsa(A[band_ids[n]].astype(np.float64), B[band_ids[n]].astype(np.float64))
         for n in names]
x = np.arange(len(names))
ax1.bar(x - 0.19, r_sam, 0.36, color=BLUE, label="vision-init model")
ax1.bar(x + 0.19, r_flo, 0.36, color=GRAY, label="baseline (floor)")
ax1.set_xticks(x, names); ax1.set_ylabel("RSA to visual anchor @100M")
ax1.set_title("Retention by concreteness band", loc="left")
ax1.legend(); style_ax(ax1)
groups = [("function\nwords", sync_ids), ("concrete\ncontrols", conc_ids)]
vals_r = [rsa(A[i].astype(np.float64), E[i].astype(np.float64)) for _, i in groups]
vals_f = [rsa(A[i].astype(np.float64), B[i].astype(np.float64)) for _, i in groups]
x2 = np.arange(2)
ax2.bar(x2 - 0.19, vals_r, 0.36, color=BLUE)
ax2.bar(x2 + 0.19, vals_f, 0.36, color=GRAY)
ax2.set_xticks(x2, [g for g, _ in groups])
ax2.set_title("Curated word sets", loc="left"); style_ax(ax2)
fig.tight_layout(); fig.savefig("eval/plots/abstract_retention.png"); plt.close(fig)
md += ["## Retention (RSA to visual anchor at 100M words)", "",
       "| group | n | vision-init | baseline floor |", "|--|--:|--:|--:|"]
for n in names:
    md.append(f"| band {n.replace(chr(10),' ')} | {len(band_ids[n])} | "
              f"{r_sam[names.index(n)]:.3f} | {r_flo[names.index(n)]:.3f} |")
md.append(f"| function words | {len(sync_ids)} | {vals_r[0]:.3f} | {vals_f[0]:.3f} |")
md.append(f"| concrete controls | {len(conc_ids)} | {vals_r[1]:.3f} | {vals_f[1]:.3f} |")

# ---- Figure 2: BLiMP concreteness gradient ----
base = Path.home() / "babylm-eval/strict/results"
PAIRS = [("deberta-base-75k-sam", "deberta-base-75k"),
         ("deberta-base-75k-sam-s2", "deberta-base-75k-s2"),
         ("deberta-base-75k-sam-s3", "deberta-base-75k-s3")]
def subtasks(model):
    rep = list((base / model / "main/zero_shot/mlm").glob(
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
deltas = {k: sum(d)/3 for k, d in agg.items() if len(d) == 3}
ed = Path.home() / "babylm-eval/strict/evaluation_data"
cands = [d for d in ed.rglob("*") if d.is_dir() and "blimp" in d.name.lower()
         and "suppl" not in d.name.lower() and list(d.glob("*.jsonl"))]
pts = []
for f in sorted(cands[0].glob("*.jsonl")):
    if f.stem not in deltas: continue
    cs = []
    for line in f.open():
        d = json.loads(line)
        g = (d.get("sentence_good") or "").lower()
        bd = (d.get("sentence_bad") or "").lower()
        for w in set(re.findall(r"[a-z']+", g)) ^ set(re.findall(r"[a-z']+", bd)):
            if w in conc: cs.append(conc[w])
    if cs: pts.append((float(np.mean(cs)), deltas[f.stem], f.stem))
cx = np.array([p[0] for p in pts]); cy = np.array([p[1] for p in pts])
r = float(np.corrcoef(cx, cy)[0, 1])
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.scatter(cx, cy, s=22, color=BLUE, alpha=0.75)
m_, b_ = np.polyfit(cx, cy, 1)
xs = np.linspace(cx.min(), cx.max(), 10)
ax.plot(xs, m_ * xs + b_, color=RED, lw=1.4)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("mean concreteness of the words that vary in the phenomenon")
ax.set_ylabel("vision \u2212 baseline (pts, 3-seed mean)")
ax.set_title(f"BLiMP phenomena: effect follows concreteness (r = {r:+.2f})",
             loc="left")
style_ax(ax)
fig.tight_layout(); fig.savefig("eval/plots/abstract_gradient.png"); plt.close(fig)
md += ["", f"## BLiMP concreteness gradient: r = {r:+.2f} over {len(pts)} "
       "phenomena (fast subset)", ""]

# ---- Figure 3: MLM loss by class ----
mlm = json.load(open("eval/mlm_class_results.json"))
order = [c for c in ["function", "abstract(<2.5)", "mid(2.5-4)",
                     "concrete(>=4)", "seeded-other", "unseeded"] if c in mlm]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
y = np.arange(len(order))[::-1]
means = [np.mean(mlm[c]["deltas"]) for c in order]
cols = [BLUE if m < 0 else RED for m in means]
ax.barh(y, means, 0.55, color=cols)
for yi, c in zip(y, order):
    ax.scatter(mlm[c]["deltas"], [yi] * 3, color=INK, s=14, zorder=3)
ax.set_yticks(y, [f"{c}  (n={mlm[c]['n']})" for c in order])
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("Δ masked-token cross-entropy (vision \u2212 baseline; "
              "negative = vision better)")
ax.set_title("Held-out mask prediction by token class (bars = 3-seed mean, "
             "dots = seeds)", loc="left", fontsize=10.5)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("eval/plots/abstract_mlm.png"); plt.close(fig)
md += ["## Held-out MLM loss by token class", "",
       "| class | n | s1 | s2 | s3 | mean |", "|--|--:|--:|--:|--:|--:|"]
for c in order:
    ds = mlm[c]["deltas"]
    md.append(f"| {c} | {mlm[c]['n']} | " + " | ".join(f"{d:+.4f}" for d in ds)
              + f" | {np.mean(ds):+.4f} |")

# ---- word lists ----
outw = Path("eval/abstract_words"); outw.mkdir(exist_ok=True)
with (outw / "function_words_seeded.csv").open("w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["word", "token_id", "corpus_freq",
                                     "cos_to_anchor_100M"])
    for w, t in sorted(sync_pairs):
        An = A[t] / (np.linalg.norm(A[t]) + 1e-9)
        En = E[t] / (np.linalg.norm(E[t]) + 1e-9)
        w_.writerow([w, t, freq.get(w, ""), round(float(An @ En), 3)])
with (outw / "abstract_band_seeded.csv").open("w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["word", "concreteness", "corpus_freq"])
    rows = sorted(((tok_word[t], conc[tok_word[t]], freq.get(tok_word[t], 0))
                   for t in band_ids[names[0]]), key=lambda x: -x[2])
    w_.writerows(rows)
md += ["", "Word lists: `eval/abstract_words/function_words_seeded.csv`, "
       "`eval/abstract_words/abstract_band_seeded.csv`.",
       "", "Conclusions: see `docs/grounding_scope.md`."]
Path("eval/abstract_results.md").write_text("\n".join(md) + "\n")
print("wrote eval/abstract_results.md + 3 figures + 2 word lists")
