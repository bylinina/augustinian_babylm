#!/usr/bin/env python
"""Reconstruct the synthetically-grounded word list from published coverage.

The original list lived in region_synth/region_embeddings.parquet, which was
purged from scratch. But sam/75k and sam_ext/75k coverage tables are both on
the Hub, so the 737 newly-seeded token ids are recoverable; a word counts as
synthetically grounded if its tokens are among them.

Writes eval/synth_words.txt (one word per line).
Expected: ~1,151 words, giving VP-Swap groups 6,242 real / 833 synthetic /
341 unseeded (Appendix D).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from vpswap_score import load_tok

REPO = "augustinian-babylm/token-embeddings"
PROPS = ["color", "material", "relative_size", "shape"]


def coverage(tag):
    p = hf_hub_download(REPO, f"{tag}/coverage.parquet", repo_type="dataset")
    return pd.read_parquet(p)


orig = coverage("sam/75k")
ext = coverage("sam_ext/75k")
seeded_orig = set(orig.loc[orig.seeded, "token_id"])
seeded_ext = set(ext.loc[ext.seeded, "token_id"])
new_tokens = seeded_ext - seeded_orig
print(f"seeded tokens: orig {len(seeded_orig):,}  ext {len(seeded_ext):,}  "
      f"new {len(new_tokens):,}   (expected +737)")

tok = load_tok("augustinian-babylm/babylm-bpe-75k", "main")

# candidate words: every VP-Swap noun that was NOT seeded originally
cands = set()
meta_rows = {}
for prop in PROPS:
    f = Path("eval/vpswap_bb24") / f"vp_swap_{prop}_pairs.meta.jsonl"
    for i, line in enumerate(l for l in f.open() if l.strip()):
        m = json.loads(line)
        meta_rows[(prop, i)] = m
        for w, sd in ((m["w1"], m["seeded_w1"]), (m["w2"], m["seeded_w2"])):
            if not sd:
                cands.add(w.lower())
print(f"unseeded VP-Swap nouns to classify: {len(cands):,}")


def ids(word, lead_space):
    text = (" " + word) if lead_space else word
    return [i for i in tok.encode(text, add_special_tokens=False)]


for lead_space in (True, False):
    hits = set()
    for w in cands:
        t = ids(w, lead_space)
        if t and all(i in seeded_ext for i in t) and any(i in new_tokens for i in t):
            hits.add(w)
    # count resulting VP-Swap groups
    groups = {"real": 0, "synthetic": 0, "unseeded": 0}
    for prop in PROPS:
        f = Path("eval/vpswap_bb24") / f"vp_swap_{prop}_pairs.meta.jsonl"
        for i, line in enumerate(l for l in f.open() if l.strip()):
            m = json.loads(line)
            for which in (1, 2):
                w = (m["w1"] if which == 1 else m["w2"]).lower()
                sd = m["seeded_w1"] if which == 1 else m["seeded_w2"]
                if sd:
                    groups["real"] += 1
                elif w in hits:
                    groups["synthetic"] += 1
                else:
                    groups["unseeded"] += 1
    print(f"\nlead_space={lead_space}:  {len(hits):,} words matched")
    print(f"   groups -> real {groups['real']:,}  synthetic {groups['synthetic']:,}  "
          f"unseeded {groups['unseeded']:,}")
    print(f"   target -> real 6,242  synthetic 833  unseeded 341")
    if groups["synthetic"] == 833 and groups["unseeded"] == 341:
        out = Path("eval/synth_words.txt")
        out.write_text("\n".join(sorted(hits)) + "\n")
        print(f"   MATCH — wrote {out}")
