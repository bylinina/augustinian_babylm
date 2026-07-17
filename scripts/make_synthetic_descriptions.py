#!/usr/bin/env python
"""
Generate synthetic-scene descriptions covering concrete zero-support words.

Reads analysis/out/zero_support_words.csv + analysis/concreteness.txt,
keeps words with Brysbaert Conc.M >= --concreteness (any POS, any freq),
and produces scene descriptions (Claude, cached, resumable) such that each
word appears in >= --min_mentions descriptions.

Output: --outdir/descriptions.jsonl
  {"id": "r0_c003", "offered": [...], "description": "...", "matched": [...]}
and coverage_report.json (per-word mention counts).

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts/make_synthetic_descriptions.py --outdir analysis/out/synth
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re, time
from collections import Counter
from pathlib import Path

PROMPT = (
    "Write ONE short, realistic visual scene description (1-3 sentences, "
    "at most 55 words -- it will be cut off beyond that) for an "
    "image-generation model. Use as many of these words as fit naturally, "
    "each exactly as written, each referring to a clearly visible physical "
    "thing or attribute in the scene: {words}. The scene must be coherent "
    "and photographable. Output ONLY the description, no preamble."
)

class Claude:
    def __init__(self, model, cache_path, max_retries=3):
        import anthropic
        self.c = anthropic.Anthropic()
        self.model, self.max_retries = model, max_retries
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.open():
                try:
                    r = json.loads(line); self.cache[r["k"]] = r["v"]
                except Exception: pass
        self.fh = cache_path.open("a")

    def complete(self, prompt, temperature=0.8, max_tokens=500):
        k = hashlib.sha1(f"{self.model}|{temperature}|{prompt}".encode()).hexdigest()
        if k in self.cache: return self.cache[k]
        last = None
        for i in range(self.max_retries):
            try:
                r = self.c.messages.create(model=self.model, max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text").strip()
                self.cache[k] = text
                self.fh.write(json.dumps({"k": k, "v": text}) + "\n"); self.fh.flush()
                return text
            except Exception as e:
                last = e; time.sleep(2 * (i + 1))
        raise RuntimeError(f"LLM failed: {last}")

def matched_words(text, words):
    low = text.lower()
    out = []
    for w in words:
        if re.search(rf"\b{re.escape(w)}(s|es)?\b", low):
            out.append(w)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero_support", default="analysis/out/zero_support_words.csv")
    ap.add_argument("--norms", default="analysis/concreteness.txt")
    ap.add_argument("--concreteness", type=float, default=4.0)
    ap.add_argument("--outdir", default="analysis/out/synth")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--min_mentions", type=int, default=3)
    ap.add_argument("--max_topup_rounds", type=int, default=6)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    conc = {}
    with open(args.norms) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try: conc[row["Word"].lower()] = float(row["Conc.M"])
            except (ValueError, KeyError): pass
    words = []
    with open(args.zero_support) as f:
        for row in csv.DictReader(f):
            w = row["word"].strip().lower()
            if w.isalpha() and len(w) >= 3 and conc.get(w, 0) >= args.concreteness:
                words.append(w)
    words = sorted(set(words))
    print(f"target words: {len(words)}")

    llm = Claude(args.model, outdir / "cache_descriptions.jsonl")
    rng = random.Random(args.seed)
    mentions = Counter()
    out_f = (outdir / "descriptions.jsonl").open("w")
    n_desc = 0

    def run_chunks(word_pool, tag):
        nonlocal n_desc
        pool = list(word_pool); rng.shuffle(pool)
        for ci in range(0, len(pool), args.chunk):
            offered = pool[ci:ci + args.chunk]
            if not offered: continue
            desc = llm.complete(PROMPT.format(words=", ".join(offered)))
            got = matched_words(desc, offered)
            for w in got: mentions[w] += 1
            rec = {"id": f"{tag}_c{ci // args.chunk:04d}",
                   "offered": offered, "description": desc, "matched": got}
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()
            n_desc += 1
            if n_desc % 25 == 0:
                cov = sum(1 for w in words if mentions[w] >= args.min_mentions)
                print(f"  {n_desc} descriptions | {cov}/{len(words)} words at "
                      f">={args.min_mentions} mentions", flush=True)

    for r in range(args.rounds):
        run_chunks(words, f"r{r}")
    for t in range(args.max_topup_rounds):
        needy = [w for w in words if mentions[w] < args.min_mentions]
        if not needy:
            break
        print(f"top-up round {t}: {len(needy)} words under target")
        run_chunks(needy, f"t{t}")

    report = {"n_words": len(words), "n_descriptions": n_desc,
              "covered": sum(1 for w in words if mentions[w] >= args.min_mentions),
              "uncovered": [w for w in words if mentions[w] == 0],
              "mentions": dict(mentions)}
    (outdir / "coverage_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{n_desc} descriptions | covered >= {args.min_mentions}x: "
          f"{report['covered']}/{len(words)} | never matched: "
          f"{len(report['uncovered'])}")
    print(f"wrote {outdir}/descriptions.jsonl + coverage_report.json")

if __name__ == "__main__":
    main()
