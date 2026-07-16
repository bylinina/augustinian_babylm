#!/usr/bin/env python
"""
build_vpswap.py
===============
Construct a VP-Swap (Visual Property Swap) benchmark from OUR training
corpus (bb24.train) and coverage inventory, in the exact pair-file format
of EgoBabyVLM's swapbench (LT-Swap "visualswap" convention), so the
resulting files drop straight into swapbench_score.py.

Faithful to apps/swapbench/visual_property_swap/generate.py (pipeline
stages, prompts, frequency bins, 4x A/B filtering with both presentation
orders) with three deliberate adaptations for our project:

  1. The stage-1 "is it a physical object?" LLM gate is replaced by our
     Brysbaert concreteness scores from the coverage analysis
     (--gate concreteness, default; --gate llm reproduces their protocol).
  2. Word frequencies/bins come from bb24.train (our actual pretraining
     corpus), not the official BabyLM distribution.
  3. Every word carries a seeded/unseeded tag (image_freq >= threshold in
     our grounding data), written to a .meta.jsonl sidecar per property --
     this enables the seeded x init difference-in-differences analysis.

LLM backends (for generation + filtering, and the optional llm gate):
  --backend anthropic   Anthropic API (ANTHROPIC_API_KEY env var)
  --backend openai      any OpenAI-compatible endpoint (vLLM etc.)

All LLM calls are cached on disk (cache_*.jsonl in --outdir); rerunning
resumes. --dry_run builds all deterministic stages and prints call-count
estimates without touching an LLM.

Outputs per property P in {color, material, relative_size, shape}:
  vp_swap_P_pairs.txt        bin|VISUAL|w1|s1|i1|w2|s2|i2   (scorer input)
  vp_swap_P_pairs.meta.jsonl one JSON per pair-file line: words, bin,
                             corpus freqs, seeded flags
  plus their intermediate stage files (byte-compatible) for debugging.

Example (pilot):
  python build_vpswap.py \
      --inventory analysis/coverage_inventory.csv \
      --outdir eval/vpswap_bb24 \
      --pairs_per_bin 40 --backend anthropic --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ----------------------------------------------------------------------
# Constants mirrored from egobabyvlm visual_property_swap/generate.py
# ----------------------------------------------------------------------
FREQ_BIN_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # bins 0..9
SUPPORTED_PROPERTIES = ("color", "material", "relative_size", "shape")
MIN_SPLIT_SENTENCE_LEN = 3

GENERATION_TEMPLATES = {
    "color": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and state "
        "the most typical real-world color of that word. Encapsulate both "
        "sentences together within brackets. Do not relate the two "
        "sentences together."
    ),
    "material": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and state "
        "the most typical real-world material that word is made of. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
    "relative_size": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and "
        "describe its typical real-world size relative to an adult human. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
    "shape": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and "
        "describe its canonical geometric shape in the real world. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
}
FILTER_TEMPLATE = (
    "Given the two sentences A and B: "
    "<start of sentence A> {s1} <end of sentence A> "
    "<start of sentence B> {s2} <end of sentence B> "
    "Which of the two sentences, A or B, is more physically accurate? "
    "Write your answer (A or B) in the brackets."
)
PHYSICAL_OBJECT_TEMPLATE = (
    "Is the word '{word}' representing something physical? "
    "Answer only by yes or no, in brackets."
)
INANIMACY_TEMPLATE = (
    "Is the word '{word}' an inanimate physical object -- not a person, "
    "profession, animal role, action, or event? "
    "Answer only by yes or no, in brackets."
)
PROPERTY_NOUN = {"color": "color", "material": "material",
                 "relative_size": "size", "shape": "shape"}
ATTRIBUTION_TEMPLATE = (
    "Consider the sentence: '{s}'. Does this sentence describe the typical "
    "{propnoun} of the {word} itself, rather than the {propnoun} of some "
    "other object mentioned in the sentence? "
    "Answer only by yes or no, in brackets."
)
#: Syntactic frames, rotated per pair (idx %% 4) for balanced variety.
FRAMES = [
    ("copular", "a predicate after 'is' or 'are', as in 'A femur is white'"),
    ("attributive", "an adjective directly before the noun, as in "
     "'She picked up the white femur'"),
    ("existential", "inside a 'There was/were ...' sentence, as in "
     "'There was a white femur on the table'"),
    ("relative", "inside a relative clause modifying the noun, as in "
     "'The femur, which was white, lay on the table'"),
]


def bin_for_freq(freq: int) -> int | None:
    """Their _bin_for_freq: index of largest edge <= freq (None if freq < 1)."""
    if freq < FREQ_BIN_EDGES[0]:
        return None
    return bisect.bisect_right(FREQ_BIN_EDGES, freq) - 1


# ----------------------------------------------------------------------
# Ported parsing helpers (logic identical to upstream, trimmed)
# ----------------------------------------------------------------------
def split_two_sentences(generation: str) -> tuple[str, str] | None:
    start, end = generation.rfind("["), generation.rfind("]")
    if start == -1 or end == -1:
        return None
    body = generation[start + 1 : end]
    body = body.replace("\\", "").replace('"', "").replace("'", "")
    body = " ".join(filter(None, body.split(" ")))
    for pattern in (".", "!", "?", "/", ", but", ", while", ", whereas",
                    ", and ", ",", ";"):
        if pattern in body[:-1]:
            idx = body.find(pattern)
            s1 = body[:idx].strip()
            s2 = body[idx + len(pattern) + 1 :].strip()
            if (len(s1) >= MIN_SPLIT_SENTENCE_LEN
                    and len(s2) >= MIN_SPLIT_SENTENCE_LEN):
                return s1, s2
    return None


def word_indices(s1: str, s2: str, w1: str, w2: str):
    i1, i2 = s1.find(w1), s2.find(w2)
    if i1 == -1 or i2 == -1:
        i1, i2 = s1.find(w2), s2.find(w1)
        if i1 == -1 or i2 == -1:
            return None
    return i1, i2


def parse_yes_no(response: str) -> bool:
    start, end = response.rfind("["), response.rfind("]")
    payload = response[start + 1 : end] if (start != -1 and end != -1) else response
    return payload.strip().lower().startswith("y")


def parse_ab(response: str) -> str | None:
    start, end = response.rfind("["), response.rfind("]")
    if start == -1 or end == -1:
        return response if response in ("A", "B") else None
    payload = response[start + 1 : end].replace(" ", "").upper()
    return payload if payload in ("A", "B") else None


# ----------------------------------------------------------------------
# LLM backends with on-disk cache
# ----------------------------------------------------------------------
class LLMBackend:
    """Cached, retrying completion caller. Cache key: sha1(model|temp|prompt)."""

    def __init__(self, kind: str, model: str, cache_path: Path,
                 base_url: str | None, api_key: str | None,
                 max_tokens: int, max_retries: int):
        self.kind, self.model = kind, model
        self.max_tokens, self.max_retries = max_tokens, max_retries
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            with cache_path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._cache_fh = cache_path.open("a")

        if kind == "anthropic":
            import anthropic  # pip install anthropic
            self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
        elif kind == "openai":
            import openai  # pip install openai
            self._client = openai.OpenAI(base_url=base_url,
                                         api_key=api_key or "dummy")
        else:
            raise ValueError(f"unknown backend {kind!r}")

    def _key(self, prompt: str, temperature: float, model: str) -> str:
        return hashlib.sha1(
            f"{model}|{temperature}|{prompt}".encode()).hexdigest()

    def complete(self, prompt: str, temperature: float,
                 model: str | None = None) -> str:
        mdl = model or self.model
        key = self._key(prompt, temperature, mdl)
        if key in self.cache:
            return self.cache[key]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.kind == "anthropic":
                    resp = self._client.messages.create(
                        model=mdl,
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(
                        b.text for b in resp.content if b.type == "text")
                else:
                    resp = self._client.chat.completions.create(
                        model=mdl,
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.choices[0].message.content or ""
                text = text.replace("\n", " ").replace("|", "/").strip()
                self.cache[key] = text
                self._cache_fh.write(
                    json.dumps({"k": key, "v": text}) + "\n")
                self._cache_fh.flush()
                return text
            except Exception as e:  # noqa: BLE001 -- retry then surface
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_err}")


def run_pool(backend: LLMBackend, jobs: list[tuple[str, str]],
             temperature: float, workers: int, label: str,
             model: str | None = None) -> dict[str, str]:
    """jobs: (job_id, prompt). Returns {job_id: response}. Cached calls are free."""
    out: dict[str, str] = {}
    todo = [(jid, p) for jid, p in jobs]
    done_n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(backend.complete, p, temperature, model): jid
                for jid, p in todo}
        for fut in as_completed(futs):
            jid = futs[fut]
            try:
                out[jid] = fut.result()
            except RuntimeError as e:
                out[jid] = f"<ERROR> {e}"
            done_n += 1
            if done_n % 50 == 0 or done_n == len(todo):
                print(f"  [{label}] {done_n}/{len(todo)}", flush=True)
    return out


# ----------------------------------------------------------------------
# Inventory loading
# ----------------------------------------------------------------------
def load_inventory(args) -> list[dict]:
    """Load the coverage inventory; returns [{word, freq, concreteness,
    pos, seeded}] rows. Column names are configurable because the
    inventory schema lives in analysis/ outputs."""
    import csv
    path = Path(args.inventory)
    delim = "\t" if path.suffix in (".tsv", ".txt") else ","
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = reader.fieldnames or []
        if args.peek:
            print(f"inventory columns: {cols}")
            for i, row in enumerate(reader):
                print(row)
                if i >= 4:
                    break
            sys.exit(0)
        needed = [args.word_col, args.freq_col]
        for c in needed:
            if c not in cols:
                sys.exit(f"column {c!r} not in inventory ({cols}); "
                         f"use --peek and the --*_col flags")
        for row in reader:
            word = (row[args.word_col] or "").strip()
            if args.lowercase:
                word = word.lower()
            if not word.isalpha() or len(word) < args.min_word_len:
                continue
            try:
                freq = int(float(row[args.freq_col]))
            except (ValueError, TypeError):
                continue
            conc = None
            if args.concreteness_col in cols:
                try:
                    conc = float(row[args.concreteness_col])
                except (ValueError, TypeError):
                    conc = None
            pos = (row.get(args.pos_col) or "").strip().upper() \
                if args.pos_col in cols else ""
            seeded = False
            if args.seeded_col in cols:
                try:
                    seeded = float(row[args.seeded_col]) >= args.seeded_threshold
                except (ValueError, TypeError):
                    seeded = False
            rows.append({"word": word, "freq": freq, "concreteness": conc,
                         "pos": pos, "seeded": seeded})
    return rows


def gate_words(rows: list[dict], args, backend: LLMBackend | None,
               outdir: Path) -> dict[str, dict]:
    """Stage 1: keep nouns representing physical objects.
    concreteness mode: Brysbaert score >= threshold (+ POS if present).
    llm mode: their physical-object prompt, temp 0."""
    # dedupe (first occurrence wins, like theirs)
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["word"], r)
    candidates = {w: r for w, r in seen.items()
                  if bin_for_freq(r["freq"]) is not None
                  and (not r["pos"] or r["pos"].startswith("NOUN")
                       or r["pos"] == "NN")}
    if args.gate == "concreteness":
        kept = {w: r for w, r in candidates.items()
                if r["concreteness"] is not None
                and r["concreteness"] >= args.concreteness_threshold}
    else:
        if backend is None:
            sys.exit("--gate llm requires a backend (not available in --dry_run)")
        prompts_f = outdir / "physical_object_prompts.txt"
        resp_f = outdir / "physical_object_responses.txt"
        with prompts_f.open("w") as f:
            for w, r in candidates.items():
                f.write(f"{w}|{bin_for_freq(r['freq'])}|"
                        f"{PHYSICAL_OBJECT_TEMPLATE.format(word=w)}\n")
        jobs = [(w, PHYSICAL_OBJECT_TEMPLATE.format(word=w))
                for w in candidates]
        responses = run_pool(backend, jobs, 0.0, args.workers, "gate", model=args.cheap_model)
        with resp_f.open("w") as f:  # their 4-col format
            for i, (w, r) in enumerate(candidates.items()):
                f.write(f"{i}|{w}|{bin_for_freq(r['freq'])}|"
                        f"{responses.get(w, '')}\n")
        kept = {w: r for w, r in candidates.items()
                if parse_yes_no(responses.get(w, ""))}
    print(f"gate ({args.gate}): kept {len(kept)} / {len(candidates)} "
          f"candidate nouns")
    if not args.loose_validity and backend is not None:
        jobs = [(w, INANIMACY_TEMPLATE.format(word=w)) for w in kept]
        responses = run_pool(backend, jobs, 0.0, args.workers, "inanimacy", model=args.cheap_model)
        kept = {w: r for w, r in kept.items()
                if parse_yes_no(responses.get(w, ""))}
        print(f"inanimacy gate: kept {len(kept)} inanimate-object nouns")
    return kept


def build_pairs(kept: dict[str, dict], args) -> dict[int, list[tuple[str, str]]]:
    """Stage 2: (w1, w2) pairs within frequency bins (their pairing logic)."""
    pairs_per_bin: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    words_sorted = sorted(kept)
    bins = {w: bin_for_freq(kept[w]["freq"]) for w in words_sorted}
    for i, w1 in enumerate(words_sorted):
        b1 = bins[w1]
        for w2 in words_sorted[i + 1:]:
            b2 = bins[w2]
            if abs(b2 - b1) > args.max_bin_distance:
                continue
            key = "|".join(sorted((w1, w2)))
            if key in seen:
                continue
            seen.add(key)
            pairs_per_bin[min(b1, b2)].append((w1, w2))
    rng = random.Random(args.seed)
    capped = {}
    for b in sorted(pairs_per_bin):
        ps = pairs_per_bin[b]
        rng.shuffle(ps)
        capped[b] = ps[: args.pairs_per_bin]
    return capped


# ----------------------------------------------------------------------
# Per-property stages 3-7
# ----------------------------------------------------------------------
def run_property(prop: str, pairs_by_bin, kept, args, backend,
                 outdir: Path) -> None:
    gen_prompts_f = outdir / f"vp_swap_{prop}_sentence_prompts.txt"
    gen_out_f = outdir / f"vp_swap_{prop}_sentence_generations.txt"
    filt_prompts_f = outdir / f"vp_swap_{prop}_pairs_filtering_prompts.txt"
    filt_out_f = outdir / f"vp_swap_{prop}_pairs_to_be_filtered.txt"
    final_f = outdir / f"vp_swap_{prop}_pairs.txt"
    meta_f = outdir / f"vp_swap_{prop}_pairs.meta.jsonl"

    # stage 3: generation prompts (their row format)
    n_gen = 0
    with gen_prompts_f.open("w") as f:
        for b in sorted(pairs_by_bin):
            for w1, w2 in pairs_by_bin[b]:
                fname, finstr = FRAMES[n_gen % len(FRAMES)]
                prompt = GENERATION_TEMPLATES[prop].format(w1=w1, w2=w2)
                prompt += (f" Write both sentences so that the "
                           f"{PROPERTY_NOUN[prop]} description appears as "
                           f"{finstr}.")
                f.write(f"{b}|{w1}|{w2}|{fname}|VISUAL|{prompt}\n")
                n_gen += 1
    print(f"[{prop}] stage 3: {n_gen} generation prompts")
    if args.dry_run:
        print(f"[{prop}] dry_run: would make {n_gen} generation calls "
              f"+ up to {4 * n_gen} filter calls")
        return

    # stage 4: LLM generation (temp 0.7, theirs)
    jobs, meta = [], {}
    with gen_prompts_f.open() as f:
        for i, raw in enumerate(f):
            b, w1, w2, frame, _rule, prompt = raw.rstrip("\n").split("|", 5)
            jid = str(i)
            jobs.append((jid, prompt))
            meta[jid] = (b, w1, w2, frame)
    responses = run_pool(backend, jobs, args.gen_temperature, args.workers,
                         f"{prop} gen")
    with gen_out_f.open("w") as f:  # their pool output shape (+frame in pos slot)
        for jid, (b, w1, w2, frame) in meta.items():
            f.write(f"{jid}|{b}|{w1}|{w2}|{frame}|VISUAL|"
                    f"{responses.get(jid, '')}\n")

    # stage 5: swap + 4 A/B filter prompts (both sentences, both orders)
    filter_rows = []  # (pair_idx, metadata, [(prompt, gt) x4])
    with gen_out_f.open() as f:
        for raw in f:
            parts = raw.rstrip("\n").split("|")
            if len(parts) < 7:
                continue
            _idx, b, w1, w2, frame, _rule = parts[:6]
            generation = "|".join(parts[6:])
            sents = split_two_sentences(generation)
            if sents is None:
                continue
            s1, s2 = sents
            idxs = word_indices(s1, s2, w1, w2)
            if idxs is None:
                continue
            i1, i2 = idxs
            ss1 = s1[:i1] + w2 + s1[i1 + len(w1):]
            ss2 = s2[:i2] + w1 + s2[i2 + len(w2):]
            if w1 in ss1 or w2 in ss2:
                continue
            metadata = f"{b}|VISUAL|{w1}|{s1}|{i1}|{w2}|{s2}|{i2}|{frame}"
            prompts_gt = [
                (FILTER_TEMPLATE.format(s1=s1, s2=ss1), "A"),
                (FILTER_TEMPLATE.format(s1=ss1, s2=s1), "B"),
                (FILTER_TEMPLATE.format(s1=s2, s2=ss2), "A"),
                (FILTER_TEMPLATE.format(s1=ss2, s2=s2), "B"),
            ]
            filter_rows.append((metadata, prompts_gt))
    with filt_prompts_f.open("w") as f:
        for metadata, pg in filter_rows:
            ps = "/".join(p for p, _ in pg)
            gts = "/".join(g for _, g in pg)
            f.write(f"{metadata}|{ps}/{gts}\n")
    print(f"[{prop}] stage 5: {len(filter_rows)} pairs -> "
          f"{4 * len(filter_rows)} filter prompts")

    # stage 5b: attribution check -- the property must be predicated of
    # the target word itself, not of some other object in the sentence.
    if not args.loose_validity:
        propnoun = PROPERTY_NOUN[prop]
        jobs = []
        for i, (metadata, _pg) in enumerate(filter_rows):
            _b, _r, w1, s1, _i1, w2, s2, _i2 = metadata.split("|")[:8]
            jobs.append((f"a{i}-1", ATTRIBUTION_TEMPLATE.format(
                s=s1, propnoun=propnoun, word=w1)))
            jobs.append((f"a{i}-2", ATTRIBUTION_TEMPLATE.format(
                s=s2, propnoun=propnoun, word=w2)))
        att = run_pool(backend, jobs, 0.0, args.workers, f"{prop} attrib", model=args.cheap_model)
        before = len(filter_rows)
        filter_rows = [
            row for i, row in enumerate(filter_rows)
            if parse_yes_no(att.get(f"a{i}-1", ""))
            and parse_yes_no(att.get(f"a{i}-2", ""))
        ]
        print(f"[{prop}] stage 5b: attribution kept "
              f"{len(filter_rows)} / {before} pairs")

    # stage 6: LLM filter (temp 0)
    jobs = []
    for i, (_metadata, pg) in enumerate(filter_rows):
        for j, (prompt, _gt) in enumerate(pg):
            jobs.append((f"{i}-{j}", prompt))
    responses = run_pool(backend, jobs, 0.0, args.workers, f"{prop} filter", model=args.cheap_model)
    with filt_out_f.open("w") as f:  # their {i}-{j}|metadata|gt|response rows
        for i, (metadata, pg) in enumerate(filter_rows):
            for j, (_prompt, gt) in enumerate(pg):
                f.write(f"{i}-{j}|{metadata}|{gt}|"
                        f"{responses.get(f'{i}-{j}', '')}\n")

    # stage 7: accept pairs with 4/4 correct verdicts; write final + meta
    accepted, seen_pairs = 0, set()
    with final_f.open("w") as dst, meta_f.open("w") as mdst:
        for i, (metadata, pg) in enumerate(filter_rows):
            ok = all(
                parse_ab(responses.get(f"{i}-{j}", "")) == gt
                for j, (_p, gt) in enumerate(pg)
            )
            if not ok:
                continue
            parts = metadata.split("|")
            b, _rule, w1, s1, i1, w2, s2, i2 = parts[:8]
            if w1 == w2:
                continue
            key = "-".join(sorted((w1, w2)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            dst.write(f"{b}|VISUAL|{w1}|{s1}|{i1}|{w2}|{s2}|{i2}\n")
            mdst.write(json.dumps({
                "property": prop, "bin": int(b),
                "frame": parts[8] if len(parts) > 8 else "",
                "w1": w1, "freq_w1": kept[w1]["freq"],
                "seeded_w1": kept[w1]["seeded"],
                "w2": w2, "freq_w2": kept[w2]["freq"],
                "seeded_w2": kept[w2]["seeded"],
            }) + "\n")
            accepted += 1
    print(f"[{prop}] stage 7: accepted {accepted} / {len(filter_rows)} pairs "
          f"-> {final_f.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True,
                    help="coverage-analysis word inventory (csv/tsv)")
    ap.add_argument("--word_col", default="word")
    ap.add_argument("--freq_col", default="corpus_freq",
                    help="RAW COUNT in bb24.train (not per-million)")
    ap.add_argument("--concreteness_col", default="concreteness")
    ap.add_argument("--pos_col", default="pos")
    ap.add_argument("--seeded_col", default="image_freq")
    ap.add_argument("--seeded_threshold", type=float, default=1.0)
    ap.add_argument("--peek", action="store_true",
                    help="print inventory columns + first rows, exit")
    ap.add_argument("--outdir", default="vpswap_out")
    ap.add_argument("--properties", nargs="+", default=list(SUPPORTED_PROPERTIES),
                    choices=list(SUPPORTED_PROPERTIES))
    ap.add_argument("--gate", choices=["concreteness", "llm"],
                    default="concreteness")
    ap.add_argument("--concreteness_threshold", type=float, default=4.0)
    ap.add_argument("--min_word_len", type=int, default=3)
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--pairs_per_bin", type=int, default=40,
                    help="theirs uses 2000; start small, scale after pilot")
    ap.add_argument("--max_bin_distance", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["anthropic", "openai"],
                    default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--cheap_model", default="claude-haiku-4-5",
                    help="model for gates/attribution/judging (generation "
                         "uses --model)")
    ap.add_argument("--base_url", default=None,
                    help="openai backend: e.g. http://localhost:8000/v1")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--loose_validity", action="store_true",
                    help="reproduce their exact protocol (no inanimacy or "
                         "attribution gates)")
    ap.add_argument("--dry_run", action="store_true",
                    help="deterministic stages only; print call estimates")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_inventory(args)
    print(f"inventory: {len(rows)} usable rows")

    backend = None
    if not args.dry_run:
        backend = LLMBackend(args.backend, args.model,
                             outdir / "cache_llm.jsonl",
                             args.base_url, args.api_key,
                             args.max_tokens, args.max_retries)

    kept = gate_words(rows, args, backend, outdir)
    n_seeded = sum(1 for r in kept.values() if r["seeded"])
    print(f"seeded/unseeded among kept: {n_seeded} / {len(kept) - n_seeded}")

    pairs_by_bin = build_pairs(kept, args)
    for b in sorted(pairs_by_bin):
        print(f"  bin {b} (freq >= {FREQ_BIN_EDGES[b]}): "
              f"{len(pairs_by_bin[b])} pairs")

    for prop in args.properties:
        run_property(prop, pairs_by_bin, kept, args, backend, outdir)

    if not args.dry_run:
        print("\nDone. Pair files + .meta.jsonl sidecars in", outdir)
        print("Feed vp_swap_<prop>_pairs.txt to swapbench_score.py; join "
              "meta by line number for the seeded-split analysis.")


if __name__ == "__main__":
    main()
