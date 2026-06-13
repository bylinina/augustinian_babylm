#!/usr/bin/env python
"""
Dump COMPLETE lists of words/tokens with little/no visual support, sorted by
corpus frequency (most frequent under-supported first). Derived from existing
coverage CSVs -- no re-tokenization or tagging needed.

--max_image_count N : include units with image_count <= N (default 0 = strictly
                      zero support; the set Stage 2 leaves un-seeded). Use e.g.
                      4 for "weakly supported" (<5 image occurrences).

Outputs (analysis/out/), suffixed by threshold when N>0:
  zero_support_words[_leN].csv
  zero_support_tokens_{50k,75k,100k}[_leN].csv
"""
import argparse
from pathlib import Path
import pandas as pd

def dump(in_csv, out_csv, unit_col, max_ic):
    df = pd.read_csv(in_csv)
    z = df[(df.text_only_count > 0) & (df.image_count <= max_ic)].copy()
    z = z.sort_values("text_only_count", ascending=False)
    z.to_csv(out_csv, index=False)
    total = (df.text_only_count > 0).sum()
    print(f"{out_csv.name}: {len(z):,} under-supported {unit_col} "
          f"(image_count<={max_ic}; {len(z)/total:.1%} of {total:,}); "
          f"top: {z[unit_col].head(8).tolist()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/out"))
    ap.add_argument("--max_image_count", type=int, default=0)
    a = ap.parse_args()
    d, n = a.out_dir, a.max_image_count
    suf = "" if n == 0 else f"_le{n}"
    dump(d / "word_coverage.csv", d / f"zero_support_words{suf}.csv", "word", n)
    for v in ["50k", "75k", "100k"]:
        f = d / f"token_coverage_{v}.csv"
        if f.exists():
            dump(f, d / f"zero_support_tokens_{v}{suf}.csv", "token_str", n)
        else:
            print(f"skip {v}: {f} not found")

if __name__ == "__main__":
    main()
