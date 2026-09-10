#!/usr/bin/env python3
"""Publish the two large artifact sets to the Hub.

  vpswap-checkpoint-scores   per-item VP-Swap correctness, 9 models x 20 revs
  synthetic-grounding-images the 3,162 SDXL images + manifest

Run from the repo root, with a write token:
    huggingface-cli login          # or: export HF_TOKEN=...
    python scripts/upload_datasets.py --scores
    python scripts/upload_datasets.py --images
    python scripts/upload_datasets.py --scores --images   # both
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ORG = "augustinian-babylm"
PAPER = "https://openreview.net/forum?id=B4TD4XdlwF"
GITHUB = "https://github.com/bylinina/augustinian_babylm"

SCORES_CARD = f"""---
license: cc-by-4.0
tags:
- babylm
- visual-grounding
- training-dynamics
- minimal-pairs
---

# VP-Swap checkpoint scores

Per-item correctness on the VP-Swap benchmark for nine models at twenty
points in training. This is the raw material behind Figures 4-6 of
*Augustinian BabyLM* ([paper]({PAPER}), [code]({GITHUB})).

## Layout

`<model>/<revision>.jsonl`, one line per benchmark item:

```json
{{"property": "color", "line": 6, "which": 1,
 "pll_orig": -21.4213, "pll_swap": -23.9077, "correct": true}}
```

`property` and `line` identify the source line in
`eval/vpswap_bb24/vp_swap_<property>_pairs.txt`; `which` is 1 or 2, since
each line yields two items (each sentence is scored against the other's
noun). Join to `vp_swap_<property>_pairs.meta.jsonl` by line number for
corpus frequency, seeded status, and syntactic frame.

`pll_orig` and `pll_swap` are MLM pseudo-log-likelihoods: mask each token
in turn, sum the log-probability of the original. An item is `correct`
when the compatible sentence scores higher.

## Models and revisions

Nine models: `deberta-base-75k` and its `-s2` / `-s3` seed replicates,
`deberta-base-75k-sam` and its replicates, and the three synthetic-extension
models `deberta-base-75k-sam_ext-s1/s2/s3`.

Twenty revisions each: `step0`, `chck_1M` through `chck_10M`, then
`chck_20M` through `chck_100M`. Revision names match the branches on the
corresponding model repos, so scores and weights can be lined up directly.

## Why this might be useful

Minimal-pair benchmarks are usually reported as a single accuracy at the end
of training. Here every item is scored at twenty checkpoints across three
random seeds, with per-item metadata, so you can ask when an effect appears,
which words carry it, whether it is stable across seeds, and how it relates
to corpus frequency — without retraining anything.

`step0` scores sit at chance (0.49-0.51) in every model, which is the
unbiasedness check for the probe.

## License

CC BY 4.0. Please cite the paper above.
"""

IMAGES_CARD = f"""---
license: cc-by-4.0
tags:
- babylm
- visual-grounding
- synthetic-data
---

# Synthetic grounding images

3,162 images generated to extend visual grounding to concrete words that no
photograph dataset covers, for *Augustinian BabyLM*
([paper]({PAPER}), [code]({GITHUB})).

## How they were made

Starting from 1,986 concrete words with no image support, an LLM
(`claude-sonnet-4-6`, temperature 0.8) wrote short scene descriptions
placing as many target words as fit naturally into one scene. Each of the
1,054 resulting descriptions was rendered three times with SDXL-Turbo
(2 inference steps, guidance scale 0, generator seeds 1000/1001/1002).
Targets were then localized with OWLv2 at score threshold 0.25, and SAM
features pooled inside the detected boxes; words the detector never found
were dropped.

The scene descriptions and the LLM response cache are committed in the
GitHub repository, so the whole set regenerates deterministically without
API calls.

## Layout

- `images/<id>_v<seed>.png` — three variants per description
- `index.jsonl` — one line per image: `file`, `desc_id`, and `matched`,
  the target words the description was written to contain

## Caveats

Generated images and automatic detection are noisier than human region
annotations, and the per-word benefit of synthetic grounding is roughly half
that of real grounding. Detection near the score threshold is not perfectly
stable across runs, so the set of successfully grounded words can vary by a
few items between regenerations.

The generator declined to place a small number of anatomical and taboo words
in scenes, so the coverage gap is not random: it is systematically the
vocabulary a safety-tuned model avoids.

## License

CC BY 4.0. Please cite the paper above.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--private", action="store_true",
                    help="create private, flip to public later")
    args = ap.parse_args()
    if not (args.scores or args.images):
        ap.error("pass --scores and/or --images")

    api = HfApi()
    home = Path.home()

    if args.scores:
        rid = f"{ORG}/vpswap-checkpoint-scores"
        src = Path("results/vpswap")
        n = len(list(src.glob("*/*.jsonl")))
        print(f"{rid}: {n} score files")
        api.create_repo(rid, repo_type="dataset", exist_ok=True,
                        private=args.private)
        api.upload_folder(
            folder_path=str(src), repo_id=rid, repo_type="dataset",
            ignore_patterns=["*.prearticle", "*.tmp"],
            commit_message="Per-item VP-Swap scores across checkpoints")
        api.upload_file(
            path_or_fileobj=SCORES_CARD.encode(), path_in_repo="README.md",
            repo_id=rid, repo_type="dataset", commit_message="Add card")
        print(f"   -> https://huggingface.co/datasets/{rid}")

    if args.images:
        rid = f"{ORG}/synthetic-grounding-images"
        src = home / "synth_images"
        n = len(list(src.glob("*.png")))
        print(f"{rid}: {n} images")
        api.create_repo(rid, repo_type="dataset", exist_ok=True,
                        private=args.private)
        api.upload_folder(
            folder_path=str(src), repo_id=rid, repo_type="dataset",
            path_in_repo="images", allow_patterns=["*.png"],
            commit_message="SDXL-Turbo synthetic grounding images")
        api.upload_file(
            path_or_fileobj=str(src / "index.jsonl"), path_in_repo="index.jsonl",
            repo_id=rid, repo_type="dataset", commit_message="Add manifest")
        api.upload_file(
            path_or_fileobj=IMAGES_CARD.encode(), path_in_repo="README.md",
            repo_id=rid, repo_type="dataset", commit_message="Add card")
        print(f"   -> https://huggingface.co/datasets/{rid}")


if __name__ == "__main__":
    main()
