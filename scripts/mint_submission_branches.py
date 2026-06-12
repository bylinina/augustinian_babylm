#!/usr/bin/env python
"""
mint_submission_branches.py
===========================
Create BabyLM submission branches (chck_1M, chck_2M, ..., per the word-based
spec) on an existing model repo, as git REFS pointing at the nearest existing
Pythia-style stepN branch. No checkpoint data is duplicated: a branch ref to
an existing commit costs nothing.

Run AFTER training is complete and all stepN branches are pushed.

Usage:
  python scripts/mint_submission_branches.py \
      --repo augustinian-babylm/deberta-base-50k \
      --corpus_words 9876543 \
      --steps_per_epoch 2574 \
      --epochs 10 \
      [--dry_run]

Get corpus_words:    wc -w < path/to/train.txt   (the bb24.train file)
Get steps_per_epoch: from the training log ("Training: N train ... | S steps")
                     steps_per_epoch = total_steps / epochs (= 25740/10 = 2574)

Writes checkpoint_map.json (branch -> step -> words, with mapping error)
locally and uploads it to the repo's main branch for provenance.
"""
import argparse
import json
import re

from huggingface_hub import HfApi


def build_targets(total_words):
    """BabyLM spec: every 1M for first 10M, every 10M to 100M, every 100M to 1B.
    Targets beyond what was trained are skipped (spec allows partial)."""
    targets = (
        [k * 1_000_000 for k in range(1, 10)]
        + [k * 10_000_000 for k in range(1, 11)]
        + [k * 100_000_000 for k in range(2, 11)]
    )
    return [w for w in targets if w <= total_words]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus_words", type=int, required=True,
                    help="word count of ONE pass over the training corpus")
    ap.add_argument("--steps_per_epoch", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the mapping, create nothing")
    args = ap.parse_args()

    api = HfApi()
    refs = api.list_repo_refs(args.repo, repo_type="model")
    existing = {b.name for b in refs.branches}
    steps = sorted(
        int(m.group(1)) for b in refs.branches
        if (m := re.fullmatch(r"step(\d+)", b.name))
    )
    if not steps:
        raise SystemExit(f"No stepN branches found on {args.repo}; "
                         "run this after training has pushed checkpoints.")

    words_per_step = args.corpus_words / args.steps_per_epoch
    total_words = args.corpus_words * args.epochs
    print(f"{args.repo}: {len(steps)} step branches (max step{steps[-1]}) | "
          f"{words_per_step:,.0f} words/step | total {total_words:,} words")

    mapping = {}
    for w in build_targets(total_words):
        ideal = w / words_per_step
        nearest = min(steps, key=lambda s: abs(s - ideal))
        branch = f"chck_{w // 1_000_000}M"
        words_at = nearest * words_per_step
        mapping[branch] = {
            "target_words": w,
            "ideal_step": round(ideal),
            "mapped_step": nearest,
            "mapped_branch": f"step{nearest}",
            "words_at_mapped_step": round(words_at),
            "relative_word_error": round(abs(words_at - w) / w, 4),
        }

    for branch, m in mapping.items():
        flag = " (!)" if m["relative_word_error"] > 0.10 else ""
        print(f"  {branch:<10} -> step{m['mapped_step']:<7} "
              f"(target {m['target_words']/1e6:.0f}M words, actual "
              f"{m['words_at_mapped_step']/1e6:.2f}M, "
              f"err {m['relative_word_error']:.1%}){flag}")
        if args.dry_run:
            continue
        if branch in existing:
            print(f"    exists, skipping (delete it first to remint)")
            continue
        api.create_branch(repo_id=args.repo, repo_type="model",
                          branch=branch, revision=m["mapped_branch"])
        print(f"    created {branch} -> {m['mapped_branch']}")

    out = f"checkpoint_map_{args.repo.split('/')[-1]}.json"
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"wrote {out}")
    if not args.dry_run:
        api.upload_file(path_or_fileobj=out, path_in_repo="checkpoint_map.json",
                        repo_id=args.repo, repo_type="model",
                        commit_message="submission branch mapping (chck_NM -> stepN)")
        print("uploaded checkpoint_map.json to main")


if __name__ == "__main__":
    main()
