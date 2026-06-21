#!/usr/bin/env python
"""Enumerate (repo, stepN-branch) pairs for the eval sweep.
Excludes `main` and `best` (best duplicates some stepN)."""
import re
from huggingface_hub import HfApi

# baseline repos (all vocabs) + vision-init variants (50k, per encoder)
REPOS = [f"augustinian-babylm/deberta-base-{v}" for v in ["50k", "75k", "100k"]]
REPOS += [f"augustinian-babylm/deberta-base-{v}-{e}" for v in ["50k", "75k", "100k"] for e in ["dinov3", "sam", "ibot"]]
api = HfApi()
for repo in REPOS:
    refs = api.list_repo_refs(repo, repo_type="model")
    steps = sorted(
        int(m.group(1)) for b in refs.branches
        if (m := re.fullmatch(r"step(\d+)", b.name))
    )
    for s in steps:
        print(f"{repo} step{s}")
