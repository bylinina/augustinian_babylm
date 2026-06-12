#!/usr/bin/env python
"""Enumerate (repo, stepN-branch) pairs for the eval sweep.
Excludes `main` and `best` (best duplicates some stepN)."""
import re
from huggingface_hub import HfApi

VOCABS = ["50k", "75k", "100k"]
api = HfApi()
for v in VOCABS:
    repo = f"augustinian-babylm/deberta-base-{v}"
    refs = api.list_repo_refs(repo, repo_type="model")
    steps = sorted(
        int(m.group(1)) for b in refs.branches
        if (m := re.fullmatch(r"step(\d+)", b.name))
    )
    for s in steps:
        print(f"{repo} step{s}")
