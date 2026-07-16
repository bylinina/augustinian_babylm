#!/usr/bin/env python
"""
Vision-init vs baseline deltas on the official BabyLM 2026 evaluation.

Reads the babylm-eval results tree (best revision, full zero-shot + GLUE)
for all 9 encoder x vocab combinations vs their same-vocab baselines.
Writes eval/plots/official_deltas.png and eval/official_results.md.

Usage: python eval/analyze_official.py \
         [--results_dir ~/babylm-eval/strict/results]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENCODERS = ("sam", "dinov3", "ibot")
VOCABS = ("50k", "75k", "100k")
ZS_TASKS = {"blimp": "blimp/blimp_filtered",
            "supplement": "blimp/supplement_filtered",
            "ewok": "ewok/ewok_filtered",
            "entity_tracking": "entity_tracking/entity_tracking",
            "comps": "comps/comps"}
FT = ("boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc")
FT_METRIC = {"mrpc": "f1", "qqp": "f1"}


def report_avg(rep: Path) -> float | None:
    val, in_avg = None, False
    for line in rep.read_text().splitlines():
        s = line.strip()
        if s.startswith("### "):
            in_avg = "AVERAGE" in s
            continue
        if in_avg and re.fullmatch(r"[-\d.]+", s):
            val = float(s)
    return val


def report_subtasks(rep: Path) -> dict[str, float]:
    out, section = {}, None
    for line in rep.read_text().splitlines():
        s = line.strip()
        if s.startswith("### "):
            section = s
            continue
        m = re.fullmatch(r"([\w\-./ ]+):\s*([\d.]+)", s)
        if m and "AVERAGE" not in (section or "") \
                and m.group(1) != "TEMPERATURE":
            out[m.group(1)] = float(m.group(2))
    return out


def zs_score(results: Path, model: str, glob: str, sub: bool):
    reps = list((results / model / "best/zero_shot/mlm").glob(
        glob + "/best_temperature_report.txt"))
    if not reps:
        return None
    return report_subtasks(reps[0]) if sub else report_avg(reps[0])


def glue_score(results: Path, model: str) -> float | None:
    scores = []
    for t in FT:
        f = results / model / "best/finetune" / t / "results.txt"
        if not f.exists():
            return None
        d = dict(re.findall(r"(\w+):\s*([\d.]+)", f.read_text()))
        scores.append(float(d.get(FT_METRIC.get(t, "accuracy"), 0)) * 100)
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path,
                    default=Path.home() / "babylm-eval/strict/results")
    ap.add_argument("--plots_dir", type=Path, default=Path("eval/plots"))
    ap.add_argument("--out_md", type=Path,
                    default=Path("eval/official_results.md"))
    args = ap.parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    R = args.results_dir

    md = ["# Vision-init vs baseline: official BabyLM 2026 evaluation", "",
          "Deltas = encoder model minus same-vocab baseline, `best` revision.",
          ""]

    # ---- per-task deltas over all 9 combos ----
    task_deltas: dict[str, list[float]] = defaultdict(list)
    for v in VOCABS:
        base_zs = {t: zs_score(R, f"deberta-base-{v}", g, sub=False)
                   for t, g in ZS_TASKS.items()}
        base_glue = glue_score(R, f"deberta-base-{v}")
        for e in ENCODERS:
            m = f"deberta-base-{v}-{e}"
            for t, g in ZS_TASKS.items():
                s = zs_score(R, m, g, sub=False)
                if s is not None and base_zs[t] is not None:
                    task_deltas[t].append(s - base_zs[t])
            s = glue_score(R, m)
            if s is not None and base_glue is not None:
                task_deltas["glue"].append(s - base_glue)

    md += ["## Per-task deltas (9 encoder x vocab combinations)", "",
           "| task | mean delta | positive |", "|--|--:|--:|"]
    print("=== per-task deltas ===")
    order = sorted(task_deltas, key=lambda t: -sum(task_deltas[t]) /
                   len(task_deltas[t]))
    names, means, poss = [], [], []
    for t in order:
        ds = task_deltas[t]
        mean = sum(ds) / len(ds)
        pos = sum(1 for d in ds if d > 0)
        md.append(f"| {t} | {mean:+.2f} | {pos}/{len(ds)} |")
        print(f"  {t:16s} mean {mean:+6.2f}  positive {pos}/{len(ds)}")
        names.append(t); means.append(mean); poss.append(pos)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["tab:green" if p == 9 else "tab:blue" if m > 0 else "tab:red"
              for m, p in zip(means, poss)]
    ax.bar(names, means, color=colors)
    for i, (m, p) in enumerate(zip(means, poss)):
        ax.text(i, m + (0.05 if m >= 0 else -0.12), f"{p}/9",
                ha="center", fontsize=9)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("mean delta (vision − baseline, pts)")
    ax.set_title("Official BabyLM eval: vision-init deltas by task\n"
                 "(green = positive in all 9 encoder×vocab combinations)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "official_deltas.png", dpi=150)
    plt.close(fig)

    # ---- subtask drill-down for comps + ewok ----
    for task, glob in (("comps", ZS_TASKS["comps"]),
                       ("ewok", ZS_TASKS["ewok"])):
        deltas = defaultdict(list)
        for v in VOCABS:
            base = zs_score(R, f"deberta-base-{v}", glob, sub=True)
            for e in ENCODERS:
                enc = zs_score(R, f"deberta-base-{v}-{e}", glob, sub=True)
                if base and enc:
                    for s in base:
                        if s in enc:
                            deltas[s].append(enc[s] - base[s])
        md += ["", f"## {task} subtask deltas", "",
               "| subtask | mean delta | positive |", "|--|--:|--:|"]
        print(f"\n=== {task} subtasks ===")
        for s, ds in sorted(deltas.items(),
                            key=lambda kv: -sum(kv[1]) / len(kv[1])):
            mean = sum(ds) / len(ds)
            pos = sum(1 for d in ds if d > 0)
            md.append(f"| {s} | {mean:+.2f} | {pos}/{len(ds)} |")
            print(f"  {s[:40]:40s} {mean:+6.2f}  {pos}/{len(ds)}")

    args.out_md.write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out_md} and official_deltas.png")


if __name__ == "__main__":
    main()
