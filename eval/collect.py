#!/usr/bin/env python
"""
Flatten BabyLM fast-eval results into one long-format CSV for plotting.

Usage:  python collect.py --results_dir $HOME/babylm-eval/strict/results \
                          --out results_dynamics.csv

Columns: vocab, step, task, section, item, value
  task    = blimp_fast / supplement_fast / ewok_fast / entity_tracking_fast / reading
  section = average | field | uid | linguistics_term | score | correlation
  item    = paradigm/field/measure name ("--" for averages)
"""
import argparse
import csv
import re
from pathlib import Path

def parse_model_name(model):
    """deberta-base-50k -> (vocab=50k, init=baseline);
       deberta-base-50k-dinov3 -> (vocab=50k, init=dinov3)."""
    rest = model.replace("deberta-base-", "")
    parts = rest.split("-", 1)
    vocab = parts[0]
    init = parts[1] if len(parts) > 1 else "baseline"
    return vocab, init


SECTION_MAP = {
    "FIELD ACCURACY": "field",
    "UID ACCURACY": "uid",
    "LINGUISTICS_TERM ACCURACY": "linguistics_term",
    "AVERAGE ACCURACY": "average",
}


def parse_report(path):
    rows, section = [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"### (.+)$", line)
        if m:
            section = SECTION_MAP.get(m.group(1).strip())
            continue
        if line.startswith("TEMPERATURE"):
            continue
        if re.fullmatch(r"[\d.]+\s+[\d.]+", line):   # "1.0  56.02" header
            continue
        m = re.match(r"(.+?):\s*([-\d.]+)$", line)
        if m and section:
            rows.append((section, m.group(1).strip(), float(m.group(2))))
        elif section == "average" and re.fullmatch(r"[-\d.]+", line):
            rows.append(("average", "--", float(line)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results_dynamics.csv"))
    args = ap.parse_args()

    out_rows = []

    # four ranking tasks
    for report in sorted(args.results_dir.rglob(
            "zero_shot/mlm/*/*/best_temperature_report.txt")):
        parts = report.parts
        zi = parts.index("zero_shot")
        model, rev = parts[zi - 2], parts[zi - 1]
        task = parts[-2]
        vocab, init = parse_model_name(model)
        ms = re.fullmatch(r"step(\d+)", rev)
        step = int(ms.group(1)) if ms else -1
        for section, item, value in parse_report(report):
            out_rows.append((vocab, init, step, task, section, item, value))

    # reading task
    for rdir in sorted(args.results_dir.rglob("zero_shot/mlm/reading")):
        parts = rdir.parts
        zi = parts.index("zero_shot")
        model, rev = parts[zi - 2], parts[zi - 1]
        vocab, init = parse_model_name(model)
        ms = re.fullmatch(r"step(\d+)", rev)
        step = int(ms.group(1)) if ms else -1
        rep = rdir / "report.txt"
        if rep.exists():
            for line in rep.read_text().splitlines():
                m = re.match(r"(.+?) SCORE:\s*([-\d.]+)$", line.strip())
                if m:
                    item = m.group(1).strip().lower().replace(" ", "_")
                    out_rows.append((vocab, init, step, "reading", "score",
                                     item, float(m.group(2))))
        cor = rdir / "correlations.txt"
        if cor.exists():
            for line in cor.read_text().splitlines():
                m = re.match(r"(\S+)\s+([-\d.]+)$", line.strip())
                if m:
                    out_rows.append((vocab, init, step, "reading", "correlation",
                                     m.group(1), float(m.group(2))))

    out_rows.sort()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vocab", "init", "step", "task", "section", "item", "value"])
        w.writerows(out_rows)
    print(f"wrote {args.out}: {len(out_rows)} rows from "
          f"{len(set((r[0], r[1], r[2]) for r in out_rows))} checkpoints")


if __name__ == "__main__":
    main()
