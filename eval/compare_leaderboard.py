#!/usr/bin/env python
"""
Merge our local results with BabyLM-Leaderboard-2026 strict-small submissions.
Leaderboard column = mean of subtask scores x100 (verified against the UI).
Usage: python compare_leaderboard.py --results_dir results [--ours_only]
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

TASKS = ["blimp", "blimp_supplement", "ewok", "entity_tracking", "comps", "glue"]
FT = ["boolq","mnli","mrpc","multirc","qqp","rte","wsc"]
FT_METRIC = {"mrpc":"f1","qqp":"f1"}

def load_leaderboard():
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download("BabyLM-community/leaderboard-2026-all-results",
                               repo_type="dataset"))
    latest = {}   # model_name -> (timestamp, row)
    for f in sorted(d.rglob("*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("track") != "strict-small":
            continue
        res = j.get("results", {})
        row = {}
        for t in TASKS:
            v = res.get(t)
            if isinstance(v, dict) and v:
                vals = [x for x in v.values() if isinstance(x,(int,float))]
                if vals:
                    row[t] = 100*sum(vals)/len(vals)
        if not row:
            continue
        name = j.get("config",{}).get("model_name") or f.stem
        ts = f.name  # timestamped filenames sort chronologically
        if name not in latest or ts > latest[name][0]:
            latest[name] = (ts, row)
    return {n: r for n,(ts,r) in latest.items()}

def get_avg(report):
    val, in_avg = None, False
    for line in report.read_text().splitlines():
        s = line.strip()
        if s.startswith("### "):
            in_avg = "AVERAGE" in s; continue
        if in_avg and re.fullmatch(r"[-\d.]+", s):
            val = float(s)
    return val

def our_scores(results_dir, model, debug=False):
    out = {}
    base = results_dir/model/"best"/"zero_shot"/"mlm"
    if base.exists():
        for rep in base.rglob("best_temperature_report.txt"):
            p = str(rep.relative_to(base)).lower()
            task = ("blimp_supplement" if "supplement" in p else
                    "entity_tracking" if "entity" in p else
                    "comps" if "comps" in p else
                    "ewok" if "ewok" in p else
                    "blimp" if "blimp" in p else None)
            if task:
                v = get_avg(rep)
                if v is not None:
                    out.setdefault(task, []).append(v)
                if debug:
                    print(f"  [{model}] {task:18s} <- {p}  avg={v}")
        out = {k: sum(v)/len(v) for k,v in out.items()}
    ft = results_dir/model/"best"/"finetune"
    sc = []
    for t in FT:
        f = ft/t/"results.txt"
        if f.exists():
            dd = dict(re.findall(r"(\w+):\s*([\d.]+)", f.read_text()))
            sc.append(float(dd.get(FT_METRIC.get(t,"accuracy"),0))*100)
    if len(sc)==7:
        out["glue"] = sum(sc)/7
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--ours_only", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = {} if args.ours_only else load_leaderboard()
    n_lb = len(rows)
    for m in sorted(p.name for p in args.results_dir.iterdir() if p.is_dir()):
        if (args.results_dir/m/"best").exists():
            s = our_scores(args.results_dir, m, args.debug)
            if s:
                rows[f"*OURS* {m}"] = s
    print(f"leaderboard entries: {n_lb} | ours: {len(rows)-n_lb}\n")

    def pavg(r):
        vals=[r[t] for t in TASKS if t in r]
        return sum(vals)/len(vals) if vals else 0
    ranked = sorted(rows.items(), key=lambda kv:-pavg(kv[1]))
    hdr = f"{'#':>3s} {'model':44s}{'p-avg':>7s}|"+"".join(f"{t[:12]:>13s}" for t in TASKS)
    print(hdr); print("-"*len(hdr))
    for i,(name,r) in enumerate(ranked,1):
        cells="".join(f"{r[t]:13.2f}" if t in r else f"{'--':>13s}" for t in TASKS)
        mark=" <<<" if name.startswith("*OURS*") else ""
        print(f"{i:3d} {name[:44]:44s}{pavg(r):7.2f}|{cells}{mark}")
    print("\np-avg = mean over shown columns (Reading & AoA excluded).")
    print("CAVEAT: ours = MLM pseudo-likelihood, local eval copy; theirs = official pipeline.")

if __name__ == "__main__":
    main()
