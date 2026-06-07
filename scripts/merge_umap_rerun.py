#!/usr/bin/env python3
"""Merge the recovered umap-rerun results into the main 100-seed escalation dataset.

The original 100-seed escalation BrokenProcessPool'd the umap family (~9% ok). The rerun
(eval_output/benchmark_100seed_umap_rerun, workers=1/4 + numba=1) recovered them. This merges the
rerun's OK umap records (+ their position .pt files) into the main escalation dir, replacing the
errored umap entries. Backs up the main results.json first; --confirm required to write.

Usage:
  python scripts/merge_umap_rerun.py            # dry-run (report only)
  python scripts/merge_umap_rerun.py --confirm  # actually merge (after backup)
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "eval_output/benchmark_100seed_escalation_final"
RERUN = ROOT / "eval_output/benchmark_100seed_umap_rerun"
CONFIRM = "--confirm" in sys.argv


def umap_key(rec):
    e = rec.get("engine_name", "")
    return "umap" in e


def main():
    main_results = json.load(open(MAIN / "results.json"))
    rerun_results = json.load(open(RERUN / "results.json"))

    # rerun OK umap records to bring over
    bring = {k: v for k, v in rerun_results.items() if umap_key(v) and v.get("status") == "ok"}
    print(
        f"main rows: {len(main_results)} | rerun rows: {len(rerun_results)} | "
        f"rerun OK umap records to merge: {len(bring)}"
    )

    # before: umap ok in main
    before = Counter()
    for v in main_results.values():
        if umap_key(v):
            before[v.get("status")] += 1
    print(f"main umap status BEFORE: {dict(before)}")

    # how many are replacements (key existed, was not ok) vs new
    repl = sum(1 for k in bring if k in main_results and main_results[k].get("status") != "ok")
    newk = sum(1 for k in bring if k not in main_results)
    keep_ok = sum(1 for k in bring if k in main_results and main_results[k].get("status") == "ok")
    print(f"  -> replacing non-ok: {repl}, new keys: {newk}, already-ok (skip): {keep_ok}")

    # verify each record's .pt exists in rerun
    missing_pt = [
        k
        for k, v in bring.items()
        if v.get("positions_file") and not (RERUN / v["positions_file"]).exists()
    ]
    if missing_pt:
        print(
            f"WARNING: {len(missing_pt)} rerun records reference a missing .pt -- will skip those"
        )
        for k in missing_pt:
            bring.pop(k, None)

    if not CONFIRM:
        after = dict(before)
        after["ok"] = before.get("ok", 0) + repl + newk
        for s in ("error", "timeout", "skipped"):
            after[s] = max(0, before.get(s, 0) - 0)  # replacements reduce non-ok
        # recompute properly: replaced non-ok become ok
        print(f"DRY RUN -- would merge {len(bring)} records (+copy .pt). Re-run with --confirm.")
        return 0

    # backup main results.json
    bak = MAIN / "results.json.prebumap.bak"
    shutil.copy2(MAIN / "results.json", bak)
    print(f"backed up main results.json -> {bak}")

    # copy .pt files + merge records
    copied = 0
    (MAIN / "positions").mkdir(exist_ok=True)
    for k, v in bring.items():
        pf = v.get("positions_file")
        if pf:
            src = RERUN / pf
            dst = MAIN / pf
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
        main_results[k] = v

    tmp = MAIN / "results.json.tmp"
    json.dump(main_results, open(tmp, "w"))
    tmp.replace(MAIN / "results.json")
    print(f"merged {len(bring)} umap records, copied {copied} .pt files")

    # validate
    after = Counter()
    for v in json.load(open(MAIN / "results.json")).values():
        if umap_key(v):
            after[v.get("status")] += 1
    print(f"main umap status AFTER: {dict(after)}")
    print(f"main total rows AFTER: {len(json.load(open(MAIN / 'results.json')))}")


if __name__ == "__main__":
    sys.exit(main())
