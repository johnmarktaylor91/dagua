#!/usr/bin/env python3
"""Re-run the 6 umap variants that failed with BrokenProcessPool in the 100-seed escalation.

Root cause (2026-06-06): umap-learn's internal numba parallelism x 18 worker processes ->
nested-parallelism thread/memory explosion -> a pool worker died -> BrokenProcessPool -> ~82-91%
of umap combos errored (only ~16 tiny-graph combos survived).

Fix: --workers 1 + NUMBA_NUM_THREADS=1 (single-threaded numba, no worker pool to break). Runs into
a SEPARATE dir (benchmark_100seed_umap_rerun) so the main 541k-row dataset is untouched until a
validated merge. --resume makes it interruptible. On completion: text JMT (ready to merge).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
FAILMAP = ROOT / ".project-context/research/sprint_rng_matching/failing_map_final.json"
OUT = "eval_output/benchmark_100seed_umap_rerun"
SEND = os.path.expanduser("~/.claude/scripts/send-to-jmt.sh")

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = "/home/jtaylor/anaconda3/envs/py311/lib:" + env.get("LD_LIBRARY_PATH", "")
# single-threaded everything: numba is the umap culprit; OMP/MKL pinned too
for k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    env[k] = "1"


def main():
    mp = json.load(open(FAILMAP))
    umap = {e: m for e, m in mp.items() if "umap" in e}
    total = sum(len(m["graphs"]) for m in umap.values())
    print(f"=== UMAP rerun started {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    print(
        f"umap variants: {len(umap)}, (variant,graph) combos: {total}, workers=1 numba=1, out={OUT}"
    )
    for i, eng in enumerate(sorted(umap), 1):
        ref = umap[eng]["ref"]
        graphs = umap[eng]["graphs"]
        sel = eng if not ref else f"{eng},{ref}"
        print(
            f"\n--- [{i}/{len(umap)}] {eng} ({len(graphs)} graphs) {time.strftime('%H:%M:%S')} ---",
            flush=True,
        )
        cmd = [
            "python3",
            "scripts/run_benchmark.py",
            "--engines",
            sel,
            "--graphs",
            ",".join(graphs),
            "--seeds",
            "100",
            "--seed-start",
            "42",
            "--variants",
            "--output-dir",
            OUT,
            "--resume",
            "--workers",
            "1",
            "--timeout",
            "300",
            "--watchdog-timeout",
            "420",
        ]
        for attempt in range(1, 4):
            rc = subprocess.run(cmd, env=env).returncode
            if rc == 0:
                break
            print(f"    {eng} attempt {attempt} exit {rc}; retry 30s", flush=True)
            time.sleep(30)
        else:
            print(f"    {eng} FAILED after 3 attempts -- continuing", flush=True)
    print(f"\n=== UMAP rerun done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===", flush=True)
    try:
        subprocess.run(
            [
                SEND,
                f"dagua umap rerun complete (workers=1, numba=1) -- {total} combos in {OUT}. "
                f"CC will validate + merge into the main 100-seed dataset, then we're clean "
                f"for the fidelity analysis.",
            ],
            timeout=150,
        )
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
