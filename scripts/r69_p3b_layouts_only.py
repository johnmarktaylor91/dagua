#!/usr/bin/env python3
"""TARGETED 100-seed LAYOUTS ONLY (no TOST/analysis -- JMT will choose the fidelity
analysis after the layouts land).

Runs the 100-seed benchmark on ONLY the failing (engine, graph) combos from the FULL
all-graphs 5-seed triage -- the full net: every stochastic combo that was neither a
bit-match nor a timeout (3,955 combos / 64 engines). Per-engine loop restricted to that
engine's failing graphs (run_benchmark's --graphs filter is global), --resume so the
~3-day run survives interruption. Reference + reimpl both run at matched seeds/params.

Reads the PERSISTENT failing map (not /tmp, which can be cleared mid-run).
On completion: texts JMT that the LAYOUTS are done. Does NOT run consolidate/TOST/report.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

FAILMAP = ROOT / ".project-context/research/sprint_rng_matching/failing_map_final.json"
BENCH = "eval_output/benchmark_100seed_escalation_final"
SEND = os.path.expanduser("~/.claude/scripts/send-to-jmt.sh")
DISK_FLOOR_GB = 15  # stop gracefully if free space drops below this (protect a 3-day run)

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = "/home/jtaylor/anaconda3/envs/py311/lib:" + env.get("LD_LIBRARY_PATH", "")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    env[k] = "1"


def notify(msg):
    try:
        subprocess.run([SEND, msg], timeout=150)
    except Exception:
        pass


def free_gb(path="/"):
    st = shutil.disk_usage(path)
    return st.free / 1e9


def main():
    mp = json.load(open(FAILMAP))
    engines = sorted(mp)
    total_combos = sum(len(mp[e]["graphs"]) for e in engines)
    print(f"=== 100-seed LAYOUTS-ONLY started {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    print(
        f"engines: {len(engines)}, failing (engine,graph) combos: {total_combos}, "
        f"out-dir: {BENCH}, free-disk: {free_gb():.1f} GB"
    )

    for i, eng in enumerate(engines, 1):
        if free_gb() < DISK_FLOOR_GB:
            msg = (
                f"100-seed layouts ABORTED at engine {i}/{len(engines)} ({eng}): "
                f"free disk {free_gb():.1f} GB < {DISK_FLOOR_GB} GB floor. --resume after cleanup."
            )
            print(msg)
            notify(msg)
            sys.exit(2)
        ref = mp[eng]["ref"]
        graphs = mp[eng]["graphs"]
        if not graphs:
            continue
        sel = eng if not ref else f"{eng},{ref}"
        print(
            f"\n--- [{i}/{len(engines)}] {eng}  ({len(graphs)} graphs)  "
            f"{time.strftime('%H:%M:%S')}  free={free_gb():.1f}GB ---",
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
            BENCH,
            "--resume",
            "--workers",
            "18",
            "--timeout",
            "300",
            "--watchdog-timeout",
            "420",
        ]
        for attempt in range(1, 4):
            rc = subprocess.run(cmd, env=env).returncode
            if rc == 0:
                break
            print(f"    engine {eng} attempt {attempt} exit {rc}; retry in 30s", flush=True)
            time.sleep(30)
        else:
            print(f"    engine {eng} FAILED after 3 attempts -- continuing", flush=True)

    print(
        f"\n=== 100-seed LAYOUTS-ONLY done {time.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"free={free_gb():.1f}GB ===",
        flush=True,
    )
    notify(
        f"dagua 100-seed LAYOUTS complete -- all {total_combos} non-bit-exact/non-timeout combos "
        f"(64 engines) laid out at 100 seeds in {BENCH}. NO analysis run yet (your call on the "
        f"fidelity analysis). Ready when you are."
    )


if __name__ == "__main__":
    sys.exit(main())
