#!/usr/bin/env python3
"""R69 P3b -- TARGETED 100-seed TOST: only the (engine, graph) combos that were
NOT bit-exact at 5 seeds. Fixes the P3 over-escalation (which ran every escalation
engine on ALL graphs). run_benchmark's --graphs filter is global, so we loop
per-engine, each restricted to that engine's failing graphs. --resume reuses any
100-seed data already collected in the escalation dir.

Reads /tmp/r69_failing_map.json : {engine: {"ref": refname, "graphs": [...]}}
Then consolidate -> TOST -> combined report.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
BENCH = "eval_output/benchmark_100seed_escalation"
STAGE1 = "eval_output/fidelity_report_r69/stage1"
TOST_OUT = "eval_output/fidelity_report_r69/tost"
REPORT = "eval_output/fidelity_report_r69/report.md"
SEND = os.path.expanduser("~/.claude/scripts/send-to-jmt.sh")

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = "/home/jtaylor/anaconda3/envs/py311/lib:" + env.get("LD_LIBRARY_PATH", "")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    env[k] = "1"


def notify(msg):
    try:
        subprocess.run([SEND, msg], timeout=150)
    except Exception:
        pass


def main():
    mp = json.load(open("/tmp/r69_failing_map.json"))
    engines = sorted(mp)
    print(f"=== R69 P3b targeted 100-seed started {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    print(
        f"engines: {len(engines)}, total failing (engine,graph) combos: "
        f"{sum(len(mp[e]['graphs']) for e in engines)}"
    )
    for i, eng in enumerate(engines, 1):
        ref = mp[eng]["ref"]
        graphs = mp[eng]["graphs"]
        if not graphs:
            continue
        sel = eng if not ref else f"{eng},{ref}"
        stamp = time.strftime("%H:%M:%S")
        print(f"\n--- [{i}/{len(engines)}] {eng}  ({len(graphs)} graphs)  {stamp} ---")
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
            print(f"    engine {eng} attempt {attempt} exit {rc}; retry in 30s")
            time.sleep(30)
        else:
            print(f"    engine {eng} FAILED after 3 attempts -- continuing")

    print(f"\n--- consolidate {time.strftime('%H:%M:%S')} ---")
    subprocess.run(
        [
            "python3",
            "scripts/consolidate_positions_hdf5.py",
            "--input",
            BENCH,
            "--output",
            f"{BENCH}/positions.h5",
        ],
        env=env,
    )

    print(f"--- TOST {time.strftime('%H:%M:%S')} ---")
    Path(TOST_OUT).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python3",
            "scripts/r68_tost_followup.py",
            "--per-variant-json",
            f"{STAGE1}/per_variant.json",
            "--results",
            f"{BENCH}/results.json",
            "--positions",
            f"{BENCH}/positions.h5",
            "--output",
            TOST_OUT,
        ],
        env=env,
    )

    print(f"--- combined report {time.strftime('%H:%M:%S')} ---")
    subprocess.run(
        [
            "python3",
            "scripts/r68_combined_report.py",
            "--per-seed",
            STAGE1,
            "--tost",
            TOST_OUT,
            "--output",
            REPORT,
        ],
        env=env,
    )

    print(f"=== R69 P3b done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    notify(
        f"R69 P3b COMPLETE (targeted 100-seed TOST on failing combos only). "
        f"Report: {REPORT} -- orchestrator finalizes 4-tier verdict (P4)."
    )


if __name__ == "__main__":
    sys.exit(main())
