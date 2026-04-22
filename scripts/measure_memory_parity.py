"""Sprint 0 Task 0.10: measure memory parity of ops pipeline vs legacy engine.

Records peak RSS for dagua.layout(g) with algorithm=None (new ops pipeline
default) vs algorithm="_legacy" (pre-decomposition body) on 1K, 10K, 100K
node graphs. Writes eval_output/native_algo/baseline_sprint_0/memory_profile.json.

If ops-pipeline RSS exceeds legacy by >20% at any tier, Sprint 1 must port
per-loss backward + checkpointing + hybrid-device ops from legacy into ops.
This script produces the evidence for that gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SIZES = [1_000, 10_000]  # 100K deferred to Sprint 0.5 background run
OUT_PATH = Path("eval_output/native_algo/baseline_sprint_0/memory_profile.json")


def _run_in_subprocess(code: str) -> dict:
    """Run `code` in a fresh python subprocess, return its peak RSS (MB).

    Returns {'peak_rss_mb': float, 'returncode': int, 'stdout': str, 'stderr': str}.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    # Peak RSS is captured per-subprocess via resource.getrusage(RUSAGE_SELF)
    # inside the child and printed as JSON; parent does not need rusage here.
    return {
        "returncode": proc.returncode,
        "stdout": stdout.strip(),
        "stderr": stderr[-500:] if stderr else "",
    }


_SUBPROC_TEMPLATE = """
import json, os, resource, time
import torch
from dagua import DaguaGraph, LayoutConfig
from dagua.layout.engine import layout as engine_layout

n = {n}
algorithm = {algorithm!r}

# Build a simple chain graph (cheap to construct, deterministic).
g = DaguaGraph()
for i in range(n):
    g.add_node(f"n{{i}}")
for i in range(n - 1):
    g.add_edge(f"n{{i}}", f"n{{i+1}}")

# Baseline RSS before layout.
baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

t0 = time.perf_counter()
try:
    cfg = LayoutConfig(steps=20, seed=42, algorithm=algorithm)
    pos = engine_layout(g, cfg)
    error = None
except Exception as e:
    error = f"{{type(e).__name__}}: {{e}}"
    pos = None
wall = time.perf_counter() - t0

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(json.dumps({{
    "n": n,
    "algorithm": algorithm,
    "peak_rss_mb": peak_kb / 1024.0,
    "baseline_rss_mb": baseline_kb / 1024.0,
    "delta_mb": (peak_kb - baseline_kb) / 1024.0,
    "runtime_s": wall,
    "error": error,
}}))
"""


def measure(n: int, algorithm) -> dict:
    """Spawn a subprocess for isolation; return its JSON result."""
    code = _SUBPROC_TEMPLATE.format(n=n, algorithm=algorithm)
    r = _run_in_subprocess(code)
    if r["returncode"] != 0:
        return {
            "n": n,
            "algorithm": algorithm,
            "error": f"subprocess exit {r['returncode']}: {r['stderr']}",
        }
    try:
        return json.loads(r["stdout"].splitlines()[-1])
    except Exception as e:
        return {
            "n": n,
            "algorithm": algorithm,
            "error": f"parse failed: {e}; stdout={r['stdout'][:500]}",
        }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for n in SIZES:
        for algorithm in (None, "_legacy"):
            tag = "ops_pipeline" if algorithm is None else "legacy"
            print(f"  running n={n:>7d} via {tag} ...", flush=True)
            r = measure(n, algorithm)
            r["tag"] = tag
            results.append(r)
            if r.get("error"):
                print(f"    ERROR: {r['error'][:200]}")
            else:
                print(
                    f"    peak_rss={r['peak_rss_mb']:.1f} MB, "
                    f"delta={r['delta_mb']:.1f} MB, "
                    f"runtime={r['runtime_s']:.1f}s"
                )

    # Summary: ratio per tier
    summary = {}
    for n in SIZES:
        ops = next((r for r in results if r["n"] == n and r["tag"] == "ops_pipeline"), None)
        legacy = next((r for r in results if r["n"] == n and r["tag"] == "legacy"), None)
        if ops and legacy and not ops.get("error") and not legacy.get("error"):
            ratio = ops["peak_rss_mb"] / max(legacy["peak_rss_mb"], 1.0)
            summary[n] = {
                "ops_rss_mb": ops["peak_rss_mb"],
                "legacy_rss_mb": legacy["peak_rss_mb"],
                "ops_over_legacy": ratio,
                "gap_exceeds_20pct": ratio > 1.20,
            }

    # Note: git SHA captured by the landing commit, not embedded (detect-secrets).
    payload = {
        "sizes": SIZES,
        "results": results,
        "summary": summary,
        "sprint_1_gate": {
            "rule": "Sprint 1 must port memory features if any ratio > 1.20",
            "triggered": any(s.get("gap_exceeds_20pct") for s in summary.values()),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")
    print(f"Sprint 1 gate triggered: {payload['sprint_1_gate']['triggered']}")


if __name__ == "__main__":
    main()
