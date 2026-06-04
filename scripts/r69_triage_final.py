#!/usr/bin/env python3
"""4-tier triage for the FINAL all-graphs 5-seed run (eval_output/*_final).

Adapts scripts/r69_triage.py to the _final dirs AND produces the TARGETED failing
map (/tmp/r69_failing_map.json = {engine: {"ref": refname, "graphs": [failing...]}})
that scripts/r69_p3b_targeted.py consumes -- so the 100-seed escalation runs ONLY the
specific (stochastic engine, graph) combos that ran-but-were-not-bit-exact, NOT whole
engines on all graphs (the P3 over-escalation bug).

Tiers (per (engine,graph) combo, summarized per engine):
  1 BIT_IDENTICAL          -- max per-seed RMSD < 1e-3.
  2 TIMEOUT/ERROR          -- combo dominated by timeout/error (no verdictable pairs).
  3<-ESCALATE (stochastic) -- ran, not bit-identical, STOCHASTIC -> 100-seed TOST decides 3 vs 4.
  4 DETERMINISTIC_DIFFERENT-- ran, not bit-identical, DETERMINISTIC (no seeds; TOST N/A) -> tier 4.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "eval_output/benchmark_5seed_final"
REPORT = ROOT / "eval_output/fidelity_report_final"
OUT = REPORT / "triage_final.md"
FAILMAP = Path("/tmp/r69_failing_map.json")
BIT = 1e-3

pv = json.load(open(REPORT / "per_variant.json"))
verdict = {}
for row in pv["summary"]:
    vid, verd, mean, median, mx, n = row
    verdict[vid] = {"verdict": verd, "mean": mean, "median": median, "max": mx, "n": n}
failures = pv.get("failures", {})  # {vid: [[graph, seed, rmsd], ...]}

res = json.load(open(BENCH / "results.json"))
status = defaultdict(Counter)
for v in res.values():
    e = v.get("engine_name", "")
    if e.startswith("classic_"):
        status[e][v.get("status")] += 1

from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name  # noqa: E402

reg = {}
for v in VARIANT_REGISTRY:
    if v.variant_id.startswith("classic_"):
        reg[v.variant_id] = {"stochastic": v.is_stochastic, "ref": original_variant_name(v)}

tiers = defaultdict(list)
fail_map = {}

for vid, info in sorted(reg.items()):
    ref = info["ref"]
    st = status.get(vid, Counter())
    ok = st.get("ok", 0)
    bad = st.get("error", 0) + st.get("timeout", 0)
    skipped = st.get("skipped", 0)
    total = sum(st.values())
    vd = verdict.get(vid)

    if ref is None:
        tiers["NO_REFERENCE"].append((vid, "no paired reference"))
        continue
    if vd is None:
        if total > 0 and ok < max(1, 0.25 * total) and (bad + skipped) >= ok:
            tiers["TIMEOUT"].append((vid, f"ok={ok} err/to={bad} skip={skipped} (no verdict)"))
        else:
            tiers["UNVERDICTED_OTHER"].append((vid, f"ok={ok} err/to={bad} skip={skipped}"))
        continue

    mx, median = vd["max"], vd["median"]
    if total > 0 and bad >= 0.5 * total and ok < 0.4 * total:
        tiers["TIMEOUT"].append((vid, f"ok={ok} err/to={bad} (verdict max={mx:.2e})"))
        continue
    if mx < BIT:
        tiers["BIT_IDENTICAL"].append((vid, f"max={mx:.2e}"))
        continue

    # not bit-identical
    if info["stochastic"]:
        # failing graphs = unique graphs in pv.failures (these RAN ok but were not bit-exact;
        # timeouts/errors have no RMSD pair so are already excluded -> they stay Tier 2).
        fgraphs = sorted({row[0] for row in failures.get(vid, [])})
        tiers["ESCALATE_STOCHASTIC"].append(
            (vid, f"median={median:.2e} max={mx:.2e} fail_graphs={len(fgraphs)}")
        )
        if fgraphs:
            fail_map[vid] = {"ref": ref, "graphs": fgraphs}
    else:
        tag = "median<1e-3 (bit-exact except outliers)" if median < BIT else f"median={median:.2e}"
        tiers["DETERMINISTIC_DIFFERENT"].append((vid, f"max={mx:.2e} {tag}"))

FAILMAP.write_text(json.dumps(fail_map, indent=2) + "\n")

order = [
    "BIT_IDENTICAL",
    "ESCALATE_STOCHASTIC",
    "DETERMINISTIC_DIFFERENT",
    "TIMEOUT",
    "NO_REFERENCE",
    "UNVERDICTED_OTHER",
]
counts = {k: len(tiers[k]) for k in order}
lines = ["# 4-Tier Triage -- FINAL all-graphs 5-seed\n", "## Counts\n"]
for k in order:
    lines.append(f"- **{k}**: {counts[k]}")
lines.append(f"\nTotal classic_ variants: {sum(counts.values())}")
escal_combos = sum(len(v["graphs"]) for v in fail_map.values())
lines.append("\n## Escalation scope (100-seed TARGETED)\n")
lines.append(f"- escalation engines (stochastic, non-bit-exact): {len(fail_map)}")
lines.append(f"- escalation (engine,graph) COMBOS: {escal_combos}")
for k in order:
    lines.append(f"\n## {k} ({counts[k]})\n")
    for vid, note in tiers[k]:
        lines.append(f"- `{vid}` -- {note}")
OUT.write_text("\n".join(lines) + "\n")

print("=== 4-TIER TRIAGE (FINAL) ===")
for k in order:
    print(f"  {k}: {counts[k]}")
print(f"  total classic_ variants: {sum(counts.values())}")
print("\n*** SANITY GATE -- 100-seed escalation scope ***")
print(f"  escalation engines: {len(fail_map)}")
print(f"  escalation (engine,graph) combos: {escal_combos}")
print("  per-engine failing-graph counts:")
for vid, m in sorted(fail_map.items(), key=lambda x: -len(x[1]["graphs"])):
    print(f"    {vid}: {len(m['graphs'])} graphs (ref={m['ref']})")
print(f"\n  failing map -> {FAILMAP}")
print(f"  triage -> {OUT}")
print("\n--- DETERMINISTIC_DIFFERENT (-> Tier 4, NO 100-seed) ---")
for vid, note in tiers["DETERMINISTIC_DIFFERENT"]:
    print(f"  {vid}: {note}")
print("\n--- TIMEOUT (-> Tier 2) ---")
for vid, note in tiers["TIMEOUT"]:
    print(f"  {vid}: {note}")
