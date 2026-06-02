#!/usr/bin/env python3
"""R69 triage: classify every classic_* variant into the 4-tier scheme.

Reads the 5-seed Procrustes verdicts (stage1/per_variant.json), the benchmark
status (results.json), and the variant registry (is_stochastic + reference).

Tiers:
  1 BIT_IDENTICAL  -- max per-seed RMSD < 1e-3.
  2 TIMEOUT        -- runs dominated by timeout/error (few/no ok pairs to verdict).
  -> ESCALATE      -- not bit-identical, STOCHASTIC (seed distribution exists) -> P3 100-seed TOST.
  4 DETERMINISTIC_DIFFERENT -- not bit-identical, DETERMINISTIC (no seed distribution; TOST N/A).
                              (sub-noted if median < 1e-3 = bit-exact-except-outliers.)
  NO_REFERENCE/NO_PORT -- no paired reference or no fidelity port (excluded from verdict).

Writes the stochastic-escalation engine list (reimpl + reference) to
/tmp/r69_escalation_engines.txt for P3, and a triage summary to
eval_output/fidelity_report_r69/triage.md.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "eval_output/benchmark_5seed_fidelity"
STAGE1 = ROOT / "eval_output/fidelity_report_r69/stage1"
OUT = ROOT / "eval_output/fidelity_report_r69/triage.md"
ESC_FILE = Path("/tmp/r69_escalation_engines.txt")
BIT = 1e-3

# --- load verdicts (summary rows: [vid, verdict, mean, median, max, N]) ---
pv = json.load(open(STAGE1 / "per_variant.json"))
verdict = {}
for row in pv["summary"]:
    vid, verd, mean, median, mx, n = row
    verdict[vid] = {"verdict": verd, "mean": mean, "median": median, "max": mx, "n": n}

# --- load benchmark status per variant ---
res = json.load(open(BENCH / "results.json"))
status = defaultdict(Counter)
for v in res.values():
    e = v.get("engine_name", "")
    if e.startswith("classic_"):
        status[e][v.get("status")] += 1

# --- load registry: is_stochastic + reference ---
from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name  # noqa: E402

reg = {}
for v in VARIANT_REGISTRY:
    if v.variant_id.startswith("classic_"):
        reg[v.variant_id] = {
            "stochastic": v.is_stochastic,
            "ref": original_variant_name(v),
        }

tiers = defaultdict(list)
escalate_engines = set()

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
        # no verdict -> either all timed out, or all skipped, or measurement gap
        if total > 0 and ok < max(1, 0.25 * total) and (bad + skipped) >= ok:
            tiers["TIMEOUT"].append((vid, f"ok={ok} err/to={bad} skip={skipped} (no verdict)"))
        else:
            tiers["UNVERDICTED_OTHER"].append(
                (vid, f"ok={ok} err/to={bad} skip={skipped} no-verdict")
            )
        continue

    mx, median = vd["max"], vd["median"]
    # timeout-dominated even if it got a few verdicts
    if total > 0 and bad >= 0.5 * total and ok < 0.4 * total:
        tiers["TIMEOUT"].append((vid, f"ok={ok} err/to={bad} (verdict max={mx:.2e})"))
        continue

    if mx < BIT:
        tiers["BIT_IDENTICAL"].append((vid, f"max={mx:.2e}"))
        continue

    # not bit-identical
    if info["stochastic"]:
        tiers["ESCALATE_STOCHASTIC"].append((vid, f"median={median:.2e} max={mx:.2e}"))
        escalate_engines.add(vid)
        if ref:
            escalate_engines.add(ref)
    else:
        tag = "median<1e-3 (bit-exact except outliers)" if median < BIT else f"median={median:.2e}"
        tiers["DETERMINISTIC_DIFFERENT"].append((vid, f"max={mx:.2e} {tag}"))

# --- write escalation engine list for P3 ---
ESC_FILE.write_text(",".join(sorted(escalate_engines)) + "\n")

# --- summary ---
lines = ["# R69 Triage -- 4-tier classification (from 5-seed Procrustes)\n"]
order = [
    "BIT_IDENTICAL",
    "ESCALATE_STOCHASTIC",
    "DETERMINISTIC_DIFFERENT",
    "TIMEOUT",
    "NO_REFERENCE",
    "UNVERDICTED_OTHER",
]
counts = {k: len(tiers[k]) for k in order}
lines.append("## Counts\n")
for k in order:
    lines.append(f"- **{k}**: {counts[k]}")
lines.append(f"\nTotal classic_ variants classified: {sum(counts.values())}")
lines.append(f"\nStochastic escalation -> P3 (engines incl. refs): {len(escalate_engines)}\n")
for k in order:
    lines.append(f"\n## {k} ({counts[k]})\n")
    for vid, note in tiers[k]:
        lines.append(f"- `{vid}` -- {note}")
OUT.write_text("\n".join(lines) + "\n")

print("=== R69 TRIAGE ===")
for k in order:
    print(f"  {k}: {counts[k]}")
print(f"  total: {sum(counts.values())}")
print(
    f"\nescalation engines (reimpl-only count): "
    f"{len([e for e in escalate_engines if e.startswith('classic_')])}"
)
print(f"escalation list written to {ESC_FILE}")
print(f"triage summary -> {OUT}")
print("\n--- ESCALATE_STOCHASTIC variants ---")
for vid, note in tiers["ESCALATE_STOCHASTIC"]:
    print(f"  {vid}: {note}")
print("\n--- DETERMINISTIC_DIFFERENT variants ---")
for vid, note in tiers["DETERMINISTIC_DIFFERENT"]:
    print(f"  {vid}: {note}")
print("\n--- TIMEOUT ---")
for vid, note in tiers["TIMEOUT"]:
    print(f"  {vid}: {note}")
print("\n--- UNVERDICTED_OTHER (needs a look) ---")
for vid, note in tiers["UNVERDICTED_OTHER"]:
    print(f"  {vid}: {note}")
