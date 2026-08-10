"""Sprint2 training-121 gate tally: fresh native regen vs the G-3 scored pool.

Protocol = the G-3 exact re-tally (findings/G3_RETALLY_REPORT.md): REAL frozen V3
(``score_position(..., ruler="v3")`` -> ``v3_tiered``), tie band 0.5, symmetric-G6
field eligibility via ``native_sprint_score.best_rows_by_graph`` / ``classify``,
field side reused verbatim from the G-3 scored pool (scoring signature verified
identical at launch). Only the NATIVE side is fresh: positions regenerated at the
sprint2 dev-endpoint sha under the deterministic envelope, scored here from
tensors.

Gate: any BEHIND row fails; strict/tied churn vs 104/17/0 is reported per row.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_sprint_score as nss  # noqa: E402

G3_DIR = Path.home() / ".claude/research/dagua/glados_prep/g3_retally_out"
G3_TALLY = {"strict": 104, "tied": 17, "behind": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", default="eval_output/sprint2_train121/native_regen")
    parser.add_argument("--out", default="eval_output/sprint2_train121/retally_summary.json")
    args = parser.parse_args()

    sig = nss.scoring_signature()
    g3 = json.loads((G3_DIR / "g3_retally_summary.json").read_text())
    if sig != g3["scoring_signature"]:
        print(f"FATAL: scoring signature drift {sig} != {g3['scoring_signature']}")
        return 2

    competitor_rows = []
    with gzip.open(G3_DIR / "competitor_v3_rows.jsonl.gz", "rt") as fh:
        for line in fh:
            competitor_rows.append(json.loads(line))

    field_best = nss.best_rows_by_graph(competitor_rows, "v3_tiered", engine=None)
    graph_map = nss.build_graph_map(sorted(field_best))
    native_dir = Path(args.native_dir)
    results = json.loads((native_dir / "results.json").read_text())
    native_rows = []
    for key, row in results.items():
        if row.get("engine_name") != "dagua":
            continue
        if str(row.get("status", "ok")).lower() != "ok":
            print(f"NATIVE ROW NOT OK: {key} {row.get('status')} {row.get('error')}")
            return 2
        graph = row["graph_name"]
        if graph not in graph_map:
            continue  # regen covers 129 rows; the tally scope is the 121 pool graphs
        pos_path = str(native_dir / row["positions_file"])
        scored = nss.score_position(graph_map[graph], pos_path, "dagua", sig, ruler="v3")
        native_rows.append({"engine": "dagua", "graph": graph, **scored})

    native_best = nss.best_rows_by_graph(native_rows, "v3_tiered", engine="dagua")

    graphs = sorted(field_best)
    missing = [g for g in graphs if g not in native_best]
    counts = {"strictly_best": 0, "tied": 0, "behind": 0}
    details = []
    for graph in graphs:
        if graph in missing:
            continue
        nrow, frow = native_best[graph], field_best[graph]
        delta = float(nrow["v3_tiered"]) - float(frow["v3_tiered"])
        status = nss.classify(delta)
        counts[status] += 1
        details.append(
            {
                "graph": graph,
                "status": status,
                "delta": delta,
                "native_v3_tiered": float(nrow["v3_tiered"]),
                "field_v3_tiered": float(frow["v3_tiered"]),
                "field_engine": frow.get("engine"),
            }
        )

    out = {
        "scoring_signature": sig,
        "tie_band": 0.5,
        "tally": {**counts, "total": len(details)},
        "vs_g3": G3_TALLY,
        "missing_native": missing,
        "behind_rows": [d for d in details if d["status"] == "behind"],
        "flips_vs_g3": {
            "tied_now": sorted(d["graph"] for d in details if d["status"] == "tied"),
            "g3_tied": sorted(r["graph"] if isinstance(r, dict) else r for r in g3["tied_rows"]),
        },
        "details": details,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(
        f"TALLY: {counts['strictly_best']} strict / {counts['tied']} tied / "
        f"{counts['behind']} behind of {len(details)} (missing {len(missing)})"
    )
    for d in out["behind_rows"]:
        print(f"BEHIND: {d['graph']} {d['delta']:+.3f} vs {d['field_engine']}")
    return 1 if (counts["behind"] or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
