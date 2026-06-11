#!/usr/bin/env python3
"""r70 control-graph pre-screen (spec sec. 8).

Selects the 8 positive-control graphs: seeded draw (sha256("r70::ctlgraphs")) from graphs
where ALL FIVE control engines have plain-Procrustes mean(W_D) in [0.05, 1.0], measured
from the existing 5-seed positions. Emits JSON to stdout and writes
eval_output/fidelity_definitive/controls/ctl_graphs.json.
"""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fast_fidelity_report import procrustes_rmsd  # noqa: E402

CONTROL_ENGINES = [
    "classic_fa2_default",
    "classic_graphopt_default",
    "classic_lgl_default",
    "classic_tsnet_default",
    "classic_linlog_default",
]
BAND = (0.05, 1.0)
N_GRAPHS = 8


def main() -> int:
    results = json.load(open("eval_output/benchmark_5seed_final/results.json"))
    h5 = h5py.File("eval_output/benchmark_5seed_final/positions.h5", "r")

    # graph -> engine -> list of ok seed keys
    by_combo: dict[str, dict[str, list[str]]] = {}
    for key, row in results.items():
        if row["engine_name"] in CONTROL_ENGINES and row["status"] == "ok":
            by_combo.setdefault(row["graph_name"], {}).setdefault(row["engine_name"], []).append(
                key
            )

    qualifying = []
    detail = {}
    for graph, engines in sorted(by_combo.items()):
        if len(engines) < len(CONTROL_ENGINES):
            continue
        means = {}
        ok = True
        for eng, keys in engines.items():
            lays = []
            for k in sorted(keys):
                if k in h5:
                    a = np.asarray(h5[k][...], dtype=np.float64)
                    if a.ndim == 2 and a.size and np.isfinite(a).all():
                        lays.append(a)
            if len(lays) < 4:
                ok = False
                break
            d = [
                procrustes_rmsd(lays[i], lays[j])
                for i in range(len(lays))
                for j in range(i + 1, len(lays))
            ]
            means[eng] = float(np.mean(d))
        if not ok:
            continue
        detail[graph] = means
        if all(BAND[0] <= m <= BAND[1] for m in means.values()):
            qualifying.append(graph)

    if len(qualifying) < N_GRAPHS:
        print(f"FAIL: only {len(qualifying)} qualifying graphs (< {N_GRAPHS})", file=sys.stderr)
        return 1

    seed = int.from_bytes(hashlib.sha256(b"r70::ctlgraphs").digest()[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(sorted(qualifying), size=N_GRAPHS, replace=False).tolist())

    out = {
        "control_engines": CONTROL_ENGINES,
        "band": BAND,
        "n_qualifying": len(qualifying),
        "qualifying": sorted(qualifying),
        "chosen": chosen,
        "w_d_means": {g: detail[g] for g in chosen},
    }
    outdir = Path("eval_output/fidelity_definitive/controls")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ctl_graphs.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_qualifying": len(qualifying), "chosen": chosen}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
