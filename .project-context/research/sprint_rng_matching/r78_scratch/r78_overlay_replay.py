import json
import os
from collections import Counter, defaultdict

base = "/home/jtaylor/projects/dagua/eval_output/"
chain = json.load(open(base + "fidelity_definitive_r77/OFFICIAL_R77_LEDGER.json"))["scoring_chain"]
# map chain names to dirs
name_map = {
    "escalation_final": "benchmark_100seed_escalation_final",
    "seeded_refs": "benchmark_100seed_seeded_refs",
    "drlref_realfix": "benchmark_100seed_drlref_realfix",
    "umap_realfix": "benchmark_100seed_umap_realfix",
    "gem_realfix": "benchmark_100seed_gem_realfix",
    "r72_fixes": "benchmark_100seed_r72_fixes",
    "fmmm_r3": "benchmark_100seed_fmmm_r3",
    "fdp_fix": "benchmark_100seed_fdp_fix",
    "r73_fixes": "benchmark_100seed_r73_fixes",
    "r75_fixes": "benchmark_100seed_r75_fixes",
    "r75_mds_topup": "benchmark_100seed_r75_mds_topup",
    "r75_topup2": "benchmark_100seed_r75_topup2",
    "r76_refs": "benchmark_100seed_r76_refs",
    "r76_gem_fix": "benchmark_100seed_r76_gem_fix",
    "r76_refs2": "benchmark_100seed_r76_refs2",
    "r76_umap_refs": "benchmark_100seed_r76_umap_refs",
    "r76_umap_refs2": "benchmark_100seed_r76_umap_refs2",
    "r76_umap_fix2": "benchmark_100seed_r76_umap_fix2",
    "r76_maar_bench": "benchmark_100seed_r76_maar_bench",
    "r76_sfdp_fix": "benchmark_100seed_r76_sfdp_fix",
    "r76_sfdp_refs": "benchmark_100seed_r76_sfdp_refs",
    "r76_sfdp_fix2": "benchmark_100seed_r76_sfdp_fix2",
    "r76_sfdp_fix3": "benchmark_100seed_r76_sfdp_fix3",
    "r76_sugiyama_topup": "benchmark_100seed_r76_sugiyama_topup",
    "r76_igraph_fix": "benchmark_100seed_r76_igraph_fix",
    "r76_refs3": "benchmark_100seed_r76_refs3",
    "r77_mds2": "benchmark_100seed_r77_mds2",
    "r77_sfdp_pack2": "benchmark_100seed_r77_sfdp_pack2",
    "r77_sugiyama_a5b": "benchmark_100seed_r77_sugiyama_a5b",
    "r77_sugiyama_final": "benchmark_100seed_r77_sugiyama_final",
    "r77_sugiyama_wired": "benchmark_100seed_r77_sugiyama_wired",
    "r77_igraph_bk": "benchmark_100seed_r77_igraph_bk",
    "r77_randomdag": "benchmark_100seed_r77_randomdag",
    "r77_era_refs": "benchmark_100seed_r77_era_refs",
}
missing = [n for n in chain if n not in name_map or not os.path.isdir(base + name_map[n])]
print("unmapped/missing dirs:", missing)
# replay overlay winner per combo (last dir with an ok row wins)
winner = {}
seedsets = defaultdict(dict)  # combo -> dir -> (n_ok_seeded, has_det)
for n in chain:
    d = base + name_map[n]
    p = d + "/results.json"
    if not os.path.exists(p):
        print("NO results.json", n)
        continue
    rows = json.load(open(p))
    ok = defaultdict(lambda: [0, False])
    for key, row in rows.items():
        if not isinstance(row, dict):
            continue
        g = str(row.get("graph_name") or key.split("::")[0])
        e = str(row.get("engine_name") or key.split("::")[1])
        if row.get("status") == "ok":
            s = row.get("seed")
            ent = ok[(g, e)]
            if s is None or (isinstance(s, str) and s.strip() in ("", "None", "deterministic")):
                ent[1] = True
            else:
                ent[0] += 1
    for combo, ent in ok.items():
        winner[combo] = n
        seedsets[combo][n] = tuple(ent)
json.dump({f"{g}::{e}": w for (g, e), w in winner.items()}, open("winners.json", "w"))
print("combos with winners:", len(winner))

print(Counter(winner.values()).most_common())
