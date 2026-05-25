# Round 41 Reference Adapter Seed Audit

## Scope and Method

Audited registered competitors in `dagua/eval/competitors/` on a path-8 graph.
`dagua` was excluded because it is the in-project engine adapter, not an external
or reference competitor.

For each adapter, the audit compared:

- Seed respecting: `seed=42` vs `seed=43`
- Reproducible: `seed=42` vs `seed=42` again

Equality was checked with `torch.equal` on the returned CPU tensors. This is a
bit-exact test, not an approximate RMSD test.

NeuLay's default recovered-reference settings did not finish the three-run audit
within the bounded subprocess timeout. The same adapter seed path was audited
with `steps=500`, `gcn_steps=50`, and `use_gcn=False`, which completed and was
bit-exact reproducible.

## Classification Summary

| Classification | Count | Meaning |
|---|---:|---|
| PASS | 44 | Different seeds produced different positions and repeated same-seed runs were bit-exact. |
| DETERMINISTIC | 17 | Different seeds produced identical positions, but the adapter path is deterministic by construction on path-8 and repeated runs were bit-exact. |
| BROKEN-SEED | 0 | No stochastic adapter returned identical positions for seeds 42 and 43 after deterministic paths were accounted for. |
| NON-DETERMINISTIC | 0 | No completed adapter returned different positions for repeated seed 42 runs. |

## Per-Adapter Results

| Adapter | Module | Classification | Seed 42 vs 43 | Seed 42 repeat | Notes |
|---|---|---|---|---|---|
| `classic_fr` | `classic_competitor.py` | PASS | different | identical | Seeded FR initialization is respected. |
| `classic_kk` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Classic KK default uses deterministic NetworkX-style/circular initialization on path-8. |
| `classic_fr_kk` | `classic_competitor.py` | PASS | different | identical | FR warm-start carries seed variation into KK refinement. |
| `classic_kk_fr` | `classic_competitor.py` | DETERMINISTIC | identical | identical | KK warm-start is deterministic, then FR refines from supplied `pos`. |
| `classic_fa2` | `classic_competitor.py` | PASS | different | identical | Seeded FA2 initialization is respected. |
| `classic_fcose` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Path-8 takes the exact distance-embedding initialization; seed is used only by the large-graph fallback. |
| `classic_stress_sgd` | `classic_competitor.py` | PASS | different | identical | Seeded Stress-SGD path is respected. |
| `classic_sugiyama` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Layered/rank assignment is deterministic on path-8. |
| `classic_spectral` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Spectral embedding ignores seed by construction. |
| `classic_classical_mds` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Classical MDS is deterministic. |
| `classic_stress_maj` | `classic_competitor.py` | PASS | different | identical | Seeded warm-start jitter is respected. |
| `classic_neato` | `classic_competitor.py` | PASS | different | identical | Seeded neato-style random start is respected. |
| `classic_pivot_mds` | `classic_competitor.py` | PASS | different | identical | Seeded pivot behavior is respected in the classic adapter. |
| `classic_rt` | `classic_competitor.py` | DETERMINISTIC | identical | identical | Reingold-Tilford tree layout is deterministic. |
| `classic_linlog` | `classic_competitor.py` | PASS | different | identical | Seeded LinLog initialization is respected. |
| `classic_gem` | `classic_competitor.py` | PASS | different | identical | Seeded GEM random stream is respected. |
| `classic_tsnet` | `classic_competitor.py` | PASS | different | identical | Seeded t-SNE network initialization is respected. |
| `classic_maxent_stress` | `classic_competitor.py` | PASS | different | identical | Seeded maxent-stress initialization is respected. |
| `classic_davidson_harel` | `classic_competitor.py` | PASS | different | identical | Seeded move/init path is respected. |
| `classic_fmmm` | `classic_competitor.py` | PASS | different | identical | Seeded FM3 path is respected. |
| `classic_graphopt` | `classic_competitor.py` | PASS | different | identical | Seeded GraphOpt initial matrix is respected. |
| `classic_drl` | `classic_competitor.py` | PASS | different | identical | Seeded DrL path is respected. |
| `classic_lgl` | `classic_competitor.py` | PASS | different | identical | Seeded LGL root/init path is respected. |
| `classic_sfdp` | `classic_competitor.py` | PASS | different | identical | Seeded SFDP path is respected. |
| `classic_umap` | `classic_competitor.py` | PASS | different | identical | UMAP seed is respected; UMAP forces `n_jobs=1` when seeded. |
| `classic_neulay` | `classic_competitor.py` | PASS | different | identical | Default path completed slowly but was bit-exact reproducible. |
| `classic_sgd2_multi` | `classic_competitor.py` | PASS | different | identical | Seeded multi-criteria SGD2 path is respected. |
| `cytoscape_fcose` | `cytoscape_fcose_competitor.py` | PASS | different | identical | R35 seed fix still holds. |
| `dagre` | `dagre_competitor.py` | DETERMINISTIC | identical | identical | Dagre layered layout ignores seed by construction. |
| `elk_layered` | `elk_competitor.py` | DETERMINISTIC | identical | identical | ELK layered layout is deterministic on path-8. |
| `fa2_ref` | `fa2_competitor.py` | PASS | different | identical | Python and NumPy RNG seeding is respected. |
| `gephi_yifanhu` | `gephi_competitor.py` | PASS | different | identical | Java wrapper receives the benchmark seed. |
| `graphviz_dot` | `graphviz_competitor.py` | DETERMINISTIC | identical | identical | `dot` is deterministic; adapter intentionally does not pass seed/start. |
| `graphviz_sfdp` | `graphviz_competitor.py` | PASS | different | identical | `seed`/`start` Graphviz attributes are respected. |
| `graphviz_neato` | `graphviz_competitor.py` | PASS | different | identical | `seed`/`start` Graphviz attributes are respected. |
| `graphviz_fdp` | `graphviz_competitor.py` | PASS | different | identical | `seed`/`start` Graphviz attributes are respected. |
| `igraph_sugiyama` | `igraph_competitor.py` | DETERMINISTIC | identical | identical | Sugiyama path is deterministic on path-8. |
| `igraph_fr` | `igraph_competitor.py` | PASS | different | identical | R35 seed matrix bridge still holds. |
| `igraph_rt` | `igraph_competitor.py` | DETERMINISTIC | identical | identical | Reingold-Tilford path is deterministic. |
| `igraph_davidson_harel` | `igraph_competitor.py` | PASS | different | identical | R35 igraph RNG/seed-matrix bridge still holds. |
| `igraph_kamada_kawai` | `igraph_competitor.py` | PASS | different | identical | R35 seed matrix bridge still holds. |
| `igraph_mds` | `igraph_competitor.py` | DETERMINISTIC | identical | identical | MDS path is deterministic. |
| `igraph_graphopt` | `igraph_competitor.py` | PASS | different | identical | Seeded initial matrix is respected. |
| `igraph_drl` | `igraph_competitor.py` | PASS | different | identical | Seeded DrL path is respected. |
| `igraph_lgl` | `igraph_competitor.py` | PASS | different | identical | Seeded LGL path is respected. |
| `linlog` | `linlog_competitor.py` | PASS | different | identical | Seeded LinLog initialization is respected. |
| `nx_spring` | `networkx_competitor.py` | PASS | different | identical | NetworkX spring `seed` keyword is respected. |
| `nx_kamada_kawai` | `networkx_competitor.py` | DETERMINISTIC | identical | identical | NetworkX KK layout does not expose a random seed. |
| `nx_spectral` | `networkx_competitor.py` | DETERMINISTIC | identical | identical | NetworkX spectral layout is deterministic. |
| `neulay` | `neulay_competitor.py` | PASS | different | identical | Audited with shortened NeuLay parameters because default triplet exceeded the bounded audit timeout. |
| `ogdf_gem` | `ogdf_competitor.py` | PASS | different | identical | OGDF runner seed is respected. |
| `ogdf_fmmm` | `ogdf_competitor.py` | PASS | different | identical | OGDF runner seed is respected. |
| `ogdf_stress` | `ogdf_competitor.py` | PASS | different | identical | OGDF runner seed is respected. |
| `ogdf_pivot_mds` | `ogdf_competitor.py` | DETERMINISTIC | identical | identical | OGDF Pivot-MDS uses deterministic max-min pivots and fixed eigensolver basis on path-8. |
| `ogdf_davidson_harel` | `ogdf_competitor.py` | PASS | different | identical | OGDF runner seed is respected. |
| `ogdf_sugiyama` | `ogdf_competitor.py` | DETERMINISTIC | identical | identical | OGDF Sugiyama layout is deterministic on path-8. |
| `sgd2` | `sgd2_competitor.py` | PASS | different | identical | `random_seed` is forwarded to `s_gd2`. |
| `sgd2_mds` | `sgd2_competitor.py` | PASS | different | identical | `random_seed` is forwarded to the MDS wrapper. |
| `sgd2_multi_ref` | `sgd2_multi_competitor.py` | PASS | different | identical | Python, NumPy, and Torch seeds are set before the reference run. |
| `tsne_graph` | `tsne_competitor.py` | PASS | different | identical | scikit-learn `random_state` is respected. |
| `umap_graph` | `umap_competitor.py` | PASS | different | identical | UMAP `random_state` is respected and forces single-threaded execution. |

## Fixes Applied

No adapter code changes were required. The initial same-output cases were
accounted for by deterministic algorithm paths rather than broken stochastic
seed plumbing.

## Unresolved

No `BROKEN-SEED` or `NON-DETERMINISTIC` adapters remain from this audit.

NeuLay default settings are expensive enough that the full default three-run
audit exceeded the bounded timeout. The seed plumbing itself was verified with
shortened settings through the same adapter and wrapper call path.

## Command Notes

The audit was run against the local environment with the registered competitor
registry. External tools present in this environment included Graphviz, Dagre,
ELK, Gephi wrapper, OGDF runner, s_gd2, UMAP, t-SNE, Cytoscape fCoSE wrapper,
and the recovered NeuLay wrapper.
