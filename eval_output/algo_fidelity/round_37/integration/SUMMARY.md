# R37 Integration Summary

## Variants Added

- `classic_sugiyama_graphviz_fidelity`
  - Base: `classic_sugiyama`
  - Reference: `graphviz_dot`
  - Params: `{"barycenter_passes": 24, "rank_sep": 1.0, "node_sep": 1.0, "fidelity_mode": "graphviz"}`
- `classic_sfdp_graphviz_fidelity`
  - Base: `classic_sfdp`
  - Reference: `graphviz_sfdp`
  - Params: `{"steps": 500, "theta": 0.6, "repulsive_exponent": -1.0, "fidelity_mode": "graphviz"}`
- `classic_fmmm_graphviz_fdp_fidelity`
  - Base: `classic_fmmm`
  - Reference: `graphviz_fdp`
  - Params: `{"steps": 200, "fidelity_mode": True}`
- `classic_neato_graphviz_fidelity`
  - Base: `classic_neato`
  - Reference: `graphviz_neato`
  - Params: `{"maxiter": 200, "epsilon": 0.0001, "pack": True, "fidelity_mode": "graphviz"}`

## Fidelity Mode Aliases Verified

- Sugiyama: `fidelity_mode="graphviz"` verified from `dagua/layout/ops/pipelines/sugiyama.py`, which accepts `{None, "igraph", "dot", "graphviz_dot", "graphviz"}` and routes `"graphviz"` to Graphviz rank/mincross behavior. R36 summaries: `dot_rank`, `dot_mincross`.
- SFDP: `fidelity_mode="graphviz"` verified from `dagua/layout/ops/pipelines/sfdp.py` and `round_36/sfdp_sequential/SUMMARY.md`; it selects Graphviz matrix hierarchy plus sequential refinement.
- FMMM/fdp: `fidelity_mode=True` verified from `dagua/layout/ops/pipelines/fmmm.py` and `round_36/fdp_recursion/SUMMARY.md`; it enables fdp recursion for clustered graphs and Graphviz component packing/attachment metadata.
- Neato: `fidelity_mode="graphviz"` verified from `dagua/layout/ops/pipelines/neato.py` and `round_36/neato_solver/SUMMARY.md`; it selects Graphviz PCA initialization plus packed-CG stress solver and enables the overlap fidelity path.

## Smoke RMSD

Command:

```bash
python eval_output/algo_fidelity/round_37/integration/smoke_check.py
```

Output:

```text
classic_sugiyama_graphviz_fidelity: 0.000000000
classic_sfdp_graphviz_fidelity: 0.023935233
classic_fmmm_graphviz_fdp_fidelity: 0.210095198
classic_neato_graphviz_fidelity: 0.442291663
```

The smoke checker uses an 8-node path for Sugiyama/SFDP/Neato and an 8-node path split into two clusters for FMMM so the fdp recursion path is exercised.

## Reference Adapter Additions

- No new adapter was required. `graphviz_fdp` was already registered in `dagua/eval/competitors/graphviz_competitor.py` as `GraphvizFdp` with `engine = "fdp"`.

## Interface Fixes

- `dagua/eval/competitors/classic_competitor.py` now forwards `graph.clusters` and `graph.cluster_parents` into `layout_fmmm_pipeline` when present, so benchmark variants can actually reach the R36 fdp recursion path.
- Added `variant_param_names` for `classic_neato`, `classic_fmmm`, and `classic_sfdp` to avoid misleading warnings when the new variant params are passed through the generic wrapper.

## Unresolved Interface Issues

- `classic_fmmm_graphviz_fdp_fidelity` smoke RMSD is `0.210095198`, above the `<0.05` sanity threshold. The variant is wired and reaches the clustered fdp recursion path, but the small clustered smoke graph is not bit-exact against the `graphviz_fdp` adapter.
- `classic_neato_graphviz_fidelity` smoke RMSD is `0.442291663`, above the `<0.05` sanity threshold. The requested `fidelity_mode="graphviz"` alias is wired, but on this smoke graph it diverges more than the older `graphviz_neato` compatibility path.
