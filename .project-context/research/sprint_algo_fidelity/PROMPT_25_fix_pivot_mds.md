<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 25 STRAGGLER FIX for **pivot_mds** family (`classic_pivot_mds` vs `ogdf_pivot_mds`).

## Round 24 measurement

5 graphs, 30 seeds (after R24 hotfix `e9c00b4` fixed PivotMDSComputeCoordinates(__init__) misplacement):
- 4 graphs PERFECT: linear_3layer_mlp 0.0, nested_shallow_enc_dec 0.0, parallel_multiedge_bundle 0.0, tl_mlp_3layer 0.0 (all < 1e-6)
- 1 graph DIVERGENT: **mixed_width_labels 0.091** (median RMSD)
- median across all graphs: 0.0182

**Why does mixed_width_labels alone diverge?** Find the special-case property of this graph that triggers the divergence.

## Your job

1. Inspect mixed_width_labels graph definition (find under `dagua/eval/graphs/` or run `python -c "from dagua.eval.graphs import get_test_graphs; g=[t for t in get_test_graphs().values() if t.graph_id=='mixed_width_labels'][0]; print(g.graph.num_nodes, g.graph.edge_index)"`)
2. Compare dagua and OGDF pivot_mds on this exact graph and identify what specifically differs.
3. Read OGDF PivotMDS reference: `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp`
4. Read dagua PivotMDS: `dagua/layout/ops/pipelines/pivot_mds.py`, `dagua/layout/ops/embed.py:_pivot_mds_coordinates`, `dagua/layout/ops/distance.py:PivotSelection`
5. Possible causes:
   - Pivot selection on this graph picks a different first pivot (deterministic vs random)
   - Pivot distance scale or normalization differs
   - SVD sign convention on this graph picks opposite sign
   - Width-aware preprocessing in mixed_width_labels triggers special handling
   - Something about the graph being labeled or having mixed widths affects node sizes which affects scale
6. Apply the fix. Verify mixed_width_labels RMSD drops below 0.01 without regressing the other 4 graphs.

## Reference

- Round 21 diff: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_pivot_mds.md`
- Round 22 residual: `.project-context/research/sprint_algo_fidelity/ROUND_22_RESIDUAL_pivot_mds.md`
- Round 23 commits: `01fe62f` (ogdf fidelity controls), `bb43daa` (sweep summary)
- OGDF source: `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp`
- Dagua: `dagua/layout/ops/pipelines/pivot_mds.py`, `dagua/layout/ops/embed.py`

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_pivot_mds ogdf_pivot_mds \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/pivot_mds/{baseline,post_fix}
```

Required: mixed_width_labels RMSD < 0.01 AND no other graph regressed above 0.01.

## Scope constraints

- **DO NOT TOUCH**: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`.
- Stage commits with explicit `git add <files>`; NO `git add -A`.
- Commit format: `feat(fidelity): round 25 pivot_mds -- <terse desc>`.

## Tests

- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "pivot"`
- Final summary: `eval_output/algo_fidelity/round_25/pivot_mds/SUMMARY.md`

</task>

<completeness_contract>
- Either measurable improvement on mixed_width_labels OR principled_residual documenting why this graph triggers a fundamental architectural floor.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Only stop for missing details that change correctness, safety, or irreversible actions.
</default_follow_through_policy>
