<task>
R34 COMPREHENSIVE reference engine audit + cleanup.

R33 refaudit covered NeuLay/sgd2_multi. This round: audit ALL ~25 reference
engines. Goals:
1. Find any other "silently broken" references (returning errors / NONE / wrong
   types) we haven't caught
2. Cleanup: per-engine smoke test on a 50-node graph; document which engines
   actually run vs which fail
3. Fix any cheap wiring bugs found

## Your job

1. List all reference engines registered in `dagua/eval/competitors/`:
   - igraph_* (drl, lgl, mds, rt, sugiyama, graphopt, davidson_harel, kk)
   - graphviz_* (dot, neato, fdp, sfdp)
   - ogdf_* (fmmm, gem, stress, pivot_mds)
   - networkx_* (spring, kamada_kawai, spectral)
   - fa2_ref, sgd2, sgd2_mds, umap_graph, tsne_graph, cytoscape_fcose,
     dagre, elk, etc.

2. For each, run a smoke test on `linear_3layer_mlp` (a stable 6-node graph):
   ```python
   from dagua.eval.competitors import get_competitor, _COMPETITORS
   from dagua.eval.graphs import get_test_graphs
   tg = [t for t in get_test_graphs() if t.name == 'linear_3layer_mlp'][0]
   for name in sorted(_COMPETITORS.keys()):
       c = get_competitor(name)
       avail = c.available()
       try:
           r = c.layout(tg.graph, seed=42)
           ok = r.pos is not None and r.error is None
           print(f'{name:50s} avail={avail} ok={ok} err={r.error[:60] if r.error else ""}')
       except Exception as e:
           print(f'{name:50s} avail={avail} EXCEPTION={str(e)[:80]}')
   ```

3. For each broken engine: check if fix is cheap (adapter bug, import path, missing
   try/except) or hard (missing upstream pkg). Apply cheap fixes.

## Output

`eval_output/algo_fidelity/round_34/full_reference_audit/SUMMARY.md` with:
- Per-engine status table (avail / runs / produces positions / matches expected dims)
- Cheap fixes applied
- Hard blockers documented

## Implementation

Use commit-safe wrapper. Commits per fix: `fix(eval): round 34 audit -- <engine> <terse>`.
</task>

<completeness_contract>
Smoke-test every reference engine. Fix every cheap-broken one. Document hard ones.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
