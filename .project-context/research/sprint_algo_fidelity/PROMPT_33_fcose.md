<task>
R33 NEW ENGINE: add fcose (Cytoscape constraint-based force-directed) to dagua.

todos.md HIGH-priority item. Adds a new fidelity-paired family to the report.

## Background

fcose = "Fast Compound Spring Embedder" from Cytoscape.js. Constraint-based extension of CoSE. Used widely in graph viz. Currently no dagua reimpl.

## Your job

### Phase A: Reference adapter (verify existing)

Check if `dagua/eval/competitors/cytoscape_fcose_competitor.py` exists. R31 plans referenced it.
If exists, verify it works:
```python
python -c "from dagua.eval.competitors import get_competitor; c = get_competitor('cytoscape_fcose'); print(c)"
```

### Phase B: Dagua reimplementation

Add `dagua/layout/ops/pipelines/fcose.py` (and any required new ops in `dagua/layout/ops/*.py`).

fcose algorithm summary (from public spec):
1. Compound graph BFS-init (place children near parent's center)
2. Spring-embedder iteration with quad-tree N-body
3. Constraints: fixed/aligned nodes, relative placement, gravity
4. Multi-level coarsening for large graphs

Source: https://github.com/iVis-at-Bilkent/cytoscape.js-fcose (Apache 2.0 license)

Implementation strategy:
- Start with a non-compound spring-embedder variant (subset of full fcose)
- Add quad-tree for N-body when needed
- Add `classic_fcose` competitor in `classic_competitor.py`
- Add variants in `dagua/eval/variants.py`

### Phase C: Verify

```bash
python scripts/algo_fidelity_live_compare.py classic_fcose cytoscape_fcose --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,small_world_100 --output-dir eval_output/algo_fidelity/round_33/fcose/post_impl
```

Expected: partial_match to strong_equivalent if implementation is faithful.

## Implementation

Commit incrementally:
1. `feat(layout): round 33 fcose -- pipeline + base ops`
2. `feat(layout): round 33 fcose -- competitor + variants`
3. `test(layout): round 33 fcose -- fidelity regression`

Use commit-safe wrapper.

## Output
`eval_output/algo_fidelity/round_33/fcose/SUMMARY.md` with bounded RMSD.
</task>

<completeness_contract>
At minimum: pipeline + competitor + variant entries that produce non-error layouts. RMSD < 0.5 on simple graphs is a pass.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Read fcose source.
</default_follow_through_policy>
