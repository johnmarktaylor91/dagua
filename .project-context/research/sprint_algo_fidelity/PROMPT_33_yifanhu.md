<task>
R33 NEW ENGINE: add YifanHu (Gephi force-directed) to dagua.

todos.md HIGH-priority item. Adds another fidelity-paired family.

## Background

YifanHu = multilevel force-directed layout by Yifan Hu (2005). Default Gephi engine. Multilevel coarsening + Barnes-Hut N-body. Currently no dagua reimpl.

## Your job

### Phase A: Reference availability

Check whether a Python reference is installable:
- `python -c "import yifanhu; print(yifanhu.__file__)"` (probably not)
- `gephi-toolkit` is Java
- igraph has no YifanHu

If no Python reference exists, this implementation is "dagua-only" — adds the algorithm to dagua but won't pair against a reference for TOST. Still useful: another algorithm in the package. Document this.

### Phase B: Dagua reimplementation

Reference paper: Yifan Hu, "Efficient and high quality force-directed graph drawing", The Mathematica Journal 10:37-71 (2005).

Algorithm summary:
1. Multilevel coarsening (matchings + clustering)
2. Force-directed embedding per level
3. Barnes-Hut tree for repulsion at each level
4. Final tuning at finest level

Add `dagua/layout/ops/pipelines/yifanhu.py` + ops.

### Phase C: Verify

If reference available:
```bash
python scripts/algo_fidelity_live_compare.py classic_yifanhu yifanhu --seeds 30 --graphs <bounded> --output-dir eval_output/algo_fidelity/round_33/yifanhu/post_impl
```

If no reference: just smoke-test that the algorithm produces non-error layouts on bounded subset.

## Implementation

Commits via commit-safe wrapper:
1. `feat(layout): round 33 yifanhu -- pipeline + base ops`
2. `feat(layout): round 33 yifanhu -- competitor + variants` (if pairable)
3. `test(layout): round 33 yifanhu -- smoke or fidelity`

## Output
`eval_output/algo_fidelity/round_33/yifanhu/SUMMARY.md`.
</task>

<completeness_contract>
At minimum: pipeline that produces non-error layouts. If reference exists, add competitor.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
