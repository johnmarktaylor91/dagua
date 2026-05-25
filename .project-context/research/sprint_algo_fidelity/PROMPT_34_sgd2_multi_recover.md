<task>
R34 RECOVER sgd2_multi reference. R33 refaudit confirmed
`gd2.py` + `criteria.py` are missing from `/tmp/graph-drawing`. Reference
adapter `sgd2_multi_ref` returns "missing gd2.py, criteria.py" error.

User said: "leave NOTHING on the table". Recover or reconstruct.

## Your job

### Phase A: Find the upstream source

The (SGD)^2 multi-criteria layout is by Tariq et al. 2022. Search for the
upstream:
1. `gh search repos "(SGD)^2" OR "sgd2_multi" OR "graph-drawing-criteria"`
2. `gh search code "gd2.py" path:graph-drawing` (looking for the original)
3. Web search: "graph drawing multi-criteria SGD" + Tariq 2022
4. Check arxiv paper https://arxiv.org/abs/2112.10527 for code links

If you find the upstream repo with gd2.py + criteria.py, clone it and verify
the files import.

### Phase B: If not found, reconstruct

The paper describes the algorithm in detail. Implement gd2.py + criteria.py
from scratch in `/tmp/graph-drawing/` (matching expected paths) OR as a new
module under dagua/eval/references/. Key components:
- `GD2(edges, num_nodes, criteria, lr, max_step, batch_size)` class with `.solve()`
- Criteria: stress, edge_uniformity, neighborhood_preservation, crossings,
  crossings_angle_maximization, aspect_ratio, vertex_resolution, gabriel,
  ideal_edge_length, angular_resolution

### Phase C: Wire the reference

Update `dagua/eval/competitors/sgd2_multi_competitor.py` adapter so it points
to your reconstructed module instead of the broken /tmp/graph-drawing path.

### Phase D: Verify

```python
from dagua.eval.competitors import get_competitor
c = get_competitor('sgd2_multi_ref')
print(c.available())
# ... run layout
```

Then bounded live_compare on 8 classic_sgd2_multi_* variants.

## Implementation

Use commit-safe wrapper.

## Output
`eval_output/algo_fidelity/round_34/sgd2_multi_recover/SUMMARY.md` documenting
whether upstream found OR reconstructed.
</task>

<completeness_contract>
Either: (a) working sgd2_multi reference producing positions, OR
(b) explicit documented blocker with thorough search trail.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
