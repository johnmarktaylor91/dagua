# Area C — Node-level global-depth y-alignment for multi-component DAGs

## Question

`disconnected_encoder_residual` (N=9, 2 components) loses by -1.62 entirely from depth_spearman 0.644 vs elk's 1.000 (= -5.4 weighted points). The components have OVERLAPPING depth ranges (component A has depths [0,1,2,3], component B has [0,1,2,3,4]), so band-permute by component-mean-depth (sprint-21c attempt) is a no-op — the spearman correlation is computed over ALL nodes globally, not within components.

elk's secret: it places depth-0 nodes from BOTH components on the same y-row, depth-1 nodes on the next row, etc. Components share global y-rows but stack horizontally. This is fundamentally different from row-major component tiling.

Sprint-21c's band-permute moved component-mean-y but didn't shift node-level y to align with global depth. The proper fix needs node-level y assignment that:
- Honors global longest-path layering (depth-i nodes share the same global y)
- Respects component boundaries (within-component layout preserved bit-for-bit by construction)
- Handles components with different depth ranges (gracefully — short components occupy a subset of rows)

## Research targets

1. **Algorithm sketch**. After per-component decomposition + tile, recompute each node's y as `y = base_y + global_depth(node) * pitch`, where `pitch` is the median between-row y-step in any component's local layout. Within-component x is preserved (just the relative x within the component's tile-x-offset). Across components, y is GLOBALLY synchronized.

2. **Edge case: components with different depth ranges**. If component A has depths [0..3] and B has [0..4], A's depth-3 nodes go on row 3 (occupied by both), B's depth-4 nodes go on row 4 (only B). Total height = max global depth + 1.

3. **Edge case: cycles**. Components with cycles have ill-defined depth; use FAS-then-longest-path. dagua already has this in init_placement.

4. **Implement and test**. /tmp/ script, measure deltas on:
   - disconnected_encoder_residual (target)
   - multi_component_80 (should also help)
   - disconnected_label_cycle_collage (already +1.27, verify no regression)
   - sparse_pair_50 (multi-component check: ensure no protected-win regression)
   - compound_dag_5x30 (ditto)

5. **Composite gain prediction**. depth_spearman 0.644 → 1.000 = +5.4 pts. But edge_length_cv may degrade if component widths differ a lot (rows now include nodes from short and tall components). Predict realistic net delta.

## Output

`.project-context/research/sprint_22_algo_bets/C_global_depth_alignment__<your_agent>.md`

- TL;DR
- Algorithm pseudocode (full version, not sketch)
- Real measured deltas from /tmp/ on the 5 graphs above
- Risk analysis for compound_dag_5x30, sparse_pair_50, etc.
- Implementation point in dagua/layout/ops/pipelines/dagua_native.py:1431 (where `_tile_component_positions` is called)

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed.
- Read CONTEXT.md first.
- BIGGER BET: real implementation tested empirically, not a sketch.
- 2000-4000 words.
