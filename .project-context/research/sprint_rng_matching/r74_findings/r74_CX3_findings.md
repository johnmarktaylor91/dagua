# CX3 fmmm / graphviz_fdp findings

Read-only research against:

- dagua pipeline: `dagua/layout/ops/pipelines/fmmm.py`
- OGDF reference: `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/FMMMLayout.cpp`
- Graphviz reference: `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/{tlayout.c,grid.c}`
- benchmark evidence: `eval_output/fidelity_definitive_r73/per_combo.json`

## Executive verdict

1. **OGDF component rotation claim is verified, but its impact is bounded.** OGDF calls
   `pack_subGraph_drawings()` after every divide-et-impera layout, including one-component graphs
   (`FMMMLayout.cpp:145-153`). With default `stepsForRotatingComponents(10)` (`FMMMLayout.cpp:267-270`),
   `pack_subGraph_drawings()` calls `rotate_components_and_calculate_bounding_rectangles()` whenever
   the setting is nonzero (`FMMMLayout.cpp:752-756`). That routine loops over every component
   (`FMMMLayout.cpp:828`) and tests 10 angles (`FMMMLayout.cpp:842-845`).

   Dagua has a faithful-looking single-component helper at `fmmm.py:1331-1373`, including the 10-angle
   search (`fmmm.py:1348-1353`) and square-aspect flip (`fmmm.py:1367-1369`). But the multi-component
   OGDF path lays out each component, immediately computes an unrotated bounding box
   (`fmmm.py:1837-1848`), sends those boxes to MAAR packing (`fmmm.py:1850`), and only applies the
   packing-level 90-degree `tipped` transform (`fmmm.py:1851-1863`). It does **not** run the
   per-component min-area search before packing.

   However, only 13/75 OGDF-step divergent combos are marked disconnected:
   steps10 5/39, steps100 4/17, steps200 4/19. The 15 graphs divergent in all three OGDF variants
   include only 3 disconnected graphs (`disconnected_encoder_residual`, `multi_component_80`,
   `random_dag_50`). Therefore this fix is real but cannot explain the 85-combo CX3 residual by itself.

2. **Graphviz fdp O(N^2) claim is refuted.** Graphviz fdp uses a spatial grid: `grid.c:17-19` says
   nodes are put in cells and repulsion is computed only in the node's 9 adjacent grids; `tlayout.c:177-180`
   sets default `T_Cell = 3 * T_K`; `tlayout.c:366-379` clears/fills the grid and walks it; `tlayout.c:266-279`
   applies same-cell plus eight-neighbor repulsion; `tlayout.c:237-245` gates neighbor-cell interactions
   by `dist2 < T_Cell*T_Cell`; `tlayout.c:579-586` runs the grid loop in `fdp_tLayout`.

   Dagua already ports that structure in Python lists: `fmmm.py:5189-5190` sets `cell_size = 3*K`,
   `fmmm.py:5197-5203` builds a grid, `fmmm.py:5214-5226` handles same-cell repulsion, and
   `fmmm.py:5227-5251` scans eight neighbor cells with the same radius gate. So the ~9
   `classic_fmmm_graphviz_fdp_fidelity` insufficient cases (`matched_seeds_lt_30`) are much more likely
   pure Python loop overhead than an algorithmic all-pairs bug. Barnes-Hut would be less faithful than
   Graphviz's own grid because Graphviz fdp is exact within a fixed 3K neighborhood and ignores the rest,
   while Barnes-Hut approximates long-range mass interactions Graphviz does not use.

3. **No evidence for a simple systematic scale/parameter gap across OGDF variants.** Variants pass
   matching step parameters to reference and reimpl: `classic_fmmm_steps10/100/200` use
   `{"steps": 10/100/200, "fidelity_mode": True}` and reference `{"fixed_iterations": 10/100/200}`
   (`dagua/eval/variants.py:1079-1107`). Dagua routes these to OGDF fidelity mode at
   `fmmm.py:6796-6805`. Because the analysis uses Procrustes, constant global scale should wash out.
   The divergent `plain_mean_W_D/plain_mean_W_R` medians are not constant across step variants:
   steps10 median 2.149, steps100 median 0.528, steps200 median 0.427. That is a variant-dependent
   shape/trajectory signature, not a single scale mismatch.

## Counts and likely flips

### A. Add OGDF min-plain-area component rotation before MAAR packing

Evidence:

- OGDF chooses the minimum **plain area** for multi-component graphs: in
  `rotate_components_and_calculate_bounding_rectangles`, `best_area` is initialized with
  `calculate_area(width,height,number_of_components)` (`FMMMLayout.cpp:831-833`), each candidate uses
  the same area (`FMMMLayout.cpp:853-855`), and the `PI/2` aspect-area special case is only for
  `number_of_components == 1` (`FMMMLayout.cpp:857-873`).
- Dagua multi-component path skips that candidate search entirely (`fmmm.py:1837-1850`).

Expected flips:

- High-confidence target subset: 13 OGDF divergent disconnected combos: 5 steps10, 4 steps100,
  4 steps200. Likely tier improvement: to rung 1/2 if all other component internals are already close;
  otherwise to 3/3Q. Best-case: 13/85 divergent CX3 combos.
- Related fdp divergent disconnected/multi-component subset: `disconnected_encoder_residual`,
  `disconnected_label_cycle_collage`, `multi_component_80`, `random_dag_50` in graphviz_fdp divergent
  list may also benefit from any analogous packing/rotation audit, but Graphviz fdp uses Graphviz pack,
  not OGDF rotation, so do not apply the OGDF rotation there.

Fix sketch:

- Factor `_ogdf_fmmm_pack_single_component` into a helper that returns `(rotated_positions, rect)` with
  `number_of_components`-aware area semantics.
- In `_layout_ogdf_fmmm_component_fidelity`, before appending `local_positions` and `component_boxes`,
  run the helper with `number_of_components=len(components)`; for multi-component mode compare
  `width*height` only, then do the final page-ratio 90-degree tip exactly like OGDF
  `FMMMLayout.cpp:885-903`.
- Feed the rotated component boxes into `_ogdf_maar_pack_component_transforms`.

Effort: small/medium. Confidence: high for root cause, medium for exact flip count.

### B. Torch-vectorize Graphviz fdp grid-cell repulsion

Evidence:

- Graphviz and dagua are both grid-based as cited above.
- The insufficient fdp cases are 9 `matched_seeds_lt_30`: `ba_500`, `citation_dag_300`, `er_500`,
  `grid_20x20`, `powerlaw_500`, `random_dag_200`, `rgg_500`, `sbm_5x50`, `small_world_500`.
- Successful larger/denser fdp divergent cases show Python cost growing: `rgg_100` runtime_D 6.17s vs
  reference 1.57s; `sbm_4x30` runtime_D 8.02s vs reference 2.08s. The small n=100 fdp cases are
  usually faster than reference, so the timeout cliff fits Python per-pair overhead in occupied cells.

Expected flips:

- Runtime-only: should move ~9 fdp insufficient combos to scorable. It does not inherently improve
  layout fidelity. After scorable, expect some mix of rung 1/2/4 depending on existing force-numeric
  parity.

Fix sketch:

- Keep Graphviz's cell-size and neighbor semantics exactly.
- For each occupied cell, build tensors for same-cell ordered pairs and neighbor-cell pairs, preserving
  directed accumulation semantics (`p,q` and `q,p` are both visited in same-cell Graphviz code).
- Compute deltas/dist2/force vectorized in torch, mask `dist2 < cell_size2` for neighbor cells, and
  scatter-add into displacement arrays. Use float64 for parity and only keep Python around cell
  dictionaries/order.
- Do not replace with Barnes-Hut for fidelity mode.

Effort: medium. Confidence: high on performance root cause, medium on preserving exact last-bit order.

### C. Connected OGDF force/numeric residual

Evidence:

- 62/75 OGDF-step divergent combos are connected. The persistent all-three-step divergent set has
  12/15 connected graphs: `compound_10x20`, `compound_dag_5x30`, `deep_chain_20`,
  `heavy_tail_weights_50`, `long_range_residual_ladder`, `ragged_feature_pyramid`,
  `resnet_stack_4x16`, `small_world_100`, `small_world_500`, `sparse_pair_50`,
  `tl_transformer_1layer`, `weighted_chain_20`.
- Many step10-only divergences become rung 1/2/3Q at steps100/200 (22 graph pattern), which suggests
  early trajectory/iteration sensitivity rather than terminal scaling.
- Persistent divergence displacements shrink with steps in many graphs but do not disappear:
  `deep_chain_20` 2.1399 -> 0.2552 -> 0.0784; `weighted_chain_20` 2.2518 -> 0.2272 -> 0.0809;
  `resnet_stack_4x16` 1.6620 -> 0.8641 -> 0.8637. This is not a uniform scale signature.

Likely avenues:

- Audit OGDF FMMM force-kernel arithmetic and RNG/cooling order at matched seeds on one connected
  graph that is divergent across all steps, preferably `weighted_chain_20` or `deep_chain_20` because
  they are small and structured.
- Capture per-iteration coordinates/forces from both reference and dagua on the benchmark path. The
  decisive experiment is a step-by-step trace diff: if positions match through initialization and first
  force pass then diverge under 1-ULP perturbation, it is a libm/chaos floor; if they diverge at the
  first force/placement update by a deterministic formula delta, it is still fixable.

Expected flips:

- Unknown until trace. This is the only avenue large enough to affect the connected majority
  (up to 62/75 OGDF divergent combos). Current evidence does **not** prove floor.

Effort: medium/high for tracing, potentially high for port fixes. Confidence: high that this is not
  component rotation or global scale; low/medium on exact root formula without traces.

## Floor subset and proof experiment

No subset is proven floor from the available data. The closest floor candidates are connected
persistent divergences whose displacement becomes very small by steps200, e.g. `deep_chain_20`
(`disp=0.0784`) and `weighted_chain_20` (`disp=0.0809`). But r72/r73 guardrails require FP-chaos
evidence, and per-iteration/thread disp-vs-iters traces were not present in the available files.

Proof experiment:

1. On benchmark path, choose `weighted_chain_20` for `classic_fmmm_steps200` and fixed matched seed.
2. Instrument reference OGDF and dagua to dump initialization, coarsening hierarchy, per-iteration
   force sums, and post-rotation/packing coordinates.
3. Run dagua twice with a controlled 1-ULP perturbation after the first matched state.
4. If OGDF-vs-dagua diff remains zero until the perturbation and then grows with the same Lyapunov
   curve as the 1-ULP dagua-vs-dagua run, classify as libm/chaos floor. If the first nonzero diff
   appears before perturbation or has a deterministic non-ULP formula signature, it remains fixable.
