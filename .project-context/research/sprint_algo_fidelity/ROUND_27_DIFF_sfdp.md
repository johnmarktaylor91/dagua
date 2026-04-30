# Round 27 SFDP Line-by-Line Diff

Pair: `classic_sfdp` vs `graphviz_sfdp`

Reference scope:
- Dagua dispatch: `dagua/layout/ops/pipelines/__init__.py:71`
- Dagua implementation: `dagua/layout/ops/pipelines/sfdp.py`, `dagua/layout/ops/sfdp.py`
- Graphviz reference: `_references/graphviz/lib/sfdpgen/{sfdpinit.c,spring_electrical.c,Multilevel.c,post_process.c,stress_model.c,sparse_solve.c}` and `_references/graphviz/lib/sparse/QuadTree.c`

Prior status:
- Round 6 already identified the attractive-force distance factor as a likely lever, tried it, and reverted because median RMSD regressed (`ROUND_6_RESIDUAL.md`).
- Round 9 fixed Graphviz seed plumbing (`-Gseed` + `-Gstart`), causing sfdp to classify as `equivalent_at_1x` against Graphviz's stochastic floor.
- This pair was not line-by-line diffed in Rounds 19/21.

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/sfdp/baseline
```

Baseline result:
- Graphs: 5
- Requested Dagua seeds: 30
- Overall dagua-vs-Graphviz graph-median RMSD: `0.018990`
- p25 / p75 / p95: `0.018987` / `0.042178` / `0.114511`
- Worst graph: `parallel_multiedge_bundle` at median RMSD `0.132595`
- Per-graph dagua-vs-Graphviz medians:
  - `linear_3layer_mlp`: `0.018987`
  - `nested_shallow_enc_dec`: `0.018990`
  - `tl_mlp_3layer`: `0.017441`
  - `mixed_width_labels`: `0.042178`
  - `parallel_multiedge_bundle`: `0.132595`
- TOST note: `tl_mlp_3layer` had one Graphviz seed in this run and was `not_tested`; the other four graph summaries were `not_equivalent` because their within-Graphviz floor was zero or near-zero on this bounded subset.

## Ranked Divergences

### 1. Spring-Electrical Inner Loop

1. **Attractive spring law omits Graphviz's current-distance factor.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:584-616`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:321-329`, `471-477`, `601-607`
   - Divergence: Graphviz uses `CRK * (x_i - x_j) * dist`; Dagua uses `attractive_scale * weight * (x_i - x_j)`.
   - Status: already addressed in Round 6 as an attempted lever, but reverted due median regression.
   - Fix size / risk: ~8-20 net lines; high risk because it changes every iteration and previously worsened median.

2. **Dagua batches force computation; Graphviz's default `spring_electrical_embedding` updates nodes sequentially within an iteration.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:1010-1031`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:599-649`
   - Divergence: Graphviz computes and immediately applies each node's normalized displacement in the main/default path. Dagua computes all attractive + repulsive forces from the same pre-step positions, then applies a synchronous update.
   - Status: missed entirely; Round 7 notes called this the likely invasive residual but no line-by-line diff existed.
   - Fix size / risk: ~80-180 net lines; high risk and slower unless carefully optimized.

3. **Non-adaptive cooling on finer levels is wrong: Graphviz still cools by `0.90`, Dagua freezes step size.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:1077-1079`, `1491`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:171-175`, `1176-1179`
   - Divergence: after prolongation Graphviz sets `adaptive_cooling=false`, and `update_step(false, step, ...)` returns `cool * step`. Dagua disables `SFDPAdaptiveCool`, so `current_step` stays `0.1` for all fine-level iterations.
   - Status: missed entirely.
   - Fix size / risk: ~10-25 net lines; medium-high risk, likely high RMSD impact.

4. **Cooling convergence metric differs: Dagua uses global Euclidean force norm; Graphviz sums per-node force magnitudes.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:1021-1032`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:337-343`, `640-648`
   - Divergence: Graphviz accumulates `Fnorm += sqrt(f_i dot f_i)`. Dagua stores `torch.linalg.vector_norm(total_force)`, which is `sqrt(sum_i ||f_i||^2)`.
   - Status: missed entirely.
   - Fix size / risk: ~3-8 net lines; medium risk.

5. **Dagua recenters after every iteration; Graphviz does not.**
   - Label: `convention`
   - Dagua: `dagua/layout/ops/sfdp.py:1028-1031`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:334-343`, `640-649`
   - Divergence: Graphviz positions drift during the force solve and are centered later by `pcp_rotate`. Dagua subtracts the centroid every step.
   - Status: missed entirely.
   - Fix size / risk: ~1-4 net lines; medium risk because quadtree boxes and floating-point paths change.

6. **Exact-vs-quadtree switch threshold is not Graphviz's default.**
   - Label: `parameter-default`
   - Dagua: `dagua/layout/ops/sfdp.py:61-62`, `902-913`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:38-43`, `550-556`, `_references/graphviz/lib/sfdpgen/sfdpinit.c:215`
   - Divergence: Graphviz default `quadtree=normal` uses quadtree for `n >= 45`; Dagua uses exact all-pairs until `n >= 10000`.
   - Status: missed entirely.
   - Fix size / risk: ~4-12 net lines; high runtime risk but likely fidelity-relevant.

7. **Dagua's public `theta` parameter is not how Graphviz exposes quadtree control.**
   - Label: `scaffolding`
   - Dagua: `dagua/layout/ops/pipelines/sfdp.py:28-36`, `86-96`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:165-193`, `211-215`
   - Divergence: Graphviz exposes quadtree scheme (`none`, `normal`, `fast`) and hard-codes `bh=0.6`; Dagua exposes only `theta`.
   - Status: missed entirely.
   - Fix size / risk: ~20-50 net lines; medium risk.

8. **Force storage/reset semantics differ.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:605-616`, `643-651`, `1010-1021`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:578-648`
   - Divergence: Graphviz's default path uses per-node scratch `f` reset for each node. Dagua builds dense force tensors and relies on `index_add_` ordering.
   - Status: missed entirely.
   - Fix size / risk: bundled with item 2; medium risk.

### 2. Repulsion and Quadtree

1. **Barnes-Hut algorithm is structurally different.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:654-875`
   - Graphviz: `_references/graphviz/lib/sparse/QuadTree.c:131-310`
   - Divergence: Graphviz computes symmetric cell-cell interactions, then accumulates cell forces down to leaves. Dagua does node-query Barnes-Hut traversal independently for each node.
   - Status: missed entirely.
   - Fix size / risk: ~150-300 net lines; high risk.

2. **Graphviz's default normal quadtree uses `QuadTree_get_supernodes`; Dagua has no supernode-list path.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:837-875`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:611-628`; `_references/graphviz/lib/sparse/QuadTree.c:52-109`
   - Divergence: Graphviz normal mode gathers accepted supernodes for each node and accumulates forces from their weights/distances. Dagua recursively returns one force directly.
   - Status: missed entirely.
   - Fix size / risk: ~100-220 net lines; high risk.

3. **Graphviz quadtree box width uses `0.52 * max_span`; Dagua uses `0.5 * span + min_span`.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:671-678`
   - Graphviz: `_references/graphviz/lib/sparse/QuadTree.c:339-346`
   - Divergence: Graphviz pads the half-width by 4%; Dagua uses exact half-span plus a tiny epsilon.
   - Status: missed entirely.
   - Fix size / risk: ~2-6 net lines; low-medium risk.

4. **Quadtree leaf insertion and coincident-point behavior differ.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:727-755`, `826-834`
   - Graphviz: `_references/graphviz/lib/sparse/QuadTree.c:437-519`
   - Divergence: Graphviz stores linked lists at max depth and inserts previous resident points while maintaining weighted averages. Dagua partitions all indices recursively and leaves all max-depth occupants in a Python list.
   - Status: missed entirely.
   - Fix size / risk: ~60-120 net lines; medium risk.

5. **Quadtree depth optimizer is absent.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:61`, `727`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:101-185`, `590-591`, `652-657`
   - Divergence: Graphviz dynamically adjusts max qtree level based on measured work; Dagua uses fixed depth 10.
   - Status: missed entirely.
   - Fix size / risk: ~50-100 net lines; medium risk, mostly large graphs.

6. **Dagua uses `float32`; Graphviz uses `double`.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:376-378`, `403`, `557`, `790-834`, `1557`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:249-254`, `401-406`, `528-535`
   - Divergence: most Dagua tensors are `float32`; Graphviz force and coordinate math is `double`.
   - Status: missed entirely.
   - Fix size / risk: ~20-80 net lines; medium runtime/memory risk.

7. **Distance floor differs.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:54`, `644`, `832`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:459-461`, `625-627`; `_references/graphviz/lib/sparse/QuadTree.c:188-194`
   - Divergence: Dagua clamps squared distance at `1e-9`; Graphviz uses `distance_cropped`/`MINDIST` from common helpers. The exact floor is not replicated here.
   - Status: missed entirely.
   - Fix size / risk: ~10-25 net lines after confirming `MINDIST`; medium risk.

### 3. Multilevel Coarsening and Prolongation

1. **Coarsening algorithm misses Graphviz supervariable-first clustering.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:474-537`
   - Graphviz: `_references/graphviz/lib/sfdpgen/Multilevel.c:56-144`
   - Divergence: Graphviz decomposes to supervariables and clusters them before random heavy-edge matching. Dagua does only random-order heaviest-neighbor matching.
   - Status: missed entirely.
   - Fix size / risk: ~80-180 net lines; high risk.

2. **Coarsening reduction loop composes multiple internal coarsenings per level; Dagua rejects a too-small single step.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:524-537`, `1246-1257`
   - Graphviz: `_references/graphviz/lib/sfdpgen/Multilevel.c:204-240`
   - Divergence: Graphviz loops internal coarsening until `nc <= 0.75*n`, composing `P/R`. Dagua returns `None` when one HEM pass leaves more than `0.75*n`, stopping the hierarchy.
   - Status: missed entirely.
   - Fix size / risk: ~60-140 net lines; high risk.

3. **Coarse graph construction is not `R * A * P`.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:416-471`
   - Graphviz: `_references/graphviz/lib/sfdpgen/Multilevel.c:183-193`
   - Divergence: Graphviz builds interpolation/restriction matrices, divides rows of `R` by degree, multiplies `RAP`, marks symmetric, and removes diagonal. Dagua aggregates edge weights by coarse endpoint pair.
   - Status: missed entirely.
   - Fix size / risk: ~80-200 net lines; high risk.

4. **Random permutation source differs.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:500`, `1238-1239`
   - Graphviz: `_references/graphviz/lib/sfdpgen/Multilevel.c:102-104`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:280-283`
   - Divergence: Graphviz uses Graphviz/C RNG helpers (`gv_permutation`, `srand`, `drand`). Dagua uses `torch.Generator` / `torch.randperm` / `torch.rand`.
   - Status: partly addressed by Round 9 only on the Graphviz adapter side; Dagua RNG stream is still different.
   - Fix size / risk: ~30-100 net lines; medium-high risk.

5. **Initial random positions differ by RNG and dtype.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:540-557`, `1306-1311`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:280-286`, `567-573`
   - Divergence: both use unit-square random positions, but Dagua uses Torch `float32`; Graphviz uses C `drand()` doubles seeded by `srand`.
   - Status: partly addressed by Round 9 for reference seed plumbing; Dagua stream remains different.
   - Fix size / risk: ~20-60 net lines; medium risk.

6. **Prolongation matrix multiply is replaced with direct parent copy.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:1129`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:855-858`
   - Divergence: Graphviz computes `y = P * x`; because `P` may be composed across internal coarsening passes, this is not equivalent to one direct fine-to-coarse array in all cases.
   - Status: missed entirely.
   - Fix size / risk: bundled with multilevel rewrite; high risk.

7. **Interpolation smoothing is close but not ordered identically.**
   - Label: `numerical`
   - Dagua: `dagua/layout/ops/sfdp.py:1133-1145`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:832-853`, `859`
   - Divergence: both use `alpha=0.5`, but Graphviz mutates `x` in-place while scanning nodes. Dagua computes from an unmodified `positions` snapshot into `smoothed`.
   - Status: missed entirely.
   - Fix size / risk: ~8-20 net lines; medium risk.

8. **Prolongation jitter grouping differs.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:1146-1155`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:860-869`
   - Divergence: Graphviz iterates rows of `R` and jitters `ja[j]` for `j = ia[i]+1..ia[i+1]`; Dagua groups all fine nodes by coarse parent and jitters every group member after the first.
   - Status: missed entirely.
   - Fix size / risk: ~15-40 net lines; medium risk.

9. **K decay is aligned but state restoration semantics differ.**
   - Label: `scaffolding`
   - Dagua: `dagua/layout/ops/sfdp.py:1454-1457`, `1497-1499`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:1176-1180`, `1202-1204`
   - Divergence: both multiply `K` by `0.75`; Graphviz restores `ctrl0` at function exit, while Dagua persists final `state.ideal_length`.
   - Status: already mostly aligned; persistence difference is missed but probably low impact.
   - Fix size / risk: ~0-5 net lines; low risk.

10. **No maximum-level control equivalent to Graphviz's `levels` attribute.**
    - Label: `parameter-default`
    - Dagua: `dagua/layout/ops/sfdp.py:1246-1257`
    - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:211-214`; `_references/graphviz/lib/sfdpgen/Multilevel.c:248-280`
    - Divergence: Graphviz parses `levels` and passes `maxlevel`; Dagua has no public max-level control.
    - Status: missed entirely.
    - Fix size / risk: ~15-40 net lines; low-medium risk.

### 4. Graph Construction and Input Semantics

1. **Connected components are not laid out separately and packed.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/pipelines/sfdp.py:166-180`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:268-288`
   - Divergence: Graphviz lays each connected component independently through `sfdpLayout`, then packs subgraphs. Dagua lays disconnected components in one force system.
   - Status: missed entirely.
   - Fix size / risk: ~120-250 net lines; high risk and affects disconnected benchmarks.

2. **Dagua collapses parallel edges to weights; Graphviz matrix construction may preserve different real weights/pattern semantics.**
   - Label: `algorithm-correctness`
   - Dagua: `dagua/layout/ops/sfdp.py:372-399`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:92-104`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:1098-1102`
   - Divergence: Dagua sums duplicate undirected edges into one weight. Graphviz's `makeMatrix`, symmetrization, and diagonal removal path owns this behavior; the Dagua logic is not line-equivalent and can diverge on multiedge bundles.
   - Status: missed entirely.
   - Fix size / risk: ~20-80 net lines after confirming `makeMatrix`; medium risk.

3. **Edge weights are used by Dagua attraction; Graphviz sfdp default adjacency values may not be equivalent to user-supplied `edge_weights`.**
   - Label: `convention`
   - Dagua: `dagua/layout/ops/sfdp.py:612-615`
   - Graphviz: `_references/graphviz/lib/sfdpgen/Multilevel.c:78-87`, `187-193`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:321-329`
   - Divergence: Graphviz attraction loops ignore `A->a` in `spring_electrical_embedding`; weights affect coarsening through `a[j]` and matrix construction. Dagua multiplies spring force by edge weight at every level.
   - Status: missed entirely.
   - Fix size / risk: ~10-35 net lines; medium risk.

4. **Dagua validates/keeps directed edge order for final orientation; Graphviz treats sfdp as undirected geometry.**
   - Label: `convention`
   - Dagua: `dagua/layout/ops/sfdp.py:270-348`, `1550-1554`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:54`, `272-273`, `283-285`
   - Divergence: Dagua uses directed source-target flow to choose orientation; Graphviz lays geometry undirected and later splines edges.
   - Status: missed entirely.
   - Fix size / risk: ~10-30 net lines; medium risk for directed readability metrics.

### 5. Post-Processing and Output Coordinates

1. **Dagua normalizes to a synthetic extent; Graphviz returns coordinates after PCP rotation/overlap/packing without this normalization.**
   - Label: `convention`
   - Dagua: `dagua/layout/ops/sfdp.py:174-203`, `1555-1557`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:1188-1200`
   - Divergence: Graphviz does not scale to `sqrt(N)*5` or node-size-derived extents in `multilevel_spring_electrical_embedding`. Dagua always recenters/scales final output.
   - Status: missed entirely.
   - Fix size / risk: ~10-40 net lines; medium risk because comparator Procrustes hides some scale but not all downstream metrics.

2. **Graphviz always applies `pcp_rotate`; Dagua conditionally chooses among original/reflected/PCA candidates.**
   - Label: `convention`
   - Dagua: `dagua/layout/ops/sfdp.py:206-348`, `1550-1556`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:896-946`, `1192-1196`
   - Divergence: Dagua's PCA math is close, but it may keep the unrotated solution or reflect it to maximize directed flow. Graphviz always runs `pcp_rotate` for 2D, then applies explicit `rotation` only if requested.
   - Status: partly addressed by prior implementation of PCA rotation, but missed conditional-selection mismatch.
   - Fix size / risk: ~8-25 net lines; medium risk.

3. **Optional stress/post smoothing is unimplemented.**
   - Label: `algorithm-correctness`
   - Dagua: no equivalent in `dagua/layout/ops/sfdp.py`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:211-215`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:1188`; `_references/graphviz/lib/sfdpgen/post_process.c:108-307`, `579-620+`; `_references/graphviz/lib/sfdpgen/stress_model.c:10-47`
   - Divergence: Graphviz supports `smoothing` modes and a stress model using sparse CG. Default is `SMOOTHING_NONE`, so this is a parameter/API gap more than default RMSD gap.
   - Status: missed entirely.
   - Fix size / risk: ~200-500 net lines; high risk, low default impact.

4. **Overlap handling and node-size behavior differ.**
   - Label: `scaffolding`
   - Dagua: `dagua/layout/ops/sfdp.py:1555-1557`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:94-101`, `247-263`, `272-283`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:1199-1200`
   - Divergence: Graphviz computes node sizes/padding for overlap removal depending on graph attributes and adjustment mode. Dagua uses `node_sizes` only to choose final extent.
   - Status: missed entirely.
   - Fix size / risk: ~120-250 net lines; medium-high risk, maybe outside headless layout scope.

5. **Edge-label-node shortcut/attachment path is absent.**
   - Label: `scaffolding`
   - Dagua: no equivalent in `dagua/layout/ops/sfdp.py`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:978-1071`, `1104-1122`
   - Divergence: Graphviz can remove edge-label nodes, solve a shortened graph, reattach labels at neighbor averages, and remove overlap. Dagua treats all nodes as normal graph nodes.
   - Status: missed entirely.
   - Fix size / risk: ~120-260 net lines; medium risk, low impact unless label nodes are present.

6. **Graphviz supports explicit rotation attribute; Dagua supports direction-oriented reflection instead.**
   - Label: `parameter-default`
   - Dagua: `dagua/layout/ops/pipelines/sfdp.py:95`, `dagua/layout/ops/sfdp.py:247-348`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:218`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:948-976`, `1196`
   - Divergence: Graphviz has `rotation` degrees; Dagua has `direction`.
   - Status: missed entirely.
   - Fix size / risk: ~15-40 net lines; low-medium risk.

### 6. Defaults and Dispatch

1. **Default core constants mostly align.**
   - Label: `parameter-default`
   - Dagua: `dagua/layout/ops/pipelines/sfdp.py:28-35`, `dagua/layout/ops/sfdp.py:54-65`
   - Graphviz: `_references/graphviz/lib/sfdpgen/spring_electrical.c:34-67`
   - Status: already addressed/aligned for `C=0.2`, `bh/theta=0.6`, `p=-1`, `maxiter=500`, `step=0.1`, `K<0 => average_edge_length`, `max_qtree_level=10`, `K *= 0.75`, `jitter=K*0.001`.
   - Residual: the constants are aligned but many call sites are not equivalent.
   - Fix size / risk: none.

2. **Dagua dispatch is a direct pipeline call; Graphviz goes through graph attributes and component/packing lifecycle.**
   - Label: `scaffolding`
   - Dagua: `dagua/layout/ops/pipelines/__init__.py:71`, `dagua/layout/ops/pipelines/sfdp.py:86-184`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:226-296`
   - Divergence: Dagua receives tensors and returns positions. Graphviz initializes graph/node/edge records, parses attributes, handles components, routes splines, and postprocesses.
   - Status: missed entirely, but partly intentional for a headless engine.
   - Fix size / risk: not a single fix; high scope.

3. **Graphviz `start=random` handling is stricter.**
   - Label: `scaffolding`
   - Dagua: `dagua/layout/ops/pipelines/sfdp.py:91`, `166-180`
   - Graphviz: `_references/graphviz/lib/sfdpgen/sfdpinit.c:200-209`; `_references/graphviz/lib/sfdpgen/spring_electrical.c:51-62`
   - Divergence: Graphviz warns and forces random start; Dagua has no warm-start input path in this pipeline.
   - Status: Graphviz seed plumbing already addressed in Round 9; Dagua warm-start parity remains absent.
   - Fix size / risk: ~20-60 net lines; low-medium risk.

## Highest-Expected-Impact Fix Order for Round 28

1. Implement correct fine-level cooling (`adaptive_cooling=false` still multiplies step by `0.90`). Small, line-local, likely high effect.
2. Match Graphviz `Fnorm` as sum of per-node norms. Small and directly tied to cooling.
3. Remove per-iteration recentering or gate it behind a compatibility option.
4. Revisit attractive distance factor only after 1-3, because Round 6 tested it in isolation and it regressed median.
5. Lower/parameterize quadtree threshold to Graphviz's `45` and add normal-mode traversal if runtime is acceptable.
6. Decide whether Round 28 is allowed to attack the invasive sequential-update path. Without it, exact line parity is unlikely.
7. Treat component packing and matrix-based coarsening as larger follow-up rounds unless Round 28 explicitly accepts a broad rewrite.

## Baseline Artifacts

- Output directory: `eval_output/algo_fidelity/round_27/sfdp/baseline`
- Summary JSON: `eval_output/algo_fidelity/round_27/sfdp/baseline/multi_seed_summary.json`
- Pairwise CSV: `eval_output/algo_fidelity/round_27/sfdp/baseline/multi_seed_rmsd.csv`
