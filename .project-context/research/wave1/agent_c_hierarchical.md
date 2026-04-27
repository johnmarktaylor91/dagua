Assumption: “configurable parameters” means each public `layout_*` signature; internal helper defaults and module constants are listed under hardcoded candidates unless the file actually exposes them.

Catalog reference: [DESIGN.md](/home/jtaylor/projects/dagua/dagua/layout/ops/DESIGN.md#L217)

**Sugiyama**
Source: [sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/classic/sugiyama.py#L40)

1. Validate `edge_index`, `num_nodes`, `node_sizes`, `edge_weights`; reject negative `trace_every`; alias `layer_sep -> rank_sep`.
2. Choose output device from `edge_index` or `node_sizes`; coerce `node_sizes` to CPU `float32`, defaulting to zeros `[N,2]`.
3. Call `make_acyclic_robust()` to reverse DFS back-edges, then fall back to greedy FAS if needed.
4. Run heap-based Kahn longest-path layering, then iteratively promote nodes downward to the deepest legal layer.
5. Expand every edge spanning multiple layers into dummy-node chains; append dummy nodes to per-layer lists; copy edge weights onto every expanded segment; record `edge_paths`.
6. Build `parents`, `children`, `parent_weights`, and `child_weights` over the expanded DAG.
7. Run `barycenter_passes` full down/up sweeps: compute weighted barycenters from already ordered adjacent layers, stable-sort each layer, optionally snapshot traces by running full coordinate assignment.
8. Assign `y = layer_idx * rank_sep`, then compute `x` with Brandes-Kopf: four orientations (`ul`,`ur`,`dl`,`dr`), type-1 conflict detection, vertical alignment, horizontal compaction, alignment normalization, median-of-four balancing, final centering.
9. Strip dummy-node coordinates/traces back to the first `num_nodes`.
10. If requested, reconstruct per-edge routes from `edge_paths`, flipping paths for reversed input edges.

Config: `edge_index: torch.Tensor`, `num_nodes: int`, `node_sizes: Optional[torch.Tensor]=None`, `rank_sep: float=1.0`, `node_sep: float=1.0`, `layer_sep: Optional[float]=None`, `seed: int=42`, `barycenter_passes: int=24`, `trace_every: int=0`, `edge_weights: Optional[torch.Tensor]=None`, `return_edge_routes: bool=False`.

State: `_ExpandedLayeredGraph(edge_index[2,E'], layers[list[list[int]]], node_sizes[N',2], edge_paths[list[list[int]]], num_nodes)`; `layer_assignments[N]`; `parents/children: list[list[int]]`; `parent_weights/child_weights: list[dict[int,float]]`; BK arrays `rank_of`, `pos_of`, `root`, `align`, `sink`, `shift`, `x`; `positions[N',2]`; `traces[list[T[N',2]]]`.

Solver: no numeric optimizer. It is a discrete pipeline: acyclic orientation, longest-path layering, weighted barycenter ordering, then Brandes-Kopf compaction.

Convergence: fixed `barycenter_passes`; no crossing-based early stop. Layer promotion loops until no node moves.

Shared motifs: `MakeAcyclic`, `BuildAdjacency`, `LongestPathLayering`, `LayerPromotion`, `BarycenterSweep`, `BrandesKopf4Pass`, `StripDummyNodes`, `ReconstructEdgeRoutes`.

Hardcoded candidates: dummy size `[0,0]`; four BK orientations; median-of-four balancing rule; stable tie handling by sorted node id; `_NO_SHIFT = inf`; no public direction transform despite cataloging; trace snapshots always run full coordinate assignment.

RNG: `seed` is not consumed. `layout_sugiyama()` passes it into `_barycenter_ordering()`, which immediately `del seed`; there are no `torch`/`random` RNG calls in this file, and [make_acyclic_robust()](/home/jtaylor/projects/dagua/dagua/layout/cycle.py#L193) is deterministic.

Catalog: strong match for `MakeAcyclic`, `LongestPathLayering`, `LayerPromotion`, `InsertDummyNodes`, `BarycenterSweep`, `BrandesKopf4Pass`, `StripDummyNodes`, `ReconstructEdgeRoutes`. Missing/mismatch: cataloged `DirectionTransform` for Sugiyama is absent here; real code also bundles a non-cataloged greedy cycle-breaking fallback.

**Reingold-Tilford**
Source: [reingold_tilford.py](/home/jtaylor/projects/dagua/dagua/layout/classic/reingold_tilford.py#L564)

1. Validate `num_nodes` and optional `edge_weights`; choose output device; return early for empty graphs; raise recursion limit to `max(current, num_nodes * 2)`.
2. Build undirected adjacency via the shared helper and rank roots by `(zero_indegree_first, indegree, node_id)`.
3. BFS each connected component to build a deterministic forest: `roots`, `children[list[list[int]]]`, `depths[list[int]]`.
4. Derive `sibling_spacing` from max node width or `1.0`, `layer_spacing` from max node height or `1.5`, and `component_gap = 2 * sibling_spacing`.
5. For each root, recursively build `_WalkerNode` objects with `parent`, `children`, `prelim`, `mod`, `shift`, `change`, `thread`, `ancestor`, and sibling metadata.
6. Run Buchheim/Walker first walk with `distance=1.0` to apportion contour conflicts and assign prelim/mod values.
7. Run second walk to accumulate modifiers into final x-coordinates; normalize each component so its minimum x starts at the current component offset; advance offset by component width plus `1.0 + component_gap`.
8. Materialize `positions[node,0] = preliminary_x * sibling_spacing`, `positions[node,1] = depth * layer_spacing`, center the whole layout, and optionally swap axes when `horizontal=True`.

Config: `edge_index: torch.Tensor`, `num_nodes: int`, `node_sizes: Optional[torch.Tensor]=None`, `seed: int=42`, `horizontal: bool=False`, `edge_weights: Optional[torch.Tensor]=None`.

State: BFS `adjacency[list[list[(neighbor,weight)]]]`, `roots[list[int]]`, `children[list[list[int]]]`, `depths[list[int]]`; `_WalkerNode` forest with `prelim/mod/shift/change/thread/ancestor`; `preliminary_x[list[float]]`; `positions[N,2]`.

Solver: deterministic Buchheim linear-time tidy-tree traversal over a BFS forest, not an optimizer.

Convergence: none; single BFS pass plus one first-walk and one second-walk per component.

Shared motifs: `DeterministicInit` with Sugiyama; adjacency building and final centering with the force-directed layouts.

Hardcoded candidates: root ranking heuristic; first-walk `distance=1.0`; default spacings `1.0` and `1.5`; `component_gap = 2 * sibling_spacing`; next-component extra `+1.0`; recursion-limit multiplier `2`; `horizontal` is really axis swap, not a horizontal flip.

RNG: `seed` is ignored via `_ = seed`; there are no RNG calls.

Catalog: `BucheimWalkerTree` matches. Missing/mismatch: cataloged `HorizontalFlip` is wrong for this code path, which swaps axes; non-cataloged real steps include `RootSelection`, `BFSForestExtract`, and component packing.

**SFDP**
Source: [sfdp.py](/home/jtaylor/projects/dagua/dagua/layout/classic/sfdp.py#L926)

1. Validate shapes/ranges; choose output device; early-return for `N=0` or `N=1`.
2. Create one CPU `torch.Generator` and `manual_seed(seed)`.
3. Collapse the input to a unique undirected weighted graph: `_GraphData(num_nodes, edge_index[2,E], edge_weight[E], adjacency[list[list[(neighbor,float)]]])`.
4. Repeatedly coarsen with heavy-edge matching in random node order until coarsening is not worthwhile.
5. Initialize the coarsest graph with uniform random positions in `[0,1]^2`.
6. Estimate coarsest `ideal_length` as the average current edge length.
7. Run Hu spring-electrical refinement on the coarsest graph: linear weighted spring attraction, inverse-power repulsion, exact all-pairs below `10_000` nodes or Barnes-Hut otherwise, normalize each node’s force to a direction, move by `current_step`, recenter, and adapt `current_step` from total-force progress.
8. Uncoarsen from coarse to fine: decay `ideal_length` by `0.75`, copy coarse positions by `fine_to_coarse`, smooth each node halfway toward its neighbor mean, add tiny random jitter to all but the first fine node in each coarse group, then rerun refinement with fixed step size and no adaptive cooling.
9. Center and scale the final coordinates to a graph-size- and node-size-based extent.

Config: `edge_index: torch.Tensor`, `num_nodes: int`, `node_sizes: Optional[torch.Tensor]=None`, `steps: int=500`, `seed: int=123`, `theta: float=0.6`, `repulsive_exponent: float=-1.0`, `edge_weights: Optional[torch.Tensor]=None`.

State: `_GraphData`; `mappings[list[fine_to_coarse[N_fine]]]`; `_QuadTreeNode(center[2], half_width, indices, level, mass, center_of_mass[2], children)`; `positions[N,2]`; `attractive[N,2]`; `repulsive[N,2]`; `total_force[N,2]`.

Solver: Hu-style fixed-step spring-electrical iteration. It is not gradient-based; movement is `positions += step * normalized_force`.

Convergence: per-level stop is `steps` or `current_step < 1e-3`; coarsening stops when the matched graph is too small, shrinks too little, or fails to reduce node count.

Shared motifs: `RandomUniformInit` with Davidson-Harel; `HeavyEdgeMatching`, `BuildQuadTree`, `BarnesHutForce`, multilevel `DirectMapping`, `NeighborSmoothing`, centering/normalization with FMMM.

Hardcoded candidates: `_FORCE_SCALING=0.2`; `_DEFAULT_STEP=0.1`; `_DEFAULT_TOLERANCE=1e-3`; `_MIN_COARSE_SIZE=4`; `_MIN_COARSEN_REDUCTION=0.75`; `_BARNES_HUT_THRESHOLD=10_000`; `_MAX_QUADTREE_DEPTH=10`; `_PROLONGATION_NOISE_SCALE=1e-3`; `_PROLONGATION_SMOOTHING=0.5`; `_REFINEMENT_K_DECAY=0.75`; adaptive-cooling factors `0.90/1.0/1.1`.

RNG: one CPU `torch.Generator`.
1. `generator.manual_seed(seed)`.
2. Each `_heavy_edge_matching()` call that reaches matching consumes one `torch.randperm(num_nodes, generator=generator)`.
3. Coarsest init consumes one `torch.rand((N_coarse, 2), generator=generator)`.
4. Each `_prolongate_positions()` consumes one `torch.rand((2,), generator=generator)` for every fine node after the first in each coarse group, in group insertion order from `fine_to_coarse.tolist()`.
5. No other random calls occur.

Catalog: matches `RandomUniformInit`, `HeavyEdgeMatching`, `BuildQuadTree`, `BarnesHutForce`, `DirectMapping`, `NeighborSmoothing`, `AdaptiveCool`, `NormalizePositions`. Missing/mismatch: the real repulsion is a general inverse-power law, not just `CoulombRepulsion`; real code also has non-cataloged `AverageEdgeLengthInit`, `IdealLengthDecay`, and a fused “normalize-force then move by fixed step” update.

**FMMM**
Source: [fmmm.py](/home/jtaylor/projects/dagua/dagua/layout/classic/fmmm.py#L1341)

1. Validate inputs; choose output device; early-return for `N=0/1`; compute final `extent` and `refinement_area = (2*extent)^2`.
2. Build a fine-to-coarse hierarchy: collapse to unique undirected edges with averaged desired lengths and summed weights, then repeatedly solar-system coarsen while `current_nodes > 50`.
3. In each solar-system coarsen step, compute star masses, maintain `_RandomNodeSet`, repeatedly choose a sun by multi-try random sampling biased toward high star mass, assign adjacent planets, remove planets and their neighbors from the selectable set, then assign leftover nodes as moons or new suns.
4. Aggregate coarse edges/desired lengths/weights and store `_HierarchyStep(mapping, node_types, dedicated_sun, dedicated_sun_distance, pm_nodes, moon_children, lambda_values, neighbor_suns)`.
5. Initialize the coarsest graph by calling `layout_fr()` with `steps=max(50, steps)` and the same `seed`; unwrap traces if that call returns them.
6. Start a fresh `random.Random(seed)` for prolongation and set `level_budget = max(10, steps // len(levels))`.
7. If there is a hierarchy, refine the coarsest layout with per-edge desired lengths, Barnes-Hut or exact repulsion, attractive forces, and temperature-limited movement.
8. Uncoarsen level by level: copy coarse positions by mapping, keep suns exact, place planets/moons by lambda interpolation toward neighbor suns with 5% random waggle or random sector fallback, place planet-with-moons nodes from moon and neighbor-sun candidates, barycenter those candidates, then refine again with per-edge desired lengths.
9. If no hierarchy was built, run one single-level refinement with uniform ideal-length attraction.
10. Center and scale the final coordinates to `extent`.

Config: `edge_index: torch.Tensor`, `num_nodes: int`, `node_sizes: Optional[torch.Tensor]=None`, `steps: int=100`, `seed: int=42`, `edge_weights: Optional[torch.Tensor]=None`.

State: `_LevelGraph(edge_index[2,E], edge_lengths[E], num_nodes, edge_weights[E])`; `_HierarchyStep(mapping[N], node_types[list[int]], dedicated_sun[list[int]], dedicated_sun_distance[list[float]], pm_nodes[list[int]], moon_children[list[list[int]]], lambda_values[list[list[float]]], neighbor_suns[list[list[int]]])`; `_RandomNodeSet(nodes, positions, last_selectable_index, star_masses)`; `_QuadCell(bounds, indices, center_of_mass[2], mass, children)`; `positions[N,2]`; `repulsive[N,2]`; `attractive[N,2]`.

Solver: multilevel FM^3-style force-directed refinement. Coarsest init is delegated to FR; refinement uses exact or Barnes-Hut repulsion plus spring attraction, then clamps each node’s displacement to a decaying temperature.

Convergence: hierarchy stops when node count drops to `<=50`, coarsening fails to shrink, or edge-count reduction is too weak for too many consecutive levels; refinement is fixed-step only, with no early stop besides `steps <= 0`.

Shared motifs: `FromAlgorithmInit`, `BarnesHutForce`, `MovementClamp`, `ExponentialCool`, `NormalizePositions` with SFDP; `RandomUniformInit` indirectly via FR.

Hardcoded candidates: `_COARSE_TARGET=50`; `_MAX_TREE_DEPTH=10`; `_COOLING_FACTOR=0.99`; `_SOLAR_RANDOM_TRIES=20`; `_WAGGLE_FACTOR=0.05`; exact-repulsion cutoff `N<=500`; `theta=1.0` hardcoded in `layout_fmmm`; `level_budget=max(10, steps//levels)`; FR bootstrap `steps=max(50, steps)`; hierarchy quality guard `current_edge_count > 0.8 * previous_edge_count` for 6 bad levels; helper `_hierarchy()` hardcodes `seed=42`.

RNG: three independent seeded streams.
1. `_build_hierarchy()` creates `rng = random.Random(seed)`.
2. Every sun selection calls `rng.randint(0, last_try_index)` exactly `min(_SOLAR_RANDOM_TRIES, active_nodes)` times inside `_RandomNodeSet.get_random_node_with_highest_star_mass()`.
3. `layout_fr(..., seed=seed)` then creates a separate NumPy `RandomState(seed)` and calls `rand(num_nodes, 2)` once for coarsest initialization.
4. After FR, `layout_fmmm()` creates a new independent `random.Random(seed)` for prolongation.
5. Every `_waggled_inbetween_position()` consumes two `rng.random()` calls: one for radius scaling, one for angle in `_create_random_position()`.
6. Every fallback `_create_random_position()` consumes one `rng.random()` call for angle.
7. Candidate evaluation order is deterministic: ascending node id, then stored `moon_children` / `lambda_values` / `neighbor_suns` order from coarsening.

Catalog: matches `FromAlgorithmInit`, `SolarSystemCoarsen`, `LambdaInterpolation`, `BarnesHutForce`, `CoulombRepulsion`, `SpringAttraction`, `MovementClamp`, `ExponentialCool`, `NormalizePositions`. Missing/mismatch: real code needs a separate `DesiredLengthSpringAttraction`; prolongation is more than `LambdaInterpolation` because it also does random sector placement and barycentric blending.

**Davidson-Harel**
Source: [davidson_harel.py](/home/jtaylor/projects/dagua/dagua/layout/classic/davidson_harel.py#L342)

1. Validate inputs; choose output device; early-return for `N=0/1`.
2. Estimate drawing `extent` from `sqrt(N)` and optional `node_sizes`.
3. Initialize random positions uniformly in `[-extent, extent]^2`.
4. Collapse the input into unique undirected edges and summed weights.
5. Compute the current energy as five normalized terms: inverse pairwise node distribution, border repulsion, weighted squared edge lengths, edge crossings, and node-edge proximity.
6. Set `initial_temperature = max(0.1 * current_energy, 1e-3)` and create a fresh seeded CPU generator.
7. For each round, perform exactly `num_nodes` move attempts: sample one node, sample a random 2D delta scaled by current temperature, clamp the moved node to the box, recompute full energy, and accept downhill moves immediately.
8. For uphill moves, compute `exp(-delta_E / temperature)` and accept with a random threshold draw.
9. Multiply temperature by `0.75` after each round, then center and scale the final positions to `extent`.

Config: `edge_index: torch.Tensor`, `num_nodes: int`, `node_sizes: Optional[torch.Tensor]=None`, `rounds: int=100`, `seed: int=42`, `edge_weights: Optional[torch.Tensor]=None`.

State: `positions[N,2]`; `edges[list[(src,dst)]]`; `unique_edge_weights[E_u]`; `candidate[N,2]`; energy scalars; `border_distances[N,4]`; pair-index tensors from `torch.triu_indices`; `penalties[list[torch.Tensor]]`.

Solver: simulated annealing with single-node Metropolis proposals; no gradient or optimizer object.

Convergence: fixed `rounds`; no early stopping. Total proposals are exactly `rounds * num_nodes`.

Shared motifs: `RandomUniformInit` with SFDP; `BoundaryClamp`, `ExponentialCool`, centering/scaling with the force-directed layouts.

Hardcoded candidates: `_BORDER_WEIGHT=0.1`; `_EDGE_LENGTH_WEIGHT=0.2`; `_CROSSING_WEIGHT=2.0`; `_NODE_EDGE_WEIGHT=0.5`; `_COOLING_FACTOR=0.75`; move scale factor `0.25`; initial-temperature factor `0.1`; one move per node per round; `_COLLINEAR_EPSILON=1e-10`; `_MIN_DISTANCE=1e-3`.

RNG: two independent CPU `torch.Generator`s, both seeded with `seed`.
1. `_initialize_positions()` creates a generator, `manual_seed(seed)`, then calls `torch.rand((num_nodes, 2), generator=generator)` once.
2. `layout_davidson_harel()` creates a second generator, `manual_seed(seed)` again.
3. Every proposal consumes `torch.randint(0, num_nodes, (1,), generator=generator)` once, then `torch.rand((2,), generator=generator)` once.
4. Only uphill moves consume an additional `torch.rand((1,), generator=generator)` for the Metropolis threshold.
5. Because the second generator is re-seeded, proposal randomness is independent of initialization randomness.

**Cross-Algorithm Patterns**
Source: [DESIGN.md](/home/jtaylor/projects/dagua/dagua/layout/ops/DESIGN.md#L217)

- Reused motifs in real code: deterministic CPU-side preprocessing, explicit device selection for final tensors, edge deduplication/aggregation before force models, multilevel `coarsen -> solve coarse -> prolong -> refine`, final centering/scaling, and “temperature/step-limited movement” instead of optimizer objects.
- Strong catalog matches: Sugiyama’s layering/order/coordinate/route ops; RT’s Walker layout op; SFDP’s `HeavyEdgeMatching`, `BuildQuadTree`, `BarnesHutForce`, `AdaptiveCool`; FMMM’s `FromAlgorithmInit`, `SolarSystemCoarsen`, `LambdaInterpolation`; Davidson-Harel’s `RandomUniformInit`, `BoundaryClamp`, `EnergyFn(5-term)`.
- Clear missing ops: `GreedyFeedbackArcFallback` for Sugiyama; `RootSelection`, `BFSForestExtract`, `ComponentPack`, and `AxisSwap` for RT; `AverageEdgeLengthInit`, `InversePowerRepulsion`, and `IdealLengthDecay` for SFDP; `DesiredLengthSpringAttraction`, `RandomSectorPlacement`, and `HierarchyQualityGuard` for FMMM; `RandomMoveProposal` and `MetropolisAcceptance` for Davidson-Harel.
- Ops that should be split: `EnergyFn(5-term)` should become five loss ops plus a weighted combiner; `LambdaInterpolation` should split from FM^3’s random waggle/fallback placement; `SpringAttraction` should split into uniform-`k` and per-edge-desired-length variants; SFDP’s prolongation should separate `DirectMapping`, `NeighborSmoothing`, and `Jitter`.
- Ops that should be merged or clarified: `NormalizePositions` already includes centering in these implementations, so `CenterPositions`/`ScalePositions`/`NormalizePositions` need stricter boundaries or a merged “center+normalize” op; RT’s cataloged `HorizontalFlip` should be replaced by a more accurate axis-transform op.

Codex session ID: 019d4fcd-bfa8-7a40-8f25-630631900c30
Resume in Codex: codex resume 019d4fcd-bfa8-7a40-8f25-630631900c30
