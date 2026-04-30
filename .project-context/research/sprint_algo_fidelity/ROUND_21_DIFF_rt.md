# Round 21 diff: `classic_rt` vs `igraph_rt`

Diagnosis-only adversarial diff for the Reingold-Tilford (`rt`) family. No source
changes were made.

## 1. Files read

Dagua implementation and wiring:

- `dagua/layout/ops/coordinate.py`
  - shared constants and imports: lines 1-32
  - `_node_spacing()`: lines 983-1007
  - `_root_candidates()`: lines 1010-1033
  - `_bfs_forest()`: lines 1036-1080
  - Walker/Buchheim helper implementation: lines 1083-1339
  - `ReingoldTilfordTreeConfig`: lines 1342-1372
  - `BucheimWalkerTree` neighbor implementation for contrast: lines 1619-1706
  - `ReingoldTilfordTree`: lines 1709-1815
- `dagua/layout/ops/pipelines/reingold_tilford.py`
  - pipeline constructor and public function: lines 1-104
- `dagua/layout/ops/graph_utils.py`
  - `layout_device()`: lines 136-158
  - `build_undirected_adjacency()`: lines 226-268
- `dagua/layout/ops/state.py`
  - `LayoutProblem`: lines 113-165
- `dagua/eval/competitors/classic_competitor.py`
  - `classic_rt` function mapping: lines 199-203
  - `ClassicRT` adapter: lines 1151-1180
  - `_run_classic_layout()` call path: lines 1600-1620
- `dagua/eval/competitors/igraph_competitor.py`
  - `_graph_to_igraph()`: lines 53-76
  - `_igraph_pos_to_tensor()`: lines 79-99
  - generic igraph adapter call path: lines 135-184
  - `IgraphRT`: lines 214-220
- `dagua/eval/variants.py`
  - `classic_rt_default` variant pair: lines 1221-1231
  - `classic_rt_horizontal` unpaired variant: lines 1232-1242
  - stochasticity/heavy flags: lines 1820-1875
- `eval_output/fidelity_report/report.md`
  - current verdict row for `rt_default`: line 70
  - artifact index: lines 194-198
- `eval_output/fidelity_report/data/algorithm_summary.csv`
  - `rt_default` row: `strong_equivalent`, 103 paired OK, 2 insufficient, median
    Procrustes RMSD 0.06471293978393078, max 0.4112982749938965.
- `eval_output/fidelity_report/data/per_graph_detail.csv`
  - `rt_default` rows for high-residual examples and igraph failures, including
    `center_port_backedge_hub`, `edge_label_braid`, `bipartite_4_3_4`,
    `grid_20x20`, and `grid_50x50`.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`
  - sprint context/headline result: lines 1-45.

Reference implementation and binding surface:

- `/home/jtaylor/projects/_references/igraph/src/layout/reingold_tilford.c`
  - unreachable-node edge insertion: lines 35-103
  - internal vertex state: lines 106-121
  - internal single-root RT layout: lines 132-227
  - coordinate propagation: lines 229-244
  - postorder contour algorithm: lines 246-443
  - directed cluster degree root helper: lines 448-487
  - automatic root selection: lines 489-661
  - public `igraph_layout_reingold_tilford()`: lines 713-937
  - circular variant, not used by current Dagua pairing: lines 973-1011
- Runtime Python igraph binding inspection:
  - `igraph.Graph.layout_reingold_tilford.__doc__` in installed igraph 1.0.0.
    The docstring exposes `mode`, `root`, and `rootlevel`, and states automatic
    root selection prefers low eccentricity for graphs with fewer than 500
    vertices and high degree for larger graphs. This docstring is consistent in
    intent with the C comment at lines 690-699, although the checked C code's
    ternary appears inverted at lines 750-755. See Section 11.

## 2. Overall pipeline structure

Dagua `classic_rt`:

1. `classic_rt` is registered in the generic classic adapter with import path
   `dagua.layout.ops.pipelines.reingold_tilford` and function
   `layout_reingold_tilford_pipeline` (`classic_competitor.py:199-203`).
2. `ClassicRT.layout()` simply delegates to `layout_with_variant()` and declares
   the layout deterministic (`classic_competitor.py:1151-1180`).
3. `_run_classic_layout()` imports the function dynamically, forwards
   `edge_index`, `graph.num_nodes`, `graph.node_sizes`, `seed`, and optional
   `edge_weights` (`classic_competitor.py:1600-1619`).
4. `layout_reingold_tilford_pipeline()` validates only `num_nodes` and
   `edge_weights`, creates a `LayoutProblem`, creates a CPU `RuntimeContext`,
   applies a one-op pipeline, and returns `final_state.pos`
   (`pipelines/reingold_tilford.py:41-101`).
5. `build_reingold_tilford_pipeline()` contains exactly one op:
   `ReingoldTilfordTree(ReingoldTilfordTreeConfig(horizontal=horizontal))`
   (`pipelines/reingold_tilford.py:19-38`).
6. `ReingoldTilfordTree.apply()` derives spacing, builds a BFS forest over an
   undirected adjacency, assigns preliminary x positions component by component,
   builds `float32` positions, mean-centers the layout, optionally swaps axes for
   horizontal output, and returns `float32` on the chosen input device
   (`coordinate.py:1754-1815`).

igraph `igraph_rt`:

1. `IgraphRT` maps to `layout_algo = "reingold_tilford"` with no explicit
   kwargs (`igraph_competitor.py:214-220`).
2. `_graph_to_igraph()` always constructs an igraph `Graph(directed=True)`,
   adds all Dagua edges in edge-index order, and copies edge weights only as an
   attribute (`igraph_competitor.py:53-76`).
3. `_IgraphBase.layout_with_variant()` calls `ig.layout(self.layout_algo,
   **kwargs)` with no root, rootlevel, or mode override for RT
   (`igraph_competitor.py:163-179`).
4. `_igraph_pos_to_tensor()` converts the returned igraph layout to a CPU
   torch tensor and multiplies both coordinates by 50 (`igraph_competitor.py:79-99`).
5. The C implementation, once entered, normalizes undirected inputs to
   `IGRAPH_ALL` only if the graph is not directed (`reingold_tilford.c:738-740`).
   Because the adapter always creates a directed graph (`igraph_competitor.py:68`),
   the Python binding's default `mode` determines whether the internal BFS uses
   outgoing, incoming, or all edges.
6. The public C wrapper obtains roots automatically when none are supplied
   (`reingold_tilford.c:742-756`), may add artificial root/rootlevel vertices
   (`reingold_tilford.c:760-885`), adds shortcut edges to make unreachable
   nodes reachable from the chosen real root (`reingold_tilford.c:887-899`),
   runs the internal single-root algorithm (`reingold_tilford.c:903-905`), then
   removes artificial vertices from the result (`reingold_tilford.c:907-923`).

Main structural mismatch:

- Dagua is a deterministic forest layout built from an undirected BFS spanning
  forest (`coordinate.py:1036-1080`), then component-packed and centered
  (`coordinate.py:1792-1810`).
- igraph is a directed-capable layout with automatic root selection,
  artificial super-rooting, and unreachable shortcut insertion before running a
  single-root contour algorithm (`reingold_tilford.c:713-937`).

The mega-run still marks the family `strong_equivalent` (`report.md:70`), but
the residual median RMSD is nonzero (0.0647), the max is 0.4113, and 103/103
paired graphs are marked anomaly graphs in the algorithm summary. That means the
pair is visually/functionally close under the current equivalence test, not bit
or algorithm equivalent.

## 3. Energy / loss / objective

Neither side has an iterative energy, loss scalar, or optimizer objective.
This is a combinatorial coordinate assignment algorithm.

Dagua objective-equivalent terms:

- Sibling/subtree separation is encoded as a fixed Walker distance of `1.0`
  (`_WALKER_BASE_DISTANCE = 1.0`, `coordinate.py:30`).
- Component padding is encoded as `_COMPONENT_PADDING = 1.0`
  (`coordinate.py:31`).
- If explicit `sibling_sep` is absent, Dagua uses
  `max(max_node_width * 1.5, 1.0)` (`coordinate.py:983-1007`,
  `coordinate.py:1763-1769`).
- If explicit `layer_sep` is absent, Dagua uses
  `max(max_node_height * 1.5, 1.5)` (`coordinate.py:983-1007`,
  `coordinate.py:1770-1774`).
- If explicit `component_gap` is absent, Dagua uses
  `sibling_sep * 2.0` (`ReingoldTilfordTreeConfig.default_component_gap_multiplier`,
  `coordinate.py:1356-1371`; application at `coordinate.py:1775-1779`).
- Contour overlap resolution is the Buchheim/Walker style inequality
  `shift = left_contour_x - right_contour_x + distance`; if positive, move the
  right subtree by `shift` (`coordinate.py:1245-1251`).
- Internal child centering is the midpoint of first and last child prelim x:
  `0.5 * (first.prelim + last.prelim)` (`coordinate.py:1282-1290`).

igraph objective-equivalent terms:

- Minimum separation is a hard constant:
  `const igraph_real_t minsep = 1` (`reingold_tilford.c:250-252`).
- For each non-first child, initial proposed root separation is
  `vdata[leftroot].offset + minsep` (`reingold_tilford.c:302-304`).
- During contour traversal, if both contours remain active and
  `roffset - loffset < minsep`, igraph increases `rootsep` by
  `minsep - roffset + loffset` and sets `roffset = loffset + minsep`
  (`reingold_tilford.c:388-395`).
- Parent centering is the arithmetic average of child root offsets:
  `avg = (avg * j) / (j + 1) + rootsep / (j + 1)`, followed by subtracting
  `avg` from child offsets and contour/extreme offsets
  (`reingold_tilford.c:402-440`).

Objective divergences:

- Dagua uses midpoint centering between leftmost and rightmost child after
  Walker shifts (`coordinate.py:1282-1290`); igraph uses the mean of immediate
  child offsets (`reingold_tilford.c:402-440`). For binary nodes these can
  coincide; for higher fanout and contour-threaded subtrees they can differ.
- Dagua's default scale is node-size-aware (`coordinate.py:1763-1779`);
  igraph's internal tree unit is a constant `1` (`reingold_tilford.c:250-252`)
  and the adapter scales the final layout by a fixed `50.0`
  (`igraph_competitor.py:94-99`).
- Dagua explicitly separates independent BFS components with component padding
  and component gap (`coordinate.py:1792-1802`); igraph instead uses synthetic
  roots/edges to connect roots/unreachable parts before layout
  (`reingold_tilford.c:853-899`).

## 4. Force / gradient computation

Not applicable. Dagua `ReingoldTilfordTree` is an `Op` that directly writes
`state.pos` and does not create a tensor requiring gradients
(`coordinate.py:1710-1815`). The pipeline contains no `LossOp`, optimizer,
annealing op, or repeat block (`pipelines/reingold_tilford.py:35-38`).

igraph similarly performs BFS, recursive postorder contour placement, and a
recursive top-down coordinate propagation (`reingold_tilford.c:172-199`,
`reingold_tilford.c:229-244`, `reingold_tilford.c:246-443`). There is no force
or gradient path.

## 5. Initialization

Dagua:

- No random initialization. `seed` is accepted and explicitly ignored
  (`pipelines/reingold_tilford.py:45-60`, `pipelines/reingold_tilford.py:78`).
- Starts with a validated CPU long `edge_index` (`coordinate.py:1754-1756`).
- Empty graph initializes an empty `[0, 2]` `float32` tensor on the output device
  (`coordinate.py:1757-1761`).
- Root ordering is deterministic by `(indegree != 0, indegree, node_id)`
  (`coordinate.py:1010-1033`).
- BFS starts each unvisited root in that order and records children in sorted
  undirected adjacency order (`coordinate.py:1036-1080`;
  adjacency sorting at `graph_utils.py:226-268`).
- Preliminary x values start as `[0.0] * num_nodes` and component offset starts
  at `0.0` (`coordinate.py:1792-1794`).

igraph:

- No random initialization for RT. The `IgraphRT` adapter does not set
  `uses_igraph_rng` and does not pass a seed matrix (`igraph_competitor.py:214-220`,
  generic RNG only at `igraph_competitor.py:170-178`).
- `_graph_to_igraph()` always creates a directed graph and inserts all vertices
  before edges (`igraph_competitor.py:68-76`).
- Public C wrapper initializes `newedges` and determines mode/root/rootlevel
  behavior (`reingold_tilford.c:735-756`).
- Internal single-root implementation allocates `vdata`, initializes every
  parent/level/offset/contour/extreme field, and then sets the real root's
  parent to itself and level to zero (`reingold_tilford.c:149-170`).
- BFS initialization pushes `(root, 0)` into an integer deque
  (`reingold_tilford.c:172-175`).

Initialization divergence:

- Dagua picks candidate roots from indegree first (`coordinate.py:1010-1033`).
  igraph's documented automatic roots depend on graph size and degree/eccentricity
  heuristics (`reingold_tilford.c:690-699`; binding docstring), with directed
  component logic in `igraph_roots_for_tree_layout()` (`reingold_tilford.c:489-661`).
- Dagua uses an undirected adjacency regardless of graph direction
  (`coordinate.py:1054-1058`). igraph is mode-sensitive (`reingold_tilford.c:684-689`)
  and the adapter supplies a directed graph (`igraph_competitor.py:68`).

## 6. Iteration / convergence

Dagua:

- One pass through a single-op pipeline (`pipelines/reingold_tilford.py:35-38`,
  `pipelines/reingold_tilford.py:98`).
- The recursive Walker implementation has:
  - postorder first walk (`coordinate.py:1270-1290`);
  - contour apportioning while both inner contours continue
    (`coordinate.py:1226-1267`);
  - top-down second walk (`coordinate.py:1293-1311`);
  - per-component assignment (`coordinate.py:1314-1339`).
- There is no convergence test, learning-rate schedule, or iteration limit.
  Recursion limit is raised to `num_nodes * 2` by default
  (`coordinate.py:1361-1372`, `coordinate.py:1781-1786`).

igraph:

- Public wrapper performs a bounded sequence: automatic roots, optional graph
  extension, unreachable shortcuting, internal layout, artificial vertex removal
  (`reingold_tilford.c:742-936`).
- Internal layout performs one BFS from `real_root`
  (`reingold_tilford.c:172-192`), one recursive postorder placement
  (`reingold_tilford.c:194-196`, `reingold_tilford.c:246-443`), and one recursive
  coordinate propagation (`reingold_tilford.c:198-199`, `reingold_tilford.c:229-244`).
- No convergence or annealing.

## 7. Hyperparameter alignment table

| Parameter / semantic | Dagua default | igraph/reference default | Match? | Evidence |
|---|---:|---:|---|---|
| Algorithm family | Reingold-Tilford/Walker tidy tree | Reingold-Tilford tidy tree | Partial | Dagua one-op RT pipeline (`pipelines/reingold_tilford.py:35-38`); igraph RT docs/source (`reingold_tilford.c:664-680`) |
| Root argument | Not exposed | `root`/`roots` exposed; auto if absent | No | Dagua config has no root field (`coordinate.py:1342-1372`); igraph root param docs (`reingold_tilford.c:690-699`) |
| Root selection | Sort by `(indegree != 0, indegree, node_id)` | Automatic roots by degree/eccentricity and components | No | Dagua (`coordinate.py:1010-1033`); igraph (`reingold_tilford.c:489-661`, `reingold_tilford.c:750-756`) |
| Direction/mode | Always undirected BFS | Mode can be OUT/IN/ALL; directed graph keeps directed mode | No | Dagua (`coordinate.py:1054-1058`); igraph (`reingold_tilford.c:684-689`, `reingold_tilford.c:738-740`) |
| Graph directedness in adapter | Dagua op ignores direction | igraph adapter creates `Graph(directed=True)` | No | `igraph_competitor.py:68`; Dagua `_bfs_forest()` ignores direction (`coordinate.py:1036-1080`) |
| Min sibling separation | Base Walker distance 1.0, later scaled | `minsep = 1` | Yes at tree-unit level | Dagua constant (`coordinate.py:30`); igraph (`reingold_tilford.c:250-252`) |
| Node-size-aware sibling spacing | `max(width) * 1.5`, min 1.0 | None in C; adapter only final scale 50 | No | Dagua (`coordinate.py:983-1007`, `coordinate.py:1763-1769`); igraph minsep (`reingold_tilford.c:250-252`), adapter scale (`igraph_competitor.py:94-99`) |
| Layer spacing | `max(height) * 1.5`, min 1.5 | BFS depth increments by 1 | No | Dagua (`coordinate.py:1770-1774`, `coordinate.py:1804-1808`); igraph (`reingold_tilford.c:186-190`) |
| Component handling | Lay out each BFS component independently with gap | Add synthetic real root/edges and remove artificial rows | No | Dagua (`coordinate.py:1792-1802`); igraph (`reingold_tilford.c:853-923`) |
| Component gap | `2.0 * sibling_sep` when unset | Emerges from synthetic root tree geometry | No | Dagua (`coordinate.py:1356-1371`, `coordinate.py:1775-1779`); igraph (`reingold_tilford.c:871-899`) |
| Parent centering | midpoint of first/last child prelim | average of child offsets | Partial/No | Dagua (`coordinate.py:1282-1290`); igraph (`reingold_tilford.c:402-440`) |
| Child traversal order | BFS child list from sorted undirected adjacency | scan all vertex ids for `parent == node` | Often yes by id, not guaranteed semantically | Dagua (`graph_utils.py:260-268`, `coordinate.py:1070-1078`); igraph (`reingold_tilford.c:258-268`, `reingold_tilford.c:288-292`) |
| Multi-edge handling | Duplicate edges accumulated then neighbor deduped | adjacency with `IGRAPH_NO_MULTIPLE` | Mostly yes for topology | Dagua (`graph_utils.py:250-268`); igraph (`reingold_tilford.c:56`, `reingold_tilford.c:146`) |
| Self-loop handling | Ignored in undirected adjacency | adjacency with `IGRAPH_NO_LOOPS` | Yes | Dagua (`graph_utils.py:263-264`); igraph (`reingold_tilford.c:56`, `reingold_tilford.c:146`) |
| Edge weights | Accepted and validated but not used for topology | copied as attribute but not used by RT | Yes | Dagua (`pipelines/reingold_tilford.py:47-95`, `_bfs_forest()` passes `edge_weights=None` at `coordinate.py:1054-1058`); igraph adapter (`igraph_competitor.py:74-75`), C ignores weights |
| Output dtype | `torch.float32` | C `igraph_real_t` then Python doubles/lists then torch default float32 assignment | Close but not identical | Dagua (`coordinate.py:1804-1815`); adapter initializes `torch.zeros()` float32 and assigns layout values (`igraph_competitor.py:94-99`) |
| Output scaling | Native node-size/tree units, centered | igraph tree units multiplied by 50, not centered by adapter | No | Dagua (`coordinate.py:1804-1814`); igraph adapter (`igraph_competitor.py:94-99`) |
| Horizontal option | Exposed by Dagua only; unpaired in variants | circular exists, not horizontal swap | Not applicable to paired default | Dagua variant unpaired (`variants.py:1232-1242`); circular C variant (`reingold_tilford.c:973-1011`) |
| RNG/seed | accepted, ignored | no RNG for RT in current adapter | Yes | Dagua (`pipelines/reingold_tilford.py:45-60`, `pipelines/reingold_tilford.py:78`); igraph adapter (`igraph_competitor.py:214-220`) |
| Empty graph | returns empty `[0, 2]` tensor | returns success; roots cleared/layout resized | Yes | Dagua (`coordinate.py:1757-1761`); igraph roots empty (`reingold_tilford.c:561-563`), circular empty guard (`reingold_tilford.c:985-986`) |

## 8. Edge cases

Self-loops:

- Dagua skips `source == target` in shared undirected adjacency construction
  (`graph_utils.py:260-264`).
- igraph adjacency is initialized with `IGRAPH_NO_LOOPS` in both unreachable BFS
  and internal layout BFS (`reingold_tilford.c:56`, `reingold_tilford.c:146`).
- Expected parity: high.

Multi-edges:

- Dagua's adjacency map dedupes neighbors and accumulates duplicate weights
  (`graph_utils.py:250-268`). RT then discards weights by passing
  `edge_weights=None` (`coordinate.py:1054-1058`).
- igraph adjacency is initialized with `IGRAPH_NO_MULTIPLE`
  (`reingold_tilford.c:56`, `reingold_tilford.c:146`).
- Expected parity: topology-level parity, but Dagua indegree root scoring counts
  duplicate target appearances before adjacency dedupe (`coordinate.py:1025-1029`).
  igraph degree ordering likely uses igraph's degree semantics from the original
  graph at `igraph_sort_vertex_ids_by_degree()` (`reingold_tilford.c:579-582`).
  This can change roots for multigraph-like inputs even if the BFS tree later
  dedupes neighbors.

Disconnected components:

- Dagua lays out every unvisited root as a separate component and advances
  `component_offset + width + _COMPONENT_PADDING + component_gap`
  (`coordinate.py:1064-1080`, `coordinate.py:1314-1339`,
  `coordinate.py:1792-1802`).
- igraph creates a synthetic real root when there are multiple roots
  (`reingold_tilford.c:853-885`) and adds shortcut edges for vertices still
  unreachable from that root (`reingold_tilford.c:887-899`).
- This is likely the largest residual source for graphs such as
  `disconnected_encoder_residual` (RMSD 0.2911, scale ratio 70.21) and
  `disconnected_label_cycle_collage` (RMSD 0.2437, scale ratio 134.68) in
  `per_graph_detail.csv`.

Weighted edges:

- Dagua validates `edge_weights` shape and stores it in `LayoutProblem`
  (`pipelines/reingold_tilford.py:82-95`) but `_bfs_forest()` intentionally
  passes `edge_weights=None` to the adjacency builder (`coordinate.py:1054-1058`).
- igraph adapter stores weights as an edge attribute (`igraph_competitor.py:74-75`),
  but the C RT implementation has no weights parameter (`reingold_tilford.c:713-717`).
- Expected parity: weights should not affect RT. Residuals on
  `heavy_tail_weights_50`, `weighted_clusters_3x10`, and `weighted_karate_34`
  are therefore topology/root/spacing issues, not weight-force issues.

Empty graph:

- Dagua returns an empty `float32` tensor on output device (`coordinate.py:1757-1761`).
- igraph root helper clears roots for zero nodes (`reingold_tilford.c:561-563`);
  circular wrapper explicitly exits on zero nodes after calling RT
  (`reingold_tilford.c:983-986`).
- Expected parity: high unless Python binding throws before C for empty directed
  graphs; no evidence of this from current report.

Cyclic/non-tree graphs:

- Dagua always builds a BFS spanning forest from undirected adjacency
  (`coordinate.py:1036-1080`).
- igraph documentation/source says non-trees first get a BFS spanning tree
  (`reingold_tilford.c:678-679`), but the BFS is mode-sensitive and rooted from
  automatic roots (`reingold_tilford.c:172-192`, `reingold_tilford.c:684-699`).
- High residual examples are mostly cyclic/small: `center_port_backedge_hub`
  RMSD 0.4113, `edge_label_braid` RMSD 0.3877, `bipartite_4_3_4` RMSD 0.3080,
  `recurrent_feedback_cell` RMSD 0.2898, `complete_bipartite_8x12` RMSD 0.2650.

## 9. Numerical precision

Dagua:

- Uses Python floats for Walker prelim/mod/shift/change fields
  (`_WalkerNode`, `coordinate.py:1083-1099`). Python float is IEEE double.
- Converts final positions to `torch.float32` when creating the output tensor
  (`coordinate.py:1804-1815`).
- Node spacing converts input node sizes to CPU `torch.float32` before taking
  maxima (`coordinate.py:1004-1007`).
- Final centering uses `positions.mean()` in torch float32
  (`coordinate.py:1809`).

igraph:

- C uses `igraph_real_t` for offsets and matrices; in normal igraph builds this
  is double precision. Relevant fields are `offset`, contour offsets, and
  extreme offsets (`reingold_tilford.c:106-121`).
- Postorder averaging and contour arithmetic are in `igraph_real_t`
  (`reingold_tilford.c:250-252`, `reingold_tilford.c:297-304`,
  `reingold_tilford.c:388-404`).
- Python adapter then writes values into a default `torch.zeros(num_nodes, 2)`
  tensor, which is float32 under torch defaults, multiplying by `50.0`
  (`igraph_competitor.py:94-99`).

Precision divergence:

- Both outputs end as float32 tensors in the adapters, but Dagua centers after
  float32 materialization (`coordinate.py:1804-1814`), while igraph has already
  completed double-precision recursive arithmetic before the adapter's float32
  conversion (`igraph_competitor.py:94-99`).
- Summation/centering order differs: Dagua subtracts torch's vectorized mean over
  all nodes (`coordinate.py:1809`); igraph does not center in the adapter, so any
  later Procrustes centering occurs in evaluation code rather than in the layout
  output.
- These are likely sub-percent contributors on tree-like graphs after
  Procrustes alignment. They do not explain the 0.2-0.4 RMSD cases.

## 10. RNG semantics

There is no meaningful RNG alignment question for the current paired default.

- Dagua accepts `seed` for interface compatibility and discards it
  (`pipelines/reingold_tilford.py:45-60`, `pipelines/reingold_tilford.py:78`).
- `variants.py` marks `classic_rt` non-stochastic (`variants.py:1820-1835`).
- `IgraphRT` does not set `uses_igraph_rng` and does not set
  `accepts_seed_matrix` (`igraph_competitor.py:214-220`). The generic igraph RNG
  context is only entered with a custom Python RNG when `uses_igraph_rng` is
  true (`igraph_competitor.py:18-50`, `igraph_competitor.py:170-178`).
- `variants.py` does not include `igraph_rt` in the stochastic engines list
  (`variants.py:1820-1868`).

Answer to the specific prompt: Dagua's torch seed cannot produce the same
sequence as reference RNG because neither side consumes a sequence for RT in
this pairing. The seed has no algorithmic effect on either adapter. Residuals
are deterministic topology/geometry differences.

## 11. Edge-case bugs and suspicious divergences

1. **Likely Dagua-vs-igraph root heuristic mismatch.**
   Dagua root candidates prefer zero-indegree, then low indegree, then low id
   (`coordinate.py:1010-1033`). igraph automatic roots are component-aware and
   use degree/eccentricity ordering (`reingold_tilford.c:489-661`). For directed
   cyclic inputs, this changes the BFS tree before coordinates are assigned.
   Expected RMSD impact: high on cyclic/non-tree graphs.

2. **Mode/direction mismatch.**
   Dagua ignores edge direction by using `_build_undirected_adjacency()`
   (`coordinate.py:1054-1058`). The igraph adapter creates a directed graph
   (`igraph_competitor.py:68`) and the C API exposes mode-dependent traversal
   (`reingold_tilford.c:684-689`). If Python igraph defaults to `OUT`, Dagua is
   fundamentally using a different BFS graph. Expected impact: high on directed
   DAG/cyclic graphs, low on symmetric/undirected-like inputs.

3. **Potential reference-source/comment inconsistency around root heuristic.**
   The installed Python igraph 1.0.0 docstring says automatic root selection
   prefers low eccentricity for graphs with fewer than 500 vertices and high
   degree for larger graphs. The C comment says the same (`reingold_tilford.c:690-699`).
   The checked C code calls
   `no_of_nodes < 500 ? IGRAPH_ROOT_CHOICE_DEGREE : IGRAPH_ROOT_CHOICE_ECCENTRICITY`
   (`reingold_tilford.c:750-756`), which appears inverted relative to the text.
   This needs confirmation against the exact compiled igraph C source/version.
   It is not a Dagua bug by itself, but it is a fidelity trap.

4. **Dagua component layout is not igraph super-root layout.**
   Dagua independently packs components with a fixed component gap
   (`coordinate.py:1792-1802`). igraph synthesizes a root and edges, lays out one
   connected augmented graph, then deletes artificial rows (`reingold_tilford.c:853-923`).
   This likely drives huge scale-ratio anomalies on disconnected graphs in the
   report.

5. **Dagua output is node-size-aware; igraph RT is not.**
   Dagua default `sibling_sep` and `layer_sep` depend on node sizes
   (`coordinate.py:1763-1779`). igraph has `minsep = 1` and BFS level y
   increments (`reingold_tilford.c:250-252`, `reingold_tilford.c:186-190`), then
   adapter multiplies by 50 (`igraph_competitor.py:94-99`). This improves Dagua
   render quality but is not reference-faithful.

6. **Parent-centering formula differs.**
   Dagua uses midpoint of outer children (`coordinate.py:1282-1290`); igraph uses
   running average of child root offsets and shifts every child by that average
   (`reingold_tilford.c:402-440`). On non-binary or uneven subtrees, this can
   change x coordinates even if roots and BFS tree match.

7. **Artificial rootlevel semantics missing.**
   igraph supports `rootlevel` and can insert chains of artificial vertices to
   place different roots at different levels (`reingold_tilford.c:700-704`,
   `reingold_tilford.c:760-824`). Dagua exposes only `horizontal` in RT config
   (`coordinate.py:1342-1372`, `pipelines/reingold_tilford.py:41-48`).
   Current variant does not pass rootlevel, so this is not an active default
   mismatch, but it blocks full API parity.

8. **Adapter scaling hides but does not solve spacing mismatch.**
   igraph output is multiplied by 50 (`igraph_competitor.py:94-99`) while Dagua
   uses node-size-derived scale and mean-centers (`coordinate.py:1804-1814`).
   Procrustes removes uniform scale for RMSD, but metric anomalies such as aspect
   ratio and edge length CV still see non-reference proportions.

9. **Duplicate edges can affect Dagua root scoring before topology dedupe.**
   `_root_candidates()` increments indegree for every target entry in
   `edge_index` (`coordinate.py:1025-1029`), while `_bfs_forest()` later uses a
   deduped adjacency (`graph_utils.py:250-268`). igraph root ordering uses its
   own degree helper (`reingold_tilford.c:579-582`) and adjacency removes
   multiples in BFS (`reingold_tilford.c:146`). Multi-edge bundles may therefore
   root differently.

10. **Dagua recursion limit side effect.**
    Dagua globally raises Python recursion limit to at least `2 * num_nodes`
    (`coordinate.py:1781-1786`) and never restores it. This is not a layout
    divergence, but it is an observable process side effect. igraph recursion is
    in C and does not mutate Python recursion limits.

## 12. Ranked fix list

Ranked by expected reduction in paired Procrustes RMSD / metric anomalies for
`classic_rt_default` vs `igraph_rt`.

1. **Add an igraph-fidelity root/mode path in Dagua RT.**
   - Impact: highest. Root/mode determines the BFS tree before all x/y geometry.
   - Evidence: Dagua root heuristic at `coordinate.py:1010-1033`; Dagua
     undirected BFS at `coordinate.py:1054-1080`; igraph automatic roots and
     mode semantics at `reingold_tilford.c:489-661` and `reingold_tilford.c:684-699`.
   - Proposed fix: add optional config fields such as `mode`, `roots`,
     `root_choice`, and `fidelity_mode="igraph"`; implement directed BFS and
     igraph root-selection semantics for the fidelity path.
   - Size estimate: medium-large (150-300 LOC plus tests).

2. **Replicate igraph synthetic super-root/unreachable shortcut handling.**
   - Impact: very high on disconnected and partially reachable graphs.
   - Evidence: Dagua component packing at `coordinate.py:1792-1802`; igraph
     artificial root and shortcut edges at `reingold_tilford.c:853-899`, row
     removal at `reingold_tilford.c:907-923`.
   - Proposed fix: before Dagua RT coordinate assignment, build an augmented
     graph matching igraph's root/rootlevel/unreachable logic, run the internal
     tree placement, then drop artificial vertices.
   - Size estimate: medium (120-220 LOC plus disconnected graph tests).

3. **Add a reference-spacing mode that disables node-size-derived spacing and
   uses unit x/y tree spacing.**
   - Impact: medium-high; likely reduces aspect-ratio and scale-ratio anomalies.
   - Evidence: Dagua node-size spacing at `coordinate.py:983-1007` and
     `coordinate.py:1763-1779`; igraph `minsep = 1` at
     `reingold_tilford.c:250-252`; igraph y depth assignment at
     `reingold_tilford.c:186-190`; adapter scale at `igraph_competitor.py:94-99`.
   - Proposed fix: in fidelity mode, set `sibling_sep=1.0`, `layer_sep=1.0`,
     `component_gap` governed by augmented-root geometry rather than Dagua gap.
   - Size estimate: small-medium (30-80 LOC plus tests).

4. **Implement igraph's offset/contour algorithm exactly for fidelity mode.**
   - Impact: medium; important for higher fanout and uneven subtrees after
     root/mode alignment.
   - Evidence: Dagua Walker/Buchheim midpoint centering at
     `coordinate.py:1208-1290`; igraph postorder contour/extreme/threading
     algorithm at `reingold_tilford.c:246-443`.
   - Proposed fix: port `reingold_tilford_vertex` state and postorder logic
     directly into a private Dagua function used only by `fidelity_mode="igraph"`.
   - Size estimate: medium-large (200-350 LOC plus direct golden tests).

5. **Expose/pass root and rootlevel through variants/adapters for controlled tests.**
   - Impact: medium; improves reproducibility and lets benchmarks isolate root
     mismatch from coordinate mismatch.
   - Evidence: current variant passes no params (`variants.py:1221-1231`);
     Dagua RT function exposes only `horizontal` beyond graph inputs
     (`pipelines/reingold_tilford.py:41-48`); igraph supports roots/rootlevel
     (`reingold_tilford.c:690-704`).
   - Proposed fix: add optional Dagua and igraph variant params for roots and
     rootlevel, then add adversarial root-fixed fixtures.
   - Size estimate: small-medium (50-120 LOC plus eval config updates).

6. **Clarify and test Python igraph's default `mode`.**
   - Impact: medium; without this, a Dagua fidelity mode may match the wrong
     traversal semantics.
   - Evidence: adapter passes no kwargs (`igraph_competitor.py:214-220`);
     C public API requires a mode (`reingold_tilford.c:713-717`); binding
     docstring documents mode but not the default in the inspected signature.
   - Proposed fix: add explicit `layout_kwargs = {"mode": "out"}` or whatever
     runtime inspection confirms as current default, then mirror that in Dagua.
   - Size estimate: small (10-40 LOC plus a tiny comparison test).

7. **Handle duplicate-edge root scoring consistently.**
   - Impact: low-medium; targeted to multiedge and dense test cases.
   - Evidence: Dagua root scoring counts target occurrences
     (`coordinate.py:1025-1029`); Dagua BFS adjacency dedupes
     (`graph_utils.py:250-268`); igraph BFS adjacency removes multiples
     (`reingold_tilford.c:146`).
   - Proposed fix: in fidelity mode, compute root degree/order from the same
     graph representation igraph uses for root selection.
   - Size estimate: small (20-60 LOC).

8. **Move final centering/scaling behind a fidelity switch.**
   - Impact: low for Procrustes RMSD, medium for raw tensor/metric anomaly flags.
   - Evidence: Dagua mean-centers at `coordinate.py:1809`; igraph adapter only
     multiplies by 50 (`igraph_competitor.py:94-99`).
   - Proposed fix: allow fidelity mode to return uncentered igraph-like tree
     units or Dagua adapter-level scaled units consistently.
   - Size estimate: small (10-30 LOC).

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle: **root/mode + spacing fidelity, but not a full
contour rewrite yet.**

Concrete scope:

1. Add `fidelity_mode: Optional[str] = None` or equivalent private config path to
   `ReingoldTilfordTreeConfig` (`coordinate.py:1342-1372`).
2. In fidelity mode, use unit spacing instead of node-size-derived spacing
   (`coordinate.py:1763-1779`) and avoid Dagua's component-gap packing where
   igraph super-rooting applies (`coordinate.py:1792-1802`).
3. Add explicit directed traversal mode and igraph-like root selection:
   Dagua currently roots by indegree (`coordinate.py:1010-1033`) and traverses
   undirected adjacency (`coordinate.py:1054-1058`); igraph does component-aware
   root selection and mode-sensitive traversal (`reingold_tilford.c:489-661`,
   `reingold_tilford.c:684-699`).
4. Confirm runtime Python igraph default mode with a tiny local probe and encode
   that default explicitly in `IgraphRT.layout_kwargs` if needed
   (`igraph_competitor.py:214-220`).
5. Add adversarial tests for:
   - a directed arborescence where OUT vs ALL differs;
   - disconnected two-tree forest;
   - multi-root DAG;
   - duplicate-edge root scoring;
   - high-fanout tree with unequal subtree widths.

Do not include the full direct port of igraph's contour algorithm in the same
round unless root/mode/spacing fixes leave a large residual. The current max
residuals line up more strongly with BFS/root/component semantics than with
floating-point or centering details.

## Current verdict interpretation

`rt_default` is currently `strong_equivalent` in the report
(`report.md:70`) with median RMSD about 0.0647 and max about 0.4113 in
`algorithm_summary.csv`. This is consistent with two deterministic tidy-tree
algorithms that often produce the same coarse hierarchy, but it is not an exact
reimplementation of igraph RT. The dominant catalogued divergences are:

- Dagua roots and traverses a deterministic undirected BFS forest
  (`coordinate.py:1010-1080`).
- igraph roots component-aware directed graphs, may synthesize a super-root, and
  adds shortcut edges for unreachable vertices (`reingold_tilford.c:489-661`,
  `reingold_tilford.c:713-937`).
- Dagua defaults are render-aware and node-size-aware (`coordinate.py:1763-1779`);
  igraph RT is unit-separation tree geometry with adapter-level scale
  (`reingold_tilford.c:250-252`, `igraph_competitor.py:94-99`).
