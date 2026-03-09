# Dagua TODO — Post-Sprint 2 Debrief

## Runtime Scaling Results

Dagua vs Graphviz (dot engine), 50 optimization steps, CPU:

| Nodes | Edges | Dagua (s) | Graphviz (s) | Winner |
|------:|------:|----------:|-------------:|--------|
| 100 | 150 | 0.23 | 0.04 | Graphviz (5x) |
| 500 | 750 | 0.57 | 0.18 | Graphviz (3x) |
| 1,000 | 1,500 | 1.78 | 0.63 | Graphviz (3x) |
| 2,000 | 3,000 | 5.18 | 1.88 | Graphviz (3x) |
| 5,000 | 7,500 | 8.36 | 14.13 | **Dagua (1.7x)** |
| 10,000 | 15,000 | 24.81 | 82.88 | **Dagua (3.3x)** |
| 20,000 | 30,000 | 86.02 | >300 (timeout) | **Dagua** |
| 50,000 | 75,000 | 502.98 | >300 (timeout) | **Dagua** |

**Crossover point: ~3-4K nodes.** Above this, Dagua's O(N) grid-based algorithms
outperform Graphviz's native C engine. At 20K+ nodes, Graphviz times out entirely
while Dagua continues to run. Dagua handles 50K nodes in ~8 minutes on CPU.

## Known Issues (fix before v0.1)

### Layout Quality
- [ ] Edge crossings on random DAGs still ~3x Graphviz at 50 nodes
  - Improved from 334→191 via multi-pass barycenter + transpose heuristic + layered crossing loss
  - Further improvement: explore median heuristic, network simplex layering
- [ ] Seed doesn't affect layout — init is fully deterministic from topology
  - Add small random perturbation to init_positions for exploration
- [ ] LR/RL direction: node_sizes not swapped before layout computation
- [ ] Back-edge routing creates wide arcs that can overlap with other nodes

### Rendering
- [ ] Multi-line node labels: secondary line font scaling is hardcoded (0.8x)
- [ ] Edge arrowheads: mutation_scale=1 makes heads very small at some zoom levels
- [ ] Cluster label position is hardcoded (top-left) — should respect ClusterStyle.label_position
- [ ] No legend for node types / edge types

### API / UX
- [ ] `from_edge_index` doesn't accept `labels` kwarg — must set manually
- [ ] No `from_dict` constructor yet
- [ ] No `to_dot` export in public API (only in graphviz_utils)
- [ ] Graph validation (check for duplicate edges, self-loops)
- [ ] DOT export: special character escaping (backslashes, angle brackets)
- [ ] compute_node_sizes is not idempotent (no-ops if sizes already set)
- [ ] No progress callback for long-running layouts

### Metrics
- [x] Direction-aware metrics (dag_fraction, edge_straightness, x_alignment)
  - Now accept `direction` parameter: TB/BT/LR/RL
- [ ] overall_quality not normalized by graph size — area penalty dominates for large graphs
- [ ] No per-metric normalization (scores not comparable across different-sized graphs)

## Completed (Sprint 2)

- [x] Multi-pass barycenter crossing reduction (10-30 passes, adaptive to graph size)
- [x] Transpose heuristic for crossing minimization (Sugiyama Phase 2 refinement)
- [x] Layered crossing loss with virtual node decomposition for multi-span edges
- [x] Crossing loss `.sum()` instead of `.mean()` (proper gradient magnitude)
- [x] Grid-based spatial hashing for overlap detection (O(N) expected)
- [x] Grid-based overlap projection for large graphs
- [x] Negative sampling repulsion with self-index exclusion
- [x] O(1) edge construction via lazy tensor finalization
- [x] Input validation in from_edge_index
- [x] Direction-aware metrics (TB/BT/LR/RL)
- [x] Runtime scaling benchmark (Dagua vs Graphviz, 100 → 100K nodes)
- [x] Extended TorchLens architecture sampling (12 models across 10 categories)
- [x] Tests for BT/RL layout directions
- [x] Tests for self-loops, disconnected components, wide/dense graphs
- [x] Tests for from_torchlens integration
- [x] Sampled count_crossings (125K random pairs for >500 edges)
- [x] Grid-based count_overlaps metric for large graphs

## Feature Roadmap

### Tier 1 (next sprint)
- [ ] Interactive SVG output with tooltips and hover effects
- [ ] Learnable bezier control points (Tier 2 edge routing)
- [ ] Edge-node crossing avoidance
- [ ] Parameter auto-tuning (Optuna integration in eval/sweep.py)
- [ ] Network simplex layering (better than longest-path for crossing reduction)
- [ ] CUDA acceleration benchmarks

### Tier 2 (medium priority)
- [ ] Edge bundling (FDEB algorithm)
- [ ] Edge confluence (shared source/target control point merging)
- [ ] Multi-edge spacing (same source-target pair)
- [ ] Self-loop rendering
- [ ] G2 curvature smoothness loss
- [ ] Hobby's algorithm for edge post-processing
- [ ] Cluster hierarchy nesting (recursive layout within clusters)
- [ ] Cross-cluster edge routing with cluster borders as obstacles

### Tier 3 (long-term)
- [ ] Graphviz DOT export with pos="x,y!" for neato -n2
- [ ] Web-based interactive viewer (D3.js or similar)
- [ ] Animation of layout optimization (frame-by-frame)
- [ ] GPU batch layout for multiple graphs simultaneously
- [ ] Integration with PyTorch Geometric for GNN visualization
- [ ] 100K+ node support verified with benchmarks

## Architecture Decision Records

### ADR-1: All loss functions are pure functions
Loss functions take tensors and return scalars. No state, no side effects.
This makes them independently testable and composable.
**Decision**: Keep this pattern for all future constraints.

### ADR-2: Direction handling via post-transform
Layout always computed in TB space, then rotated for LR/RL/BT.
**Trade-off**: Simpler engine code, but node sizes need careful handling for LR.
**TODO**: Consider computing node_sizes in layout space to handle width/height swap.

### ADR-3: Heuristic bezier routing (Tier 1)
Control points computed from geometry after layout, not optimized.
**Rationale**: Good enough for MVP, keeps routing fast.
**TODO**: Tier 2 will make control points learnable parameters.

### ADR-4: Composite quality score design
Current: `-100*overlaps - 10*crossings + 50*dag_fraction - variance - 2*x_align - 0.5*area/n`
**Issue**: Area penalty dominates for large graphs. Need per-category normalization.
**TODO**: Normalize by node count, add Graphviz-relative scoring.

### ADR-5: Evaluation test graphs are synthetic + TorchLens
Expanded from 4 to 12+ TorchLens models covering all structural categories.
No hand-curated "golden" layouts yet.
**TODO**: Create reference layouts for key graphs, compute similarity scores.

### ADR-6: Scalability strategy
- N ≤ 500: exact O(N²) algorithms (repulsion, overlap, projection)
- N ≤ 2000: exact repulsion, grid-based overlap/projection
- N > 2000: sampled repulsion (k=128), grid-based everything
- Crossover vs Graphviz at ~3-4K nodes on CPU

## Choice Points (deferred decisions)
1. Should `layout()` return CPU tensors always, or match input device?
   - Currently: returns on compute device
   - Preference: probably CPU for convenience
2. Should `draw()` accept LayoutConfig kwargs directly?
   - `dagua.draw(g, steps=200, w_dag=5.0)` vs `dagua.draw(g, config=LayoutConfig(...))`
3. Should we add a `Graph.layout()` method or keep it as free function?
   - Currently: free function `dagua.layout(graph)`
   - Alternative: `graph.layout()` mutates in-place
4. Edge type classification in TorchLens conversion — how to handle unrecognized types?
5. Should cluster compactness use centroid attraction or bbox minimization?
