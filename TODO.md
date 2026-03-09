# Dagua MVP TODO — Post-Build Debrief

## Known Issues (fix before v0.1)

### Layout Quality
- [ ] Edge crossings on random DAGs (334 crossings on 50-node random DAG vs 53 for Graphviz)
  - Crossing loss with sigmoid proxy needs temperature tuning
  - Consider Sugiyama-style crossing minimization as preprocessing
- [ ] Seed doesn't affect layout — init is fully deterministic from topology
  - Add small random perturbation to init_positions for exploration
- [ ] LR/RL direction: layout is computed as TB then axes swapped — node sizes should also swap
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
- [ ] Graph validation (check for duplicate edges, self-loops, etc.)

## Feature Roadmap

### Tier 1 (next sprint)
- [ ] Interactive SVG output with tooltips and hover effects
- [ ] Learnable bezier control points (Tier 2 edge routing)
- [ ] Edge-node crossing avoidance
- [ ] Parameter auto-tuning (Optuna integration in eval/sweep.py)
- [ ] LaTeX report generation (eval/reference.py)
- [ ] HTML dashboard generation with interactive charts

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
- [ ] 100K+ node support with spatial indexing (quadtree repulsion)

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

### ADR-5: Evaluation test graphs are all synthetic + TorchLens
No hand-curated "golden" layouts yet.
**TODO**: Create reference layouts for key graphs, compute similarity scores.

## Performance Observations
- 50 nodes: ~2s (500 steps)
- 200 nodes: ~8s (500 steps)
- Exact repulsion O(N²) is the bottleneck above 200 nodes
- Negative sampling kicks in at 5000 nodes (configurable)
- Crossing loss O(E²) — sampled at 1000 edges

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
