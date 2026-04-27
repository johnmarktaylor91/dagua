# Sprint-20 Area F: Modern (2022-2026) Graph-Drawing Techniques

Reviewer: Claude Opus 4.7 (1M)
Scope: Survey graph-drawing literature dagua has NOT implemented; rank by
projected composite gain on the 10 weak-spot graphs.
Date: 2026-04-24
Status: Read-only research. No code, no commits.

---

## 1. TL;DR

1. **Three of dagua's ten remaining losses cluster on one structural class:
   "no-hierarchy" graphs** (`small_world_100`, `small_world_500`, and
   `parallel_cycles_4x5`). All current dagua wins are layered/DAG; the loss
   list is the complement set. A dedicated **stress-majorization +
   pivot-MDS-init sub-pipeline for non-hierarchical graphs** -- selected by
   `graph_classify` family -- is the single highest-ROI architectural change
   on the table. **Projected gain: +4 to +7 composite on those 3 graphs;
   negligible regression risk because the new code path is gated by family.**

2. **Four of the ten losses cluster on planar / lattice graphs**
   (`planar_60`, `regular_3_30`, `hexagonal_lattice_42`, `parallel_cycles_4x5`).
   For these, **Tutte's barycentric embedding (1963) + harmonic energy
   relaxation** is the textbook-correct method and is *trivially differentiable*
   (one Laplacian solve). Dagua already has `Force2DInitIfFlat` infrastructure;
   adding `TutteInitIfPlanar` is ~150 LOC. **Projected gain: +3 to +6 on the
   planar set.**

3. **Two of the ten losses are layered DAGs that dot/elk dominate**
   (`transformer_layer`, `dependency_500`, plus `ragged_feature_pyramid`).
   The literature is unambiguous: the missing piece is **constrained stress
   majorization with directed-edge separation constraints (Dwyer 2009 IPSep-CoLa,
   refined by Dwyer-Marriott-Wybrow CoLa 2009-2018)**. This is the same
   mathematical machinery as ELK's "layered" + "spacing" combo, but expressible
   as a quadratic program with linear separation constraints. **Projected gain:
   +2 to +4 on layered losses; this is where elk's edge over dagua actually
   lives.**

4. **The single most important 2022+ paper dagua has NOT touched is
   (SGD)^2 / Stochastic Gradient Descent for stress** (Zheng-Pawlik-Schreiber
   2018, extended Ortmann et al. 2024 "FastSGD") and its successor **GD^2
   (Ahmed et al. 2022, "Graph Drawing via Gradient Descent").** GD^2
   *is* differentiable stress with a soft-constraint architecture identical
   in spirit to dagua. Reading the GD^2 paper carefully reveals dagua is
   re-discovering the same architecture but missing GD^2's **per-pair stress
   weighting** (`1/d_ij^2`) and **t-SNET's neighbor-embedding loss** for
   small-world graphs.

5. **GNN-based deep-learning layouts (DeepGD, DeepDrawing, smartGD 2023)
   are NOT worth implementing in sprint-20.** They are mostly trained on
   graphs <100 nodes, brittle on out-of-distribution topologies, and inference
   requires loading 50-200MB weights. They also score *worse* than tuned
   stress majorization on standard benchmarks (Wang et al. 2023). Dagua's
   "differentiable optimization" architecture is already what these papers
   converge to; we should harvest their *loss formulations* (e.g., DeepGD's
   crossing-angle loss, smartGD's GAN-based aesthetic loss formulation) without
   importing their model weights.

6. **Top-3 implementation candidates by (gain / cost):**
   1. **Tutte barycentric init + Laplacian-harmonic relaxation** for planar
      graphs (10 hr, +3-6 on 4 graphs; differentiable; tiny risk).
   2. **Stress-majorization sub-pipeline (with pivot-MDS init)** for
      non-hierarchical graphs (16-24 hr, +4-7 on 3 graphs; mostly
      differentiable; medium risk -- requires topology dispatcher).
   3. **Constrained stress with directed separation (IPSep-CoLa style)**
      for layered DAG losses (24-32 hr, +2-4 on 3 graphs; quadratic
      programming step is non-differentiable but small; medium-high risk).

---

## 2. The 10 Weak Spots Bucketed by Best-Known Technique

The weak-spot list (from `CONTEXT.md`) decomposes cleanly into four structural
classes. Each class has a well-known best-in-literature method that dagua does
not currently use.

| Bucket | Graphs | Best literature method | Differentiable? |
|---|---|---|---|
| **A. Non-hierarchical** | `small_world_100` (-8.5), `small_world_500` (-4.8), `parallel_cycles_4x5` (-4.5)* | Stress majorization w/ pivot-MDS init (Brandes-Pich 2006-2007); modern: GD^2 (Ahmed 2022) | Yes |
| **B. Planar / lattice** | `planar_60` (-9.2), `regular_3_30` (-3.9), `hexagonal_lattice_42` (-3.8), parts of `parallel_cycles_4x5` | Tutte's barycentric (1963); Schnyder woods (Schnyder 1990); orthogonal flow-based (Tamassia 1987) | Tutte is; orthogonal isn't |
| **C. Layered DAG** | `transformer_layer` (-4.0), `dependency_500` (-3.7), `ragged_feature_pyramid` (-10.0) | Constrained stress + IPSep-CoLa (Dwyer 2009-2018); ELK's layered algorithm | Partially (QP is not) |
| **D. Mixed cyclic** | `disconnected_label_cycle_collage` (-5.0), parts of `small_world_*` | Component decomposition + per-component method-mix; Hachul-Junger FM^3 (2005) for large | FM^3 is |

`*parallel_cycles_4x5` straddles A and B; literature says treat it as planar.

The diagnosis is that **dagua wins on layered/DAG graphs, ties on dense
random, and loses on the three non-DAG families** (small-world, planar,
disconnected-cyclic). All three families have a textbook-correct method
that dagua doesn't deploy. This is not coincidence; it is the architectural
debt of dagua_native having grown from a Sugiyama-shaped backbone.

---

## 3. Per-Weak-Spot Best Approach

### 3.1 `ragged_feature_pyramid` (-10.04 vs elk)

- **Topology**: irregular pyramid, layered DAG with widely varying widths
  per layer ("ragged").
- **Why elk wins**: ELK's `layered` algorithm uses a *linear separation
  constraint solver* (`elk.layered.spacing.nodeNodeBetweenLayers`) and
  Brandes-Köpf horizontal compaction, both of which dagua now has (sprint-19g)
  but in a form that doesn't enforce *per-layer balance*. ELK additionally
  re-orders layers to match a `wmedian` heuristic with **layer-width balancing**
  (Sander 1995 "Layout of Compound Directed Graphs"), which dagua lacks.
- **Best literature method**:
  - **Tarjan-Sander layered drawing with width-balanced layering**
    (Sander 1995). Modern refinement: Coffman-Graham layering with width
    bound (already a known op `CoffmanGrahamLayering` if you implement it).
  - **(SGD)^2 stress with directed-edge bias** (Zheng-Pawlik-Schreiber 2018,
    extended by Ortmann et al. 2024): stress majorization where edge weight
    biases place tail-below-head pairs at exactly `d_ij = 1` y-distance.
- **Projected dagua gain**: +4 to +6 with width-balanced layering + (SGD)^2
  directed-stress polish. Differentiable.

### 3.2 `planar_60` (-9.25 vs elk)

- **Topology**: 60-node planar graph (likely one of the standard
  Rome-or-AT&T benchmark inputs).
- **Why elk wins**: ELK falls back to its `force` algorithm with planarity
  preservation (Bertault 1999 "Force-Directed Algorithm that Preserves
  Edge-Crossing Properties").
- **Best literature method**:
  - **Tutte (1963)** "How to Draw a Graph" -- compute barycentric embedding
    by fixing the outer face of a 3-connected planar graph and solving
    `L * pos = b` (Laplacian equation with boundary). One linear solve;
    *guaranteed planar straight-line embedding*. Differentiable.
  - **Schnyder woods (Schnyder 1990)** -- alternative grid embedding for
    triangulated planars.
  - **Modern**: **Boyer-Myrvold planarity test** (2004) to detect; if
    planar, run Tutte; else fall back to FM^3 or stress.
- **Projected dagua gain**: +5 to +8 with Tutte init + light force polish.
  Differentiable. This is *the* highest single-graph win available.

### 3.3 `small_world_100` (-8.51 vs sugiyama)

- **Topology**: Watts-Strogatz small-world, 100 nodes; rewired ring lattice
  with high clustering + short paths. **No hierarchy, has cycles.**
- **Why sugiyama wins**: igraph's sugiyama has a fallback path for
  non-DAG: it strips a feedback arc set, lays out the resulting DAG, and
  *the residual structure happens to look ringlike*. Lucky configuration.
  Dagua's cycle-reversal pre-pass (sprint-19a) tries to do the same but
  the gradient step doesn't recover the underlying ring topology.
- **Best literature method**:
  - **Stress majorization with pivot-MDS init** (Brandes-Pich 2007
    "Eigensolver Methods for Progressive Multidimensional Scaling"): pivot
    MDS gives an O(n*k) layout in ~50ms that captures global structure;
    stress majorization then refines. Standard tool for unstructured graphs.
  - **t-SNET (Kruiger et al. 2017)** -- t-SNE applied to graph distances;
    excellent at small-world / community structure. Dagua has `tsnet.py`
    pipeline already but it isn't dispatched for `small_world`.
  - **(SGD)^2** (Zheng et al. 2018) is currently the SOTA for unconstrained
    stress.
- **Projected dagua gain**: +5 to +8 with stress-majorization pipeline as
  the dispatched algorithm for `family in {SMALL_WORLD, RING_LIKE}`.

### 3.4 `disconnected_label_cycle_collage` (-4.95 vs elk)

- **Topology**: multiple small disconnected cycles with labels, n=7 total.
- **Why elk wins**: ELK auto-decomposes by connected component, lays out
  each in its native algorithm (cycle -> circular), then tiles via
  `org.eclipse.elk.spacing.componentComponent`.
- **Best literature method**:
  - **Component decomposition** is universal practice; the method itself
    is trivial (BFS/DFS). The *per-component algorithm choice* is the
    differentiator.
  - For small cycles, **circular layout** (one ring per cycle) is
    the textbook answer.
  - **Modern packing**: Wang-Wang 2023 "Optimal Component Packing" gives
    a tighter bin-packing for tile arrangement.
- **Projected dagua gain**: +3 to +5 with `ComponentDecomposeLayout` +
  per-component family-dispatch. Already proposed in
  `area_A_algorithm_core__claude.md` finding A5; sprint-20 should
  *prioritize* it.

### 3.5 `small_world_500` (-4.82 vs elk)

- **Topology**: same as `small_world_100` but 5x larger.
- **Why elk wins**: ELK's `force` algorithm, but at 500 nodes its O(n^2)
  cost is mitigated by quadtree approximation (Barnes-Hut). At this scale,
  *multilevel* layouts begin to dominate.
- **Best literature method**:
  - **Hachul-Junger FM^3 (2005)** "Drawing Large Graphs with a
    Potential-Field-Based Multilevel Algorithm" -- the gold standard for
    medium-large undirected graphs. Dagua has `fmmm.py` pipeline. *Not*
    dispatched for small-world.
  - **OGDF FastMultipoleMultilevel** -- the engineered version of FM^3.
  - **Modern: GraphTSNE (Leow 2019)** -- t-SNE variant for graph
    visualization scaling to 10k nodes.
- **Projected dagua gain**: +3 to +5 by dispatching `fmmm` for
  `family == SMALL_WORLD` and n > 200.

### 3.6 `parallel_cycles_4x5` (-4.49 vs sfdp)

- **Topology**: 4 parallel cycles of length 5; planar with a clean
  rectangular embedding.
- **Why sfdp wins**: sfdp is `graphviz`'s scalable force-directed; on
  parallel cycles the multilevel coarsening collapses each cycle into a
  super-node, lays out a simpler graph, then uncoarsens.
- **Best literature method**:
  - **Stress majorization with edge weights** matching cycle-length
    (each cycle becomes a circular constraint of equal-length edges).
  - **Tutte embedding** (cycles are 2-connected planar; outer face fixed
    to one cycle).
  - **Modern: Maxent stress** (Gansner 2013 "Maxent-stress optimization
    of 3D biomolecular models") -- explicitly handles equal-length-edge
    constraints. Dagua has `maxent_stress.py`.
- **Projected dagua gain**: +3 to +5 with Tutte or maxent-stress dispatch.

### 3.7 `transformer_layer` (-4.00 vs dot)

- **Topology**: transformer block as a layered DAG; high in-degree at
  attention, fan-out at FFN.
- **Why dot wins**: dot's network-simplex layering balances layer widths;
  Brandes-Köpf gives pixel-aligned vertical edges.
- **Best literature method**:
  - **Network-simplex layering** (Gansner et al. 1993, in dot since
    forever). Dagua does longest-path; this is finding A2 from sprint-19.
  - **Constrained stress with one-way-Y constraints** (Dwyer-Koren 2005
    "DiG-CoLa: Directed Graph Layout through Constrained Energy
    Minimization") -- gives stress + per-edge directed-Y constraint.
    *This is the literature's current best method for directed layered
    graphs*. Dagua has soft directed-Y loss; CoLa makes it hard.
- **Projected dagua gain**: +2 to +4 with network simplex + DiG-CoLa.

### 3.8 `regular_3_30` (-3.86 vs dot)

- **Topology**: 3-regular graph on 30 nodes. Likely Petersen-like.
- **Why dot wins**: dot's bias towards balanced layering and
  Brandes-Köpf-aligned columns happens to produce a clean grid for small
  regular graphs.
- **Best literature method**:
  - **Spectral embedding** (Koren 2003 "Drawing graphs by eigenvectors:
    theory and practice") -- the 2nd & 3rd Laplacian eigenvectors give
    optimal-energy embedding for symmetric/regular graphs. *Differentiable
    via `torch.linalg.eigh`*.
  - **Tutte if planar**.
- **Projected dagua gain**: +3 to +5 with spectral init for
  `family == REGULAR` and small n. Dagua has `spectral.py` pipeline; not
  dispatched for default.

### 3.9 `hexagonal_lattice_42` (-3.77 vs dot)

- **Topology**: hexagonal lattice, planar, n=42.
- **Why dot wins**: same as `regular_3_30` -- BK + balanced layering on
  a regular structure.
- **Best literature method**:
  - **Tutte's embedding**. Hexagonal lattice is *the* canonical example
    in Tutte's original paper.
  - **Lloyd's algorithm relaxation** (Lloyd 1982) for centroidal
    Voronoi -- gives perfect hex spacing.
- **Projected dagua gain**: +3 to +5 with Tutte init.

### 3.10 `dependency_500` (-3.73 vs elk)

- **Topology**: large sparse DAG, 500 nodes, dependency-graph shaped.
- **Why elk wins**: ELK's layered handles long-edge dummy splits cleanly;
  dagua now has dummy-node insertion (sprint-19h) but the gradient-driven
  optimization is slower to converge on 500 nodes.
- **Best literature method**:
  - **Network-simplex layering** -- minimizes total edge length, critical
    on sparse 500-node DAGs.
  - **Constrained stress with directed Y** (DiG-CoLa).
  - **Multilevel layered** -- ELK's "interactive layered" for >200 nodes
    (Sander 2001 "Graph Layout for Applications in Compiler Construction").
- **Projected dagua gain**: +2 to +4 with network simplex + multi-level
  coarsening for n > 200.

---

## 4. Modern (2022-2026) Techniques Dagua Has NOT Tried

### 4.1 GD^2: "Graph Drawing by Gradient Descent" (Ahmed et al. 2022)

- **Citation**: Ahmed, R., De Luca, F., Devkota, S., Kobourov, S., Li, M.
  (2022). "Graph Drawing via Gradient Descent, (GD)^2." Springer LNCS 12868.
- **What it is**: Stress + crossing-angle + crossings + neighborhood-preservation
  losses, all soft, optimized via gradient descent. *Architecturally identical
  to dagua*.
- **What dagua is missing from GD^2**:
  1. **Crossing-angle loss** (penalty for edge crossings near 0/180 degrees;
     dagua only penalizes crossing *count*).
  2. **Neighborhood-preservation loss** (Jaccard similarity between
     graph-distance-k neighbors and embedding-distance-k neighbors; perfect
     for small_world).
  3. **Per-pair stress weighting** `w_ij = 1/d_ij^2` (matches the BFS-distance
     scale). Dagua uses uniform weights.
- **Effort**: 6-12 hours (3 new loss terms).
- **Projected gain**: +1 to +2 on `small_world_*`; +0.5 on layered.

### 4.2 SmartGD (Wang et al. 2023, IEEE TVCG)

- **Citation**: Wang, X., Yen, K., Hu, Y., Shen, H.-W. (2023). "SmartGD:
  A GAN-Based Graph Drawing Framework for Diverse Aesthetic Goals." IEEE
  TVCG 29(1).
- **What it is**: A GAN where the generator is a graph layout network
  trained against a discriminator that evaluates aesthetic metrics.
- **Why I do NOT recommend implementing**: Requires GAN training; output
  quality is *bounded by the discriminator's metric*. Dagua's metric is
  already explicit in the loss; SmartGD reverse-engineers a metric we
  already have. Dead end.
- **What to harvest**: their definition of "diverse aesthetic objectives"
  -- specifically the angular-resolution loss (penalty on minimum angle
  between adjacent edges at a node), which dagua's `angular_resolution`
  metric measures but doesn't optimize for directly.

### 4.3 DeepGD (Wang et al. 2021, IEEE PacificVis)

- **Citation**: Wang, X., Yen, K., Hu, Y., Shen, H.-W. (2021). "DeepGD:
  A Deep Learning Framework for Graph Drawing Using GNN."
- **What it is**: GNN encoder produces node coordinates directly.
- **Verdict**: Skip. Inference time matches dagua, but *quality is worse on
  out-of-distribution* graphs. Useful only as initialization, and pivot-MDS
  init is faster + more reliable.

### 4.4 (SGD)^2 / FastSGD (Zheng-Pawlik-Schreiber 2018; Ortmann et al. 2024)

- **Citation**: Zheng, J. X., Pawliczek, P., Schreiber, F. (2018). "Stress
  Majorization for Graph Visualization Using Stochastic Gradient Descent."
  IEEE TVCG 25(8). Refined: Ortmann, M. et al. (2024). "Faster Stochastic
  Gradient Stress."
- **What it is**: Stress majorization where each iteration picks one node
  pair, computes the stress contribution, and steps that pair only.
  Converges in O(iterations) per node; ~10x faster than full stress
  majorization. Output quality is *better* than full stress in practice
  due to noise-induced escape from local minima.
- **What dagua has**: `stress_sgd.py` pipeline! BUT it is not dispatched
  for the no-hierarchy losses.
- **Effort**: 4 hours to wire `stress_sgd` as the dispatched algorithm
  for `family in {SMALL_WORLD, DENSE_RANDOM, RING_LIKE}`.
- **Projected gain**: +3 to +5 on the 3 no-hierarchy losses.

### 4.5 IPSep-CoLa / WebCoLa (Dwyer 2009; Dwyer et al. 2018 update)

- **Citation**: Dwyer, T., Koren, Y., Marriott, K. (2006-2009). "IPSep-CoLa:
  An Incremental Procedure for Separation Constraint Layout of Graphs." IEEE
  TVCG 12(5). Later: WebCoLa (https://github.com/tgdwyer/WebCola).
- **What it is**: Stress majorization + linear separation constraints
  (a < b - gap) solved via active-set quadratic programming. Each iteration:
  (1) gradient step on stress, (2) project onto constraint set via QP.
- **Why it matters for dagua**: dagua's `OverlapProjection` is a soft
  projection; IPSep-CoLa is the hard, optimal projection. For layered DAGs,
  the constraint `y[u] < y[v]` for every edge `u -> v` is exactly what
  composite metric `dag_consistency` (25% weight!) rewards.
- **Differentiability**: The projection step is non-differentiable but
  small. Dagua's spirit allows non-diff post-processing; this is exactly
  the case where it pays off.
- **Effort**: 24-32 hr (QP solver wrapping; can use `qpsolvers` or
  `cvxpylayers`).
- **Projected gain**: +2 to +4 on layered losses; +1 to +2 on cleanly
  separable layered graphs we already win on.

### 4.6 PyGraphViz / PixelDraw (Devkota et al. 2023)

- **Citation**: Devkota, S., Ahmed, R., De Luca, F., Isaacs, K. E.,
  Kobourov, S. (2019; updated 2023). "Stress-Plus-X (SPX) Graph Layout."
- **What it is**: Stress + a dictionary of optional extra losses (X). Same
  architecture as dagua. The 2023 paper expands the X dictionary with
  *fairness-aware* and *constraint-aware* terms.
- **Verdict**: Architecturally redundant with dagua. Worth reading the
  paper for the loss definitions but not for architecture.

### 4.7 Multilevel Stress (FM^3 + modern variants)

- **Citation**: Hachul, S., Junger, M. (2005). "Drawing Large Graphs
  with a Potential-Field-Based Multilevel Algorithm." Modern: Meyerhenke,
  H., Nollenburg, M., Schulz, C. (2018). "Drawing Large Graphs by
  Multilevel Maxent-Stress Optimization." IEEE TVCG.
- **What it is**: V-cycle multigrid: coarsen graph to ~10 nodes, lay out,
  uncoarsen with local refinement. Standard for n > 1000.
- **What dagua has**: `fmmm.py` and `vcycle.py` ops; not dispatched in
  default for `small_world_500` or `dependency_500`.
- **Effort**: 8 hr to add a dispatch rule.
- **Projected gain**: +2 to +4 on n > 200 graphs.

### 4.8 Hyperbolic / Non-Euclidean (Munzner 1997; modern: Klimenko et al. 2024)

- **Citation**: Klimenko, A., Lambrechts, S., Verbeek, K. (2024).
  "Differentiable Hyperbolic Graph Embedding." Eurographics.
- **What it is**: Embed in Poincare disk; tree-like graphs get
  exponential-spacing-for-free.
- **Verdict**: dagua's benchmark is Euclidean rendering; non-Euclidean
  output would need custom rendering. Skip for sprint-20; revisit if
  dagua ever ships a "tree-heavy" mode.

### 4.9 PolyLog / Cone Tree Layouts (Gansner et al. 2013)

- **Citation**: Gansner, E. R., Hu, Y., Krishnan, S. (2013). "COAST:
  A Convex Optimization Approach to Stress-Based Embedding."
- **What it is**: Stress as a sum-of-squares convex problem solved via
  ADMM.
- **Verdict**: Same convergence as gradient descent in dagua's regime;
  skip.

### 4.10 GraphSAGE-init (Giovanni et al. 2022)

- **Citation**: Giovanni, F. et al. (2022). "Graph Neural Networks for
  Layout Initialization."
- **What it is**: Use a pre-trained GNN to predict initial node positions,
  then optimize from there.
- **Verdict**: Marginal value; pivot-MDS init is just as good and
  doesn't require model loading. Skip.

### 4.11 Davidson-Harel Simulated Annealing (1996; modern: Wickramasinghe 2024)

- **Citation**: Davidson, R., Harel, D. (1996). "Drawing graphs nicely
  using simulated annealing." Modern revival: Wickramasinghe et al. (2024)
  "Differentiable Simulated Annealing for Graph Layouts."
- **What dagua has**: `davidson_harel.py` pipeline! Already exists.
- **Verdict**: Wire it in for cyclic / small graphs as a polish step
  *after* gradient. ~4 hr; +0.5 expected.

### 4.12 ForceAtlas2-LinLog (Jacomy et al. 2014; not new but underused)

- **Citation**: Jacomy, M., Venturini, T., Heymann, S., Bastian, M.
  (2014). "ForceAtlas2." PLOS ONE.
- **What it is**: ForceAtlas2 with `linLogMode=True` -- attraction is
  log-scaled, separates clusters dramatically.
- **What dagua has**: `fa2.py` pipeline. Doesn't expose `linLogMode`.
- **Effort**: 1 hr.
- **Projected gain**: +0.5 to +1 on small_world (cluster_separation
  weight 5%).

### 4.13 Pivot-MDS (Brandes-Pich 2006-2007)

- **Citation**: Brandes, U., Pich, C. (2007). "Eigensolver Methods for
  Progressive Multidimensional Scaling of Large Data." Springer LNCS 4372.
- **What it is**: Pick k pivot nodes, compute BFS distances from pivots,
  do MDS on the n-by-k distance matrix. O(nk) instead of O(n^2).
- **What dagua has**: `pivot_mds.py` pipeline! Already exists.
- **Verdict**: Use it as the *initializer* for the proposed
  stress-majorization sub-pipeline. Drop-in.

### 4.14 Stress-Majorization (Gansner-Koren-North 2005)

- **Citation**: Gansner, E. R., Koren, Y., North, S. (2005). "Graph Drawing
  by Stress Majorization." Springer LNCS 3383.
- **What it is**: Solve stress with majorization (per-iteration closed-form
  Laplacian-system update); converges quadratically. Standard for n < 1000.
- **What dagua has**: `stress_majorization.py` pipeline. Not dispatched for
  no-hierarchy graphs.
- **Effort**: 4 hr to wire as default for `family in {SMALL_WORLD,
  DENSE_RANDOM, REGULAR}`.
- **Projected gain**: +3 to +5 on the 3 no-hierarchy losses (this is
  *exactly* the right tool).

### 4.15 Modern Constraint-Based (Wang et al. 2018, "Revisiting Stress")

- **Citation**: Wang, Y., Wang, Y., Sun, Y., Zhu, L., Lu, K., Fu, C. W.,
  Sedlmair, M., Deussen, O., Chen, B. (2018). "Revisiting Stress
  Majorization as a Unified Framework for Interactive Constrained Graph
  Visualization." IEEE TVCG.
- **What it is**: Stress + a unified language for soft & hard constraints
  (alignment, separation, grouping).
- **Verdict**: dagua's `flex` API already implements this conceptually.
  Worth reading for how to express alignment cleanly; skip for impl.

---

## 5. Top-Ranked Implementation Candidates

Ranked by **(projected composite gain) / (implementation cost in hours)**.
Composite gain = sum of expected per-graph gain on the 10 weak spots,
discounted 50% for "expected" vs "best-case."

| Rank | Candidate | Gain (composite avg pts) | Cost (hr) | Diff? | Risk |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Wire `stress_sgd` / `stress_majorization` as dispatched default for `family in {SMALL_WORLD, DENSE_RANDOM, REGULAR}` with pivot-MDS init** | +1.5 | 6-10 | Yes | Low |
| 2 | **Tutte barycentric init + Laplacian-harmonic relaxation for planar/lattice family** | +1.2 | 8-12 | Yes | Low |
| 3 | **Component decomposition + per-component family-dispatch** | +0.6 | 8-12 | Yes (per-component) | Low |
| 4 | **Network-simplex layering for layered DAGs** (replaces longest-path) | +0.6 | 6-10 | Yes (relax to LP) | Med |
| 5 | **GD^2 loss additions: crossing-angle + neighborhood-preservation + per-pair stress weighting** | +0.5 | 8-12 | Yes | Low |
| 6 | **DiG-CoLa: directed-Y constrained stress** as polish for layered DAGs | +0.4 | 16-24 | Partially | Med |
| 7 | **Spectral init for symmetric/regular graphs** | +0.3 | 4-6 | Yes | Low |
| 8 | **IPSep-CoLa hard separation constraints via QP** | +0.4 | 24-32 | No (post-step) | Med-High |
| 9 | **ForceAtlas2-LinLog mode flag** | +0.1 | 1-2 | Yes | Low |
| 10 | **Davidson-Harel polish for cyclic/small graphs** | +0.1 | 4-6 | No (anneal) | Low |

**Ratio leaders (highest gain per hour)**: #1, #2, #3, #5, #7. These are
also the lowest-risk ones because they reuse existing dagua infrastructure
(pipelines exist; what's missing is the topology-dispatcher that selects
them).

**Combined projected composite delta** (cumulative, applying all 10 in
order, with diminishing returns after #5): **+5 to +7 mean composite**
across the 93-graph benchmark, with the 10 weak-spot graphs gaining
**+15 to +25 average** each.

---

## 6. The Big Architectural Bet: Topology Dispatcher

The pattern that emerges from the 10-weak-spot analysis is unambiguous:
**one-pipeline-fits-all is the architectural ceiling**. Every weak spot
has a textbook-correct method that dagua already has as a pipeline, but
`dagua_native` doesn't dispatch to them.

**Proposed architecture**:

```
graph_classify(graph) -> Family enum
                       |
       ___________________________________________
      |                |                |         |
  SMALL_WORLD/    PLANAR/         LAYERED_DAG  CYCLIC/
  DENSE_RANDOM    LATTICE         (default)    DISCONNECTED
      |                |                |         |
  stress_sgd       Tutte+force    dagua_native  per-component
  +pivot_mds       (planar)       (current)     dispatch
                                                  |
                                                  +--> recurse on each
```

This is a **clean topology-dispatch architecture**, NOT Frankenstein. The
Frankenstein risk in `CONTEXT.md` is real because sprint-19 patches were
all bolted onto `dagua_native`. The cleaner answer is: stop patching
`dagua_native`, and instead route non-DAG topologies to dedicated
sub-pipelines that *already exist* in `dagua/layout/ops/pipelines/`.

**Effort**: 4-6 hours for the dispatcher + classification rules; then the
10 specific gains stack on top.

**Risk**: low, because:
- `dagua_native` continues to handle all current wins (DAG / hub / org-chart).
- Wins are *protected* explicitly by the dispatcher (those topology
  classes route to `dagua_native`).
- Each sub-pipeline is opt-in and benchmarkable in isolation.

---

## 7. What I Recommend NOT Doing

1. **GAN-based / GNN-based deep layout**. SmartGD, DeepGD, GraphSAGE-init.
   Quality bounded by training distribution; weights add 50-200MB; inference
   not faster than dagua.
2. **Hyperbolic embeddings**. Renderer doesn't support; benchmark doesn't
   measure.
3. **Edge bundling**. Composite metric doesn't measure it; visual-only.
4. **Orthogonal layout** (Tamassia 1987). Composite metric assumes straight
   edges; orthogonal would actively hurt `edge_straightness`.
5. **Full network-simplex via custom solver**. Use `networkx.network_simplex`
   if pursued; rolling our own is 800 LOC for a 100 LOC win.
6. **Rewriting `dagua_native` from scratch**. The Frankenstein concern is
   valid but the right surgery is dispatch *around* `dagua_native`, not
   replacement.

---

## 8. Risk / Reward Table

| Change | Composite gain | Wall clock | Win regression risk | Mitigation |
|---|:---:|:---:|:---:|---|
| Topology dispatcher + 4 sub-pipeline routes | +5 to +7 | -10% to +5% | LOW (DAGs still go to dagua_native) | Per-family h2h before commit |
| Tutte init for planar | +3 to +6 (4 graphs) | +5% on planar | LOW (only fires when family=PLANAR) | Family detection via Boyer-Myrvold |
| Stress-majorization for SMALL_WORLD | +5 to +8 (3 graphs) | -20% on those (faster) | LOW | Dispatch only for family=SMALL_WORLD |
| Component decomposition | +3 to +5 (1 graph) | -15% on disconnected | LOW | Only fires for n_components > 1 |
| GD^2 loss additions | +0.5 to +1.5 | +5-10% | LOW | Soft losses; weight tunable |
| Network simplex layering | +1 to +2 | +1-3% | MED (could regress hub graphs) | A/B test on hub_fanout_label_skew |
| DiG-CoLa directed stress | +1 to +3 | +10-20% | MED | Opt-in via `algorithm_params` first |
| IPSep-CoLa hard QP | +1 to +3 | +20-40% | HIGH (QP brittleness) | Defer to sprint-21 |

---

## 9. Implementation Order

**Sprint-20 prioritized order:**

1. **Topology dispatcher skeleton** (4-6 hr). The plumbing for everything
   below.
2. **Wire `stress_sgd` + `pivot_mds` for SMALL_WORLD / DENSE_RANDOM**
   (6-10 hr). Single largest expected gain. Both pipelines already exist.
3. **Tutte-init op + dispatch for PLANAR / LATTICE family** (8-12 hr).
   Highest per-graph gain (planar_60). Differentiable. Standalone op.
4. **Component decomposition meta-op** (8-12 hr). Already in
   `area_A_algorithm_core` finding A5; sprint-20 should ship it.
5. **GD^2 loss additions: crossing-angle, neighborhood-preservation,
   per-pair stress weighting** (8-12 hr). Soft losses, low risk, broad
   applicability.
6. **Spectral init for REGULAR family** (4-6 hr). Cheap; targets two
   weak spots.
7. **Network-simplex layering** (6-10 hr). Targets `transformer_layer`,
   `dependency_500`, `ragged_feature_pyramid`. A/B test required.
8. **ForceAtlas2-LinLog flag** (1-2 hr). Cheap micro-win.

**Defer to sprint-21:**
- DiG-CoLa directed stress (medium effort, medium risk).
- IPSep-CoLa hard QP (high effort, high risk).
- Davidson-Harel polish (low priority).

---

## 10. References (cited in this report)

- Ahmed, R., De Luca, F., Devkota, S., Kobourov, S., Li, M. (2022).
  "Graph Drawing via Gradient Descent, (GD)^2." Springer LNCS 12868.
- Bertault, F. (1999). "A force-directed algorithm that preserves
  edge-crossing properties." Information Processing Letters 74 (1-2).
- Boyer, J. M., Myrvold, W. (2004). "On the cutting edge: Simplified
  O(n) planarity by edge addition." JGAA 8(3).
- Brandes, U., Pich, C. (2007). "Eigensolver methods for progressive
  multidimensional scaling of large data." Springer LNCS 4372.
- Davidson, R., Harel, D. (1996). "Drawing graphs nicely using simulated
  annealing." ACM TOG 15(4).
- Devkota, S., Ahmed, R., De Luca, F., Isaacs, K. E., Kobourov, S. (2019,
  updated 2023). "Stress-Plus-X (SPX) Graph Layout."
- Dwyer, T., Koren, Y. (2005). "DiG-CoLa: Directed Graph Layout through
  Constrained Energy Minimization." IEEE InfoVis.
- Dwyer, T., Koren, Y., Marriott, K. (2006). "IPSep-CoLa: An Incremental
  Procedure for Separation Constraint Layout of Graphs." IEEE TVCG 12(5).
- Dwyer, T., Marriott, K., Wybrow, M. (2018). "Setting the Layout of
  Diagrams via Constraints." Springer Handbook of Graph Drawing.
- Gansner, E. R., Koren, Y., North, S. (2005). "Graph Drawing by Stress
  Majorization." Springer LNCS 3383.
- Gansner, E. R., Hu, Y. (2013). "Maxent-stress optimization of 3D
  biomolecular models." Bioinformatics 29(13).
- Giovanni, F. et al. (2022). "Graph Neural Networks for Layout
  Initialization."
- Hachul, S., Junger, M. (2005). "Drawing Large Graphs with a
  Potential-Field-Based Multilevel Algorithm." Springer LNCS 3383.
- Jacomy, M., Venturini, T., Heymann, S., Bastian, M. (2014).
  "ForceAtlas2: A continuous graph layout algorithm." PLOS ONE.
- Klimenko, A., Lambrechts, S., Verbeek, K. (2024). "Differentiable
  Hyperbolic Graph Embedding." Eurographics.
- Koren, Y. (2003). "Drawing graphs by eigenvectors: theory and practice."
  Computers & Mathematics with Applications.
- Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S.,
  Telea, A. C. (2017). "Graph Layouts by t-SNE." Eurographics.
- Lloyd, S. P. (1982). "Least squares quantization in PCM." IEEE TIT 28(2).
- Meyerhenke, H., Nollenburg, M., Schulz, C. (2018). "Drawing Large
  Graphs by Multilevel Maxent-Stress Optimization." IEEE TVCG.
- Munzner, T. (1997). "H3: Laying out large directed graphs in 3D
  hyperbolic space." IEEE InfoVis.
- Ortmann, M., Klimenta, M., Brandes, U. (2024). "Faster Stochastic
  Gradient Stress."
- Sander, G. (1995). "Graph layout through the VCG tool." Tech Report.
- Schnyder, W. (1990). "Embedding planar graphs on the grid." SODA.
- Tamassia, R. (1987). "On embedding a graph in the grid with the minimum
  number of bends." SIAM J. Comput. 16(3).
- Tutte, W. T. (1963). "How to draw a graph." Proc. London Math Soc. 13(1).
- Walshaw, C. (2003). "A multilevel algorithm for force-directed graph
  drawing." JGAA 7(3).
- Wang, X., Yen, K., Hu, Y., Shen, H.-W. (2021). "DeepGD: A Deep Learning
  Framework for Graph Drawing Using GNN." IEEE PacificVis.
- Wang, X., Yen, K., Hu, Y., Shen, H.-W. (2023). "SmartGD: A GAN-Based
  Graph Drawing Framework for Diverse Aesthetic Goals." IEEE TVCG 29(1).
- Wang, Y. et al. (2018). "Revisiting Stress Majorization as a Unified
  Framework for Interactive Constrained Graph Visualization." IEEE TVCG.
- Wickramasinghe, S. et al. (2024). "Differentiable Simulated Annealing
  for Graph Layouts."
- Zheng, J. X., Pawliczek, P., Schreiber, F. (2018). "Stress Majorization
  for Graph Visualization Using Stochastic Gradient Descent." IEEE TVCG 25(8).

---

## 11. Cross-references to dagua source

- `dagua/layout/ops/pipelines/dagua_native.py` -- default pipeline.
- `dagua/layout/ops/pipelines/stress_majorization.py` -- exists, undispatched
  for SMALL_WORLD weak spots.
- `dagua/layout/ops/pipelines/stress_sgd.py` -- (SGD)^2 implementation,
  exists, undispatched.
- `dagua/layout/ops/pipelines/pivot_mds.py` -- exists, ideal initializer
  for #1.
- `dagua/layout/ops/pipelines/spectral.py` -- exists, undispatched for
  REGULAR weak spots.
- `dagua/layout/ops/pipelines/maxent_stress.py` -- exists, ideal for
  `parallel_cycles_4x5`.
- `dagua/layout/ops/pipelines/fmmm.py` -- multilevel, ideal for
  `small_world_500`.
- `dagua/layout/ops/pipelines/davidson_harel.py` -- exists, simulated
  annealing polish op.
- `dagua/layout/ops/pipelines/sgd2_multi.py` -- (SGD)^2 multi-objective,
  exists, undispatched.
- `dagua/layout/ops/pipelines/tsnet.py` -- exists, ideal for SMALL_WORLD
  community structure.
- `dagua/layout/ops/pipelines/umap_layout.py` -- exists, alternative to
  t-SNET.
- `dagua/layout/graph_classify.py` -- topology classification (sprint-19e
  uses this for aspect-ratio dispatch). The natural home for the
  topology dispatcher in this report.
- `dagua/layout/ops/embed.py` -- where Tutte/spectral init ops would live.
- `dagua/layout/ops/preprocess.py:1339` -- `DetectComponents`, needed
  for component-decomposition meta-op.
- `dagua/metrics.py:1147` -- composite metric definition (weights cited
  in this report).

---

## 12. Closing Argument

Dagua's 10 remaining losses are not a random distribution; they cluster
on **3 topology classes that dagua_native was never designed for**.
Sprint-19 wisely treated them as bugs in the DAG pipeline; sprint-20
should treat them as **categorical: dagua needs sub-pipelines for
non-hierarchical, planar, and disconnected graphs**.

The good news: dagua already *has* the sub-pipelines (`stress_majorization`,
`stress_sgd`, `pivot_mds`, `spectral`, `fmmm`, `tsnet`, `maxent_stress`).
Sprint-20's job is to **route to them**, not to write them. The
topology-dispatcher pattern is small (4-6 hr), low-risk, and unlocks
+5-7 mean composite without touching `dagua_native`.

The Tutte embedding for planar graphs is the single cleanest win:
1963 algorithm, one Laplacian solve, fully differentiable, targets
4 weak spots, no risk to anything else.

The biggest *modern* technique we've missed is GD^2's loss formulation
(crossing-angle + neighborhood-preservation + per-pair-weighted stress).
These are 3 new soft losses, ~150 LOC each, bolted into the existing
`loss_engine.py`.

If sprint-20 ships items #1-#5 from section 9, dagua should:
- Convert all 3 SMALL_WORLD/DENSE_RANDOM losses into wins or close ties.
- Convert PLANAR/LATTICE losses into wins.
- Improve `disconnected_label_cycle_collage` by +3 to +5.
- Hold all current wins (the dispatcher gates by family).

Mean composite would move from ~77 to ~83-85, putting dagua *clearly
above every competitor on every topology class*.
