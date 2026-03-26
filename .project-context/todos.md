# Task & Bug Tracker

## Active Tasks
- [ ] [HIGH] Benchmark finishing (~95.5%) -- run fidelity pipeline when done
- [ ] [HIGH] Continue Graphviz theme calibration until critics reach min>=8, mean>=9
- [ ] [MED] Close fidelity gaps -- add reimplementations for unpaired algorithms:
  - [ ] Reimplement fcose (Cytoscape force-directed) in PyTorch
  - [ ] Reimplement YifanHu (Gephi force-directed) in PyTorch
  - [ ] Fix LinLog original pairing -- OGDF says "unsupported"
- [ ] [LOW] Fix UMAP on disconnected graphs: replace inf shortest-path distances with 2*max(finite_distances)

## Bugs
- [ ] [HIGH] Arrowheads placed INSIDE target node boundary instead of OUTSIDE. Root cause: edge router computes endpoint at node boundary, arrowhead extends backward into gap but tip overlaps node surface.
- [ ] [HIGH] Cluster API overhaul -- clusters as first-class members with auto-detected nesting, strict tree validation, auto-propagated membership
- [ ] [MED] Cluster label collision: sibling labels merge into unreadable text. Auto-shrink font on colliding labels.
- [ ] [MED] Long label text overflow: ellipse aspect ratio not adjusting correctly for very wide text with overflow_policy="expand_node"
- [ ] [LOW] Arrowhead sizing slightly smaller than Graphviz's chunky triangles

## Cosmetic Polish (Maybe)
- [ ] Text background corners match node corner_radius
- [ ] Cloud/organic shape text bg follows contour
- [ ] Dash spacing adapts to curvature
- [ ] Hub arrowhead distribution around perimeter
- [ ] Shadow contour matching for elliptical nodes
- [ ] Self-loop arc height tightening
- [ ] Cluster shape variants: pill, cloud, convex hull

## Roadmap (Post-Benchmark)

### Architecture
- [ ] [HIGH] Original algorithm backend: `dagua.layout(g, algorithm="fr", backend="original")` -- user-facing transparency, runs literal originals via competitor adapters
- [ ] [HIGH] Composable ops migration: decompose classic/ into dagua/layout/ops/. Monoliths become dagua/layout/_reference/ as frozen test oracles. Bitwise-identical seed-matched validation.
- [ ] [HIGH] Pixel-unit overrides: "2pt" syntax for fixed-size elements
- [ ] [HIGH] Expose text rendering in style fields: text_background, text_underline, text_strikethrough, label_outline

### Layout Quality
- [ ] [HIGH] Theme-suggested layout spacing: suggested_node_sep/rank_sep on Theme
- [ ] [MED] Auto algorithm selection: benchmark all competitors per graph structure, auto-pick winner
- [ ] [MED] Incremental layout: warm-start from previous positions on graph changes
- [ ] [MED] Adaptive fine-level refinement for large-scale graphs

### Rendering
- [ ] [HIGH] Port constraints: named ports on nodes for circuit/flowchart diagrams
- [ ] [HIGH] Edge label collision avoidance
- [ ] [MED] Orthogonal edge routing with obstacle avoidance
- [ ] [MED] Table/swimlane nodes for BPMN
- [ ] [MED] Self-loop style: configurable (semicircular vs compact side-attached)
- [ ] [MED] Edge endpoint offset for arrowhead placement (Graphviz matching)

### Infrastructure
- [ ] [MED] Image galleries via GitHub Pages
- [ ] [MED] Import adapters: n8n, Airflow, dbt, Dagster, GitHub Actions, Step Functions, Terraform, Argo
- [ ] [MED] Interactive rendering: pan/zoom, tooltips, collapsible clusters (WebGL/Canvas)
- [ ] [MED] 3D graph rendering
- [ ] [LOW] yFiles benchmark comparison (eval license)
- [ ] [LOW] Print-quality PDF with embedded fonts

### Performance (Large Scale)
- [ ] Fix 2K performance cliff
- [ ] Fused edge loss CUDA kernel
- [ ] V-cycle pipeline parallelism
- [ ] GPU-accelerated coarsening
- [ ] Fix fanout_distribution_loss hub count mismatch at 200M+ scale

## Completed (This Sprint)
- [x] Full variant benchmark: 104 variants x 91 graphs x 30 seeds (432K evals)
- [x] Benchmark pitstop: skip-after-3 errors, graph-size timeout, rolling submission, SIGINT handler, watchdog executor
- [x] 4 new competitors: cytoscape fcose, gephi yifanhu, FR->KK chain, KK->FR chain
- [x] Edge weights on all 20/20 classic algorithms
- [x] 306 themes (44 -> 306) with 13 categories + list_themes()/theme_categories() API
- [x] Semicircle/semi-ellipse node shape with 4 orientations
- [x] aspect_ratio field on NodeStyle
- [x] Fidelity analysis pipeline + LaTeX report generator (adversarially critiqued)
- [x] Cosmetic feature recipe: docs/COSMETIC_FEATURE_RECIPE.md
- [x] FA2 reference adapter fix (kwarg filtering)
- [x] SGD2 multi reference adapter fix (criteria patches, vis, evaluate, .pos)
- [x] Parallel heavy engines (was serial, 86GB free)
- [x] Gallery audits Phase 1-3 + cosmetic polish sprint
