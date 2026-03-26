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

## Feature Parity Gaps (static/cosmetic, non-interactive)
- [ ] [HIGH] Record/table nodes -- structured multi-field content inside a node (Graphviz records, yFiles table nodes, database/UML diagrams)
- [ ] [MED] Text on path -- edge labels that curve along the edge path instead of sitting in a box (yFiles, D3)
- [ ] [MED] Intra-node compositional layout -- sub-elements within a node (text + icon + bar chart); GoJS "panels" (horizontal, vertical, table, spot)
- [ ] [MED] Edge bundling rendering -- visually merge parallel edges into a thick band that splits at endpoints
- [ ] [LOW] ~20 niche node shapes from Graphviz: house, invhouse, folder, component, promoter, cds, septagon, egg, point, etc. (bio/UML-specific)

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
- [ ] [HIGH] Edge-following rotated labels: rotate edge labels to curve tangent (~30 lines)
- [ ] [MED] Orthogonal edge routing with obstacle avoidance
- [ ] [MED] Octilinear edge routing: 45-degree angle mode between ortho and polyline (~100 lines)
- [ ] [MED] Edge bundling: group parallel edges with configurable bundling strength (~200 lines)
- [ ] [MED] Polyline routing with user-defined waypoints/bends (~100 lines)
- [ ] [MED] Icons/images in labels: inline images with text (~150 lines)
- [ ] [MED] Node badges/decorators: overlay indicators at configurable positions (~100 lines)
- [ ] [MED] Pattern fills (tiling image): tile small image across node fill (~50 lines)
- [ ] [MED] Tab header on cluster groups: folder-tab protrusion with label (~80 lines)
- [ ] [MED] Table/swimlane nodes for BPMN
- [ ] [MED] Self-loop style: configurable (semicircular vs compact side-attached)
- [ ] [MED] Edge endpoint offset for arrowhead placement (Graphviz matching)
- [ ] [HARD] Text wrapping to non-rectangular shapes (ellipse, diamond, triangle) (~200 lines)
- [ ] [HARD] Rich text via HTML/markup subset in labels (~300 lines)
- [ ] [HARD] Swimlane/table node with internal grid layout (~500 lines)

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
