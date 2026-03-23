# Task & Bug Tracker

## Active Tasks
- [ ] [HIGH] Benchmark analysis: when benchmark finishes, run compare_reimpl_vs_original.py for fidelity report
- [ ] [HIGH] Monitor current benchmark run (97 variants x 91 graphs x 30 seeds, ~10.6% at 7hrs)
- [ ] [MED] Add edge weights to remaining 8 algos: sfdp, fmmm, gem, drl, davidson_harel, sugiyama, neulay, umap (AFTER current run)
- [ ] [MED] Warm-start chain variants FR->KK, KK->FR (AFTER current run)
- [x] [DONE] Gallery audit Phase 1: 133/133 individual cards at 9+ critic score
- [x] [DONE] Gallery audit Phase 2: 38/38 combo cards at 9+ critic score
- [x] [DONE] Gallery audit expansion: 40 combos + 20 evil cards added and iterated to 9+/7+
- [x] [DONE] Cosmetic polish sprint: 7 rounds, 113 below-9 -> 32, 17 new 10/10s, 900 LOC across 3 commits
- [ ] [HIGH] Continue Graphviz theme calibration until critics reach min>=8, mean>=9

## Bugs
- [ ] [HIGH] Arrowheads placed INSIDE target node boundary instead of OUTSIDE. Edge endpoint should be offset outward by arrowhead length so arrowhead fills the gap between edge body and node, matching Graphviz. Root cause: edge router computes endpoint at node boundary, arrowhead extends backward into gap but tip overlaps node surface.
- [ ] [MED] Cluster label collision: sibling cluster labels ("Right Branch", "Left Branch") merge into unreadable text in nested_clusters test graph. Labels need spacing or collision detection.
- [ ] [MED] Long label text overflow: nodes with long labels (e.g., "BatchNormalization2d(128, eps=1e-05, momentum=0.1)") show text extending beyond ellipse boundary. overflow_policy="expand_node" should expand the ellipse to contain text, but ellipse aspect ratio may not be adjusting correctly for very wide text.
- [ ] [LOW] Arrowhead sizing still slightly smaller than Graphviz's chunky triangles. Current graphviz_strict uses arrow_length=7, arrow_width=4.5 -- may need to increase to ~10x7 once the placement bug is fixed.

## Cosmetic Polish (Maybe)
- [ ] [MAYBE] Text background corners: round text bg corners to match node corner_radius on rounded shapes
- [ ] [MAYBE] Cloud/organic shape text bg: shape text backgrounds to follow cloud/hexagon contour instead of rectangular
- [ ] [MAYBE] Dash spacing on curved surfaces: adapt dash pattern spacing to curvature (cylinder caps, high-curvature edges)
- [ ] [MAYBE] Hub arrowhead distribution: when many edges converge on one node, distribute arrowheads around the perimeter instead of stacking
- [ ] [MAYBE] Graphviz font size matching: increase default font size in comparison cards to better match Graphviz's chunkier proportions
- [ ] [MAYBE] External label font scaling: reduce external label font size relative to node labels (currently dominates)
- [ ] [MAYBE] Shadow contour matching: shadow blur should follow elliptical/rounded node contour precisely
- [ ] [MAYBE] Crossing sharp style: make the angular kink more dramatic for better visual distinction
- [ ] [MAYBE] Gradient center smoothing: lighten the radial gradient center hotspot slightly for less concentration
- [ ] [MAYBE] Star point sharpening: slightly sharper star points for a more iconic look
- [ ] [MAYBE] Tab protrusion sizing: make the tab shape's protrusion taller/wider so it reads as a file tab at small sizes
- [ ] [MAYBE] Overflow/shrink demo labels: use longer test labels that actually trigger the overflow/shrink policies visibly
- [ ] [MAYBE] Fill-pattern card node proportions: fix extreme width:height ratio on striped/gradient reference cards (deep layout issue)
- [ ] [MAYBE] Self-loop arc height: tighten self-loop radius so loops don't extend 2x node height above the shape
- [ ] [MAYBE] Taxi self-loop routing: implement orthogonal segments for self-loops (currently falls back to smooth arc)
- [ ] [MAYBE] Italic shear angle: increase synthetic italic shear by 1-2 degrees for demo/reference cards where the point is to showcase italic

## Improvements (Nice-to-Have)
- [ ] [MED] Image galleries via GitHub Pages: build_gallery.py generates PNGs, pushes to gh-pages branch. Keep images out of main branch. Milestone snapshots as GitHub Releases.

- [ ] [LOW] Add semicircle node shape (flat bottom, curved top -- useful for state machines, architectural diagrams)
- [ ] [MED] Add Van Essen and Cajal themes -- neuroscience-inspired aesthetics (Van Essen: clean cortical connectivity diagrams with anatomical coloring; Cajal: hand-drawn ink-wash style inspired by Santiago Ramon y Cajal's neural illustrations)
- [ ] [MED] Interactive graph rendering: pan/zoom, hover tooltips, click handlers, collapsible nodes/clusters (expand/collapse subgraphs). WebGL or HTML5 Canvas backend alongside matplotlib.
- [ ] [MED] 3D graph rendering: z-coordinate support, 3D force-directed layout, perspective/orthographic projection. Consider Three.js export or matplotlib 3D axes.
- [ ] [MED] Add Cytoscape.js fcose as competitor layout algorithm (npm install cytoscape cytoscape-fcose, Node.js subprocess adapter, same pattern as ELK/dagre)
- [ ] [MED] Add Gephi/YifanHu as competitor layout algorithm (Java Gephi Toolkit JAR, subprocess adapter)
- [x] [DONE] Edge jump styles at crossings (arc/gap/sharp). Implemented with detection + rendering.
- [x] [DONE] Pie chart node fills. Implemented with donut hole support.
- [ ] [MED] Import adapters for workflow/pipeline tools: parse native file formats into DaguaGraph. n8n (.json workflows), Airflow (DAG .py -> task graph), dbt (manifest.json -> model lineage), Dagster (asset graph), GitHub Actions (.yml -> job/step DAG), AWS Step Functions (ASL .json -> state machine), Terraform (.tfstate -> resource graph), Argo Workflows (workflow .yaml -> DAG). Each adapter: read file -> extract nodes/edges/metadata -> return DaguaGraph with appropriate theme auto-applied.
- [ ] [MED] Warm-start chain variants: benchmark FR->KK and KK->FR two-pass layouts using pos= forwarding. Does warm-starting KK from FR output beat either alone? Add as variant entries once plumbing is validated.
- [ ] [HIGH] Pixel-unit overrides: support "2pt" syntax for fixed-size elements (Union[float, str] values in styles). Users who want pixel-based sizing for specific elements.
- [ ] [HIGH] Expose text rendering capabilities in style fields: NodeStyle.text_background, text_underline, text_strikethrough, EdgeStyle.label_outline, ClusterStyle.label_outline. Capabilities exist in dagua/render/text/ but not exposed in styles.
- [ ] [MED] Graphviz comparison pipeline: fix DOT generation for multi-variant grid tests (Graphviz flattens grids to strips). May need separate DOT sources per grid cell.
- [ ] [MED] Self-loop style: dagua uses semicircular arcs, Graphviz uses compact side-attached loops. The shapes are semantically correct but stylistically different. Consider making self-loop style configurable.
- [ ] [MED] Edge endpoint offset for arrowhead placement: when arrowheads are enabled, offset the edge target endpoint outward by the arrowhead length so the arrowhead sits in the gap between edge and node boundary (matching Graphviz behavior).
- [ ] [MED] Cluster proportions for graphviz_strict: nested clusters are wider/taller than Graphviz's more compact rendering. May need cluster-specific sizing adjustments.
- [ ] [LOW] Node X-alignment on linear chains: dagua's edge router may offset nodes horizontally on simple vertical chains. Graphviz centers them on a single vertical axis.
- [ ] [LOW] Dash pattern calibration: dagua's dashed borders use shorter dashes than Graphviz's bolder long-dash pattern. May need separate dash defaults for borders vs edges.
- [ ] [LOW] graph_4in scaling test: data-coordinate text at very small figure sizes produces tiny but proportional output. Consider minimum readable text size floor for export sizes below ~4 inches.
- [ ] Auto algorithm selection: benchmark all competitor algos per graph structure/size, auto-pick the best one for each regime. If FR beats dagua on small dense graphs, just use FR there. Dagua becomes a meta-engine that always picks the winner.
- [ ] Non-rectangular collision detection: ellipse proxy for round shapes, SDF for arbitrary shapes. Currently bbox only.
- [ ] Parallel BFS layering (Kahn's algorithm) for further layering speedup on wide graphs
- [ ] Fix 2K performance cliff: still 14s vs 10s at 5K. Consider lowering multilevel threshold or fewer auto-steps
- [ ] Install OGDF and add subprocess-based reference tests for Maxent-Stress and FM^3. Thin C++ wrapper: takes graph file + algorithm name, outputs positions as JSON. Same pattern as Graphviz competitor.
- [ ] Competitor aesthetic themes: implement ~10 themes mimicking the visual defaults of Graphviz, D3.js, Mermaid, Draw.io, ELK, yEd, Cytoscape, NetworkX/matplotlib, TikZ/LaTeX, Obsidian. Side-by-side visual comparison gallery.
- [ ] Adaptive fine-level refinement: detect regions needing correction (high overlap, crossing hotspots) and only optimize those patches. Keeps differentiability but avoids global optimization on fine levels at 1B scale.
- [ ] Fused edge loss CUDA kernel: compute all 5 edge losses in one pass (one pos gather, one backward). Eliminates 4x redundant memory reads per step.
- [ ] V-cycle pipeline parallelism: overlap disk reload + CPU prep with GPU optimization using double-buffering and CUDA streams.
- [ ] GPU-accelerated coarsening: vectorize the Python matching loop in coarsen_once with segmented radix sort.
- [ ] Fix fanout_distribution_loss hub count mismatch properly. At 200M+ scale, boundary_offsets and hub_degrees_v have different lengths due to edge batching filtering. Current fix truncates to min_len (band-aid). Root cause: edge batch may not include all edges for every hub, causing inconsistent hub filtering between the angle computation and the degree computation paths.
- [ ] Investigate 325s unexplained overhead in Phase 1 at 50M. Coarsening only took 65s but Phase 1 total was 390s. The ~325s gap is likely CSR argsort (75M edges) + frontier BFS (7070 waves) + layer propagation + offload logic. Add fine-grained timing to isolate. Could the CUDA CSR kernel help here? Is the frontier BFS still doing unnecessary work?

## Completed (recent)
