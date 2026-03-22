# Task & Bug Tracker

## Active Tasks
- [ ] [HIGH] Benchmark analysis: when benchmark finishes, run compare_reimpl_vs_original.py for fidelity report
- [ ] [HIGH] Continue Graphviz theme calibration until critics reach min>=8, mean>=9

## Bugs
- [ ] [HIGH] Arrowheads placed INSIDE target node boundary instead of OUTSIDE. Edge endpoint should be offset outward by arrowhead length so arrowhead fills the gap between edge body and node, matching Graphviz. Root cause: edge router computes endpoint at node boundary, arrowhead extends backward into gap but tip overlaps node surface.
- [ ] [MED] Cluster label collision: sibling cluster labels ("Right Branch", "Left Branch") merge into unreadable text in nested_clusters test graph. Labels need spacing or collision detection.
- [ ] [MED] Long label text overflow: nodes with long labels (e.g., "BatchNormalization2d(128, eps=1e-05, momentum=0.1)") show text extending beyond ellipse boundary. overflow_policy="expand_node" should expand the ellipse to contain text, but ellipse aspect ratio may not be adjusting correctly for very wide text.
- [ ] [LOW] Arrowhead sizing still slightly smaller than Graphviz's chunky triangles. Current graphviz_strict uses arrow_length=7, arrow_width=4.5 -- may need to increase to ~10x7 once the placement bug is fixed.

## Improvements (Nice-to-Have)
- [ ] [MED] Add Cytoscape.js fcose as competitor layout algorithm (npm install cytoscape cytoscape-fcose, Node.js subprocess adapter, same pattern as ELK/dagre)
- [ ] [MED] Add Gephi/YifanHu as competitor layout algorithm (Java Gephi Toolkit JAR, subprocess adapter)
- [ ] [LOW] Edge jump styles at crossings (arc/gap/sharp). Draw.io. Requires crossing detection.
- [ ] [LOW] Pie chart node fills (16-slice). Cytoscape only.
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
