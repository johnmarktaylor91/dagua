# Task & Bug Tracker

## Active Tasks

## Bugs

## Improvements (Nice-to-Have)
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
