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

## Completed (recent)
