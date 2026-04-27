# Area B — Conformal / harmonic embedding for true planar lattices

## Question

D Claude (sprint-21) flagged arXiv:2506.20541 (June 2025) "conformal-rigidity-guided Tutte/harmonic refinement" as the genuinely-new post-2024 result that could close the lattice gap on `hexagonal_lattice_42` (-0.63), `triangular_lattice_36` (-1.61), `sierpinski_42` (already +1.14). Predicted +5..+7 composite.

Tutte's classical result: any 3-connected planar graph has a unique convex straight-line embedding where every interior vertex is the centroid of its neighbors, given a fixed convex outer face. This produces visually balanced, conformal-like embeddings.

For dagua's metric (dag_consistency 25, edge_length_cv 20, depth_spearman 15, edge_straightness 10, crossing_rate 10, overlap 10, angular_resolution 5, cluster_separation 5):

- Tutte gives **0 crossings** (planarity preserved) → 10 pts.
- Tutte gives **good edge_length_cv** (typically 0.15-0.30 on regular lattices, dot achieves 0.10).
- Tutte breaks **dag_consistency** because it ignores edge directions → could LOSE 25 pts.
- Tutte breaks **depth_spearman** for the same reason → could LOSE 15 pts.

So naive Tutte regresses dagua. The 2025 conformal-rigidity refinement adapts Tutte to preserve directional structure while maintaining lattice regularity.

## Research targets

1. **Read the arXiv paper** (2506.20541) and 2-3 cited follow-ups. Understand exactly what "conformal rigidity" guarantees and what it costs.

2. **Adapt the algorithm to dagua's directed metric**. The classical result is for undirected planar graphs. For dagua, we need to preserve y = depth(node) up to a monotonic mapping. Two-step approach: (a) classical Tutte → x coordinates only; (b) y from BFS-layer index. Or a constrained variant that respects partial orderings.

3. **Implement a working version**. Real pseudocode, not a sketch. Use scipy / numpy / pure torch. The Tutte system is a sparse linear solve at heart: `Lx = b` where `L` is a graph Laplacian.

4. **Empirical validation**. Test on hex_lattice_42, triangular_lattice_36, sierpinski_42, parallel_cycles_4x5. Also test non-lattice planar graphs (planar_60, transformer_layer if planar) to verify no regression.

5. **Detection gate**. classify_graph already exposes `is_planar` and `topology_tags`. Recommended: `lattice_like` tag + `is_planar=True` + N >= 12.

## Output

`.project-context/research/sprint_22_algo_bets/B_conformal_embedding__<your_agent>.md`

- TL;DR with the single biggest call
- Key paper findings with proper citations
- Working pseudocode for Tutte-with-depth-anchor
- Measured composite delta from /tmp/ on at least 5 lattice / planar graphs
- Risk: which Tutte-broken metrics need a guard
- Recommended gate + implementation point

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed for measurement.
- Read CONTEXT.md first.
- BIGGER BET: do this properly. Read the paper. If Tutte's-with-depth-anchor isn't right, find the correct adaptation.
- 3000-5000 words if needed.
