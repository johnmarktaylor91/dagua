# Area D - Modern algorithmic literature scan, post-2024

Agent: codex
Date: 2026-04-25
Scope: research only; no source changes.

## TL;DR

- **Best bet: conformal-rigidity / edge-isometric spectral detection plus a gated harmonic finisher for lattice-like graphs.** This targets `hexagonal_lattice_42` and likely `triangular_lattice_36` by attacking the actual residual metric gap: edge-length CV while preserving y-depth order.
- **Second bet: gap-constrained layered ordering for large DAGs.** The 2025 layered-drawing work on "few gaps and few crossings" is directly relevant to `dependency_500`, whose loss is edge-length CV after a saturated gradient solve.
- **Third bet: deterministic candidate scoring, not more stochastic polish.** Use exact crossings for the small/medium candidate picker and deterministic low-discrepancy pairs for large graphs. This reduces selection noise in `_best_of_polish` without changing the metric contract.
- **GNN/DR papers are useful as init generators, not as replacement solvers.** CoRe-GD, NNP-NET, and Word2VecGD are plausible for `dependency_500` and `small_world_500`, but they are less aligned with dagua's directed-depth and straightness metrics than the first two bets.
- **For Petersen/non-planar regular graphs, there is no clear 2024-2026 paper that specifically solves the exact gap.** The modern relevant work is beyond-planarity/SAT and orthogonal shape-first layout. It may provide a candidate generator, but risk is higher because dagua scores straight-line node positions.
- **Anti-recommendation: edge bundling and metric-fooling papers are valuable warnings but bad integration targets.** Dagua's benchmark does not reward bundled-edge readability, and metric-equivalent adversarial layouts are not a layout strategy.

## Search Notes And Assumptions

The requested `exa` MCP tools were not exposed in this session, so I used the available web search/fetch tool and prioritized primary sources: arXiv, OpenReview, Dagstuhl/LIPIcs, IEEE/PubMed metadata, and Eurographics/CGF metadata. I read `.project-context/research/sprint_21_final_push/CONTEXT.md` first and treated its sprint-20l gap table as the task baseline, while noting that the sibling Petersen report says `petersen_10` may already be closed at current HEAD. Dagua already has classical spectral, stress-SGD, Sugiyama, tsNET, UMAP, and polish candidates (`dagua/layout/ops/pipelines/__init__.py`, `dagua/layout/ops/pipelines/dagua_native.py`), so recommendations below focus on techniques not yet represented as dispatch gates or postprocessors.

## Technique 1 - Conformal-Rigidity-Guided Harmonic Lattice Finisher

**Citation.** Joao Gouveia, Stefan Steinerberger, Rekha R. Thomas, ["Conformal Rigidity and Spectral Embeddings of Graphs"](https://arxiv.org/abs/2506.20541), arXiv math.CO/math.OC, 2025. Related implementation context: Henry Forster, Stephen Kobourov, Jacob Miller, Johannes Zink, ["Drawing Trees and Cacti with Integer Edge Lengths on a Polynomial-Size Grid"](https://graphdrawing.github.io/gd2025/pages/program/), GD 2025 poster listing.

**Core idea.** The conformal-rigidity paper connects graph classes with edge-isometric spectral embeddings: for certain highly symmetric or walk-regular graphs, Laplacian eigen-embeddings can place adjacent vertices at uniform edge length. This is not a drop-in graph-drawing algorithm, but it gives dagua a structural detector: if the graph is regular/lattice-like and has the right spectral signature, generate one deterministic candidate by spectral coordinates, choose a boundary cycle or hull, and solve a harmonic/Tutte-style linear system with boundary fixed. The objective is not "prettier planar drawing" in general; it is a candidate that directly minimizes edge-length CV for lattice-like straight-line drawings.

**Predicted impact.** Highest on `hexagonal_lattice_42` (-2.52, known CV 0.43 vs graphviz_dot 0.10) and likely `triangular_lattice_36`. The important caveat from area A is that pure low CV can destroy depth/order; therefore this should be a post-polish candidate scored by `composite(full(...))`, not a replacement layout. For lattice DAGs, keep y-ranks from the current pipeline and solve/refit only x or a lightly constrained 2D harmonic coordinate. Expected gain: +2.5 to +5.5 on hex, +1 to +3 on triangular, near zero elsewhere.

**Implementation difficulty.** 250-400 LOC. Detector: degree histogram, E/N band, layer regularity, connectedness, spectral multiplicity/eigen-gap checks. Candidate: sparse Laplacian solve for interior coordinates, boundary extraction fallback to convex hull if face data is unavailable, and final scale/translate. Integration: add to `_best_of_polish` candidate list or a lattice-only `best_of_structural_polish` wrapper.

**Risk of regression.** Low if candidate-scored. Medium if dispatched unconditionally, because Sierpinski/fractal and dependency-like graphs can look lattice-ish but need rank/depth more than unit edges. Guard with tags plus score gate.

## Technique 2 - Gap-Constrained Layered Ordering For Large DAGs

**Citation.** Alexander Dobler and Jakob Roithinger, ["Layered Graph Drawing with Few Gaps and Few Crossings"](https://arxiv.org/abs/2502.20896), arXiv cs.CG, 2025. Soren Domros and Reinhard von Hanxleden, ["Determining Sugiyama Topology with Model Order"](https://drops.dagstuhl.de/storage/00lipics/lipics-vol320-gd2024/LIPIcs.GD.2024.48/LIPIcs.GD.2024.48.pdf), GD 2024 poster.

**Core idea.** The 2025 paper extends one-sided crossing minimization so long edges do not weave freely through layers and create many gaps. That matches dagua's `dependency_500` diagnosis: the layout is already DAG-consistent, but edge-length CV is worse than ELK's. A gap cap is a discrete layered constraint that current continuous losses cannot recover after saturation. The 2024 model-order paper is less directly useful for arbitrary benchmark graphs, but it reinforces the same lesson: the topological phases of Sugiyama layout can be solved or constrained deterministically before geometry, instead of expecting gradient descent to repair topology later.

**Predicted impact.** `dependency_500` is the main target. A gap-aware order can shorten long-layer spans and regularize dummy-chain corridors, improving edge-length CV without sacrificing depth Spearman. It may also help `transformer_layer`, `ragged_feature_pyramid`, and wide DAG close losses. Expected gain: +1.5 to +3.5 on `dependency_500`, smaller +0.5 to +1.5 on wide/skip DAGs.

**Implementation difficulty.** 300-600 LOC if added as a true Sugiyama ordering variant; 120-200 LOC if implemented as a post-order local search over existing layers. Practical first step: after current layer assignment, identify long edges, compute per-layer gap counts, then run adjacent swaps that lower a weighted objective `crossings + alpha * gaps + beta * edge_span_cv`. Reuse existing barycenter ordering and only add a deterministic local repair phase.

**Risk of regression.** Medium. Over-constraining gaps can increase crossings and damage angular resolution. Mitigation: only run for DAG-ish graphs with N >= 200, and candidate-score against current baseline.

## Technique 3 - Deterministic Candidate Scoring And Exact Medium-Graph Crossings

**Citation.** Martin Noellenburg, Sebastian Roeder, Markus Wallinger, ["GdMetriX - A NetworkX Extension For Graph Drawing Metrics"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2024.45), GD 2024. Gavin J. Mooney, Helen C. Purchase, Michael Wybrow, Stephen G. Kobourov, ["The Multi-Dimensional Landscape of Graph Drawing Metrics"](https://slinky.cs.arizona.edu/people/kobourov/gd-metrics2024.pdf), PacificVis 2024. Oriol Sole Pi, ["Approximating the Crossing Number of Dense Graphs"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2024.54), GD 2024 poster / arXiv 2024.

**Core idea.** Dagua's polish picker chooses among candidates by a metric bundle that includes sampled metrics. That is defensible for suite throughput, but risky for close losses: a stochastic crossing/angular estimate can choose the wrong candidate near the 0.5 margin. Current graph sizes in the residual bucket are small/medium enough that exact pairwise segment crossings are feasible for candidate selection. For larger graphs, use deterministic pair schedules: lexicographic edge-pair blocks, Sobol/Halton pair indices, or hash-stable stratified samples. The recent metrics papers do not prescribe this exact engineering, but they strongly support metric standardization and careful normalization; Sole Pi's dense crossing work is a useful reminder that deterministic crossing approximations are an active 2024 topic, even if his theorem targets graph crossing number rather than a fixed drawing's crossing count.

**Predicted impact.** This does not create better geometry by itself; it reduces false negatives in candidate selection. It is specifically relevant to the stochastic-noise question and to the current polish op where variants are rejected if composite appears worse. Expected suite gain: +0.5 to +2.0 by allowing good candidates through, especially on close losses; bigger value is lower research noise.

**Implementation difficulty.** 80-180 LOC. Add `composite_for_picker(..., deterministic=True)`: exact crossings for E <= threshold, exact angular resolution for all nodes below threshold, deterministic stratified fallback above threshold. No algorithmic source changes outside metrics/picker would be needed in a later implementation sprint.

**Risk of regression.** Low. Runtime risk is the main issue; keep exact mode only for candidate picking and bounded E, not for every benchmark metric call.

## Technique 4 - CoRe-GD / GNN Hierarchical Init, Not Replacement Layout

**Citation.** Florian Groetschla, Joel Mathys, Robert Veres, Roger Wattenhofer, ["CoRe-GD: A Hierarchical Framework for Scalable Graph Visualization with GNNs"](https://openreview.net/forum?id=vtyasLn4RM), ICLR 2024. Loann Giovannangeli, Frederic Lalanne, David Auber, Romain Giot, Romain Bourqui, ["Toward Efficient Deep Learning for Graph Drawing (DL4GD)"](https://pubmed.ncbi.nlm.nih.gov/36378788/), IEEE TVCG 30(2), 2024.

**Core idea.** CoRe-GD combines graph coarsening, GNN layout refinement, and positional rewiring to learn stress-optimizing embeddings at sub-quadratic runtime. DL4GD similarly shows unsupervised graph-layout learning without ground-truth layouts. Dagua already has multilevel and a NeuLay archive/competitor path, but the specific modern idea not tried in the native dispatcher is **GNN as a candidate initializer for saturated graphs**, followed by dagua's existing directed losses and polish. This avoids turning dagua into a trained-layout system while using the GNN to escape bad continuous basins.

**Predicted impact.** Main candidates: `dependency_500` and `small_world_500`. CoRe-GD optimizes stress, which correlates with local/global distance preservation but not necessarily depth Spearman. Therefore use it only when current solve is saturated and edge CV remains bad. Expected gain: +1 to +3 on large cyclic/flat graphs; uncertain +0.5 to +2 on `dependency_500`.

**Implementation difficulty.** 500-900 LOC if built natively with a tiny PyTorch GCN and coarsening; lower if reusing archived NeuLay code, but still substantial because the integration must be deterministic and testable.

**Risk of regression.** Medium-high. Learned/parameterized init may fight directed-depth metrics and increase runtime. Candidate-score and restrict to N >= 200.

## Technique 5 - NNP-NET / Fast tsNET And Word2VecGD As Large-Graph Candidate Generators

**Citation.** Ilan Hartskeerl, Tamara Mchedlidze, Simon van Wageningen, Peter Vangorp, Alexandru Telea, ["NNP-NET: Accelerating t-SNE Graph Drawing for Very Large Graphs by Neural Networks"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.22), GD 2025. Amyra Meidiana, Seok-Hee Hong, Kwan-Liu Ma, ["BH-tsNET, FIt-tsNET, L-tsNET: Fast tsNET Algorithms for Large Graph Drawing"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.54), GD 2025 poster. Minglai Yang and Reyan Ahmed, ["Word2VecGD: Neural Graph Drawing with Cosine-Stress Optimization"](https://arxiv.org/abs/2509.17333), arXiv cs.CG/cs.LG, 2025.

**Core idea.** NNP-NET accelerates tsNET by replacing t-SNE projection with a neural projection method. BH/FIt/L-tsNET reduce tsNET's runtime with Barnes-Hut, interpolation, and linear-time entropy/probability approximations. Word2VecGD replaces exact graph-distance stress with random-walk embeddings and cosine-stress. Dagua already has tsNET/UMAP pipelines, so the new value is not "add tsNET"; it is to compute a fast alternative initial embedding for large graphs where exact APSP or stochastic stress is too costly or too noisy.

**Predicted impact.** Potentially useful for `small_world_500`, maybe `dependency_500` if used as x-order init while preserving layers. It is less promising for lattice graphs because neighborhood embeddings tend to round/cluster rather than preserve deterministic grids. Expected gain: +0.5 to +2 on large close-loss graphs, with uncertain direction.

**Implementation difficulty.** 300-700 LOC for a deterministic Word2VecGD-lite using seeded walks or deterministic BFS-context windows; 700+ LOC for NNP-NET-style neural projection.

**Risk of regression.** Medium. These methods optimize neighborhood preservation, not dagua's composite. They should remain candidate initializers, never default replacement dispatch.

## Technique 6 - Beyond-Planarity / Shape-First SAT For Non-Planar Regular Graphs

**Citation.** Sergey Pupyrev, ["OOPS: Optimized One-Planarity Solver via SAT"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.14), GD 2025. Giordano Andreola et al., ["A Walk on the Wild Side: A Shape-First Methodology for Orthogonal Drawings"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.35), GD 2025. Timo Brand et al., ["Using Reinforcement Learning to Optimize the Global and Local Crossing Number"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.56), GD 2025 poster.

**Core idea.** None of these papers is "Petersen layout" specifically. They do, however, give modern candidate-generation tools for small non-planar graphs: encode a beyond-planar structure, solve or heuristically improve crossing constraints, then realize coordinates. For dagua, the conservative version is a tiny-N candidate path: for regular non-planar graphs, generate circular/symmetric, one-planar/SAT-informed, and layered candidates, then let deterministic composite choose.

**Predicted impact.** Only relevant if `petersen_10` is truly still a loss in the active benchmark. If sibling measurements are right and Petersen is already a win, this should not be prioritized. Expected gain on Petersen if needed: +1 to +3; suite-wide expected gain small.

**Implementation difficulty.** 250 LOC for brute-force/circular/symmetric candidates; 800+ LOC plus a SAT dependency for OOPS-like logic.

**Risk of regression.** Low if candidate-scored for N <= 12; high if SAT/orthogonal drawings leak into general straight-line scoring.

## Anti-Recommendations

- **Bundling-aware graph drawing.** Daniel Archambault et al., ["Bundling-Aware Graph Drawing"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2024.15), GD 2024, improves bundled-edge quality metrics, but dagua's composite counts straight-line crossings/length/depth. Bundling can improve human readability while leaving the benchmark unchanged or worse.
- **Metric-fooling / arbitrary-shape work as an optimizer.** Simon van Wageningen, Tamara Mchedlidze, Alexandru Telea, ["Same Quality Metrics, Different Graph Drawings"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.7), GD 2025, is a warning that metric bundles can be gamed. It argues for deterministic validation and visual checks, not for generating adversarial layouts.
- **Pure stress minimization as a universal fix.** Gavin Mooney et al., ["Stress in Graph Drawings: Perception, Preference, and Performance"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.38), GD 2025, reinforces that stress is important but not equivalent to dagua's composite. Dagua already has stress routes; residual failures are more about directed layering, grid regularity, and scoring stability.
- **General DR/GD framework papers as implementation targets.** Fernando Paulovich, Alessio Arleo, Stef van den Elzen, ["When Dimensionality Reduction Meets Graph (Drawing) Theory"](https://diglib.eg.org/handle/10.1111/cgf70105), CGF 2025, is useful conceptual framing, but not a concrete sprint-21 integration.

## Recommended Implementation Order

1. Deterministic candidate scoring for polish and structural candidates. It makes every follow-up experiment more trustworthy.
2. Lattice harmonic/conformal candidate gated behind existing composite picker.
3. Gap-constrained layered local search for N >= 200 DAGs.
4. GNN/Word2Vec/NNP initializers only if `dependency_500` or `small_world_500` remain losses after 1-3.
5. Petersen/beyond-planarity candidate path only if current HEAD still loses on `petersen_10` under the canonical seeded harness.
