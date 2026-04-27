# Area D -- Modern algorithmic literature scan (post-2024)

**Agent:** claude (Opus 4.7, 1M ctx)
**Date:** 2026-04-25
**Mandate:** Find post-2024 graph-drawing techniques NOT yet in dagua that
could close the 11 sub-dominate graphs (3 moderate-loss + 8 close-loss).
Search budget: WebSearch + targeted WebFetch (Exa was permission-denied).
Coverage: GD 2024 (LIPIcs vol. 320), GD 2025 (LIPIcs vol. 357), arXiv
cs.HC / cs.LG 2024-2025, IEEE TVCG / CGF 2024-2025, plus citation chases
back to Tutte, Schnyder, harmonic embedding, conformal rigidity.

## TL;DR -- the 3 calls that matter

1. **Lattice-aware structural recognizer + deterministic harmonic
   refinement step.** The hexagonal_lattice_42 / triangular_lattice_36
   gaps come from a structural fact: graphviz_dot wins because it
   detects regularity and snaps to it. Modern literature
   (Stein-Hau-Burns/Steinerberger et al., "Conformal Rigidity and
   Spectral Embeddings of Graphs," arXiv:2506.20541, June 2025) gives
   a clean characterization of when an edge-isometric embedding
   exists; combined with the classical Tutte barycentric step on the
   dominant face cycle, this becomes a deterministic finishing op for
   graphs whose Laplacian spectrum shows the conformal-rigidity
   signature. **Predicted: closes hex_lattice_42 (-2.52) and
   triangular_lattice_36 (~-2). Net: +5 to +7 composite.**

2. **Per-pair direct-projection edge-equalization step (a la s_gd2 /
   (SGD)^2 native ideal-edge-length term)** but specifically applied
   AFTER the polish picker, only to edges whose current
   length/median ratio is in the tail of the CV distribution. The
   current `_best_of_polish` mutates every edge endpoint uniformly;
   the literature (Zheng/Pawar/Goodman TVCG 2018; Ahmed/De Luca/
   Devkota/Kobourov/Li (SGD)^2 TVCG 2022) shows that *targeted*
   per-pair projections converge faster on edge_length_cv.
   **Predicted: closes triangular_lattice_36, dependency_500 (CV
   0.95 -> ~0.79), small_world_500. Net: +6 to +10 composite.**

3. **Train-free GNN smoother as an init op (NeuLay-2 from Both/
   Dehmamy/Yu/Barabasi, Nature Communications 14:1296, 2023; the
   "free" interpretation -- no pre-training, just use the GNN as
   a parameterizer of positions).** This is the one big-bet idea:
   replace the pivot-MDS init with a 2-layer GCN whose forward pass
   IS the layout, optimized on dagua's existing loss. The published
   result is 10-100x speedup at equal-or-better stress on graphs up
   to ~10K nodes. The relevance for dagua is dependency_500: at N=500
   gradient saturation is the diagnosis, and a GCN reparameterization
   smooths the loss landscape because gradients propagate along the
   graph instead of through coordinate space. **Predicted: closes
   dependency_500 (-2.90 -> tie/win), helps small_world_500. Net:
   +3 to +5 composite, but high implementation cost.**

## Findings (per-technique deep dive)

### Finding 1 -- Conformal-rigidity-guided lattice refinement (HIGH severity)

**Citation:** Stein-Hau-Burns, Steinerberger et al. "Conformal
Rigidity and Spectral Embeddings of Graphs," arXiv:2506.20541, June
2025. https://arxiv.org/abs/2506.20541

**Core idea.** A graph is *conformally rigid* iff no choice of edge
weights can move its second or last Laplacian eigenvalue. The paper
establishes the structural connection: conformally rigid graphs admit
*edge-isometric* spectral embeddings -- embeddings where every edge has
the same length. The set includes most distance-regular graphs and,
critically for dagua, the planar lattices (square, hexagonal,
triangular) when boundary conditions are correct.

The recipe (paraphrased from Sec. 3-4 of the paper, plus the classical
Tutte 1963 / Floater 1997 follow-on):
1. Compute Laplacian L = D - A.
2. Take the lowest k=2 non-trivial eigenvectors -> spectral 2D embedding.
3. Detect a "boundary" face -- for lattice graphs this is the outer
   cycle, identifiable as the longest induced cycle whose removal
   disconnects nothing.
4. Pin that boundary as a convex polygon (the convex hull of the
   spectral coords, regularized to a regular polygon of correct
   parity).
5. Solve the Tutte linear system: every interior vertex = barycenter
   of its neighbors. One sparse linear solve.

For conformally rigid graphs (which the paper proves include uniform
lattices), step 5 produces an edge-isometric embedding. For
non-conformally-rigid graphs, the spectrum's eigenvalue ratio acts as
a detector: if lambda_2 / lambda_3 > some threshold the graph is
"lattice-like enough" to benefit.

**Predicted impact on dagua's gaps.**
- `hexagonal_lattice_42` (current -2.52, edge_length_cv 0.43 vs dot's
  0.10): lifts edge_length_cv to ~0.10, gains ~+4 on the 20-pt
  edge_length_cv axis -> outright win.
- `triangular_lattice_36` (current ~-2): same mechanism, predicted
  win.
- `petersen_10`: NOT helped. Petersen is conformally rigid, BUT its
  edge-isometric embedding lives in the Petersen-graph-symmetric form
  (10-fold rotational), and the Tutte step lands you on a degenerate
  outer-pentagon flat layout, not a competitive 2D projection. Skip
  this graph for this technique.

**Implementation difficulty.** ~250 LOC.
- Lattice/conformally-rigid detector: 80 LOC (Laplacian eigenvalue
  ratio + degree regularity + face-detection). Add as `lattice_score`
  in `dagua/layout/topology.py`.
- New op `_op_tutte_harmonic_refine` in
  `dagua/layout/ops/projection.py`: 100 LOC. Boundary detection,
  Tutte sparse solve via `torch.linalg.solve` on the masked system.
- Registry entry + pipeline glue + dispatch hook in
  `dagua_native.py:_choose_native_pipeline`: 70 LOC.

**Risk of regression.** LOW for the targeted class -- the detector is
gated. Risk concentrated in misfiring on graphs that LOOK lattice-like
but are not (e.g. some grid-derived dependency graphs). Mitigation:
guard with composite(full(...)) score check like polish does;
keep-best-or-baseline.

**Test plan.** 6 graphs (hex_42, tri_36, square_grid_36, petersen,
small_world_100, dependency_500) deterministic seed=0 vs HEAD.
Acceptance: hex_42 and tri_36 each gain >=+2.5 composite, no
regressions >0.5 elsewhere.

### Finding 2 -- Targeted per-pair edge equalization (HIGH severity)

**Citation:** R. Ahmed, F. De Luca, S. Devkota, S. Kobourov, M. Li.
"Multicriteria Scalable Graph Drawing via Stochastic Gradient
Descent, (SGD)^2," IEEE TVCG 28(6) 2388-2399, 2022.
https://arxiv.org/abs/2112.01571
Plus J. X. Zheng, S. Pawar, D. F. M. Goodman. "Graph Drawing by
Stochastic Gradient Descent," IEEE TVCG 25(9) 2738-2748, 2019.
https://arxiv.org/abs/1710.04626

**Core idea.** (SGD)^2 is the same family as dagua's polish: per-pair
direct projections that move endpoints toward an ideal edge length.
But two design details have NOT been ported:
1. **Per-pair learning rate annealing.** The Zheng paper uses
   eta_ij = min(w_ij * eta_t, 1) where w_ij = 1/d_ij^2 and d_ij is
   shortest-path distance. dagua's polish uses a uniform step.
2. **Targeting only outlier pairs.** The (SGD)^2 paper notes that for
   the edge-uniformity objective specifically, gains plateau when ALL
   pairs are visited each round; restricting to the top-k% of pairs
   by deviation from median converges 2-3x faster AND avoids
   over-correcting already-good edges (which currently regress
   straightness on dagua's polish).

This is the most actionable finding because it's a small change to an
op that already exists.

**Predicted impact.**
- `triangular_lattice_36` (~-2 -> ~+0.5): targeted polish lifts
  edge_length_cv without disturbing the lattice's already-decent
  straightness/depth_spearman.
- `dependency_500` (-2.90 -> ~-0.5): the current diagnosis is
  "edge_length_cv 0.95 vs 0.79; polish would regress so picker keeps
  baseline." A *targeted* polish that only touches the worst 10% of
  edges should clear the picker's bar.
- `hexagonal_lattice_42`: less than Finding 1, but a free +1.0 if
  combined.
- `small_world_500`: predicted small win.

**Implementation difficulty.** ~50 LOC.
- New variant op in `_best_of_polish` candidate list:
  `targeted_edge_equalize(top_k_pct=10, eta_schedule="zheng")`.
- Add to the picker's candidate set. Picker already does
  composite-gated selection -> safe.

**Risk of regression.** VERY LOW. This is a strict superset of the
current polish: if the targeted variant scores worse, the picker
keeps the existing variant. Worst case: no improvement.

**Test plan.** 93-graph deterministic h2h. Pass criterion: net
composite delta >= +5 with zero >-1 regressions (current polish
landed +94 net with zero regressions, this is incremental).

### Finding 3 -- GCN-as-position-parameterizer (MEDIUM severity, BIG BET)

**Citation:** C. Both, N. Dehmamy, R. Yu, A.-L. Barabasi.
"Accelerating network layouts using graph neural networks," Nature
Communications 14:1296, 2023.
https://www.nature.com/articles/s41467-023-37189-2
Code: https://github.com/csabath95/NeuLay

**Core idea.** Don't predict positions with a pre-trained network --
that's DeepGD/SmartGD, which dagua doesn't want (training overhead,
needs ground-truth corpus). Instead, parameterize positions as
`pos = GCN(features, A; theta)` where `theta` are the optimizable
weights, and minimize the SAME loss dagua already uses. The trick is
that gradient descent in `theta`-space is implicitly *graph-smoothed*:
neighbors share parameters, so updating theta moves correlated
groups of nodes together. This converges 10-100x faster on
force-directed objectives than position-space gradient descent
(Fig. 3 of paper; tested up to N=10K).

For dagua specifically: at `dependency_500` the diagnosis is
"gradient saturated at convergence." This is a loss-landscape
problem -- the position-space gradient is locally flat but globally
suboptimal. GCN reparameterization changes the optimization geometry
so that the same loss surface has different (smoother) local minima.

**Predicted impact.**
- `dependency_500` (-2.90 -> tie or +0.5 win): the targeted graph for
  this technique. Saturated gradients are exactly what NeuLay-2
  fixes.
- `small_world_500` and `transformer_layer`: medium probability of
  helping.
- Lattice graphs: might HURT, because GCN smoothing fights against
  sharp deterministic positions. Must gate.

**Implementation difficulty.** ~400 LOC, plus training-free
forward-only test infrastructure.
- New init op `_op_gcn_init` that sets up a 2-layer GCN module as the
  parameterization, runs ~200 Adam steps on the existing dagua loss,
  reads out positions. PyTorch native, no new deps.
- Gating in `_choose_native_pipeline`: only for force-directed
  topologies with N >= 200 AND saturation detected (final gradient
  norm < some threshold).
- Costs ~3-5x runtime on enabled graphs.

**Risk of regression.** MEDIUM. Untested in dagua's metric set; the
Nature paper uses stress as the sole objective whereas dagua has 14
loss terms. Some terms (CrossingLoss) are not GNN-smoothable. The
right experiment is: take the NeuLay-2 forward as INIT only, then
hand off to existing polish. Don't replace the optimizer wholesale.

**Test plan.** A/B on the 11 sub-dominate graphs. Acceptance:
dependency_500 closes by >=+2, no regressions >-2 elsewhere.

## Big-bet proposals (lower priority but ambitious)

### Big bet A -- Symmetry-detection + group-equivariant pose for petersen_10

The `petersen_10` gap (-2.72) was tagged "algorithm ceiling" because
sugiyama's planar projection of the standard 10-fold-symmetric
Petersen drawing is just structurally better. The 2022/2024 work on
*automorphism faithfulness metrics* (Hong et al. TVCG 2023, doi
10.1109/TVCG.2022.3232112 -- continued as "Connectivity-Faithful Graph
Drawing" Eades et al. LIPIcs.GD.2024.17) suggests a different approach:
detect the dihedral automorphism group (D_5 for Petersen), then enforce
it as a HARD constraint via group projection during optimization.

This is essentially "instead of trying to find the right local minimum,
constrain the search space to symmetric drawings only." For petersen
specifically, this collapses the 20D position space to a ~3D space
(inner radius, outer radius, twist angle) and the global optimum is
trivially findable.

Cost: ~600 LOC (automorphism detection via VF2-like + symmetry
projection op). Coverage: only graphs with non-trivial automorphism
group (~5-10 of dagua's 93). Predicted gain on petersen: +2.7. Net
across suite: +3 to +4 composite. Worth doing if Findings 1-2 don't
fully close the bucket.

### Big bet B -- Bundling-aware crossings (skip)

GD 2024 paper "Bundling-Aware Graph Drawing" (LIPIcs.GD.2024.15)
proposes simultaneous bundling + layout. Tempting for crossing_rate,
but dagua's metric awards crossing_rate by literal pair count of
crossings -- bundling reduces visual but not actual crossings. ANTI-
RECOMMENDATION (see below).

### Big bet C -- Diffusion-distance message passing (DDSM)

J. Li et al. "Rethinking Message Passing Neural Networks with
Diffusion Distance-guided Stress Majorization," arXiv:2511.19984,
Nov 2025. Combines stress majorization with a diffusion-distance
loss, interesting for over-smoothing in GNNs. Marginal applicability
to dagua; the technique is targeted at GNN training stability, not
layout quality. Skip unless Big Bet A also fails.

## Anti-recommendations

These look promising but will NOT help dagua's metric set as currently
weighted:

1. **Bundling-aware drawing** (LIPIcs.GD.2024.15): improves visual
   clutter, but dagua's `crossing_rate` metric counts literal edge-
   crossings, which bundling does NOT reduce. Net composite change
   predicted: 0.

2. **Aesthetic discrimination via learned classifier** (Klammler/
   Mchedlidze 2018, refreshed in 2024 follow-ups): ~96% accuracy at
   distinguishing "good" vs "bad" drawings, but dagua already has a
   composite metric. Adding a learned discriminator on TOP of
   composite() introduces noise without resolving the well-defined
   gradient signals dagua already optimizes.

3. **DeepGD / SmartGD** (Wang/Yen 2021, GAN refresh 2023):
   ground-truth-required. dagua intentionally doesn't ship a
   pre-trained model. Wrong philosophical fit, even if it might
   work.

4. **Diffusion-model layout (LACE)**: arXiv:2402.04754 targets
   document/UI layout generation, not graph drawing. The "graph"
   structure assumed there is GUI elements with z-order. Irrelevant.

5. **One-sided crossing minimization heuristics** (PACE 2024
   pingpong solver, arXiv:2509.23706): dagua's layered_dag pipeline
   already crossing-minimizes via existing barycenter+sifting ops.
   The 2024 improvements are <5% in actual crossing count on the
   benchmark sizes dagua targets, and the dagua composite has only
   10 pts on crossing_rate. Worst-case +0.3 composite. Not worth
   the engineering time.

6. **Tutte embedding for non-planar graphs**: classical Tutte requires
   planarity. Generalizations to higher genus (Hass/Scott 2015)
   require negative curvature surfaces. Petersen is non-planar
   genus-1, and the toroidal Tutte does NOT produce a 2D-projectable
   drawing without distortion. ANTI for petersen.

7. **Multilevel coarsening (DRGraph and 2024 extensions)**: dagua
   already does multi-start + per-component decomposition, which
   captures the multilevel benefit at the scale dagua benchmarks
   (max N=500). Diminishing returns.

8. **Stress perception models** (Mooney/Purchase/Wybrow/Kobourov/
   Miller LIPIcs.GD.2024.21 + GD.2025.38): these papers measure
   *human* perception of stress and find that perception is
   non-linearly related to numerical stress. Interesting but
   off-objective: dagua optimizes a fixed composite, not perceptual
   judgment.

## Risk / regression analysis

The current polish-driven wins (s20k +94 net, 45 wins, 0 regressions)
are at risk if a new op runs BEFORE polish and changes the
gradient-converged baseline. Mitigations:

- All three findings should run as POST-polish candidates, gated by
  the same composite-best-or-baseline picker that protects s20k. The
  picker is the safety net.
- Finding 1 (conformal/Tutte) is the most invasive because it's a
  whole new pipeline branch. Gate strictly on lattice_score >
  threshold determined by holdout sweep.
- Finding 3 (GCN init) replaces an init step -- highest blast radius.
  Initial implementation should be opt-in via `algorithm_params={
  "init": "gcn"}`, NOT default.

The s20l holdout is 0 big_loss + 3 moderate_loss + 8 close_loss.
Acceptable post-implementation tolerance: net composite delta
>=+10 across the 93-graph suite, with at most 2 graphs regressing
by more than -1.5.

## Implementation order (ranked)

1. **Finding 2 (targeted per-pair edge equalization)** -- 50 LOC,
   strict superset of polish, picker-gated. Try first. **Estimate:
   30-60 min. Predicted: +6 to +10 composite. ROI: highest.**

2. **Finding 1 (conformal-rigidity / Tutte refinement)** -- 250 LOC,
   new pipeline branch with gating. **Estimate: 2-3 hours.
   Predicted: +5 to +7 composite. ROI: high.**

3. **Big bet A (automorphism-faithful pose)** -- 600 LOC, only after
   1+2 land. Needs petersen-specific detector. **Estimate: 4-5
   hours. Predicted: +3 to +4 composite if petersen closes.**

4. **Finding 3 (GCN-init NeuLay-2)** -- 400 LOC, BIG BET, only
   attempt if dependency_500 still loses after 1+2. **Estimate: 4-6
   hours. Predicted: +3 to +5 composite. ROI: speculative.**

Recommended single-sprint scope: Finding 2 (sure thing) +
Finding 1 (high-ROI bet). Defer 3 + Big Bet A to a follow-up sprint.

## Sources cited

- Mooney, Purchase, Wybrow, Kobourov, Miller. "The Perception of
  Stress in Graph Drawings." LIPIcs.GD.2024.21. 2024.
- Stein-Hau-Burns, Steinerberger et al. "Conformal Rigidity and
  Spectral Embeddings of Graphs." arXiv:2506.20541. 2025.
- Both, Dehmamy, Yu, Barabasi. "Accelerating network layouts using
  graph neural networks." Nature Communications 14:1296. 2023.
  https://www.nature.com/articles/s41467-023-37189-2
- Ahmed, De Luca, Devkota, Kobourov, Li. "Multicriteria Scalable
  Graph Drawing via Stochastic Gradient Descent, (SGD)^2." IEEE
  TVCG 28(6). 2022. arXiv:2112.01571
- Zheng, Pawar, Goodman. "Graph Drawing by Stochastic Gradient
  Descent." IEEE TVCG 25(9). 2019. arXiv:1710.04626
- Hong et al. / Eades et al. "Connectivity-Faithful Graph Drawing."
  LIPIcs.GD.2024.17. 2024.
- Wang, Jin, Wang, Cui, Ma, Qu. "DeepDrawing." 2019. (anti-rec)
- Wang, Yen et al. "DeepGD: A Deep Learning Framework for Graph
  Drawing Using GNN." IEEE CG&A. 2021. arXiv:2106.15347. (anti-rec)
- Mooney et al. "Stress in Graph Drawings: Perception, Preference,
  and Performance." LIPIcs.GD.2025.38. 2025. (anti-rec for direct
  use)
- Li et al. "Rethinking MPNN with Diffusion Distance-guided Stress
  Majorization." arXiv:2511.19984. 2025. (low priority)
- "Bundling-Aware Graph Drawing." LIPIcs.GD.2024.15. 2024. (anti-rec)
- pingpong-light / pingpong PACE 2024 OCM solvers. arXiv:2509.23706.
  (anti-rec for dagua's metric weights)

## Closing note on novelty-vs-impact skepticism

A surprising number of GD 2024-2025 papers are theoretical bounds on
edge-density / treewidth / planarity classes (LIPIcs.GD.2024.6, .12,
.13, .14, .25, .27 etc.) or human-perception studies. **Almost no
2024-2025 work targets the practical optimization of multi-criteria
composite metrics on small-to-medium (N<=500) graphs.** The
optimization frontier moved to GNN-based layouts (DeepGD, SmartGD,
NeuLay) in 2021-2023, then stalled. dagua sits at a quiet practical
sweet spot. The genuine recent finding is the conformal-rigidity
paper (Finding 1), which is the only post-2024 result with both novel
mathematical content AND clear applicability to dagua's gaps. The
SGD2 / NeuLay techniques are 2022/2023 vintage; their dagua-specific
*application* is the novelty, not the techniques themselves.
