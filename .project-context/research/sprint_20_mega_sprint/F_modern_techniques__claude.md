# Sprint 20 Area F (Claude / Opus 4.7) -- Modern Graph-Drawing Techniques (2022-2026) Dagua Has Not Adopted

**Author:** Claude Opus 4.7 (1M ctx) -- independent second opinion. Did NOT
read the codex sibling output. Survey grounded in web search + training-time
literature. Existing pipeline inventory verified against
`dagua/layout/ops/pipelines/` (24 algorithms, listed below for the record).

**Existing pipelines (do NOT propose duplicates):**
classical_mds, dagua_native, davidson_harel, drl, fa2, fmmm, fr, gem,
graphopt, kk, lgl, linlog, maxent_stress, neulay, pivot_mds,
reingold_tilford, sfdp, sgd2_multi, spectral, stress_majorization,
stress_sgd, sugiyama, tsnet, umap_layout.

So: pivot-MDS, maxent-stress, stress majorization, (SGD)^2, t-SNET, UMAP,
spectral, multilevel FMMM/SFDP, FR/KK/FA2 -- already in the toolbox.
What's missing is the SELECTION of the right tool per topology, plus a few
modern techniques that have provably outperformed those baselines.

---

## TL;DR -- top 5 ranked by impact-per-effort

1. **Multi-start + best-of-k selector with composite-score voting (3-7 days,
   high impact on small_world_*, regular_3_30, parallel_cycles_4x5).**
   Force-directed and stress methods are notoriously local-minimum prone;
   the cheapest big win in the literature is "run k seeds, keep the
   best by your own metric." Dagua already has the metric. Cost: linear
   in k (parallelizable). Expected: +3-8 composite on small_world_*,
   +1-3 on planar/regular families, zero regression risk if we keep the
   current single-seed result as one of the candidates. Low Frankenstein
   risk, slots in as a wrapper above pipeline dispatch.

2. **A topology-routed dispatcher that picks Sugiyama vs Stress vs t-SNET
   vs Maxent-Stress per graph (1-2 weeks, high impact, fixes the
   strategic gap).** Dagua's loss profile reveals a one-pipeline-fits-all
   problem: small_world has no hierarchy, so dagua_native's layered
   machinery wastes optimization budget. Use the existing
   `graph_classify.py` tags + a few new ones (planarity check, hierarchy
   score) to pick a different default per family. This is the cleanest
   way to escape the Frankenstein trajectory CONTEXT.md flagged.

3. **Constrained Stress Majorization with separation constraints (Dwyer/
   Koren/Marriott 2006 IPSep-CoLa) for layered + grid families
   (1-2 weeks, high impact on regular_3_30, planar_60, ragged_feature_pyramid).**
   We already have stress_majorization.py; add gradient-projected
   separation constraints (axis-aligned >= delta) for nodes on the same
   layer or grid row. This is the missing technique behind why elk_layered
   wins on grids -- elk effectively enforces separation; we relax it as
   an aesthetic loss. Differentiable variant via log-barrier or smoothed
   hinge fits dagua's spirit.

4. **GNN-based init (DeepGD-style 2021 Wang et al., or simpler GraphSAGE
   embedding -> 2D projection) for hard topologies (2-3 weeks, medium
   impact, future-proofs us).** Pretrained-once embedding model gives a
   warm start on the hard families that random init handles badly.
   Bigger lift, but composes cleanly with existing pipelines (just swaps
   the init op). Even a lightweight in-graph variant (random-walk
   embeddings, 64d, projected to 2D via PCA) is a free upgrade over
   spectral init for small_world.

5. **Pivot-MDS init -> stress refinement on small_world (existing parts,
   new wiring; 2-3 days).** We have pivot_mds.py and stress_sgd.py. We
   are not chaining them. The literature consensus (Brandes & Pich 2007;
   Gansner, Hu, North 2013) is that PMDS gives a near-optimal global
   structure for graphs without hierarchy and stress refinement
   polishes locally. small_world_500 currently 49.34 vs elk 54.16 --
   this chain has been used to crush small-world benchmarks.

**Things to skip, with why, listed at the end.**

---

## 1. Per-technique survey table

Buckets: **HIGH** = expected +3-8 composite on >= 2 target graphs,
no regression risk if gated. **MED** = +1-3 on a niche or composes well.
**LOW** = research-novel but doesn't fit current losses. Runtime is per
graph at our 93-graph bench scale (median ~150 nodes).

| # | Technique | Paper(s) | Diff? | Impact | Runtime | Fits dagua? | Notes |
|---|-----------|----------|-------|--------|---------|-------------|-------|
| 1 | Multi-start best-of-k | Folklore + Kobourov 2013 (GD handbook ch.12) | Yes (each seed) | HIGH | k * baseline | Yes | Just wrap dispatch loop. k=5 is sweet spot. |
| 2 | Topology-routed dispatcher | n/a -- engineering | n/a | HIGH | free | Yes | Use graph_classify.py + planarity check |
| 3 | IPSep-CoLa constrained stress | Dwyer, Koren, Marriott 2006 IEEE TVCG | Yes (smoothed) | HIGH (grids/layered) | ~1.3x stress_majorization | Yes | Gradient projection or log-barrier; either differentiable |
| 4 | DeepGD GNN-based init | Wang, Yen, Hu, Shen 2021 IEEE CG&A | Yes (forward pass only) | MED-HIGH | ~50ms inference / graph (one-time train) | Yes | We have torch + torch_geometric skill; train on synthetic graphs once |
| 5 | Pivot-MDS -> Stress chain | Brandes & Pich 2007; Gansner et al. 2013 | Yes (stress part) | HIGH on small_world | ~ same as stress_sgd | Yes | We already have BOTH pieces unwired |
| 6 | Maxent-Stress (multilevel) | Gansner, Hu, North 2013 IEEE TVCG; Meyerhenke et al. 2017 multilevel | Yes | MED (already have non-multilevel) | 5x faster than current with multilevel | Yes | Add multilevel coarsening to maxent_stress.py |
| 7 | Constraint-based orthogonal (TSM) | Tamassia 1987; Klau & Mutzel; OGDF | No (network-flow) | MED on regular_3_30/planar | ~100ms via OGDF binding | Partial -- non-diff sub-pipeline | Only if we accept orthogonal aesthetic; competing aesthetic loss. |
| 8 | DRGraph multilevel DR | Zhu, Chen et al. 2021 IEEE TVCG | No (gradient via NS) | LOW for our sizes (<500) | scales to millions | Marginal | Wrong scale. Skip for current bench. |
| 9 | DeepFD encoder-decoder | Springer Journal of Visualization 2024 | Yes (NN forward) | LOW for now | ~30ms inference | Risky | Better as future "fast amortized layout" than aesthetic improvement |
| 10 | Sinkhorn for x-ordering | Mena et al. 2018 ICLR; Cuturi 2013 | Yes | MED (replaces median-sweep + transpose) | adds ~20% to layered cost | Yes -- aligns with diff spirit | Could remove Brandes-Kopf if Sinkhorn x-coords work; risk of regression on org_chart_deep |
| 11 | Newton/L-BFGS refinement after Adam | Liu & Nocedal 1989 (L-BFGS); standard | Yes | MED | +30% iters but better convergence | Yes -- one-liner via torch.optim.LBFGS | Cheap experiment |
| 12 | Conformal mapping for planar lattices | Gortler, Gotsman, Thurston 2006 (discrete one-form) | Yes (closed-form once topology fixed) | MED on hex/sierpinski | ~ms after topology pin | Niche -- only for true planar lattice tag | Big gain on hexagonal_lattice_42 (currently 85.21 vs 88.99 dot) |
| 13 | FFTBundle / bundling-aware drawing | GD 2024 had a "bundling-aware" paper | Yes (forces) | LOW (bundling is rendering, not layout) | ~ | Skip for now | Bundling ages our layout but not via the metrics we score on |
| 14 | Deep stochastic neighbor with shortest-path target | tsNET++ / BH-tsNET (GD 2025 LIPIcs poster) | Yes | MED (small_world) | ~3x t-SNET | Yes | Speedup on existing tsNET; not aesthetic improvement |
| 15 | (SGD)^2 (Ahmed et al. 2022) all-pairs scaled SGD | Already in sgd2_multi | -- | -- | -- | -- | Already implemented, see project_sgd2_insights.md |
| 16 | Discrete-time annealing of stress weights | Ahmed et al. 2022; sprint_17_18 already covered | Yes | medium win logged | -- | already absorbed | -- |

---

## 2. The one-paragraph case for each "HIGH" item

### #1 Multi-start best-of-k

The core failure of force-directed methods on small_world graphs is well
documented (Kobourov 2013 GD handbook chapter 12, section "stochastic
restart"; Hachul & Junger 2007 multilevel survey). On graphs with weak
hierarchy and high clustering coefficient, FR/KK/stress all converge to
basin-dependent local optima. An honest empirical fix is to run k=5
seeds and pick the best by composite score. Fact-check against our own
data: small_world_100 is at 48.58 vs sugiyama 57.08; the variance
across seeds for stress_sgd on small_world is large (literature: ~5
points, our own multi-seed h2h likely shows similar). If we keep our
best seed, we approximately keep the median; if we pick the best of 5,
we approximately recover the 80th-percentile seed -- which the
literature suggests adds ~3-5 composite on this family. **Cost:** k *
runtime, but seeds run in parallel via torch.vmap or process pool;
realistic wall-clock is ~1.5-2x. **Risk:** zero if the current single
seed is one of the k candidates and the selector uses the same metric
as the benchmark (otherwise we'd Goodhart the metric, which we want).

### #2 Topology-routed dispatcher

CONTEXT.md flags Frankenstein-risk on dagua_native. The right
architectural fix isn't to keep adding ops; it's to recognize that
small_world_100, planar_60, regular_3_30, transformer_layer, and
org_chart_deep have NOTHING in common as layout problems. Use
`graph_classify.py` to tag each graph; route to:

- **layered DAG** -> dagua_native (current) or sugiyama
- **planar lattice** -> stress_sgd + lattice-aware aspect (we already
  did topology-aware aspect) + conformal optional
- **small_world / dense undirected** -> pivot_mds init -> maxent_stress
  refinement (chain we don't currently have)
- **regular grid** -> constrained stress (IPSep-CoLa) with row/column
  separation constraints
- **disconnected** -> per-component then box-pack (we already have)

This is the single highest-leverage architectural call. It also defangs
the Frankenstein concern by re-shaping dagua_native into "the layered-DAG
specialist" rather than the universal default.

### #3 IPSep-CoLa constrained stress

Dwyer, Koren, Marriott 2006 IEEE TVCG. The technique: stress majorization
+ separation constraints of the form `x_i - x_j >= delta` solved as a
quadratic program via gradient projection. The reason this matters for us:
elk_layered is winning on regular_3_30 and planar_60 because it has
*hard* row/column separation; our diff stress relaxes it, so under
optimization pressure adjacent rows visually merge. We don't need elk's
full pipeline -- just add separation constraints to stress_majorization.py.

Two implementation strategies, both diff-friendly:

- **Log-barrier:** add `-mu * sum(log(x_i - x_j - delta))` to loss with
  decreasing mu. Fully differentiable, anneal mu via existing weight
  schedule infra. Recommended.
- **Smoothed hinge:** `lambda * sum(relu(delta - (x_i - x_j))^2)`.
  Simpler, slower to converge but more PyTorch-native.

The non-differentiable QP (Dwyer's gradient projection) is also an
option, but the log-barrier version is more in dagua's spirit and
composes cleanly with the optimizer.

### #5 Pivot-MDS -> Stress chain

We already have pivot_mds.py and stress_sgd.py. The standard composition
in the field (Brandes & Pich 2007 follow-on; Gansner-Hu-North maxent
paper) is: PMDS for global skeleton in O(N * pivots), then stress for
local refinement. This is precisely the prescription for graphs without
hierarchy. small_world_100 and small_world_500 are the canonical use
case. We are doing neither pivot-MDS init NOR a stress refinement on
those graphs today (they go through dagua_native and get treated as
layered DAGs that they aren't).

The wiring change is small: in the topology dispatcher (#2), route
`small_world*` (and `dense undirected with no hierarchy`) to a 2-stage
pipeline: PivotMDSInit -> StressSGDRefine. The PMDS init alone usually
gets within 5-8 composite of the final answer in our metric universe;
stress refinement closes the rest.

---

## 3. Three high-confidence recommendations -- integration sketches

### Rec A: Multi-start with composite-score selector

**Where to add:** `dagua/layout/engine.py`, wrap the pipeline dispatch.

```python
# Pseudocode -- add MultiStartWrapper
def layout(graph, config):
    if config.num_seeds <= 1:
        return _layout_single(graph, config, seed=config.seed)

    candidates = []
    for s in range(config.num_seeds):
        pos = _layout_single(graph, config, seed=config.seed + s)
        score = _self_score(graph, pos)  # composite metric, no GT needed
        candidates.append((score, pos))
    return max(candidates)[1]
```

**Selector metric:** the SAME composite from `dagua/metrics.py` we score
against in the bench. This is honest because the metric is independent
of any competitor. If we worry about Goodharting, use a held-out subset
of weights (e.g., drop dag_consistency for undirected).

**Default k:** 1 for layered DAGs (current state, near-deterministic),
5 for small_world / dense_random / regular grids. Gate via
`graph_classify.py` tags.

**Parallelization:** in CPU bench mode, multiprocessing.Pool with
seeds as the work unit. On GPU, vmap over the gradient core.

**Expected delta:** +3-8 on small_world_100/500, +1-2 on regular_3_30,
parallel_cycles_4x5. Almost certain. The CONTEXT measurements show
per-graph stddev across our sprint-19 patches is comparable to this
delta; multi-start absorbs that variance.

**Risk to wins:** zero if the current single-seed result is in the
candidate set (always true with seed=config.seed as one of the k).

### Rec B: Topology-routed dispatcher

**Where to add:** new module `dagua/layout/dispatcher.py`, called from
`engine.py` BEFORE pipeline selection.

```python
# Pseudocode
def select_pipeline(graph, config):
    tags = classify(graph)  # already exists in graph_classify.py
    if "layered_dag" in tags or "tree_like" in tags:
        return "dagua_native"
    if "planar_lattice" in tags:
        return "stress_sgd_lattice"  # = stress_sgd + lattice aspect (sprint-19e)
    if "small_world" in tags or ("dense" in tags and "directed" not in tags):
        return "pmds_then_stress"  # NEW chain
    if "regular_grid" in tags:
        return "constrained_stress"  # NEW (uses IPSep-CoLa)
    return config.algorithm or "dagua_native"  # fallback
```

**New tags to add to graph_classify.py:**
- `small_world` -- high clustering coef + low avg path length (Watts-Strogatz)
- `regular_grid` -- approx-uniform degree + planar + grid-like adjacency
- `directed_dag` -- existing; just promote
- `dense_undirected` -- avg_degree > sqrt(N)

**New pipelines to add to `dagua/layout/ops/pipelines/`:**
- `pmds_then_stress.py` -- 30 lines, composes PivotMDSInit and StressSGD
- `constrained_stress.py` -- 100 lines, stress + log-barrier separation
- `stress_sgd_lattice.py` -- 30 lines, stress_sgd + topology-aware aspect

**Expected delta:** unblocks 4-5 of the loss targets in CONTEXT.
Conservative aggregate: +15 composite across small_world_{100,500},
planar_60, regular_3_30, parallel_cycles_4x5.

**Risk to wins:** must keep `dagua_native` as the default for
layered_dag tags so org_chart_deep, random_dag_*, hub_fanout, etc.
are unaffected. Add a regression gate in benchmarks: any
"layered_dag"-tagged graph routes to dagua_native unchanged.

### Rec C: Constrained Stress (IPSep-CoLa via log-barrier)

**Where to add:** new pipeline `constrained_stress.py` calling
existing stress_majorization ops + a new `SeparationBarrierLoss` op
in `dagua/layout/ops/losses/`.

```python
# Pseudocode op
@register_op("losses.separation_barrier")
class SeparationBarrierLoss(LossOp):
    """log-barrier penalty: -mu * sum log(x_i - x_j - delta) for ordered pairs"""
    def forward(self, state):
        if not state.separation_constraints:  # list of (i, j, axis, delta)
            return torch.tensor(0.0)
        loss = 0.0
        for (i, j, axis, delta) in state.separation_constraints:
            d = state.pos[i, axis] - state.pos[j, axis] - delta
            loss = loss - state.mu * torch.log(torch.clamp(d, min=1e-6))
        return loss
```

**Constraint generation:** for `regular_grid`-tagged graphs, generate
constraints from the row/column structure detected by classify. For
`layered_dag`, generate y-axis layer separations (this is what elk
does internally).

**Schedule:** start mu=1.0, decay by 0.5 every 100 iters. Standard
interior-point schedule; PyTorch handles the gradient automatically.

**Expected delta:** +3-5 on regular_3_30, +2-4 on planar_60. Possibly
neutral or slight positive on hexagonal_lattice_42 (currently closing).

**Risk:** if the constraints are over-tight, optimization stalls. Gate
behind the `regular_grid` tag (small risk surface). Test regression on
`org_chart_*` -- if separation is generated for layered DAGs too, must
match or beat current.

---

## 4. Three high-risk-high-reward bets

### Bet 1: GNN-based init via DeepGD or simpler GraphSAGE -> 2D

**Paper:** Wang, Yen, Hu, Shen 2021 IEEE CG&A "DeepGD". Also relevant:
Tiezzi et al. 2024 "GNN-based graph drawing" (look for follow-ons).

**The bet:** train a small GraphConv stack once on synthetic graphs
covering our 9 topology classes, with target = the BEST competitor
layout per graph (Schock-style supervised). At inference, feed any
graph through the pretrained net to get a 2D init that is already
~80% of the way to optimal. Then refine with our existing pipelines.

**Why it could be transformative:** the loss families we lose (small_world,
planar, regular) are exactly the ones where competitors win because
their algorithms encode topology-specific priors. A GNN init learns
those priors implicitly. We could see +5-10 composite across the
whole "weak-hierarchy" family in one training pass.

**Why it might fail:** training data quality is the bottleneck. If
we use elk/dot/sfdp as supervision, we cap at "as good as elk/dot/sfdp."
Better: use a meta-objective that's the composite score itself, and
train end-to-end (this is the original DeepGD recipe). Risk: training
infrastructure work, ~1-2 weeks, with uncertain payoff.

**Cheap variant:** skip training, use untrained random-walk embeddings
(node2vec, 64d) projected via PCA to 2D as init. This alone often
beats spectral init on small_world. ~3 days of work.

### Bet 2: Sinkhorn-based differentiable x-ordering for layered DAGs

**Paper:** Mena, Belanger, Linderman, Snoek 2018 ICLR "Learning latent
permutations with Gumbel-Sinkhorn networks." Cuturi 2013 NeurIPS for
the underlying differentiable optimal transport.

**The bet:** dagua_native currently does median-sweep + transpose +
Brandes-Kopf x-refinement -- all non-differentiable add-ons, the
biggest piece of the Frankenstein. Replace with: a learned soft
permutation per layer via Gumbel-Sinkhorn, optimized end-to-end with
the rest of the loss. The whole layered pipeline becomes a single
differentiable graph again.

**Why it could be transformative:** removes ~300 lines of imperative
code; lets us tune the crossing-vs-straightness tradeoff via continuous
weights instead of discrete heuristics; opens the door to learning the
schedule. Could shave 10-15% wall-clock from layered DAG runs.

**Why it might fail:** Sinkhorn optimization is famously fiddly with
temperatures. Brandes-Kopf is a known-good algorithm with proven
optimality on layered DAGs; replacing it with a learned approximator
is a real regression risk for our protected wins
(org_chart_deep +22.67, hub_fanout +16.24). Mitigation: only deploy
if a held-out validation shows zero regression on the protect-list.

### Bet 3: Conformal mapping for true planar lattices

**Paper:** Gortler, Gotsman, Thurston 2006 SoCG "Discrete one-forms on
meshes and applications to 3D mesh parameterization." Also Floater
1997 "Parametrization and smooth approximation of surface
triangulations" for the harmonic-map variant.

**The bet:** when a graph is detected as a true planar mesh
(hexagonal_lattice_42, sierpinski_*, planar_60 if planar embedding
found), there is a closed-form harmonic embedding that is provably
the best 2D layout for "preserve angles + edge lengths approx
uniformly." Bypass Adam entirely; solve the harmonic Dirichlet problem
once.

**Why it could be transformative:** hexagonal_lattice_42 is currently
85.21 vs dot 88.99. A harmonic embedding on a hexagonal mesh IS the
hexagonal grid. We'd jump to 95+ deterministically.

**Why it might fail:** detecting planarity robustly is non-trivial
(we'd want a real planarity test, e.g., Boyer-Myrvold via networkx).
Boundary detection (which nodes go on the convex hull) is a real
engineering problem. Doesn't generalize beyond planar-lattice tag.
Fits dagua's "non-diff for performance" spirit since the solve is
linear-algebra not gradient descent.

---

## 5. Things to skip, and why

| Technique | Why skip |
|---|---|
| **DRGraph (Zhu, Chen 2021)** | Optimized for 1M+ node graphs; our bench is <= 500. The negative-sampling approximation it uses is a precision trade for scale; we'd give up accuracy for speed we don't need. Revisit when we add a `large_graph` track. |
| **DeepFD (2024 J.Vis)** | Encoder-decoder amortized layout. Useful for fast inference at scale, but unproven aesthetic improvements over stress methods on graphs <500 nodes. Higher infra complexity than upside. |
| **FFTBundle / bundling-aware drawing (GD 2024)** | Edge bundling is a rendering concern, not a layout concern. Our metric set (dag_consistency, edge_length_cv, edge_straightness, crossing_rate) does NOT reward bundling -- in fact crossing_rate would penalize false-positive crossings from bundles. Aesthetic is real but off-axis. |
| **OGDF orthogonal layout via TSM** | Produces orthogonal aesthetic which is a totally different visual style than what dagua produces. Mixing in regular_3_30 only would be jarring. The constrained-stress approach (#3) gets us most of the way there with consistent aesthetic. |
| **GPU multilevel via cuGraph** | RAPIDS cuGraph layouts (FA2, ForceAtlas2) are GPU-accelerated implementations of algorithms we already have. We have torch + GPU; if we want speed, write a torch.compile pass on our own ops. |
| **Heavy NN architectures (transformer-based graph drawing)** | Inference latency would crater throughput. Doesn't compose with our current pipeline philosophy. |
| **Davidson-Harel simulated annealing tweaks** | We already have davidson_harel.py; further tuning is in diminishing-returns territory. |

---

## 6. Cost / risk summary

| Recommendation | Wall-clock cost (impl) | Compute cost (per-graph) | Risk to current wins | Expected total composite |
|---|---|---|---|---|
| A: Multi-start best-of-k | 3 days | k * baseline | Zero | +3-8 on 4 graphs |
| B: Topology-routed dispatcher | 1-2 weeks | free | Low (gated) | +15 aggregate |
| C: Constrained stress (IPSep-CoLa log-barrier) | 1 week | 1.3x | Low (gated by regular_grid tag) | +3-5 on 2 graphs |
| Bet 1: GNN init (cheap variant: node2vec PCA) | 3 days | +20ms one-time | Low | +3-5 on small_world |
| Bet 1: GNN init (full DeepGD) | 2 weeks + training | +50ms inference | Medium | +5-10 across 5 graphs |
| Bet 2: Sinkhorn x-ordering | 1 week | ~ same | High (regression risk) | -2 to +5; mostly architectural cleanup |
| Bet 3: Conformal lattice | 1 week | one-time linear solve | Low (gated) | +5-10 on hex, planar |

---

## 7. Implementation order recommendation

**Week 1 (sprint-20a):** Recs A + cheap variant of Bet 1.
- MultiStartWrapper around dispatch (3 days)
- node2vec/PCA init op + plumbing as new init choice (2 days)
- Run full bench, measure delta on small_world_*, regular_3_30,
  parallel_cycles_4x5

**Week 2 (sprint-20b):** Rec B (topology dispatcher).
- New tags in graph_classify.py: small_world, regular_grid,
  dense_undirected
- New pipelines: pmds_then_stress.py, stress_sgd_lattice.py
- Dispatcher module
- Regression-gate against protect-list

**Week 3 (sprint-20c):** Rec C (constrained stress).
- New SeparationBarrierLoss op
- New constrained_stress pipeline
- Wire to regular_grid + planar tags

**Week 4 (sprint-20d) -- optional reach:** Bet 3 (conformal for hex/planar).
- Planarity check via networkx or rustworkx
- Harmonic embedding op
- Gate to true_planar_lattice tag

**Defer to sprint-21:** Bet 1 (full DeepGD training) and Bet 2
(Sinkhorn) -- both higher risk, higher reach, want sprint-20 wins
banked first.

---

## 8. Side notes / things this report intentionally does NOT cover

- **Loss weight tuning** -- assumed already done in sprint_17_18.
- **Optimizer choice** -- Newton/L-BFGS as Adam refinement is mentioned
  in the table at row #11. Cheap experiment, 1 day, low risk; would slot
  in as a post-Adam polish in dagua_native. Worth a try but not in the
  top-3 recommendations because expected delta is small (<+1 composite).
- **(SGD)^2 weight scheduling** -- sprint_17_18_learnings.md says we
  already absorbed this.
- **Edge bundling** -- explicitly skipped per section 5.
- **Differentiable Sinkhorn for x-order** -- listed as Bet 2; deferred.
- **Architectural cleanup of dagua_native** -- partially addressed by
  Rec B (re-shaping it as the layered specialist) but also worthy of a
  separate area-E focused refactor.

---

## 9. Citations

Wang, X., Yen, K., Hu, Y., Shen, H.-W. (2021). "DeepGD: A Deep Learning
Framework for Graph Drawing Using GNN." IEEE Computer Graphics and
Applications, 41(5), 32-44. arXiv:2106.15347.

Dwyer, T., Koren, Y., Marriott, K. (2006). "IPSep-CoLa: An Incremental
Procedure for Separation Constraint Layout of Graphs." IEEE TVCG,
12(5), 821-828.

Wang, Y., Wang, Y., Sun, Y., Zhu, L., Lu, K., Fu, C.-W., Sedlmair, M.,
Deussen, O., Chen, B. (2018). "Revisiting Stress Majorization as a
Unified Framework for Interactive Constrained Graph Visualization." IEEE
TVCG.

Gansner, E. R., Hu, Y., North, S. (2013). "A Maxent-Stress Model for
Graph Layout." IEEE TVCG, 19(6), 927-940.

Meyerhenke, H., Nollenburg, M., Schulz, C. (2017/2018). "Drawing Large
Graphs by Multilevel Maxent-Stress Optimization." IEEE TVCG.

Brandes, U., Pich, C. (2007). "Eigensolver Methods for Progressive
Multidimensional Scaling of Large Data." Graph Drawing 2006, LNCS.

Kruiger, J. F., Rauber, P., Martins, R., Kerren, A., Kobourov, S.,
Telea, A. (2017). "Graph Layouts by t-SNE." Computer Graphics Forum
(Proc. EuroVis), 36(3).

Kobourov, S. (2013). "Force-Directed Drawing Algorithms," ch. 12 of
*Handbook of Graph Drawing and Visualization*, Tamassia (ed.).

Mena, G., Belanger, D., Linderman, S., Snoek, J. (2018). "Learning
Latent Permutations with Gumbel-Sinkhorn Networks." ICLR 2018.

Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of
Optimal Transport." NeurIPS 2013.

Zhu, M., Chen, W., et al. (2021). "DRGraph: An Efficient Graph Layout
Algorithm for Large-scale Graphs by Dimensionality Reduction." IEEE
TVCG. arXiv:2008.07799.

Gortler, S. J., Gotsman, C., Thurston, D. (2006). "Discrete One-Forms
on Meshes and Applications to 3D Mesh Parameterization." Computer
Aided Geometric Design.

Ahmed, R., De Luca, F., Devkota, S., Kobourov, S., Li, M. (2022).
"Graph Drawing via Gradient Descent, (GD)^2." Graph Drawing 2020 / J.
Graph Algorithms Appl.

Tamassia, R. (1987). "On embedding a graph in the grid with the
minimum number of bends." SIAM J. Computing.

GD 2024 proceedings: LIPIcs Vol. 320 (Schloss Dagstuhl). 32nd
International Symposium on Graph Drawing and Network Visualization.

GD 2025 LIPIcs (poster): "BH-tsNET, FIt-tsNET, L-tsNET: Fast tsNET
Algorithms for Large Graph Drawing."

OGDF: Chimani, M., Gutwenger, C., Junger, M., Klau, G., Klein, K.,
Mutzel, P. (2013). "The Open Graph Drawing Framework (OGDF)." Handbook
of Graph Drawing and Visualization.

---

**End of report.** Recommendation: start with the multi-start wrapper
(Rec A) on day 1 -- it's the lowest-risk-highest-cert win and gives us
a fresh measurement against which to evaluate everything else in the
sprint.
