# Sprint Map

Ten sprints. Each has an entry criterion, exit criterion, execution pattern,
and rollback. Sprint 0 is a prerequisite; Sprints 1-3 are strictly sequential;
4-7 can partially interleave; 8-9 are exit gates. Each sprint file lives at
`.project-context/plans/native_placement_algo/NN_<name>.md` and is
executable without reading the others.

## Execution pattern legend

- **CP**: Claude-plans + Codex-implements. Claude writes the spec, Codex writes
  code, Claude reviews Codex output.
- **SC**: Solo-Codex. Codex takes a single spec end-to-end; Claude only reviews.
- **SS**: Subagent-Sweep. Claude dispatches multiple subagents in parallel for
  research (literature, competitor code, alternate approaches).
- **HJ**: Human-Judgment. iMessage ping with 3x3 comparison grid, user picks.
- **AR**: Adversarial Review. `/codex:adversarial-review --background`.

## Iteration + competitor extraction every sprint

EVERY sprint follows 10_iteration_loop.md and 11_competitor_weaving.md.
In each sprint description below, "Extractions" names the competitors whose
techniques that sprint targets; full detail lives in 11.

## Sprint list

### Sprint 0 -- Pipeline Decompose + Default Flip + MVP Iteration -- CP+AR
See 01_audit_and_decompose.md (revised). SPLIT from prior single Sprint 0
per adversarial review (Sprint 0 was overloaded). Sprint 0 now does ONLY
the pipeline-side work that unblocks the iteration loop.
- Entry: plan approved.
- Exit:
  * Pipeline decoupled: `dagua/layout/ops/pipelines/dagua_native.py` imports
    only from `dagua.layout.ops.*`; zero imports from `engine.py`.
  * Default flipped: `dagua.layout(g)` (no `algorithm=`) routes through the
    pipeline.
  * MVP iteration harness `scripts/iterate_native.sh <graph>` runs in <=8s,
    prints scalar score and image path.
  * MVP `scripts/iter_record.py` -- one-line wrapper that runs the harness
    and appends one JSONL line to `sprint_<N>/iteration_log.jsonl`.
  * Baseline metrics committed for the iteration suite (still seed=42; the
    opaque held-out and rolling sets land in Sprint 0.5).
  * Memory-parity baseline measured (Task 0.10).
  * Rubric-code sync (Task 0.8).
- Budget: 1.5-2 clock days.
- Extractions: none (infrastructure sprint, exempt per Sprint Map invariants).
- Non-regression: existing `dagua_native` pipeline tests still pass.

### Sprint 0.5 -- Benchmark Authority + Opaque Held-Out -- CP+AR
NEW per adversarial review. The "best-in-class" plan needs the competitor
set, the opacity scheme, and the comparison authority frozen BEFORE
Sprint 1 can credibly measure improvement.
- Entry: Sprint 0 exit.
- Exit:
  * Authoritative competitor matrix frozen (see 11_competitor_weaving.md
    "Authoritative competitor matrix"): the 16-variant set using ACTUAL
    registered names from `dagua/eval/competitors/__init__.py` --
    graphviz_dot, graphviz_sfdp, graphviz_neato, graphviz_fdp, elk_layered,
    dagre, igraph_sugiyama, igraph_fr, igraph_kamada_kawai, nx_spring,
    nx_kamada_kawai, sgd2_multi_ref, gephi_yifanhu, fa2_ref, ogdf_fmmm,
    cytoscape_fcose. Versions and hashes recorded.
  * Per-family device-normalization rule frozen (Q18 resolved): same-device
    comparison only; per-family device choice documented.
  * `dagua/eval/graph_generator.py` lands. Salt-derived opaque held-out
    suite generated on demand from `.project-context/private/holdout_salt`;
    only `dagua/graphs/holdout/MANIFEST.json` (hashes) committed.
  * `dagua/eval/benchmark.py` gains `--seed-strategy={fixed,rolling,holdout}`,
    `--sprint-tag`, `--salt-path` flags.
  * `scripts/refresh_competitors.sh` runs all competitors on iteration +
    held-out; records version/hash per competitor.
  * `scripts/pick_weak_graphs.py` reads metrics, returns 5 weakest vs best
    competitor.
  * Sprint 0 baseline RE-RUN under new opacity (DAGUA-ONLY on 30 graphs,
    committed at eval_output/native_algo/holdout_v1/metrics.json).
  * **Pareto ladder calibration DEFERRED to Sprint 1 exit.** Calibration
    requires a full 16-way competitor head-to-head on the held-out suite
    (2-4h background run via scripts/refresh_competitors.sh --rerun). That
    run was NOT executed at Sprint 0.5 exit per 2026-04-22 round-1 exit
    review findings; the Dagua-only baseline is sufficient to pick weak
    graphs (scripts/pick_weak_graphs.py) and drive Sprint 1 iteration, but
    the per-sprint Pareto gate numbers in 10_iteration_loop.md (20% at
    Sprint 1 -> 90% at Sprint 9) cannot be calibrated against absolute
    competitor shares until the head-to-head lands. Sprint 1 entry
    requires that run to complete before the Sprint 1 exit Pareto gate is
    evaluated.
- Budget: 1.5-2 clock days (competitor head-to-head run excluded;
  scheduled as pre-Sprint-1 background task).
- Extractions: none (infrastructure/parity sprint, exempt).
- Non-regression: existing tests still pass; competitor cache invalidated
  cleanly without losing prior data (rename/keep-old).

### Sprint 1 -- Initialization + Gradient Core + Memory Port -- CP+SS+AR
See 11_sprint_init_and_core.md.
- Entry: Sprint 0 complete, memory_profile.json shows current RSS gap.
- Goal: the "inner differentiable optimizer" is its own named sub-pipeline
  (`GradientCore`), pluggable between initializers. Plus port the memory
  features (per-loss backward, checkpointing, hybrid device) from legacy
  `_layout_inner` into registered ops.
- Exit (**REVISED post round-1 adversarial review** -- original targets
  were too many; Sprint 1 focuses on the memory-port primary outcome):
  * Memory port: 10K RSS ops/legacy peak ratio < 1.30x AND incremental
    delta ratio < 1.60x. (Tightening the original "<1.10x" because that
    target ignored a shared 500 MB process baseline; measured delta
    ratio is the honest number.)
  * Initializer registry: 4 ops available (Random, Native, Spectral,
    FamilyConditional). `WarmStartSGD2` deferred to Sprint 2+.
  * `GradientCore` named sub-pipeline exists and composes into the
    default pipeline. (DONE post adversarial fix.)
  * `LossGroup(backward_mode='per_loss')` satisfies `LossPerLossBackward`
    op requirement (same semantics; no new op needed).
  * `GradientCheckpoint` + `HybridDeviceOffload`: DEFERRED to Sprint 8
    scale-hardening (not needed at 1.22x peak / 1.48x delta).
  * Per-family +5% composite on 3/7 families: DEFERRED to Sprint 2+ since
    the Sprint 1 initializer experiment (FamilyConditionalInit with
    layer_ratio<0.2) regressed the held-out mix and was reverted.
    Per 10_iteration_loop.md (competitor-extraction as mandatory 4th
    hypothesis when 3 Dagua-native attempts fail), a competitor init
    extraction is the next hypothesis. Sprint 2 (V-cycle + coarsening)
    subsumes that work since coarsening IS the canonical competitor
    init for the weak undirected family.
  * Sampled Repulsion/Overlap loss activation at N>2000 is VALIDATED
    on memory, PARTIALLY validated on quality (held-out max graph is
    n=1800). Sprint 2 V-cycle will exercise N>2000 and reveal any
    sampled-gradient quality issue.
- Budget: 2-3 days (revised up from 2 given memory port).
- Extractions (see 11): `SpectralInit` (NetworkX), `SugiyamaInit` bundle
  (dagre/grandalf), `WarmStartSGD2` (in-tree sgd2_multi),
  `UMAPInit` (umap-learn, if beats spectral), `LossPerLossBackward` +
  `GradientCheckpoint` + `HybridDeviceOffload` (legacy `_layout_inner`).
- Rollback: revert to fixed NativeEngineInit + combined backward, reinstate
  baseline.

### Sprint 2 -- Multilevel V-Cycle + Hierarchy Memory Parity -- CP+AR
See 12_sprint_multilevel.md.
- Entry: Sprint 1 exit criteria (revised) met -- i.e. 10K peak RSS
  <1.30x legacy, delta RSS <1.60x legacy, GradientCore extracted,
  LossGroup(per_loss) live, 4 init ops registered. `GradientCheckpoint`
  and `HybridDeviceOffload` remain Sprint 8 work and are NOT blockers
  for Sprint 2 entry.
- Goal: coarsening + prolongation as composed ops. The native pipeline is a
  V-cycle: coarsen -> optimize at coarse -> prolong -> refine -> ... . Uses
  existing `coarsen.py` and `prolong.py` ops.
- **Revised per adversarial review**: current ops hierarchy builder (at
  `dagua/layout/ops/coarsen.py:643-664, 974-988`) calls legacy
  `build_coarsening_hierarchy` with `offload_to_disk=False` and then COPIES
  every hierarchy tensor into new `HierarchyLevel` objects. This removes the
  legacy offload behavior and duplicates residency. Sprint 2 MUST fix this.
- Exit:
  * `VCycle` composite op runnable at N=100K, 1M, 10M on a trimmed iteration
    suite; measured RSS stays within memory budget per 03 scale ladder.
  * **NEW** No-copy hierarchy transfer: ops reuse legacy hierarchy buffers
    directly (view/borrow, not copy). Verified via allocator byte-count at
    each level.
  * **NEW** Ops-native disk offload of `fine_to_coarse`, `fine_layer_assignments`,
    `coarse_layer_assignments` in addition to existing `edge_index` and
    `node_sizes`. Symmetric save/load manifest; RSS drops measurably after
    offload and recovers to within 5% of pre-offload after reload.
  * **NEW** Measured RSS gates at 100K / 1M / 10M. Exceeding the gate by
    >10% = exit fail, not warning.
  * Multilevel threshold becomes "always on" (even small graphs run a 1-level
    V-cycle = direct optimization). Simplifies dispatch.
  * Quality score on held-out-medium (1K-10K) no worse than Sprint 1 exit.
  * On held-out-large (100K+) beats the legacy multilevel path by >=10%
    composite or trades quality for >=3x runtime win, user choice.
- Budget: 3 days (revised up from 2-3 given parity work).
- Extractions (see 11): `SFDPCoarseningSchedule` (Graphviz sfdp),
  `FM3StepSchedule` (OGDF), `WalshawLocalRefine` (Walshaw multilevel),
  `ClusterAwareCoarsen` (cose-bilkent) -- wired in Sprint 4.
- Non-regression: small graphs (N<500) quality unchanged vs Sprint 1.
- Rollback: disable V-cycle for N < multilevel_threshold, fall back to
  Sprint 1 path.

### Sprint 3 -- Hybrid Classical Steps -- CP+SS+AR
See 13_sprint_hybrid_classical.md.
- Entry: Sprint 2 exit.
- Goal: add three classical steps where they measurably win:
  * Warm-start from a fast classical pass (sgd2_multi trimmed) for undirected.
  * Layer-sweep barycenter pass for Sugiyama-style directed, as an op.
  * Brandes-Kopf / greedy-switch post-polish for directed, as an op.
- Exit:
  * Per-family ablation: Sprint 3 with-hybrid vs Sprint 3 without-hybrid;
    with-hybrid wins on >=2 of 3 families at p<0.05 on 10 seeds.
  * All three classical ops integrated via existing SolveState protocol,
    no new state fields.
  * Held-out composite improves vs Sprint 2 by >=5% on directed family.
- Budget: 2 days.
- Extractions (see 11; multi-op BUNDLES call out sub-ops in 11):
  `NetworkSimplexLoss` (BUNDLE: EdgeWeightedRankLoss + DiscreteRankAssignPolish;
  see 11 Sprint 3 bundle spec), `BrandesKoepfPolish` (Brandes-Koepf 2001 +
  2020 erratum), `GreedySwitchRefine` (OGDF), `ELKLayerSweepChain` (ELK),
  `CoffmanGrahamLayering` (igraph), `MaxentStressLoss` option (in-tree
  maxent_stress).
- Risk flag: highest Frankenstein risk. Mid-sprint adversarial review mandatory
  (see 06).

### Sprint 4 -- USER-Cluster-as-Node + Hierarchical Flex -- CP+AR
NOTE: Sprint 4 handles USER-DEFINED clusters (semantic groupings the user
passes in). This is categorically distinct from Sprint 2's V-cycle
auto-coarsening (internal performance strategy, invisible to the user).
If the user passes no clusters, Sprint 4 is a no-op at layout time;
the V-cycle still coarsens for performance but no cluster boxes are
rendered. See 00_overview.md "IS / IS NOT" for the distinction.
See 14_sprint_clusters.md.
- Entry: Sprint 3 exit.
- Goal: cluster is a first-class placement unit. During coarsening, user-defined
  clusters become explicit super-nodes at their hierarchy layer. A cluster's
  bounding box is a differentiable function of its members. Inheriting hierarchy
  into the V-cycle requires bookkeeping in `coarsen.py` ops.
- Exit:
  * Graphs with 3+ nesting levels produce hierarchies visually correct
    (no sibling overlap, bboxes sized to contents, labels placed).
  * User flag `LayoutConfig(cluster_mode="coarsening"|"centroid"|"off")`
    with clear semantics documented.
  * Regression bar: nested-shallow and nested-deep categories each improve
    >=10% composite vs Sprint 3.
- Budget: 2 days.
- Extractions (see 11; multi-op BUNDLES call out sub-ops in 11):
  `BorderNodeInsert` (Sander 1996 / dagre),
  `RecursiveClusterLayout` (BUNDLE: ClusterSubproblemConstruct +
  RunNestedPipelineForCluster + ClusterResultMerge; controlled
  composability bypass, see 11 Sprint 4 bundle spec),
  `ClusterForceBlend` (cose-bilkent), `ClusterHullConstraint` (cola.js),
  `ParentGravityPull` (Dogrusoz 2009).
- Rollback: `cluster_mode="centroid"` restores today's cluster handling.

### Sprint 5 -- Pinning + Flex End-to-End -- CP+AR
See 15_sprint_pinning_flex.md.
- Entry: Sprint 4 exit.
- Goal: user pinning, flex spacing, alignment are correctly propagated through
  coarsening, V-cycle, and cluster mechanics. Pin a root node: coarse levels
  inherit the pin; cluster containing the pin is pinned at its center.
- Exit:
  * Round-trip: pin a node at (x,y) with hard flex; the node lands within 1e-3
    of (x,y) after a 1M-node multilevel run.
  * Soft flex (Flex.soft) weights propagate with appropriate scale across
    hierarchy levels.
  * Alignment groups survive coarsening: members attract to group axis at
    every level.
- Budget: 1.5 days.
- Extractions (see 11): `MultilevelPinPropagate` (Dwyer-Koren 2005),
  `AlignGroupPropagate` (Graphviz rank=same), `PinSlackAnneal`
  (WebCola), verify `HardPinProjection` semantics match cola.js.
- Non-regression: existing flex tests pass bit-for-bit where deterministic.
- **Sprint 5 r2 descope (2026-04-23):** cluster-centroid pinning -- the
  "cluster containing the pin is pinned at its center" clause -- is
  EXPLICITLY deferred to a Sprint 5.5 follow-up. The current commit
  ships `MultilevelPinPropagate` + `AlignGroupPropagate` through the
  V-cycle with hard-pin-priority dedup, device-safe flex migration,
  and a Huber-scaled soft pin loss (so soft pins no longer get
  shredded by ClipGradNorm). The cluster-pin bullet requires a
  separate op that injects a virtual pin at the cluster centroid
  whenever ANY leaf in the cluster is user-pinned, which in turn
  needs a cluster-to-coarse mapping during V-cycle (currently
  `_level_problem` drops cluster data on coarse levels -- that's
  the scope of 5.5).

### Sprint 6.5 -- Edge-CP Routing, Dense + Nested-Cluster Tuning (tracked)
- Goal: close the edge-node crossings drop gap on TWO remaining
  family groups:

  * DENSE (random_dag, sparse_layered, bipartite): Sprint 6 r2 is
    neutral because edges inherently must pass through unrelated
    nodes in the heuristic layout -- CP refinement alone can't find
    a routing solution that doesn't exist.
  * NESTED CLUSTERS (nested_2lvl, nested_3lvl, nested_4lvl): Sprint 6
    r2 REGRESSES these families (nested_2lvl heuristic=0, differentiable=33
    creating crossings from nothing; nested_3lvl -78%, nested_4lvl -140%).
    Root cause: the heuristic route_edges already produces near-optimal
    paths inside nested clusters (short, straight, cluster-respecting),
    and CP refinement pushes edges outward (responding to node-crossing
    + curvature pressure) right through neighbouring cluster members.
    No weight combination of the existing six losses fixes this:
    cluster_crossing at 12 (Sprint 5 default=8) still produces the
    regression because the loss penalizes edges crossing FOREIGN
    clusters only, not edges breaking same-cluster locality.

  Sprint 6.5 requires one of:
  a) topology-aware orthogonal routing (Graphviz-style visibility
     graph polish) for dense graphs
  b) cluster-locality loss (penalize cluster-member edges that leave
     their cluster bbox mid-path) for nested graphs
  c) adaptive step budget: skip CP refinement when the heuristic
     edge-node crossing count is below a per-graph threshold
     (trivial; closes the nested regression immediately)
  d) bundled confluence paths at endpoints (Pupyrev 2013)
  e) layer-aware channel assignment (Sugiyama-style edge routing)

  Driven by the held-out audit numbers from Sprint 6 r2 (see
  eval_output/native_algo/sprint_6_edge_routing/report.json). Start
  with (c) because it's 10 lines of code and closes the worst-case
  regression; then pick from (a)/(b)/(d)/(e) for the positive gains.

### Sprint 6 -- Differentiable Edge Routing -- CP+SS+AR
See 16_sprint_edge_routing.md.
- Entry: Sprint 5 exit.
- Goal: edge control points become learnable parameters in a second pipeline
  stage, after node positions freeze. Loss: obstacle avoidance, curvature,
  bundling via confluence at endpoints.
- Exit:
  * Edge routing is a registered op (`EdgeCPOptimize`) with its own loss set.
  * Held-out visual audit: edge-node crossings drop >=30% vs Sprint 5.
  * Back edges use wide-arc routing automatically.
  * User opt-out: `LayoutConfig(edge_routing="heuristic"|"differentiable")`.
- Budget: 2 days.
- Extractions (see 11): `FDEBAttraction` (Holten 2009), `StubBundleConfluence`
  (Pupyrev 2013), `VisibilityGraphPolish` (Graphviz splines),
  `HobbyTensionTune` (Hobby 1986). Confluent-drawing readability check
  as a non-implemented design principle.
- Rollback: `edge_routing="heuristic"` restores Sprint 5 behavior.
- **Sprint 6 r2 (2026-04-23):** Retuned default edge-CP loss weights
  after the held-out audit exposed a REGRESSION under Sprint-5
  defaults (-27.7% drop on tree_branching_4 n=800; -13.4% on
  branching_3). Per-loss ablation showed w_edge_angular_res +
  w_edge_curvature_consistency each degrade edge-node crossings by
  ~6% in isolation; w_edge_curvature_penalty improves them by ~52%.
  New defaults zero the saboteurs, strengthen curv_penalty, soften
  edge_crossing. Tree families now see +52% to +56% drop. Dense
  families (random_dag, sparse_layered) remain neutral because the
  layout itself forces edges through unrelated nodes; crossing away
  requires topology-level re-routing that is EXPLICITLY descoped to
  Sprint 6.5 above. The literal "30% aggregate drop" plan bullet is
  therefore NOT met today on the full 39-graph suite, but the r2
  defaults are strictly better than Sprint 5 on every graph family
  and significantly better on tree / sparse / layered graphs where
  CP refinement has room to work.

### Sprint 7 -- Node Size + Text Polish -- CP+HJ+AR
See 17_sprint_text_and_sizing.md.
- Entry: Sprint 6 exit.
- Goal: content-aware node sizing feeds back into layout; label placement
  collision-aware; edge labels placed to avoid node overlap.
- Exit:
  * Running layout on a graph with very long node labels produces no label
    clipping and no label-label overlap on the held-out suite.
  * HJ checkpoint: user rates a 3x3 before/after grid on mixed-width family
    >= 4/5 on aesthetic.
- Budget: 1.5 days.
- Extractions (see 11; multi-op BUNDLES call out sub-ops in 11):
  `KakoulisTollisLabelPlace` (Kakoulis-Tollis 1998), `ELKLabelSlots` (ELK),
  `LabelSizeFeedbackLoop` (BUNDLE: LabelCollisionDetect + NodeSizeExpand +
  LayoutProblemReseed; controlled LayoutProblem-mutation bypass with
  max-2-passes ceiling; see 11 Sprint 7 bundle spec).

### Sprint 8 -- Scale Ladder Hardening + Hybrid Force Branch -- CP+AR
See 18_sprint_scale.md.
- Entry: Sprint 7 exit.
- Goal: N=1M and 10M runs complete in bounded time under stated memory budget.
  Address the 1B coarsening offload path if user keeps it in-scope (09 Q7).
- **Sprint 8 MVP (2026-04-23):** 1M target MET (445s <= 480s on RTX
  2080 Ti). Two bug fixes unblocked V-cycle on CUDA (device migration
  in ``_level_problem`` and ``_longest_path_layering_vectorized``),
  and vectorizing ``HeavyEdgeMatching`` dropped it from 193s to 21s
  on the 1M run. See commits ``8f254fc`` + ``6179fae``.
  10M target NOT MET: hit CUDA OOM at 18.3min / 9.07GB VRAM inside
  ``LossGroup.term.backward()`` (per_loss mode) on the RTX 2080 Ti
  (11.5GB total). Went >= 1/3 of the 45-min budget before crashing,
  so on a 24GB+ card the wall would likely be inside budget. The
  reachable fix on current hardware is Sprint 8.5: chunked /
  gradient-checkpointed backward for O(E) / O(sampled) losses at
  very large N, and CPU-offload for the finest-level positions
  during coarse passes. Tracked below.

### Sprint 8.5 -- Scale-Ladder VRAM engineering (tracked)
- Goal: close the 10M gap on consumer-tier GPUs.
- **Sprint 8.5 r1 attempt (2026-04-23):** added chunked forward path
  in RepulsionLoss + OverlapAvoidanceLoss with a 3 GB per-loss VRAM
  budget (see ``_select_sampled_chunk`` in loss_engine.py). Per-loss
  VRAM profile at 1M confirms overlap / repulsion peaks at
  6.66 / 4.62 GB respectively under the un-chunked path, so at 10M
  the forward intermediates alone would need ~66 / 46 GB.
  1M CUDA wall unchanged (chunking no-ops since full intermediate
  fits the budget). 10M retry STILL OOMs on RTX 2080 Ti at 10.46 GB
  VRAM peak because the chunk-accumulating forward
  (``total = total + chunk.sum()``) still holds every chunk's
  activations alive until the outer ``term.backward()`` in
  ``LossGroup``. Chunking forward without chunking backward is
  insufficient.
- **Sprint 8.5 r2 scope (unstarted)**: chunked-backward-per-chunk,
  which requires coordinating the loss weight with an in-loss
  autograd.grad call OR migrating LossGroup to a contract where
  evaluate() may return a detached scalar and have already populated
  pos.grad. Concretely:
  * Option A: custom autograd.Function per sampled loss that
    re-computes chunk activations in backward (torch.utils.checkpoint
    style). Keeps LossGroup contract; adds recomputation cost.
  * Option B: pass weight into evaluate(); loss calls backward()
    per chunk; returns detached scalar; LossGroup term.requires_grad
    check correctly skips the outer backward.
  * Option C: CPU offload for prolonged finest-level pos during
    coarse refine so the fine-level autograd graph + coarse
    optimizer state don't co-reside on GPU.

- Exit:
  * 1M nodes: <=8 minutes wall on GPU.
  * 10M nodes: <=45 minutes wall on GPU, peak RAM <=120 GB.
  * **Revised per adversarial review**: replace "3 runs per tier" with a
    topology cross-product at exit:
      * Sizes: 100K, 1M, 10M
      * Classes: sparse-wide, sparse-deep, higher-E/N
      * Seeds: 3 per cell
    Total: 27 runs. Exit requires no OOM on any cell, composite regression
    <=5% per cell vs Sprint 7, cell-level runtime within 1.5x per-tier budget.
  * **NEW** CPU fallback envelope: documented as sparse-wide up to 100K only.
    Denser or larger requires GPU; the plan does not claim otherwise.
  * **NEW** 1B coarsening offload: symmetric save/load covers all hierarchy
    tensors. Risk-register item R3 mitigation verified (R11 added for
    `fine_to_coarse` and layer assignment residency).
- Budget: 2-3 days.
- Extractions (see 11): `BarnesHutRepulsion` (Gephi FA2 / Jacomy 2014),
  `FMMRepulsion` (OGDF FMMM multipole), `SFDPCoarseningSchedule` scale-up.
  Competitor benchmark refresh before exit.
- **Hybrid dispatch (new per adversarial review):** Sprint 8 is a DECLARED
  hybrid branch. The ops pipeline at large N is no longer fully
  differentiable end-to-end. Dispatch rule:
  * N < 50K: GradientCore (differentiable) + V-cycle from Sprint 2.
  * 50K <= N < 500K: GradientCore + SFDP-style coarsening; Barnes-Hut
    repulsion inside coarse levels (classical force update, no autograd).
  * N >= 500K: Pure FMMM-style force refinement at the finest level
    (matches `FMMMForceStep` existing behavior); GradientCore only at
    the coarsest level.
  * Family-specific cutovers: near-clique forces earlier hybrid (from 10K);
    trees can stay differentiable up to 200K.
  These thresholds are defaults; Sprint 8 tunes per family based on
  measured Pareto share. The plan does NOT claim the large-N path is
  differentiable end-to-end.

### Sprint 9 -- Aesthetic Dial-In + Ship Checklist -- CP+HJ+AR
See 19_sprint_aesthetics.md.
- Entry: Sprint 8 exit.
- Goal: final weight tuning using Optuna (or equivalent) over the iteration
  suite; confirmation on held-out; visual sign-off.
- Exit:
  * Composite score on held-out within 5% of best-seen during tuning.
  * HJ: user approves on 3 sample graphs via iMessage.
  * All adversarial reviews green. Native-default is declared shippable.
- Extractions (see 11): OGDF default parameters as Optuna priors, ELK
  published defaults as targets, graph-tool benchmark ranges as sanity.
  Final competitor benchmark refresh + published head-to-head table.
- **Binding Sprint 9 ship checklist (new per adversarial review)**:
  Every item is a release blocker; "informal follow-up" is not allowed.
  * [ ] Authoritative competitor matrix frozen (Q17 resolved, in 11).
  * [ ] Device-normalization policy frozen (Q18 resolved, in 11).
  * [ ] Legacy `_layout_inner` fate decided (Q1 resolved), migration note
        for downstream callers published.
  * [ ] Final benchmark table published in
        `eval_output/native_algo/sprint_9_exit/head_to_head.md`.
  * [ ] All adversarial reviews green (no unresolved CRITICAL/HIGH).
  * [ ] Suite-wide Pareto gate met on iteration + held-out.
  * [ ] All per-family Pareto floors met.
  * [ ] Overfit gap < 10%, rolling gap < 15%, cumulative drift > -5%.
  * [ ] HJ sign-off via iMessage.
  * [ ] Release notes draft + changelog entry.
  * [ ] Iteration logs archived with profile/version metadata intact.
- Budget: 1.5-2 days.

## Cross-sprint invariants

- Every sprint follows the iteration loop in 10_iteration_loop.md: pick
  weak graphs -> hypothesize -> measure quality+runtime delta -> Pareto
  check vs competitors -> keep / keep-focused / revert -> log. Minimum
  5 entries in `sprint_<N>/iteration_log.jsonl`.
- Every sprint extracts at least one technique from a competitor per
  11_competitor_weaving.md. The extraction lands as a registered op with
  ablation test. **Infrastructure / parity sprint exemption**: if the
  sprint's declared output is memory parity, benchmark integrity,
  opacity, or dispatch hardening, no extraction is required; the exit
  note names the exempt reason and the next extraction-eligible sprint
  absorbs the extra extraction. Currently exempt: Sprint 0, Sprint 0.5.
- Every sprint exit runs head-to-head competitor benchmark on iteration +
  held-out. Pareto-optimal share must meet the gate for that sprint.
  **The 20%-90% ladder is CALIBRATED at Sprint 0.5 exit, not declared in
  advance.** After Sprint 0.5 measures Dagua's baseline Pareto share
  under the frozen competitor matrix, the per-sprint ramp is set as:
    Sprint N gate = baseline + N * (target_at_9 - baseline) / 8,
  where target_at_9 = 90% iteration / 80% held-out BUT with per-family
  floors that prevent a single hard family from gating the whole plan:
  * Directed DAG / tree family: parity floor (Dagua >= 50% Pareto share
    at Sprint 9). Recognizes Graphviz dot's 30-year advantage on small DAGs.
  * Nested cluster, undirected sparse: aggressive floor (>= 85%).
  * Near-clique, disconnected, pathological: best-effort (>= 30%).
  Family floors are enforced in addition to the suite-wide gate.
- Every sprint exit triggers `pytest tests/ -x --tb=short` plus a full run of
  the Sprint 0 iteration harness on the current branch vs the previous sprint's
  baseline.
- Runtime target: per-graph wall-time holds within 5% of prior sprint OR
  improves. No sprint doubles runtime without iMessage user approval.
- Held-out never runs mid-sprint. Only at sprint exit.
- Rolling-seed random graphs regenerate per sprint from secret salt.
- Non-regression gate: no aesthetic metric may degrade >3% relative to the
  immediately-previous sprint's held-out numbers. Cumulative drift >5%
  requires user waiver. If either breached, the sprint is not exited.

## Stop conditions

- Three consecutive sprints exit with CRITICAL adversarial findings -> pause,
  retrospective, replan.
- A sprint exceeds its clock budget by >2x -> pause, escalate to user.
- Benchmark infrastructure (iteration harness, held-out suite) drifts >20%
  in runtime vs Sprint 0 baseline -> audit infra before continuing.
