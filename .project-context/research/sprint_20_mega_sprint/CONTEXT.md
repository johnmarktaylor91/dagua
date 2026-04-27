# Sprint 20 — Mega-Sprint Shared Context

## Mandate

JMT directive: **"This is the turning point where we can really create
something special. Everything is on the table — big additions/changes
are fine. Try as hard as possible to catch up to competitors (ideally
pass them) for the graph structures we still lose at. Different handling
for directed vs undirected is fine. Stay in the spirit of dagua's
'differentiable when possible, non-differentiable when it yields a
performance boost'. Don't regress strengths. Strike a balance — avoid
Frankenstein patchwork. Be ambitious. Iterate till you can think of no
further improvements."**

## Where dagua stands today

Branch: `feat/bench-and-aesthetics`, HEAD `ec7d4db` (sprint-19h).
Benchmark: `eval_output/variant_bench_full` (93 graphs <= 500 nodes,
cached competitor positions in `positions/`).

### Per-competitor head-to-head (post sprint-19)

| Competitor | Wins | Ties | Losses | Avg advantage |
|---|---|---|---|---|
| graphviz_dot | 63 | 12 | 18 | +4.11 |
| dagre | 74 | 13 | 6 | +7.16 |
| igraph_sugiyama | 77 | 10 | 6 | +10.38 |
| elk_layered | 74 | 7 | 12 | +10.85 |
| graphviz_sfdp | 90 | 1 | 2 | +29.59 |
| nx_spring | 93 | 0 | 0 | +34.97 |
| igraph_kamada_kawai | 93 | 0 | 0 | +30.60 |

Dagua leads on mean composite vs every competitor; closest threat is
graphviz_dot at +4.11.

### Per-graph wins to PROTECT (must not regress > 0.5 each)

| Graph | dagua | best competitor | Δ |
|---|---|---|---|
| org_chart_deep | 91.64 | elk(68.98) | +22.67 |
| random_dag_200 | 65.21 | dagre(44.33) | +20.88 |
| hub_fanout_label_skew | 92.67 | dot(76.43) | +16.24 |
| org_chart_1_5_4_8 | 95.89 | dot(80.26) | +15.64 |
| random_dag_50 | 61.30 | dagre(45.80) | +15.50 |
| random_bipartite_60 | 80.39 | elk(65.97) | +14.42 |
| edge_label_braid | 91.96 | dagre(79.35) | +12.61 |
| bipartite_4_3_4 | 80.68 | dot(68.07) | +12.61 |
| weighted_karate_34 | 71.68 | dot(59.37) | +12.31 |
| real_karate_34 | 71.68 | dot(59.37) | +12.31 |

### Per-graph losses still open — sprint-20 targets

| Graph | dagua | best competitor | Δ | Bucket |
|---|---|---|---|---|
| ragged_feature_pyramid | 69.52 | elk(79.56) | -10.04 | NEW from sprint-19, pyramid |
| planar_60 | 65.82 | elk(75.06) | -9.25 | NEW, planar mid-size |
| small_world_100 | 48.58 | sugiyama(57.08) | -8.51 | structural — no hierarchy |
| disconnected_label_cycle_collage | 74.41 | elk(79.36) | -4.95 | tiny disconnected cyclic |
| small_world_500 | 49.34 | elk(54.16) | -4.82 | structural — no hierarchy |
| parallel_cycles_4x5 | 58.24 | sfdp(62.73) | -4.49 | NEW, parallel cycles |
| transformer_layer | 76.18 | dot(80.19) | -4.00 | NEW, layered DAG |
| regular_3_30 | 68.37 | dot(72.23) | -3.86 | NEW, regular graph |
| hexagonal_lattice_42 | 85.21 | dot(88.99) | -3.77 | planar lattice (closing) |
| dependency_500 | 54.46 | elk(58.19) | -3.73 | large sparse DAG (closing) |

### Composite metric weights (sum = 100)

- dag_consistency: 25 (LOWER edges-going-against-y → higher score)
- edge_length_cv: 20 (LOWER stddev/mean → higher score)
- depth_spearman: 15 (correlation between graph-depth and y-position)
- overlap_count: 10 (binary: 0 overlaps → 10 pts, else 0)
- edge_straightness: 10 (LOWER deg from layer axis → higher score)
- crossing_rate: 10 (LOWER crossings/edge-pair → higher score)
- angular_resolution: 5 (HIGHER deg between adjacent edges)
- cluster_separation: 5

Files: `dagua/metrics.py` (composite at L1147).

## What we did in sprint-19 (avoid duplicating)

- 56c3b93 sprint-19a: cycle-reversal pre-pass for init layering.
- 64731dd sprint-19b: greedy FAS sign bug fix + cycle gate expansion.
- 8b4bead sprint-19b follow-up: relax cycle-reversal gate for small graphs.
- b423607 sprint-19c: 3 metric/cycle correctness bug fixes (segments_intersect
  sign, make_acyclic_robust self-loops, _greedy_fas).
- 8c0a332 sprint-19c pt2: metric determinism cleanup.
- 33868c2 sprint-19d: per-component decomposition wrapper.
- d1ba9ef sprint-19e: topology-aware aspect ratio (lattice/wide families
  get a different target than the default 0.25).
- 092ee79 sprint-19f: median sweep + transpose heuristic crossing reduction.
- 519f58e sprint-19g: Brandes-Köpf x-only refinement for layered DAGs.
- ec7d4db sprint-19h: dummy-node long-edge splitting in native pipeline.

Wave-1 + wave-2 research lives in
`.project-context/research/sprint_19_improvement_scan/`. Read those
to avoid retreading: area_A_algorithm_core, area_B_loss_buckets, area_C_bug_hunt,
area_D_runtime, area_E_metric_gaps, plus 5 implementation plans in
wave2_gpt55/.

## Architectural state of the codebase

- `dagua/layout/engine.py` -- entry point, dispatcher to pipelines.
- `dagua/layout/ops/pipelines/dagua_native.py` -- the default pipeline,
  now ~1500 lines after sprint-19 patches. Per-component wrapper, then
  per-component pipeline: NativeEngineInit -> Force2DInitIfFlat ->
  optional dummy expansion -> gradient core (Adam) -> BarycenterReorder ->
  MedianSweep -> TransposeHeuristic -> BrandesKopf x-refine ->
  OverlapProjection -> AspectRatioFit -> ClusterGridArrange.
- `dagua/layout/ops/pipelines/sugiyama.py` -- alternate pipeline; opt-in
  via algorithm="sugiyama". 24 algorithm pipelines in `pipelines/` total.
- `dagua/layout/ops/` -- 268+ registered composable primitives.
- `dagua/layout/graph_classify.py` -- topology classification (used by
  sprint-19e to pick aspect targets). Tags include planar_dag,
  lattice_like, dense_dag etc.

The pipeline is starting to look complex. Frankenstein-risk is real.
A research deliverable should call this out and propose a clean
topology-dispatch architecture if you think one is warranted.

## Sprint-20 mandate to research agents

This is a **research** dispatch. You produce a markdown findings file.
**Do not write code or commit.** A separate implementation pass will
follow. Aim for ambition: propose big architectural changes if the
evidence supports them, including:

- A separate force-directed sub-pipeline for graphs with no hierarchy
  (small_world, dense random, etc.) — dagua has zero story here today.
- A directed-vs-undirected split that picks different objectives.
- Borrowing modern techniques from the literature post-2022 we haven't
  tried (GraphSAGE-init, Stress Majorization, Constrained Stress, etc.).
- Even rewriting the default pipeline if the wave-1+wave-2 patches have
  accumulated into something messy.

Ground proposals in measured evidence — actual h2h numbers, actual
metric breakdowns. Don't propose a change that costs +30% runtime for
+0.2 composite — quantify the tradeoff.

## Where to write your output

`.project-context/research/sprint_20_mega_sprint/<area>_<agent>.md`

Each report MUST include:

1. **TL;DR** (4-6 bullets) — what's the call.
2. **Findings** — each with severity (high/med/low), evidence (file:line,
   real h2h numbers), proposed change.
3. **Big-bet proposals** — even if not all will land, list the ambitious
   ideas + their projected impact + what we'd give up.
4. **Risk / regression analysis** — what current wins are at risk?
5. **Implementation order** — what should be tried first vs later, and why.

## Reference commands

```
# Per-graph diagnostic
CUDA_VISIBLE_DEVICES="" python /tmp/diag_single.py <graph_name>

# Mini h2h on the worst loss + winners
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_quick.py
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_winners.py

# Full 93-graph h2h (~4 min CPU)
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_wins.py

# Inspect a competitor's cached layout
torch.load("eval_output/variant_bench_full/positions/<graph>__<engine>.pt")
```

## Cached competitor positions

`eval_output/variant_bench_full/positions/` — 8 engines x 93 graphs.
Engine names: graphviz_dot, graphviz_sfdp, elk_layered, dagre,
nx_spring, igraph_kamada_kawai, igraph_sugiyama, plus seeded variants.
