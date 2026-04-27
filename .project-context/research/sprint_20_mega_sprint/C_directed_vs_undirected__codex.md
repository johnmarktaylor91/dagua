# C - Directed vs Undirected Handling

## TL;DR

- Dagua needs a semantic directed/undirected split. `DaguaGraph` is currently documented as directed, `LayoutConfig(algorithm=None)` always remaps to `dagua_native`, and the classifier only says whether the directed edge set is acyclic. That misses graphs whose edge orientation is an arbitrary benchmark encoding detail.
- My semantic count for the 93 benchmark graphs is: 63 directed DAG-style graphs, 6 directed cyclic/feedback graphs, and 24 effectively undirected/non-hierarchical graphs. The important trap is that 21 of the 24 effectively undirected graphs are still `is_directed_acyclic=True` because the benchmark orients undirected NetworkX graphs into DAGs.
- Current composite scoring is 50% directed-axis reward before optional edge-label terms: `dag_consistency` 25, `depth_spearman_rho` 15, and `edge_straightness_mean_deg` 10. On undirected graphs this rewards arbitrary node-index orientation.
- Under a direction-agnostic composite using the remaining 50 headline points, cached Dagua loses all 24 effectively undirected graphs to a cached competitor. Mean undirected score on that subset is `igraph_kamada_kawai` 56.72, `graphviz_sfdp` 54.96, `graphviz_dot` 54.95, `elk_layered` 52.35, `dagre` 49.82, `igraph_sugiyama` 45.38, `dagua` 42.21.
- Competitor behavior is by design. KK and SFDP layouts are isotropic force/stress layouts with much better undirected geometry, while layered engines score well today because the metric rewards the arbitrary DAG orientation.
- Proposed path: add an explicit graph/config semantic hint, preserve it in importers and benchmark graph builders, extend `GraphStructure` with semantic/no-hierarchy tags, and dispatch `algorithm=None` to `dagua_native_directed`, `dagua_native_undirected`, or `dagua_native_cyclic_hybrid`.

## Measurement Method

Primary code refs:

- Public graph object: [dagua/graph.py](/home/jtaylor/projects/dagua/dagua/graph.py:67) documents `DaguaGraph` as directed and stores only `direction: str = "TB"` at [dagua/graph.py](/home/jtaylor/projects/dagua/dagua/graph.py:101), not a semantic directedness flag.
- Public draw path: [dagua/__init__.py](/home/jtaylor/projects/dagua/dagua/__init__.py:95) uses `direction` for rendering/layout direction; it does not carry graph semantics.
- Config dispatch knob: [dagua/config.py](/home/jtaylor/projects/dagua/dagua/config.py:286) exposes `algorithm`, with `None` as native/default.
- Default dispatch: [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:936) remaps `algorithm=None` to `dagua_native` at [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:941).
- Classifier: [dagua/layout/graph_classify.py](/home/jtaylor/projects/dagua/dagua/layout/graph_classify.py:31) returns structural fields including `is_directed_acyclic`, but no semantic direction.
- Composite score: [dagua/metrics.py](/home/jtaylor/projects/dagua/dagua/metrics.py:1171) assigns the current weights; `quick()` always computes direction terms at [dagua/metrics.py](/home/jtaylor/projects/dagua/dagua/metrics.py:1368).
- Benchmark undirected conversion: [dagua/eval/graphs.py](/home/jtaylor/projects/dagua/dagua/eval/graphs.py:160) orients undirected graphs into a DAG by internal node order, and [dagua/eval/graphs.py](/home/jtaylor/projects/dagua/dagua/eval/graphs.py:232) applies that to NetworkX undirected graphs.

I measured cached core positions from `eval_output/variant_bench_full/positions` for:

`dagua`, `graphviz_dot`, `graphviz_sfdp`, `elk_layered`, `dagre`, `nx_spring`, `igraph_kamada_kawai`, `igraph_sugiyama`.

There were 651 scored layouts: 93 graphs times 7 engines. `nx_spring` did not have the simple core cached files in this directory layout, so it is absent from the measurement tables. Metrics were recomputed from cached positions with `quick()`, `sampled_crossing_rate(n_samples=50_000, seed=42)`, and `angular_resolution(seed=42)`. This is not a fresh current-Dagua h2h run; the sprint-20 context already reports the current fresh-Dagua h2h. This pass answers the alternative-scoring question on the cached benchmark artifact and is deterministic apart from the explicitly sampled crossing estimate.

For the proposed undirected score, I used the five direction-agnostic headline terms from `composite()`:

```
edge_length_cv: 20
overlap_count: 10
crossing_rate: 10
angular_resolution: 5
cluster_separation: 5
```

Those sum to 50, not 35. The task text says "remaining 35 pts"; I treat that as an arithmetic slip because the same prompt lists `edge_length_cv`, `overlap_count`, `crossing_rate`, `angular_resolution`, and `cluster_separation` as direction-agnostic. Because these cached metric calls do not pass cluster IDs, I used the same neutral 2.5 cluster points that `composite()` uses when cluster separation is missing.

## Findings

### High - DAG-ness Is Not Semantic Direction

The classifier correctly reports topological properties, but it cannot tell whether direction is meaningful. That distinction matters because many benchmark graphs are constructed from undirected sources and then oriented to fit Dagua's directed tensor model.

Semantic benchmark distribution:

| Bucket | Count | Classifier `is_directed_acyclic=True` | Notes |
|---|---:|---:|---|
| Directed DAG-style | 63 | 63 | Neural nets, dependencies, org charts, tree/chain/flow DAGs, generated layered DAGs. |
| Directed cyclic/feedback | 6 | 0 | Feedback/control-flow graphs where direction is meaningful but not acyclic. |
| Effectively undirected | 24 | 21 | Social/community/random/geometric/lattice/PPI/regular/cyclic-no-hierarchy graphs. |
| Total | 93 | 84 | Acyclic orientation overcounts semantic DAGs by 21 graphs. |

The 24 effectively undirected graphs I counted are:

`chung_lu_150`, `er_100`, `er_500`, `heavy_tail_weights_50`, `hexagonal_lattice_42`, `parallel_cycles_4x5`, `petersen_10`, `planar_60`, `protein_ppi_200`, `real_football_115`, `real_karate_34`, `real_lesmis_77`, `regular_3_30`, `regular_4_40`, `rgg_100`, `rgg_500`, `sbm_4x30`, `sbm_5x50`, `sierpinski_42`, `small_world_100`, `small_world_500`, `triangular_lattice_36`, `weighted_clusters_3x10`, `weighted_karate_34`.

The 6 directed cyclic/feedback graphs are:

`braided_feedback_tails`, `center_port_backedge_hub`, `disconnected_label_cycle_collage`, `kitchen_sink_hybrid_net`, `kitchen_sink_platform_graph`, `recurrent_feedback_cell`.

I put `parallel_cycles_4x5` in effectively undirected/non-hierarchical rather than directed feedback. It is made of directed cycles, but the layout problem is circular/component geometry, not vertical flow.

### High - The Current Composite Rewards Arbitrary Direction

Current headline weights in [dagua/metrics.py](/home/jtaylor/projects/dagua/dagua/metrics.py:1174):

| Metric | Weight | Direction profile | Why it matters |
|---|---:|---|---|
| `dag_consistency` | 25 | Directed/layered | Checks whether each edge points along the configured layout axis. |
| `depth_spearman_rho` | 15 | Directed/layered | Correlates topological depth with y-position. |
| `edge_straightness_mean_deg` | 10 | Directed-axis | Rewards edges aligned with vertical or horizontal layer axis. |
| `edge_length_cv` | 20 | Direction-agnostic | Uniform Euclidean edge length. |
| `overlap_count` | 10 | Direction-agnostic | Node box overlaps. |
| `crossing_rate` | 10 | Direction-agnostic | Segment crossings ignore arrow direction. |
| `angular_res_mean_deg` | 5 | Direction-agnostic | Incident edge angle quality. |
| `cluster_mean_sep_ratio` | 5 | Direction-agnostic | Cluster geometry. |

For undirected graphs, the 25-point DAG term is not noise around zero. It strongly favors layouts that honor the arbitrary edge orientation imposed by `_undirected_to_dag()`. The same is true for the 15-point depth term. On `real_karate_34`, Dagua gets `dag_consistency=1.00` and `depth_spearman_rho=0.98` on a social network, adding about 39.7 directed points that have no social-network meaning.

Mean point contributions across the 24 effectively undirected graphs show the skew:

| Engine | DAG pts | Depth pts | Straight pts | Edge-CV pts | Crossing pts | Angular pts | Overlap pts |
|---|---:|---:|---:|---:|---:|---:|---:|
| dagua | 24.77 | 13.01 | 2.74 | 6.17 | 3.37 | 1.98 | 7.08 |
| graphviz_dot | 24.77 | 11.00 | 2.06 | 5.69 | 6.37 | 2.91 | 10.00 |
| elk_layered | 24.77 | 10.95 | 2.04 | 4.93 | 5.99 | 2.75 | 10.00 |
| dagre | 24.77 | 10.92 | 0.88 | 4.41 | 5.35 | 2.65 | 10.00 |
| igraph_sugiyama | 24.77 | 12.79 | 1.22 | 4.20 | 4.47 | 2.36 | 9.17 |
| graphviz_sfdp | 13.31 | 2.77 | 0.05 | 12.88 | 7.98 | 2.88 | 1.25 |
| igraph_kamada_kawai | 12.66 | 2.16 | 0.17 | 14.59 | 7.78 | 3.07 | 0.42 |

The force/stress engines have much better edge length and crossing terms, but current scoring buries that under arbitrary directed terms and overlap penalties caused by coordinate scale/node-size mismatch in cached competitors.

### High - Alternative Undirected Composite Reorders the Benchmark

On the 24 effectively undirected graphs, cached core layouts score as follows:

| Engine | Current composite mean | Current median | Undirected composite mean | Undirected median |
|---|---:|---:|---:|---:|
| graphviz_dot | 65.31 | 61.07 | 54.95 | 53.01 |
| elk_layered | 63.93 | 61.48 | 52.35 | 49.70 |
| dagua | 61.63 | 62.17 | 42.21 | 45.04 |
| dagre | 61.48 | 58.07 | 49.82 | 47.69 |
| igraph_sugiyama | 61.48 | 59.73 | 45.38 | 42.57 |
| graphviz_sfdp | 43.61 | 39.23 | 54.96 | 49.87 |
| igraph_kamada_kawai | 43.35 | 40.08 | 56.72 | 53.02 |

Dagua vs best cached competitor on this subset:

| Scoring profile | Graphs | Wins | Ties | Losses | Mean delta |
|---|---:|---:|---:|---:|---:|
| Current directed composite | 24 | 6 | 2 | 16 | -4.91 |
| Undirected composite | 24 | 0 | 0 | 24 | -19.17 |

Measurement caveat: the full 93-graph cached run should not be read as the
current sprint-20 h2h leaderboard because the sprint context's fresh current
Dagua run is newer than some cached `dagua` position files. It is still useful
for the scoring experiment because every engine is scored from the same cached
geometry under two formulas. Across all 93 cached graphs, current directed
scoring gives cached Dagua 19 wins, 13 ties, 61 losses, and mean delta -4.87
against the best cached competitor. Applying the undirected formula to all 93
graphs is intentionally invalid for directed graphs, but it is an informative
sanity check: Dagua falls to 2 wins, 6 ties, 85 losses, and mean delta -17.30.
That confirms the undirected formula is not a universal replacement; it must be
selected only for semantic-undirected inputs.

This also explains why metric selection and layout dispatch should land
together. If Dagua adds an undirected force path but keeps the directed
leaderboard as the only benchmark target, the better undirected layouts will
look like regressions. If the benchmark switches profiles before dispatch
exists, Dagua will expose large losses without a native path to close them.

Worst Dagua losses under undirected scoring:

| Graph | Dagua | Best competitor | Delta |
|---|---:|---:|---:|
| `rgg_100` | 9.40 | elk_layered 52.19 | -42.79 |
| `er_500` | 14.52 | graphviz_sfdp 55.11 | -40.59 |
| `rgg_500` | 15.84 | elk_layered 52.49 | -36.65 |
| `protein_ppi_200` | 19.45 | igraph_kamada_kawai 52.98 | -33.53 |
| `real_lesmis_77` | 22.50 | graphviz_dot 53.79 | -31.29 |
| `parallel_cycles_4x5` | 64.22 | graphviz_sfdp 94.62 | -30.40 |
| `petersen_10` | 52.93 | igraph_kamada_kawai 81.95 | -29.02 |
| `small_world_100` | 45.00 | igraph_kamada_kawai 69.78 | -24.78 |

This is the key benchmark answer: under a metric aligned with undirected semantics, Dagua's current native pipeline is not competitive on the undirected slice. The current headline h2h table therefore overstates Dagua quality on social, random, geometric, PPI, regular, and small-world families.

### Medium - Competitor Layouts Are Not Winning by Accident

I inspected the four requested undirected-heavy targets. The prompt names `real_football_34`, but the benchmark registry has `real_football_115`; I used the registered graph.

For `small_world_100`, `igraph_kamada_kawai` is clearly a force/stress winner, not a random coordinate accident:

| Engine | Current | Undirected | DAG | Edge CV | Crossing | Angular | Stress | Aspect / PCA anisotropy |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dagua | 57.13 | 45.00 | 0.99 | 4.00 | 0.0000 | 0.0 | 0.942 | aspect 0.00 / huge |
| igraph_kamada_kawai | 47.47 | 69.78 | 0.50 | 0.13 | 0.0001 | 48.6 | 0.896 | aspect 1.00 / 1.0 |
| graphviz_sfdp | 38.38 | 51.25 | 0.51 | 0.33 | 0.0040 | 1.6 | 0.894 | aspect 1.01 / 1.0 |
| graphviz_dot | 56.61 | 52.23 | 0.99 | 3.83 | 0.0086 | 35.8 | 0.862 | aspect 0.01 / 5578 |

Dagua and layered competitors mostly arrange this graph as a near-line because the arbitrary direction is rewarded. KK is isotropic and has much better edge-length uniformity. A same-bounding-box random baseline averaged only 16.41 undirected points and maxed at 18.09, so the KK result is not "random x,y that happens to score well."

For karate:

| Graph / Engine | Current | Undirected | DAG | Depth rho | Edge CV | Crossing | Degree-center rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| `real_karate_34` / dagua | 64.89 | 50.35 | 1.00 | 0.98 | 0.49 | 0.1439 | 0.25 |
| `real_karate_34` / graphviz_dot | 59.26 | 57.01 | 1.00 | 0.38 | 0.58 | 0.0502 | 0.10 |
| `real_karate_34` / igraph_kamada_kawai | 39.80 | 53.02 | 0.53 | 0.01 | 0.28 | 0.0344 | 0.65 |

KK puts higher-degree nodes more centrally (`degree_center_spearman=0.65`) and improves edge length/crossing geometry. Dagua wins the current score through arbitrary depth alignment and loses the undirected score.

For `real_football_115`, all engines are poor under this simple composite because angular resolution is low and edge density is high, but Dagua is especially weak: undirected score 14.22 versus graphviz_dot 37.91. The random baseline averages 26.33, so Dagua's cached layout is below random under the undirected scoring profile. This is a hard signal that the native directed path is the wrong objective for this graph family.

### High - Proposed Split

Do not try to infer everything from `is_directed_acyclic`. Add semantic intent at the API boundary and use topology only as fallback.

Proposed public shape:

```python
from typing import Literal, Optional

GraphSemantics = Literal["auto", "directed", "undirected", "directed_cyclic"]

@dataclass
class DaguaGraph:
    """A graph ready for layout and rendering."""

    semantics: GraphSemantics = "auto"
    direction: str = "TB"
```

Add the same override to `LayoutConfig`:

```python
graph_semantics: GraphSemantics = "auto"
```

Resolution rule:

```python
def resolve_graph_semantics(graph: DaguaGraph, structure: GraphStructure) -> GraphSemantics:
    """Resolve semantic direction for dispatch.

    Parameters
    ----------
    graph : DaguaGraph
        Graph carrying an optional explicit semantic hint.
    structure : GraphStructure
        Topology classification for the graph.

    Returns
    -------
    GraphSemantics
        Directed, undirected, or directed-cyclic semantic dispatch category.
    """
    if graph.semantics != "auto":
        return graph.semantics
    if "semantic_undirected" in structure.topology_tags:
        return "undirected"
    if not structure.is_directed_acyclic:
        return "directed_cyclic"
    return "directed"
```

Importer policy:

- `from_networkx(nx.Graph)` should set `semantics="undirected"`.
- `from_networkx(nx.DiGraph)` should set `semantics="directed"` unless caller overrides.
- `from_igraph(..., directed=False)` and SciPy symmetric adjacency should set `undirected`.
- Benchmark builders that call `_graph_from_undirected_networkx()` should preserve `semantics="undirected"` even though they orient the tensor.
- Synthetic DAG builders should set `directed`.
- Feedback/control-flow builders should set `directed_cyclic`.

Default dispatch in [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:936) should change from unconditional `dagua_native` to:

```python
if remapped_from_default:
    structure = classify_graph(graph.edge_index, graph.num_nodes)
    semantics = resolve_graph_semantics(graph, structure)
    config = copy.copy(config)
    if semantics == "undirected":
        config.algorithm = "dagua_native_undirected"
    elif semantics == "directed_cyclic":
        config.algorithm = "dagua_native_cyclic_hybrid"
    else:
        config.algorithm = "dagua_native_directed"
```

Concrete pipeline split:

| Semantic dispatch | Pipeline | Objective |
|---|---|---|
| `directed` | `dagua_native_directed` | Current layered pipeline: cycle prep, dummy expansion, median/transpose, BK, overlap projection, aspect fit. |
| `undirected` | `dagua_native_undirected` | KK/stress or SFDP/FA2-style initialization, undirected stress/edge-length/repulsion/angular/overlap losses, no DAG/depth/axis-straightness terms. |
| `directed_cyclic` | `dagua_native_cyclic_hybrid` | Make FAS/back-edge mask explicit, layout the acyclic backbone with directed path, route feedback/cycle edges separately, or run force seed then layered refinement only on forward edges. |

The short-term implementation can reuse existing pipelines already present in `dagua/layout/ops/pipelines/`: `kk.py`, `stress_sgd.py`, `stress_majorization.py`, `sfdp.py`, `fa2.py`, `fr.py`, and `linlog.py`. The best conservative first candidate is a stress/KK seed followed by overlap projection and component packing. Do not run BK or dummy-node expansion on semantic-undirected graphs.

### Medium - Metrics Should Dispatch Too

The benchmark should report two profiles:

```python
def composite_undirected(metrics: dict[str, float]) -> float:
    """Score direction-agnostic layout quality on a 0-100 scale.

    Parameters
    ----------
    metrics : dict[str, float]
        Metrics from quick/full evaluation.

    Returns
    -------
    float
        Direction-agnostic composite score.
    """
    raw = 0.0
    raw += 20.0 * max(0.0, 1.0 - metrics.get("edge_length_cv", 1.0))
    raw += 10.0 * (1.0 if metrics.get("overlap_count", 1) == 0 else 0.0)
    raw += 10.0 * max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10.0)
    raw += 5.0 * min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)
    raw += 5.0 * _cluster_score_or_neutral(metrics)
    return 100.0 * raw / 50.0
```

Add an explicit `profile=` argument to benchmark/eval rather than silently changing `composite()`. Reports should show:

- directed composite for directed graphs,
- undirected composite for undirected graphs,
- both scores during transition to reveal metric gaming,
- a unified "semantic composite" that selects the profile by resolved graph semantics.

Longer term, undirected scoring should add `sampled_stress` and neighborhood preservation. The current 50-point undirected composite still misses community preservation and graph-theoretic distance structure; it mainly exposes that the current directed score is invalid for these families.

## Big-Bet Proposals

1. **Semantic graph contract.** Make graph semantics first-class and propagate through IO. This is low-risk API work with high leverage: it prevents arbitrary orientation from controlling layout and scoring.

2. **Native undirected pipeline.** Build `dagua_native_undirected` around stress/KK or SFDP initialization plus differentiable overlap/repulsion refinement. Expected impact is largest on `rgg_*`, `er_*`, `protein_ppi_200`, `small_world_*`, `petersen_10`, and social/community graphs. The measurement suggests 20-40 points are available on several graphs under an undirected score.

3. **Hybrid cyclic pipeline.** Treat meaningful feedback graphs separately from undirected cyclic graphs. Use FAS/back-edge masks for `recurrent_feedback_cell` and platform/control-flow graphs, but avoid forcing `small_world_*` into a hierarchical story.

4. **Semantic benchmark leaderboard.** Keep the existing directed leaderboard for DAGs, but publish a second undirected slice. Otherwise Dagua can keep "winning" by optimizing the wrong metric.

## Risk / Regression Analysis

The protected wins in sprint-20 context are mostly directed/layered graphs: `org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`, `org_chart_1_5_4_8`, `random_dag_50`, `random_bipartite_60`, `edge_label_braid`, `bipartite_4_3_4`. Those should remain on the directed pipeline.

The two protected wins that are semantically undirected are `weighted_karate_34` and `real_karate_34`. The current protected advantage exists under the directed score. Under the measured undirected composite, cached Dagua trails graphviz_dot by 7.87 and 6.65 points respectively. That is not a regression risk; it is a measurement correction. The implementation risk is product-facing: users may expect Dagua's current vertical karate drawing if they have seen it. Mitigate with explicit `graph_semantics="directed"` override and with release notes.

Classifier-only dispatch is risky. `real_karate_34` and `weighted_karate_34` are directed-acyclic after orientation, so `is_directed_acyclic=True` would send them to the wrong path. Conversely, `recurrent_feedback_cell` is cyclic but semantically directed, so "cyclic means undirected" would also be wrong.

Runtime risk is real. KK/stress can be expensive at 500 nodes, and the current sprint already has runtime pressure. The first implementation should gate to `N <= 2_000` exact stress/KK, use SFDP/FA2/LinLog for larger semantic-undirected graphs, and keep a quick `algorithm="dagua_native_directed"` escape.

## Implementation Order

1. Add `semantics` to `DaguaGraph` and `graph_semantics` to `LayoutConfig`, defaulting to `"auto"`.
2. Preserve semantics in NetworkX/igraph/SciPy importers and benchmark graph builders. This is the step that fixes the 21 undirected-but-DAG-classified graphs.
3. Add `resolve_graph_semantics()` and route `algorithm=None` through semantic dispatch in `engine.layout()`.
4. Scaffold `dagua_native_directed` as an alias/wrapper around current `dagua_native` to avoid churn.
5. Implement `dagua_native_undirected` by reusing existing KK/stress/SFDP/FA2 ops, followed by overlap projection and aspect/component packing.
6. Add `composite_undirected()` and benchmark profile selection. During transition, report both current and semantic composites.
7. Add regression tests: NetworkX `karate_club_graph()` resolves to undirected; a synthetic dependency DAG resolves to directed; a recurrent feedback graph resolves to directed-cyclic; explicit config override wins over auto.

## Knowledge

- The benchmark contains 24 semantically undirected graphs, not just the obvious `small_world_*` cases.
- `_undirected_to_dag()` is the source of the metric illusion: it makes social/random/lattice graphs scoreable by DAG metrics without making the orientation meaningful.
- Current Dagua's strong karate score is mostly arbitrary DAG/depth reward, not social-network geometry.
- `igraph_kamada_kawai` and `graphviz_sfdp` look weak in current scores because they refuse the arbitrary y-axis story; under undirected scoring they become the engines to beat.
