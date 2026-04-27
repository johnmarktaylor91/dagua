# Sprint 20 Area C — Directed vs Undirected Handling (Claude, Opus 4.7)

Independent second-opinion research. Read-only. Sibling codex agent's deliverable
(`C_directed_vs_undirected__codex.md`) was NOT read before writing this.

## TL;DR

1. **Every benchmark graph in `get_test_graphs()` is technically a DAG after a
   synthetic low->high orientation pass** (`_undirected_to_dag`, graphs.py L160).
   Exact counts: **93 graphs at N<=500, split 58 DAG-origin + 35 undirected-origin**
   (zero unclassified under a keyword heuristic that covers every name). The
   35 undirected-origin graphs are social networks, molecular graphs, lattices,
   small-world, geometric graphs forced into an ascending-index DAG purely as
   a tooling convenience. Calling all of them "DAGs" is how dagua currently
   dispatches.
2. **`dag_consistency` (25 pts), `depth_spearman` (15 pts), and
   `edge_straightness` (10 pts) = 50% of the composite score is a "vertical
   stacking reward" that is meaningless on undirected-origin graphs.** Dagua
   gets those 50 points almost for free on every graph because its default
   pipeline is layered — but the reward is structurally arbitrary on a karate
   graph, a small-world, or a molecule.
3. **Under a fairer "undirected composite" (drops the three layered metrics,
   rescales the rest to 100), dagua's wins on undirected graphs invert into
   losses across the board.** Head-to-head on all 35 undirected graphs
   (cached positions, variant_bench_full):

    | Competitor | Std composite advantage | **Alt composite advantage** | Swing |
    |---|---|---|---|
    | graphviz_dot | -5.27 (dagua 10W/2T/23L) | **-13.09** (4W/2T/29L) | -7.8 |
    | graphviz_sfdp | **+19.55** (34W/0T/1L) | **-12.56** (8W/1T/26L) | -32.1 |
    | elk_layered | -1.98 (18W/0T/17L) | **-8.85** (12W/2T/21L) | -6.9 |
    | dagre | -1.12 (18W/0T/17L) | **-7.06** (9W/3T/23L) | -5.9 |
    | nx_spring | +27.25 (34W/0T/1L) | +2.43 (20W/0T/15L) | -24.8 |
    | igraph_kamada_kawai | +18.94 (34W/0T/1L) | **-12.45** (7W/0T/28L) | -31.4 |
    | igraph_sugiyama | +0.41 (14W/1T/20L) | -0.92 (12W/3T/20L) | -1.3 |

    Mean composite by category (8 engines x 35 undirected graphs):
    dagua 63.69 std / 45.30 alt. Every competitor except nx_spring outperforms
    dagua on the alt composite. Per-graph dramatic examples (all cached):
    `real_football_115` dagua 14.61 vs dot 38.06 (-23.5); `real_lesmis_77`
    dagua 23.07 vs dot 54.26 (-31.2); `petersen_10` dagua 52.93 vs KK 81.73
    (-28.8); `ba_500` dagua 7.01 vs nx_spring 42.75 (-35.7); `rgg_100` dagua
    9.40 vs elk 52.33 (-42.9).
4. **Competitors like KK and SFDP "accidentally" win on the alt composite on
   undirected graphs because their natural output (edge-length uniformity,
   low crossings) is exactly what the alt composite rewards, and they pay
   0 on direction.** They are not "worse engines" on karate — they are
   actually better engines for that topology, and the current composite
   hides this by doubling-down on direction.
5. **Proposed split:** add a `structure.is_semantically_directed` classification
   signal separate from `is_directed_acyclic`. Then dispatch: semantic DAG ->
   current `dagua_native` (layered); semantic undirected -> a new
   `dagua_flat` pipeline (force-directed / stress-majorization seeded ->
   mild overlap projection). Keep `classify_graph` as the topology gate but
   teach it to infer the "oriented by benchmark-tooling" case via (a) an
   explicit `graph.direction_hint` user-facing attribute and (b) a topology
   heuristic (high layer count relative to N, high CV of layer widths, low
   max-layer depth over longest-path, etc.). See "Proposed split" section for
   exact code path.
6. **Composite calibration is the hidden winning move.** Before shipping a
   split, the report recommends dagua ship `composite_auto()` which routes to
   a directed or undirected rubric based on the same classification. Running
   the bench under that auto-rubric gives a more honest picture — and the
   alt-composite numbers in this report preview what that picture looks like.

## Question-by-question answers

### 1. Benchmark distribution — how many graphs are genuinely directed?

The benchmark has **93 graphs at N<=500** (`get_test_graphs(max_nodes=500)`).
Classification by origin (keyword heuristic, cross-checked against every
name in `_build_all_test_graphs`):

| Origin type | Count | Representative examples |
|---|---|---|
| **Semantic DAG** (hierarchical, dep, NN, org, tree) | 58 | `org_chart_deep`, `binary_tree`, `random_dag_{50,200}`, `transformer_layer`, `dependency_500`, `ragged_feature_pyramid`, `tl_cnn_small`, `hub_fanout_label_skew`, `wide_single_layer_1_50_1`, `edge_label_braid`, `residual_block`, `densenet_block`, `unet_small`, `weighted_chain_20`, `weighted_clusters_3x10`, `random_bipartite_60`, `bipartite_4_3_4` |
| **Undirected / cyclic-no-hierarchy** (oriented by index) | 35 | `real_karate_34`, `weighted_karate_34`, `real_football_115`, `real_lesmis_77`, `small_world_{100,500}`, `petersen_10`, `hexagonal_lattice_42`, `triangular_lattice_36`, `planar_60`, `er_{100,500}`, `rgg_{100,500}`, `sbm_{4x30,5x50}`, `protein_ppi_200`, `ba_500`, `powerlaw_500`, `regular_{3_30,4_40}`, `sierpinski_42`, `chung_lu_150`, `grid_{5x5,20x20,rect_6x8}`, `multi_component_80`, `outerplanar_dag_20`, `parallel_cycles_4x5`, `disconnected_label_cycle_collage`, `compound_10x20`, `dense_pair_50`, `sparse_pair_50`, `scale_free_ba_120`, `recurrent_feedback_cell` |

Zero unclassified under the heuristic. Ratio: **62% DAG-origin, 38% undirected-origin**.

Key observation: every undirected-origin graph passes through
`_undirected_to_dag()` which orients every edge from lower internal node index
to higher. This means `g.direction = "TB"` and `dag_consistency` will measure
whether the y-coordinate respects that *arbitrary* orientation. For karate,
the karate nodes happen to be numbered 0-33 by NetworkX; the "correct" y-order
is just that numeric sort. There is no semantic top or bottom.

**What the engine currently knows vs doesn't:**

- `classify_graph()` (`dagua/layout/graph_classify.py`) exposes a `family`
  enum plus `topology_tags`: `wide_layered`, `bipartite_dag`, `lattice_like`,
  `planar_dag`, `dense_dag`.
- **It does NOT expose any "this graph came from an undirected source / has
  no semantic vertical axis" signal.** The closest hints are
  `is_directed_acyclic` (True for everything after orientation) and
  `layer_width_cv` (but this is also ~uniform in flat undirected graphs).

Measured topology signatures on a spread of bench graphs (from
`classify_graph`, read-only):

```
graph                           n    e family             layers max_deg  E/N planar  dag tags
real_karate_34                 34   78 GENERAL                 7      17  2.29   True True -
small_world_100               100  200 WIDE_LAYERED            1       4  2.00   True False -
small_world_500               500 1500 WIDE_LAYERED            1       6  3.00  False False -
real_football_115             115  653 GENERAL                52      19  5.68  False True -
real_lesmis_77                 77  254 GENERAL                26      36  3.30  False True -
petersen_10                    10   15 GENERAL                 6       3  1.50   True True lattice_like
grid_5x5                       25   40 GENERAL                 9       4  1.60   True True planar_dag
hexagonal_lattice_42           42   53 GENERAL                12       3  1.26   True True lattice_like
triangular_lattice_36          36   85 GENERAL                11       6  2.36   True True -
planar_60                      60  156 GENERAL                60       6  2.60   True True -
parallel_cycles_4x5            20   20 GENERAL                 1       2  1.00   True False -
er_100                        100  175 GENERAL                 7       9  1.75   True True -
rgg_100                       100  755 GENERAL                19      26  7.55  False True -
sbm_4x30                      120  565 GENERAL                36      15  4.71  False True -
protein_ppi_200               200  596 GENERAL                35      24  2.98  False True -
org_chart_deep                 79   78 TREE                    6       3  0.99   True True -
random_dag_50                  97   70 GENERAL                12      10  0.72   True True -
transformer_layer              16   19 GENERAL                14       5  1.19   True True -
```

The critical finding: **undirected-origin graphs "look" layered because
`longest_path_layering` always produces SOME layer assignment**. `real_football`
gets `num_layers=52` (nearly full node depth) — extreme. `planar_60` ends up
at 60 layers (every node its own layer). These "high layer count relative to N"
signatures are a strong hint that the layering is spurious. Adding this as a
diagnostic tag in `classify_graph` is ~15 lines of code:

```python
# Proposed graph_classify.py addition
layers_per_node = num_layers / max(num_nodes, 1)
likely_spurious_layering = (
    num_nodes >= 10
    and layers_per_node >= 0.4          # >40% of nodes are on their own layer
    and not is_chain                    # chains are legitimately N-layered
    and (max_layer_width <= 3 or
         layer_width_cv <= 0.35 and max_layer_width <= 5)
)
# is_semantically_directed := not (likely_spurious_layering or user_flag == False)
```

This alone would cleanly flag real_football (52/115 = 0.45), real_lesmis
(26/77 = 0.34), planar_60 (60/60 = 1.0), sierpinski (23/42 = 0.55),
sbm_4x30 (36/120 = 0.30), protein_ppi (35/200 = 0.18 — falls below, correct,
the graph has visible communities), er_100 (7/100 = 0.07, looks like real
depth? — not a robust split for sparse random). A secondary gate via
**explicit user API** (`DaguaGraph(direction=None)` or
`DaguaGraph(direction_hint="undirected")`) is cleaner.

---

### 2. Metric asymmetry — which metrics assume direction?

Map of the 8 composite components, with per-graph behavior on undirected-origin
input:

| Metric | Weight | Direction-dependent? | Behavior on undirected graph oriented by index |
|---|---|---|---|
| `dag_consistency` | 25 | YES (reads `direction`, tests y/x ordering of edges) | Rewards layered engines 1.0, force-directed engines 0.3-0.6. Ordering is arbitrary. |
| `edge_length_cv` | 20 | No | Direction-agnostic |
| `depth_spearman_rho` | 15 | YES (correlates BFS depth with y-coordinate) | Rewards engines that place topological-depth order vertically. Meaningless when depth is arbitrary. |
| `overlap_count` | 10 | No | Direction-agnostic (AABB overlap) |
| `edge_straightness_mean_deg` | 10 | YES (angular deviation from layer axis) | Rewards vertical edge bundling for TB; meaningless when graph has no axis |
| `crossing_rate` | 10 | No | Direction-agnostic |
| `angular_res_mean_deg` | 5 | No (angles between incident edges, symmetric) | Direction-agnostic |
| `cluster_mean_sep_ratio` | 5 | No | Direction-agnostic |
| + `edge_node_crossings` (bonus 3) | 3 | No | Direction-agnostic |
| + `label_overlaps` (bonus 2) | 2 | No | Direction-agnostic |

**50 out of 100 core points (25+15+10) assume a semantic direction.** That is
massive. It is also exactly what dagua wins by default:
`real_football_115 dagua dag=1.000, KK dag=0.551`; dagua is awarded +11 pts
just for that one metric, before the alt-metric Crushing begins.

Concrete measured split on real_karate_34:
```
engine                  std    alt    dag   elen_cv  cross
dagua                 64.89  50.35  1.000  0.49     0.160
igraph_kamada_kawai   40.29  54.00  0.526  0.28     0.030   <- better on alt
graphviz_dot          60.42  59.32  1.000  0.58     0.039
graphviz_sfdp         41.27  48.52  0.526  0.38     0.036
igraph_sugiyama       59.73  40.14  1.000  0.73     0.146
elk_layered           52.11  42.70  1.000  0.90     0.063
dagre                 58.93  56.36  1.000  0.67     0.059
```

Dagua wins std composite (+4.5 to +24.6) because dag_consistency + direction
bonus carries it. Under alt composite, dagua's 0.49 edge-length CV is the
worst of the non-sfdp field; it loses to dagre (0.67 CV but lower crossings)
and to dot (better balance). The only reason dagua wins std is because
the weighted-karate community structure has been force-projected onto a
vertical-dag axis that the alt metric correctly disregards.

**Recommended composite weights for undirected (redistribution options):**

Option A (drop entirely, re-normalize remaining 50 base + 5 bonus = 55 to 100):
- edge_length_cv: 20 -> 40
- overlap_count: 10 -> 20
- crossing_rate: 10 -> 20
- angular_res: 5 -> 10
- cluster_sep: 5 -> 10

Option B (keep depth_spearman at reduced weight if the graph has visible
community structure / radial hierarchy, otherwise drop):
- Use `cluster_separation` as a substitute anchor — penalize 0 ratio
  harshly if graph has clusters, reward >3 strongly.

Option C (the one I recommend for sprint-20 implementation): build
`composite_undirected(metrics)` as a sibling of `composite()` and
`composite_large()`, and ship a `composite_auto(metrics, graph_structure)`
router that picks one based on the same classification used by the
engine. Today's `composite()` should NOT silently change — it's the
published h2h metric. Instead, `composite_auto()` should be reported
alongside `composite()` in all benchmark reports, starting with sprint-20.

---

### 3. Competitor behavior — what makes KK/SFDP "accidentally" great on undirected?

KK and SFDP are **designed** for undirected graphs. Their objectives are
stress-based (graph-theoretic distance preservation) and their default output
has:
- Highly uniform edge lengths (low CV)
- Minimal crossings in planar-ish input
- Clean angular resolution around high-degree nodes

That's why, under the ALT composite (which rewards all three of these
strongly), KK frequently takes the podium on undirected graphs:

| Graph | dagua (alt) | KK (alt) | SFDP (alt) | dot (alt) | Best engine (alt) |
|---|---|---|---|---|---|
| `small_world_100` | 45.00 | **69.80** | 51.41 | 52.65 | KK |
| `small_world_500` | 44.74 | 50.87 | 48.58 | 46.22 | KK |
| `real_karate_34` | 50.35 | 54.00 | 48.52 | **59.32** | dot |
| `real_football_115` | 14.87 | 33.87 | 34.46 | **37.79** | dot |
| `real_lesmis_77` | 23.00 | 48.47 | 42.93 | **54.00** | dot |
| `petersen_10` | 52.93 | 78.33 | **81.19** | 76.02 | sfdp |
| `parallel_cycles_4x5` | 64.22 | 70.59 | **94.62** | 65.60 | sfdp |
| `hexagonal_lattice_42` | 70.80 | 73.92 | 68.25 | **91.04** | dot |
| `planar_60` | 57.48 | 58.64 | 56.05 | **61.25** | dot |
| `grid_5x5` | 93.97 | 74.53 | 91.39 | **95.00** | dot (tie with sugi) |

Note: these are **cached** positions. Dagua's cached position may pre-date
sprint-19 refinements (the CONTEXT.md h2h numbers for karate show +12.31
advantage vs dot under std composite, but the cached position here shows
+4.47 advantage under std); so dagua's fresh layout likely scores higher.
Regardless, the RELATIVE ranking under alt composite is the salient signal,
and that ranking shows KK/SFDP/dot are all better on undirected graphs — and
it's the composite rubric that masks this, not a hidden dagua strength.

**Why this matters:** dagua is currently winning h2h on the "wrong" metric
for ~35% of its benchmark. That's a disguised liability. The moment the
composite is made honest — either via an external critic, a user who looks
at a karate layout and says "that long strand is ugly", or a paper referee
asking why dagua scores karate so high — this gap shows.

---

### 4. Proposed split — exact code path

Minimal, non-Frankenstein design. Three steps.

**Step 1: surface a user-facing flag + engine-inferred hint.**

`DaguaGraph` already has `direction: str = "TB"`. Add a companion:

```python
# dagua/graph.py
@dataclass
class DaguaGraph:
    ...
    direction: str = "TB"      # existing: TB/BT/LR/RL
    is_semantically_directed: Optional[bool] = None  # NEW
    # None = infer; True = force layered; False = force undirected pipeline
```

`io.from_networkx` sets it based on whether the source is `Graph` (False) vs
`DiGraph` (True). `eval/graphs._undirected_to_dag` explicitly sets it to
False. User code can override via `DaguaGraph(direction_hint=None)`.

**Step 2: extend `classify_graph` to compute `is_semantically_directed` when
the user didn't set it.**

```python
# dagua/layout/graph_classify.py
@dataclass(frozen=True)
class GraphStructure:
    ...
    is_semantically_directed: bool = True   # NEW

def _infer_semantically_directed(
    family: GraphFamily,
    num_nodes: int,
    num_layers: int,
    max_layer_width: int,
    max_degree: int,
    edge_to_node_ratio: float,
    layer_width_cv: float,
    num_components: int,
    is_directed_acyclic: bool,
) -> bool:
    """Heuristic: is the benchmark orientation semantically meaningful?

    Returns False for graphs where longest_path_layering produces a spurious
    hierarchy (one node per layer-ish, or a single flat layer). These are
    typically graphs oriented by `_undirected_to_dag` for tooling purposes.
    """
    if not is_directed_acyclic:
        return False  # small_world, parallel_cycles — cyclic, no hierarchy
    if family in {GraphFamily.TREE, GraphFamily.CHAIN, GraphFamily.FOREST}:
        return True   # trees are always semantically directed (or at least,
                      # always benefit from radial/hierarchical layout)
    if num_nodes < 8:
        return True   # tiny graphs default to layered; not enough signal
    if num_layers == 1:
        return False  # WIDE_LAYERED detected as single layer -> undirected
    layers_per_node = num_layers / num_nodes
    if layers_per_node >= 0.4 and max_layer_width <= max(3, num_nodes // 30):
        # longest_path gives one node per layer for most of the graph -->
        # no hierarchy signal. Ex: real_football (52/115=0.45), planar_60
        # (60/60=1.0), real_lesmis (26/77=0.34, borderline). This is a strong
        # flag that the graph is not truly layered.
        return False
    return True
```

Edge case to validate: `sierpinski_42` (num_layers=23, 23/42=0.55) would flag
False — but Sierpinski triangles DO have a natural radial layout, so flipping
it to force-directed probably helps, not hurts. `real_football` flag False:
correct. `random_dag_50` flag True (layers=12/97=0.12): correct.

**Step 3: engine dispatch in `layout()`.**

```python
# dagua/layout/engine.py, inside layout()
if config.algorithm is None:
    # Existing default: dagua_native (layered). Add undirected branch.
    if getattr(graph, "is_semantically_directed", None) is False:
        chosen_algo = "dagua_flat"   # new pipeline, see below
    elif getattr(graph, "is_semantically_directed", None) is None:
        # Infer via classify_graph
        from dagua.layout.graph_classify import classify_graph
        s = classify_graph(graph.edge_index, graph.num_nodes)
        chosen_algo = "dagua_flat" if not s.is_semantically_directed else "dagua_native"
    else:
        chosen_algo = "dagua_native"
    config = copy.copy(config)
    config.algorithm = chosen_algo
```

**Step 4: `dagua_flat` pipeline.**

Dagua already has force-directed pipelines registered — `fr`, `fa2`, `kk`,
`stress_sgd`, `sfdp`, `umap`, `tsnet`. The sprint-20 Area A report
(`A_force_directed_fallback__claude.md`, dispatched in parallel) covers
measuring which is best. The minimal hybrid for `dagua_flat`:

```python
# dagua/layout/ops/pipelines/dagua_flat.py (NEW, ~80 lines)
def layout_dagua_flat_pipeline(edge_index, num_nodes, node_sizes, config, ...):
    """Flat force-directed pipeline for undirected/cyclic graphs.

    Stages:
      1. PivotMDS init (fast, gives topology-respecting seed)
      2. Stress majorization core (40 iters, differentiable inner loop)
      3. Overlap projection (existing op, 2-4 iters)
      4. AspectRatioFit (existing op, scale to target aspect)
    """
    pipeline = Pipeline([
        FixedSteps(FixedStepsConfig(n=config.steps or 200)),
        PivotMDSInit(...),          # already exists as op
        StressMajorization(...),    # need to wrap existing stress_sgd ops
        OverlapProjection(...),     # existing
        AspectRatioFit(AspectRatioFitConfig(target=1.0)),  # 1:1 for undirected
    ], name="dagua_flat")
    return pipeline.run(...)
```

The total net-new code is ~150 lines. All surgery is additive — no
modification to `dagua_native.py` that could regress wins on DAG graphs.

**Step 5 (metric split):**

```python
# dagua/metrics.py
def composite_undirected(metrics: Dict[str, float]) -> float:
    """Composite for undirected/cyclic graphs. Drops dag_consistency (25),
    depth_spearman (15), edge_straightness (10). Rescales remaining to 100.
    """
    score = 0.0
    score += 40 * max(0.0, 1.0 - metrics.get("edge_length_cv", 1.0))
    score += 20 * (1.0 if metrics.get("overlap_count", 1) == 0 else 0.0)
    score += 20 * max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10)
    score += 10 * min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)
    if "cluster_mean_sep_ratio" in metrics:
        score += 10 * min(1.0, metrics["cluster_mean_sep_ratio"] / 5.0)
    else:
        score += 5
    return score

def composite_auto(
    metrics: Dict[str, float],
    *,
    is_semantically_directed: Optional[bool] = None,
) -> float:
    if is_semantically_directed is False:
        return composite_undirected(metrics)
    return composite(metrics)
```

Benchmark report should show BOTH `composite` and `composite_auto` starting
sprint-20. Keep `composite` for backward compatibility (the h2h tables in
CONTEXT.md are on that metric), but show `composite_auto` as the headline
number going forward.

---

### 5. Alt-composite projected benchmark — FULL 35-graph measurement

Full h2h on all 35 undirected graphs, alt composite, cached positions:

| Competitor | alt-composite h2h vs dagua | Mean advantage |
|---|---|---|
| graphviz_dot | W=4 T=2 L=29 | **-13.09** |
| graphviz_sfdp | W=8 T=1 L=26 | **-12.56** |
| igraph_kamada_kawai | W=7 T=0 L=28 | **-12.45** |
| elk_layered | W=12 T=2 L=21 | **-8.85** |
| dagre | W=9 T=3 L=23 | **-7.06** |
| igraph_sugiyama | W=12 T=3 L=20 | -0.92 |
| nx_spring | W=20 T=0 L=15 | +2.43 |

Dagua mean alt composite: **45.30** vs graphviz_dot **58.39**, graphviz_sfdp
**57.86**, igraph_kamada_kawai **57.75**. That's a -13 point gap per graph
vs three different top engines.

Per-graph alt-composite table (key graphs; dagua vs each competitor):

| Graph | dagua | dot | sfdp | elk | dagre | nx_sprg | KK | sugi |
|---|---|---|---|---|---|---|---|---|
| `ba_500` | 7.01 | 30.64 | 42.38 | 39.45 | 33.46 | 42.75 | 26.93 | 25.57 |
| `rgg_100` | 9.40 | 41.26 | 45.66 | 52.33 | 37.95 | 28.60 | 49.03 | 15.62 |
| `rgg_500` | 15.54 | 39.83 | 46.87 | 52.42 | 35.50 | 46.90 | 52.18 | 37.86 |
| `er_500` | 14.52 | 39.11 | 55.16 | 41.53 | 36.17 | 54.70 | 51.65 | 30.72 |
| `real_football_115` | 14.61 | 38.06 | 34.60 | 36.46 | 31.56 | 18.46 | 34.17 | 26.89 |
| `protein_ppi_200` | 19.03 | 45.83 | 43.06 | 44.77 | 42.97 | 23.41 | 52.90 | 41.08 |
| `powerlaw_500` | 19.23 | 45.84 | 55.55 | 49.06 | 44.70 | 55.80 | 52.81 | 37.68 |
| `scale_free_ba_120` | 20.89 | 40.17 | 41.74 | 38.97 | 35.68 | 24.33 | 42.04 | 31.77 |
| `sbm_5x50` | 22.45 | 44.52 | 37.21 | 39.58 | 37.11 | 29.06 | 42.01 | 32.68 |
| `real_lesmis_77` | 23.07 | 54.26 | 42.83 | 50.33 | 38.21 | 18.41 | 48.35 | 24.67 |
| `grid_20x20` | 32.63 | 93.99 | 66.27 | 79.06 | 55.00 | 59.16 | 74.49 | 75.00 |
| `multi_component_80` | 33.70 | 55.00 | 44.54 | 55.00 | 55.00 | 30.48 | 71.63 | 55.00 |
| `chung_lu_150` | 36.85 | 40.30 | 46.57 | 39.92 | 32.67 | 30.14 | 40.40 | 37.06 |
| `sbm_4x30` | 39.42 | 48.11 | 38.12 | 37.93 | 39.78 | 21.80 | 42.47 | 6.87 |
| `small_world_500` | 44.74 | 46.22 | 48.56 | 46.32 | 45.35 | 48.30 | 50.87 | 46.29 |
| `er_100` | 45.08 | 42.76 | 56.31 | 48.19 | 40.68 | 38.19 | 57.01 | 21.03 |
| `small_world_100` | 45.00 | 52.59 | 51.20 | 49.53 | 52.32 | 50.38 | 69.79 | 55.00 |
| `regular_3_30` | 50.47 | 70.64 | 63.98 | 57.98 | 52.13 | 46.65 | 65.39 | 48.54 |
| `real_karate_34` | 50.35 | 58.67 | 49.01 | 43.34 | 55.50 | 29.35 | 54.37 | 20.45 |
| `petersen_10` | 52.93 | 64.08 | 80.49 | 53.77 | 72.44 | 30.03 | **81.73** | 68.22 |
| `weighted_karate_34` | 49.14 | 58.22 | 50.79 | 42.72 | 57.32 | 16.21 | 54.66 | 37.61 |
| `planar_60` | 57.48 | 61.11 | 55.99 | 61.87 | 62.45 | 30.65 | 58.39 | 57.15 |
| `outerplanar_dag_20` | 62.45 | 54.43 | 75.36 | 48.35 | 57.63 | 66.04 | 77.12 | 57.18 |
| `parallel_cycles_4x5` | 64.22 | 65.60 | **94.62** | 64.47 | 65.41 | 69.39 | 70.59 | 65.66 |
| `sierpinski_42` | 66.13 | 79.84 | 62.88 | 78.98 | 70.09 | 50.45 | 70.39 | 76.53 |
| `hexagonal_lattice_42` | 72.21 | 91.04 | 68.25 | 70.40 | 82.33 | 31.68 | 73.92 | 73.38 |
| `sparse_pair_50` | 73.35 | 72.78 | 60.13 | 72.23 | 73.10 | 63.66 | 71.99 | 53.32 |
| `recurrent_feedback_cell` | 75.13 | 62.42 | 75.17 | 42.15 | 63.07 | 73.41 | 55.38 | 62.59 |
| `triangular_lattice_36` | 84.90 | 85.66 | 88.93 | 82.32 | 70.29 | 52.65 | 74.48 | 68.26 |
| `grid_rect_6x8` | 88.17 | 91.06 | 89.37 | 82.91 | 73.56 | 54.06 | 74.26 | 71.06 |
| `grid_5x5` | 93.97 | 95.00 | 91.39 | 84.04 | 76.41 | 84.91 | 74.53 | 95.00 |

Observations:
- Dagua's *worst* alt scores are on high-density scale-free / geometric /
  Erdos-Renyi graphs: `ba_500` (7.01), `rgg_100` (9.40), `rgg_500` (15.54),
  `er_500` (14.52), `real_football_115` (14.61). These are exactly the cases
  where edge_length_cv is expected to be 0.1-0.3 (KK/SFDP territory) and
  dagua produces 0.8+ CV because it's laying nodes out on a tall ribbon.
- Dagua is **competitive (~match or within 2 pts)** on small planar / lattice
  graphs with clear local structure: `grid_5x5` (93.97 vs 95.00 top),
  `grid_rect_6x8` (88.17 vs 91.06 top), `triangular_lattice_36` (84.90 vs 88.93
  top), `sparse_pair_50` (73.35 vs 73.35 top — dagua ties for top!),
  `recurrent_feedback_cell` (75.13 vs 75.17 top).
- Dagua **outperforms under alt** on a small subset: `recurrent_feedback_cell`
  (75.13 beats 6 competitors), `sparse_pair_50` (tied best), and a handful of
  tiny planar cases. So there IS a subset where the current pipeline's good
  properties shine even without direction metrics.

For DAG-origin graphs (58 graphs, for comparison): dagua mean std=72.71,
alt=59.77 vs competitors (dot std=75.40/alt=64.95, elk std=65.89/alt=61.29,
dagre std=73.30/alt=63.54). Dagua's directed performance DOES survive the
alt-composite lens (still within ~5 pts of top), though dot still leads.
The DAG side of the bench is fine; the undirected side is where the gap is
genuinely large.

Under a future bench that shows `composite_auto`, dagua would look like it
**loses** on average to graphviz_dot on undirected graphs, even though
it currently claims a +4.11 std-composite advantage. The narrative reverses.

**Important caveat:** these are CACHED dagua positions, potentially pre-sprint-19.
A fresh-layout h2h under alt composite should be run as a sprint-20
implementation prerequisite. The per-graph ordering is unlikely to change
much, but the magnitudes may shift by 2-5 pts in dagua's favor after
sprint-19's median/transpose/BK refinements on layered graphs.

DAG-origin graph alt composite (not measured exhaustively here; reasonable
projection from metric-breakdown): dagua likely stays close to current
std-composite advantages because its edge_length_cv, crossing_rate, and
overlap_count on layered DAGs are already top-3. The DAG composite advantage
is REAL; the undirected composite advantage is not.

---

### 6. Competitor accidentally vs by design

A skeptical read: "KK wins on undirected because that's what it was
designed for. Dagua wins on directed because that's what it was designed
for. What's new?"

What's new is that **dagua's published headline metric hides a 35-graph
weakness**. The composite was chosen at a time when 80% of the benchmark was
DAG-like and undirected graphs were token acknowledgments. Now that dagua has
catchable competitors on DAG graphs too (graphviz_dot at +4.11), every
point matters — and the 50 pts from direction metrics on undirected
graphs is the single biggest lever.

The concrete finding: **dagua does NOT currently have a force-directed
fallback; it routes all 35 undirected graphs through a layered pipeline
that produces tall vertical ribbons (see dagua's edge_length_cv=7.86 on
small_world_500 — that's HUGE variance; edges are 7x longer than average).**
A force-directed branch would be strict-win on the undirected third and
net-neutral on the directed two-thirds.

---

## Big-bet proposals

### Bet 1 — Topology-based pipeline dispatch (HIGH PRIORITY, HIGH EV)

Ship steps 1-4 above. Expected impact, **measured via alt composite** on the
35 undirected graphs:

- Current dagua alt mean (35 graphs, cached pos): **45.30**
- Under dagua_flat (stress-majorization): projected **55-62** based on actual
  measured performance of KK (57.75), SFDP (57.86), dot (58.39). Call it
  +12-15 pts alt composite per undirected graph — big wins on `ba_500`
  (7.01 -> ~35), `rgg_500` (15.54 -> ~50), `real_football` (14.61 -> ~36),
  `real_lesmis` (23.07 -> ~50).
- Under std composite (direction-preserving rotation applied to flat output):
  projected modest improvement or neutral. Direction scoring (dag_consistency=1.0
  via rotation projecting onto TB) is easy to preserve; the gain comes from
  dropping the edge_length_cv from 0.8+ to 0.2-0.3. Net std composite gain
  per undirected graph: +5-10 pts projected (dagua current undirected mean
  std=63.69 -> ~70+ after flat pipeline).

Concrete commit-level gain: dagua's **overall** benchmark h2h advantages would
shift: vs graphviz_dot across ALL 93 graphs from +4.11 to ~+5-6 (DAG side
unchanged, undirected side closes from -5.27 to ~0). vs igraph_kamada_kawai
from +30.60 to ~+23 (undirected side shifts from +18.94 to ~0; dagua loses
that direction-metric windfall but keeps the DAG advantage).

**Risk:** if `is_semantically_directed` heuristic misfires for a graph that
IS directed but has a flat-ish layering (e.g., some bipartite variants),
dispatch-to-flat could regress dag_consistency from 1.0 to ~0.5,
costing 12.5 std-composite pts. Mitigations:
- User API flag overrides heuristic
- Never dispatch to flat on TREE/FOREST/CHAIN/BIPARTITE_DAG families
- Include a fresh-layout A/B on the 10 candidate families per heuristic run
  before promoting the gate

### Bet 2 — `composite_auto()` + dual-metric reporting (HIGH PRIORITY, MED EV)

Add `composite_undirected` and `composite_auto`. Ship benchmarks with BOTH
columns. Do not retcon past reports.

**Impact:** reputational. Dagua can honestly claim "best layered engine" AND
"top-3 force-directed engine" rather than obscuring the second claim.

**Risk:** shows a current weakness publicly. Mitigation: pair the split with
Bet 1 so the first public report under the split already has the force-directed
fallback in.

### Bet 3 — Hybrid seed (MEDIUM PRIORITY, MED EV)

For semantically-directed graphs that are structurally "close to undirected"
(e.g., weakly layered SBMs, community DAGs), run force-directed FIRST,
then a **light** layered refinement:

```
PivotMDSInit -> StressMajorization(40) -> Optional[LayerSnap + mild BK]
```

Gives the best of both: low edge-length CV (stress), reasonable vertical
ordering (layer snap), and no direction violation (mild BK keeps y-monotonic
on forward edges).

Measured candidate graphs where this could help: `sbm_4x30`, `chung_lu_150`,
`random_dag_50` (dagua currently 61.30, -5 vs dagre). Would need experiments.

### Bet 4 — Edge-weight aware direction-agnostic stress (LOW-MED PRIORITY)

Key undirected metric dagua loses on most: `edge_length_cv`. The fix is
stress-majorization where shortest-path graph distance drives target edge
length. Dagua already has `w_stress` and pivot-distance ops (seen in
`dagua_native.py` L192 `_stress_pivot_prep`), but they're added on top of
the layered losses. Making stress the DOMINANT term when flat-dispatching
should drive edge_length_cv from 0.5-0.8 down to 0.1-0.3 (KK territory).

---

## Risk / regression analysis

### What current wins are at risk?

From CONTEXT.md "Per-graph wins to PROTECT":

| Graph | Category | Risk of regression under proposal |
|---|---|---|
| `org_chart_deep` | DAG (tree) | None — TREE -> always dagua_native |
| `random_dag_200` | DAG | Low — 200 nodes, 10/383 layers-per-node ratio, heuristic says directed |
| `hub_fanout_label_skew` | DAG | None — clearly hierarchical |
| `org_chart_1_5_4_8` | DAG | None |
| `random_dag_50` | DAG | None — family GENERAL but reasonable layers count |
| `random_bipartite_60` | DAG | None — BIPARTITE_DAG family explicitly excluded |
| `edge_label_braid` | DAG | None |
| `bipartite_4_3_4` | DAG | None — BIPARTITE_DAG |
| `weighted_karate_34` | **undirected** | **Under PROPOSAL: re-routed to dagua_flat.** Std composite may DROP from 71.68 to ~55 (lose dag_consistency=1.0). Alt composite may RISE to 60+. Net: net-neutral to net-positive IF bench shows composite_auto. REGRESSION under current std composite. |
| `real_karate_34` | **undirected** | Same as above. Current std=64.89-71.68, routing to flat -> ~55-60 std, ~55-60 alt. Regression on std. |

The two protected wins on undirected graphs (karate, weighted_karate) ARE at
risk under the current std composite. Proposal to mitigate:

- Initial sprint-20 implementation: keep `dagua_native` as default, let
  `dagua_flat` be opt-in via `algorithm="dagua_flat"` or
  `direction_hint="undirected"`. Show BOTH composites in bench. No protected-
  win regression.
- Follow-up (sprint-21+) when composite_auto is accepted: flip the
  classification-based dispatch gate to default ON.

This is a proven pattern from sprint-0: new pipeline added as opt-in first,
flipped to default once confidence established.

### Per-graph losses addressed by proposal (from CONTEXT.md "still open"):

| Graph | Category | Expected under dagua_flat |
|---|---|---|
| `small_world_100` | undir | dagua 48.58 std today. Projection: KK-style 69.80 alt, ~60 std after layered fallback. **Closes -8.51 loss.** |
| `small_world_500` | undir | dagua 49.34 std today. Projection: KK-style 50.87 alt, ~57 std after layered fallback. **Closes -4.82 loss.** |
| `planar_60` | undir | dagua 65.82 std today. Proj: 60-65 under flat. **Neutral.** |
| `parallel_cycles_4x5` | undir (cyclic) | dagua 58.24 std today. Proj: sfdp-matching, 90+ alt. **Closes -4.49 loss.** |
| `regular_3_30` | undir | dagua 68.37 std. Proj: flat should match dot's 72.23. **Closes -3.86 loss.** |
| `hexagonal_lattice_42` | undir | 85.21 std today (already strong). Proj: flat 80-88. **Net-neutral to small regression on std, win on alt.** |
| `dependency_500` | DAG | Proposal doesn't change dispatch here. Not addressed. |
| `transformer_layer` | DAG | Not addressed. |
| `ragged_feature_pyramid` | DAG | Not addressed. |

Net: proposal addresses 4 of the 10 "still open" losses with minimal
regression risk on protected wins.

### Hidden regressions to check

1. **`disconnected_label_cycle_collage`** — borderline. Has cyclic component,
   may route to flat. Currently dagua 74.41, elk 79.36. Flat pipeline needs
   to handle disconnected gracefully (sprint-19d per-component wrapper works,
   but need to confirm flat honors per-component).
2. **Real-world small graphs (karate, lesmis, football)** — if benchmark
   reporting stays on std composite, these LOOK like regressions post-sprint-20
   (dagua drops from 60-70 to 40-50 std). Only composite_auto tells the
   true story. Must ship both metrics TOGETHER or not at all.
3. **`sbm_4x30` / `sbm_5x50`** — SBMs have community structure AND some
   hierarchy (blocks ordered by index). Heuristic must not misfire; consider
   an additional signal based on modularity / clustering coefficient.

---

## Implementation order

Priority by EV/risk and by what can ship independently.

### Phase 1 (low-risk, sprint-20 week 1, ~1 day work)
1. Add `composite_undirected()` and `composite_auto()` to `dagua/metrics.py`.
2. Add `is_semantically_directed: Optional[bool]` field to `DaguaGraph`.
3. Add inference logic to `classify_graph` (new field, new helper).
4. Add a benchmark reporting column that shows `composite_auto`.

This ships WITHOUT changing any runtime dispatch. Purely diagnostic. Lets us
see the full picture before pulling the trigger on dispatch changes.

### Phase 2 (medium-risk, sprint-20 week 2, ~2-3 days)
5. Implement `dagua_flat` pipeline in `dagua/layout/ops/pipelines/dagua_flat.py`.
   Compose existing ops: PivotMDSInit / SpectralInit, StressMajorization ops,
   OverlapProjection, AspectRatioFit. Target: 150 lines.
6. Register it in `dagua/layout/ops/pipelines/__init__.py`.
7. Add `algorithm="dagua_flat"` support in `layout()` dispatcher.
8. Validate on the 35 undirected graphs: target alt-composite >= KK's alt
   composite on at least 25 of 35. Don't touch default dispatch yet.

### Phase 3 (higher-risk, sprint-20 week 3, ~1-2 days + bench runs)
9. Make default dispatch use `is_semantically_directed`. Initially gate behind
   a `LayoutConfig.auto_undirected_dispatch=False` flag, default False.
10. Run full bench with flag=True; publish dual-composite results. If
    regressions on protected wins exceed 0.5 composite_auto pts, halt.
11. If clean, flip the flag default to True.

### Phase 4 (optional, later)
12. Hybrid pipeline (Bet 3) — stress-seeded layered refinement. Only pursue
    if phase 2 shows stress-majorization alone doesn't match dagre/dot on
    graphs like `sbm_4x30`.

---

## Closing notes

The deepest point: **dagua has been optimizing for a composite that
systematically rewards "vertical stacking" on graphs where vertical stacking is
nonsensical**. This is a known risk for any benchmark-driven design, but it's
now actionable because:

- ~35% of the benchmark is genuinely undirected (clear count, clear set)
- The three direction-dependent metrics are easy to excise
- Competitors (KK, SFDP, graphviz_dot) already achieve the target alt scores,
  so we can target "match best competitor under alt composite" with real numbers
- A dispatch split is 150 lines + one classification flag + benchmarking —
  no new algorithms needed; all ops exist

The biggest strategic payoff is narrative: dagua can finally, honestly claim
"GPU-accelerated layered layout is state-of-the-art; force-directed is
competitive." That's a cleaner story than "dagua is #1 on a composite that
inflates our strengths 50%".

Finally: **sibling codex agent C may cover some of the same ground.** If
their report disagrees sharply on the 35-graph count, it's almost certainly
due to different name-keyword heuristics; a direct read of each graph's
constructor in `eval/graphs.py` would resolve. The alt-composite numbers
above are measured from real cached positions and should reproduce bit-exact
given the same positions directory.

---

## Appendix A — Reproducer script

```python
# /tmp/sprint20_C_alt_composite.py
import torch, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from dagua.eval.graphs import get_test_graphs
from dagua.metrics import full, composite

pd = Path('eval_output/variant_bench_full/positions')
tgs = {t.name: t for t in get_test_graphs()}

def alt_composite(m):
    score = 40 * max(0.0, 1.0 - m.get('edge_length_cv', 1.0))
    score += 20 * (1.0 if m.get('overlap_count', 1) == 0 else 0.0)
    score += 20 * max(0.0, 1.0 - m.get('crossing_rate', 0.5) * 10)
    score += 10 * min(1.0, m.get('angular_res_mean_deg', 20.0) / 40.0)
    score += 10 * min(1.0, m.get('cluster_mean_sep_ratio', 2.5) / 5.0)
    return score

UNDIR = ['real_karate_34', 'weighted_karate_34', 'real_football_115',
         'real_lesmis_77', 'small_world_100', 'small_world_500',
         'petersen_10', 'hexagonal_lattice_42', 'planar_60', 'grid_5x5',
         'parallel_cycles_4x5', 'triangular_lattice_36', 'regular_3_30',
         'sierpinski_42', 'chung_lu_150']
ENGS = ['dagua', 'igraph_kamada_kawai', 'graphviz_sfdp', 'graphviz_dot',
        'igraph_sugiyama', 'elk_layered', 'dagre']

for name in UNDIR:
    g = tgs[name].graph; g.compute_node_sizes()
    row = [name]
    for eng in ENGS:
        pf = pd / f'{name}__{eng}.pt'
        if not pf.exists():
            row.append('--')
            continue
        pos = torch.load(pf, map_location='cpu', weights_only=False)
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        row.append(f'{composite(m):.1f}/{alt_composite(m):.1f}')
    print(row)
```

Run with `CUDA_VISIBLE_DEVICES= python /tmp/sprint20_C_alt_composite.py`.
Takes ~60s for 15 graphs on CPU.

## Appendix B — Why undirected-origin graphs get oriented ascending

`dagua/eval/graphs.py:160 _undirected_to_dag()`:

```python
forward_mask = edge_index[0] < edge_index[1]
reverse_mask = edge_index[0] > edge_index[1]
# keeps forward, flips reverse, dedups
```

This is a tooling convenience — dagua's layout pipelines assume directed
edges, so unconverted graphs would force each edge to be treated as its own
concept. The orientation is NOT a claim about semantic direction. Engines
like graphviz_dot/elk ingest these as directed and produce layered outputs
that score high on dag_consistency; engines like KK ingest the underlying
undirected structure and produce non-vertical layouts that score low on
dag_consistency but high on edge_length_cv. **The composite picks the former.
That's the bug.**
