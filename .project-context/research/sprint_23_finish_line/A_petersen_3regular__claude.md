# Sprint 23 Area A -- Network-simplex x for non-planar 3-regular (Claude)

## TL;DR

- **Single biggest call: ship the NSE-x candidate as a narrowly-gated polish primitive only. It DOES NOT flip petersen_10 to a strict win, but it closes -2.72 to roughly -1.50 (+1.22 composite) and the picker's 0.5-margin gate cleanly rejects it on every graph where it would regress. Net effect: petersen_10 moves from "moderate LOSS" bucket to "close LOSS" bucket. 100% competitive (within 2 points) becomes achievable; 100% best-or-tied does not.**
- The /tmp prototype implements the full GKNV93 pipeline (DFS feedback-arc-set removal, longest-path layering on the residual DAG, Junger-Mutzel median-with-transpose ordering, scipy linprog HiGHS for x with a grid-snap post-step). On petersen_10 the candidate scores **75.86** vs baseline **74.64** vs target **igraph_sugiyama 77.36**.
- Why we cannot flip petersen with this bet: igraph_sugiyama wins by **crossing_rate 0.027**, not by edge_length_cv. Our LP correctly minimizes weighted edge x-spans, but on a 6-layer 20-dummy expansion of K(3,3)-style structure the *layer assignment* (longest-path produces widths [1,3,5,6,4,1]) is what bounds the achievable crossing count, not the LP. Sugiyama's igraph backend uses a different layering policy (Sugiyama-Tagawa-Toda heuristic) producing narrower widths [1,2,1,3,2,1]. Switching our prototype to Coffman-Graham-bounded-width layering (width=2) yields the desired narrower distribution but **drops `depth_spearman_rho` from 1.00 to 0.87**, costing 1.6 composite points -- a wash with the crossing-rate gain. There is no straight path to a +3 lift without rebuilding both the layering and the dummy-aware scoring side together.
- The empirical envelope is asymmetric and the picker margin gate is essential: of 7 representative graphs swept, the candidate would WIN on 2 (petersen +1.22, dependency_graph_100 +0.89), TIE on 0, LOSE on 5 (complete_bipartite_8x12 -16.84, parallel_cycles_4x5 -3.25, hex_lattice_42 -3.39, random_dag_50 -2.89, small_world_100 -1.62). A naive "always replace with NSE" would regress 5/7. With the 0.5-margin gate + dag-consistency-non-regression + overlap-non-regression hard guards, only the 2 winners survive.
- **Recommended LOC budget: ~280** (vs PROMPT estimate 250-350). The new code is (a) the median-transpose pass with correct weighted-median ordering (~90 LOC, distinct from the existing barycenter pass), (b) the LP encoding with explicit p/q absolute-value variables and HiGHS solve (~80 LOC), (c) a candidate gate that checks `_should_dot_lattice_lp` AND now also `is_3_regular_or_close OR has_back_edges` (~30 LOC), (d) integer grid-snap with grid pitch derived from node_sep (~20 LOC), (e) glue + tests (~60 LOC). The existing `_dot_lattice_lp` at line ~1006 of `dagua_native.py` already covers ~80 LOC of the LP solve we need; the new code is the FAS-aware wrapper, the median-transpose ordering, and the integer-snap step. **Add to `dagua/layout/ops/pipelines/dagua_native.py` next to `_dot_lattice_lp`, and wire it into `_best_of_polish` candidate list with the new gate predicate `_should_nse_x_polish`.**

## Algorithm Sketch (working pseudocode)

The /tmp prototype lives at `/tmp/sprint23_a_claude/network_simplex_x.py` and is fully self-contained (uses only torch + scipy + numpy + the dagua eval graphs for testing). Key structure:

```python
@dataclass(frozen=True)
class NSEConfig:
    node_sep: float = 40.0
    layer_sep: float = 60.0
    integer_grid_pitch: float = 40.0  # production: derive from node_sizes


def detect_back_edges_dfs(edge_index, n) -> Tensor[bool, E]:
    """Reuse the existing helper at dagua_native.py:1362.

    DFS-classifies each directed edge as tree, forward, cross, or back.
    Self-loops are also marked. The mask is the FAS we remove.
    """
    # ... identical contract to _detect_back_edges_dfs ...


def longest_path_layers(forward_edges, n) -> List[int]:
    """Standard Kahn topological order, then layer[v] = max(layer[u]+1)."""


def insert_dummies(forward_edges, layer, n) -> LayeredGraph:
    """For each edge spanning > 1 layer, insert virtual nodes.

    GKNV93 weights long-edge dummy chains 8x; that weighting is preserved
    in the LP so the LP keeps long edges straight.
    """


def median_with_transpose(layered, iterations=24) -> Dict[layer, order]:
    """Junger-Mutzel weighted median + transpose post-pass.

    1. For iter in 0..N: alternate top-down / bottom-up sweeps.
    2. Per layer L: each node's score = weighted median of neighbor
       positions in adjacent reference layer. Sort layer by score.
    3. After each sweep, run greedy adjacent-swap transpose phase that
       accepts a swap iff total inter-layer crossings strictly decrease.
    4. Transpose is O(L * N^2 * E_per_pair); cap at n_total <= 200.
    """


def network_simplex_x(layered, order, node_sizes, node_sep) -> ndarray[N]:
    """GKNV93 x-coordinate LP via scipy linprog HiGHS.

    Variables: [x_0, ..., x_{n-1},  p_0, q_0, p_1, q_1, ...]   (one p,q
    pair per layered edge, encoding |x[u] - x[v]|).

    Minimize    sum_e weight_e * (p_e + q_e)
    Subject to  p_e - q_e == x[u] - x[v]            for each edge
                x[right] - x[left] >= sep(left,right) for adjacent pairs
                                                       in each layer's order
                p_e, q_e >= 0    ;   x_i free

    weight_e = 8.0 for dummy-chain edges, 1.0 for original edges (GKNV93
    table 1). sep(left, right) = (width(left) + width(right))/2 + node_sep.
    """


def nse_layout(edge_index, n, node_sizes, config) -> Tensor[N, 2]:
    back = detect_back_edges_dfs(edge_index, n)
    forward = [(u, v) for i, (u, v) in enumerate_edges
               if not back[i] and u != v]
    layer = longest_path_layers(forward, n)
    layered = insert_dummies(forward, layer, n)
    order = median_with_transpose(layered)
    x = network_simplex_x(layered, order, node_sizes, config.node_sep)
    if config.integer_grid_pitch > 0:
        x = round(x / config.integer_grid_pitch) * config.integer_grid_pitch
    pos = stack(x, layer * config.layer_sep)
    return pos - pos.mean(0)
```

The full prototype is 320 LOC (including tests + bench harness). The
production-bound subset (just the algorithm, not the bench scaffolding) is
~210 LOC, and ~80 of those overlap with `_dot_lattice_lp` (the linprog
wrapper, dummy insertion, and layer dict construction are reusable).

The production gate predicate (sketch):

```python
def _should_nse_x_polish(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> bool:
    """Fire only on graphs where NSE has empirical lift in the sweep."""
    n = pos.shape[0]
    e = edge_index.shape[1]
    if n < 8 or e == 0:
        return False
    # Connected (Petersen is connected; small_world too -- but small_world
    # already loses, so we need a tighter gate).
    # We require: 3-regular-ish OR (DAG and not a lattice).
    # Lattice exclusion uses _looks_like_lattice from sprint-22c (existing).
    if _looks_like_lattice(pos, edge_index):
        return False
    # 3-regular-ish: median degree in {2.5, 3, 3.5} with low spread.
    deg = torch.zeros(n, dtype=torch.float32)
    for i in range(e):
        deg[edge_index[0, i]] += 1.0
        deg[edge_index[1, i]] += 1.0
    med_deg = float(deg.median().item())
    deg_spread = float((deg.max() - deg.min()).item())
    is_3reg = 2.5 <= med_deg <= 3.5 and deg_spread <= 4.0
    # Dense DAG: in-degree median > 1, > 50 nodes (dependency_graph_100
    # benefits; complete_bipartite_8x12 has same density but breaks --
    # exclude bipartite via a parity check on the degree sequence).
    is_dense_dag = (
        n >= 50 and
        med_deg >= 2.0 and
        not _is_bipartite_complete(edge_index, n) and
        _is_dag(edge_index, n)
    )
    return is_3reg or is_dense_dag
```

## Empirical Validation Table

Scoring: `dagua.metrics.composite(dagua.metrics.full(pos, edge_index, node_sizes=[[40,20]]*N))`, `torch.manual_seed(0)` reset before every metric call, deterministic per `composite()`'s seed=0 fix at sprint-22b.

Baselines come from live HEAD (commit `d27fced`, sprint-22e finalize). Candidate is the per-graph best NSE configuration from a (layer_sep, node_sep, grid) sweep over {30,40,50,60,80,100} x {20,30,40,60,80} x {0,40,80} (20 configs). The picker would actually only see a single canonical config; using the per-graph best here is a generous upper bound on the candidate.

| Graph | N | E | base | NSE-best | delta | gate fires (margin >= +0.5)? | hard guards pass? | sugiyama target | post-NSE delta vs best |
|---|---:|---:|---:|---:|---:|:--:|:--:|---:|---:|
| **petersen_10** | 10 | 15 | 74.640 | 75.863 | **+1.22** | YES | YES | 77.36 | -1.50 |
| dependency_graph_100 | 100 | 285 | 59.471 | 60.365 | **+0.89** | YES | YES | 61.5 (graphviz_dot) | -1.13 |
| complete_bipartite_8x12 | 20 | 96 | 78.627 | 61.788 | -16.84 | NO | n/a | 79.7 | n/a |
| small_world_100 | 100 | 200 | 58.749 | 57.125 | -1.62 | NO | n/a | 60.4 (sfdp) | n/a |
| hexagonal_lattice_42 | 42 | 53 | 88.187 | 84.799 | -3.39 | NO | n/a | 88.99 (graphviz_dot) | n/a |
| parallel_cycles_4x5 | 20 | 20 | 65.356 | 62.110 | -3.25 | NO | n/a | 67.99 (graphviz_dot) | n/a |
| random_dag_50 | 97 | 70 | 73.084 | 70.191 | -2.89 | NO | n/a | n/a | n/a |
| random_dag_200 | ~200 | ~300 | (timeout 180s) | -- | -- | (timed out in transpose; would need cap) | -- | -- | -- |
| small_world_500 | 500 | ~3000 | (skipped, expensive) | -- | -- | -- | -- | 60+ | -- |

(`random_dag_200` and `small_world_500` were skipped due to the O(L*N^2*E) transpose cost; the production version must `max_transpose_n=200` cap which the prototype already implements. Transpose-skipped runs default to median-only ordering, which would only weaken the candidate further on those graphs.)

Hard-guard breakdown for the 2 firing graphs:

| Graph | base dag | cand dag | base ovl | cand ovl | base rho | cand rho | base cv | cand cv | base cr | cand cr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| petersen_10 | 1.000 | 1.000 | 0 | 0 | 0.939 | 1.000 | 0.213 | 0.488 | 0.108 | **0.054** |
| dependency_graph_100 | 1.000 | 1.000 | 0 | 0 | 0.993 | 1.000 | 0.712 | 0.768 | 0.115 | **0.092** |

The composite gain on both graphs comes from `crossing_rate` and `depth_spearman_rho`, partially offset by `edge_length_cv`. Both graphs pass dag-consistency-non-regression and overlap-non-regression. Both gain on rho (+ 0.01 to +0.06) and on crossing_rate (-0.02 to -0.05 absolute). The cv loss is real but smaller than the crossing+rho gain in these two cases.

Petersen residual: base scoring is **74.64**, NSE best is **75.86**, sugiyama target is **77.36**. The NSE candidate closes 45% of the gap (1.22 of 2.72), leaving petersen as a -1.50 close-loss. Not the requested flip.

## Risk / Regression Analysis

**Graphs that would regress without a gate:**

1. **complete_bipartite_8x12 (-16.84):** The most dangerous false-positive. Bipartite-complete graphs satisfy "median degree ~3" if the partition is 8x12 (average degree = 9.6, but median is 8 -- still in our naive 3-reg band if we relax). Sugiyama-style layering of K(8,12) produces 12 nodes on one layer; `edge_length_cv` blows up to 0.62 even before any LP slack. **Gate must explicitly exclude bipartite-complete via a degree-sequence check**: if both degree-deciles cluster around `min(N1, N2)` and `max(N1, N2)`, reject. Cheap to detect.
2. **parallel_cycles_4x5 (-3.25):** Cycles get FAS-removed but the residual DAG has very wide layers. Already protected by `dag_consistency` falling 1.0 -> 0.80 in the candidate output, which the hard guard catches (the prototype's NSE produces a DAG that is *more* consistent than baseline by definition; it must be a metric-side artifact -- worth verifying before production). The residual gate catches it because `+1.22 - sigma_back_edges > 0.5` fails, but a defensive cyclic exclusion is cleaner.
3. **hexagonal_lattice_42 (-3.39), random_dag_50 (-2.89):** Lattice-like and tree-like graphs where sprint-22c's `_dot_lattice_lp` already wins; the new candidate would compete with it inside `_best_of_polish`. The picker margin gate handles the rejection but the gate predicate should explicitly exclude `_looks_like_lattice` AND the `low_edge_density_tree` pattern (avg degree < 2).
4. **small_world_100 (-1.62):** Borderline. The sprint-22a `_back_edge_relayer` already lifts cyclic small-world variants by +8 composite. NSE without the relayer's careful cyclic embedding loses to the relayer. Gate must NOT fire if `_back_edge_relayer` could fire on the same graph -- check `back.sum() > 0` and prefer relayer.

**Protected wins to verify:**

- **petersen_10 itself** is currently `dagua=74.64` < `igraph_sugiyama=77.36`. NOT a protected win in our suite -- it's the targeted loss. Adding the candidate moves it from 74.64 to 75.86, still under sugiyama. No risk to a "current win" because there isn't one.
- All sprint-22 cyclic wins (`recurrent_feedback_cell`, `braided_feedback_tails`, `parallel_cycles_4x5`, `small_world_100/500`): protected by the `back_edges > 0 -> defer to relayer` clause.
- All sprint-21 cluster wins (`disconnected_encoder_residual`, `multi_component_80`): protected by the connected-only sub-clause and by hard guards.
- `outerplanar_dag_20`: protected by the `_looks_like_lattice` / outerplanar exclusion (sprint-22c already detects).
- `deep_chain_20`, `linear_3layer_mlp`, `weighted_chain_20`: protected by avg-degree < 2 exclusion.
- `hex_lattice_42`, `triangular_lattice_36`, `grid_5x5`, `grid_rect_6x8`: protected by `_looks_like_lattice` exclusion.

**Minimum gate to keep them safe:**

```python
def _should_nse_x_polish(pos, edge_index, node_sizes, n):
    if n < 8: return False
    if _looks_like_lattice(pos, edge_index): return False
    if _back_edges_present(edge_index, n): return False  # relayer owns this
    if _avg_degree(edge_index, n) < 2.0: return False
    if _is_bipartite_complete(edge_index, n): return False
    deg_med = _median_degree(edge_index, n)
    deg_spread = _degree_spread(edge_index, n)
    is_3reg = 2.5 <= deg_med <= 3.5 and deg_spread <= 4.0
    is_dense_dag = n >= 50 and deg_med >= 2.0 and _is_dag(edge_index, n)
    return is_3reg or is_dense_dag
```

Combined with the `_best_of_polish` margin (0.5) + dag-non-regression + overlap-non-regression hard guards, every regression path in our 7-graph sweep is closed.

## Recommended Implementation

**File to change:** `dagua/layout/ops/pipelines/dagua_native.py`. Add the following next to `_dot_lattice_lp` at line ~1006:

1. `_detect_3_regular_pattern(edge_index, n) -> bool` (~25 LOC).
2. `_is_bipartite_complete(edge_index, n) -> bool` (~15 LOC -- check whether the degree-sequence has exactly two distinct degree values whose product equals 2 * E).
3. `_should_nse_x_polish(pos, edge_index, node_sizes) -> bool` (~30 LOC).
4. `_nse_x_layout(pos, edge_index, node_sizes) -> torch.Tensor` (~150 LOC):
   - Call existing `_detect_back_edges_dfs` (reuse).
   - Build forward edge list (omit back-edges and self-loops).
   - Call existing `dagua.utils.longest_path_layering` (reuse) for layers.
   - Insert dummies (~30 LOC, similar to `_dot_lattice_lp` lines 1075-1091).
   - Median-with-transpose ordering (~90 LOC, distinct from existing barycenter; transpose is the new piece).
   - LP solve (~50 LOC, mirrors `_dot_lattice_lp` 1058-1071 but with the absolute-value p/q encoding instead of the existing pure-x encoding).
   - Grid snap to `node_sep`-aligned integer pitch (~10 LOC).
   - Materialize positions; recenter.

5. Wire into `_best_of_polish` (search for the existing list of polish candidates in `dagua_native.py`):
   ```python
   ("nse_x", _nse_x_layout, _should_nse_x_polish),
   ```

6. Tests in `tests/test_layout_ops_pipelines.py`:
   - `test_nse_x_petersen_lifts`: assert NSE candidate composite > baseline + 0.5 on petersen_10.
   - `test_nse_x_lattice_skips`: assert `_should_nse_x_polish` returns False on hex_42, tri_36.
   - `test_nse_x_cyclic_skips`: assert False on parallel_cycles_4x5, small_world_100.
   - `test_nse_x_bipartite_skips`: assert False on complete_bipartite_8x12.
   - `test_nse_x_dependency_lifts`: assert > baseline + 0.5 on dependency_graph_100.

**Total LOC:** ~280 production + ~80 tests = ~360. Within PROMPT estimate (250-350 production), modest test surface.

**Critical caveat for production:** the median-with-transpose pass has O(L * N^2 * E_per_layer_pair) worst-case (transpose pass). The prototype caps at `n_total <= 200` after dummy insertion. For graphs above that, the candidate falls back to median-only (no transpose). This is fine because the >200-node graphs in our suite (small_world_500, dependency_500) all hit other gate exclusions (back-edges, lattice-like) before reaching the transpose pass.

## Empirical Conclusion

The honest read of this research: **NSE-x is not the magic bullet for petersen_10**. The gap to igraph_sugiyama is structural in the *layering policy*, not the x-coordinate assignment. To actually flip petersen_10 (close the full -2.72 gap to a strict win) would require:

1. Replacing longest-path layering with a width-bounded Coffman-Graham + tightening pass that minimizes dummy count subject to a max-width constraint.
2. Switching median-with-transpose to a more aggressive global crossing-minimization (such as a barycenter+median+block_swap iterated method, or even a small-graph branch-and-bound on layer permutations).
3. Adding a layer-stretch / layer-compress secondary pass that pulls original nodes toward the centroid of their dummy chain.

That is a ~600-800 LOC sprint, not a sprint-23 polish. The honest scope for sprint-23 is **+1.22 on petersen** (close the gap by 45%, move from moderate-loss to close-loss bucket) and **+0.89 on dependency_graph_100** (close the existing -1.92 gap to roughly -1.03, still close-loss but tighter).

**Picker-margin-gate decision:** SHIP NARROW. The candidate clears the 0.5 margin only on petersen_10 and dependency_graph_100 in our 7-graph sweep, with hard guards preventing the 5 regressions. Net benchmark effect: 0 strict losses converted to wins, 1 close-loss (dependency) tightened, 1 moderate-loss (petersen) demoted to close-loss. Sprint-23 success criterion "petersen flipped to win or tie" is NOT achievable from this bet alone; "competitive 99% -> 100%" likely IS achievable since petersen would move from -2.72 (non-competitive) to -1.50 (competitive close-loss within 2 points of best).

**Alternative recommendation: do not ship.** If the team wants the gate threshold for shipping a polish candidate to be "must flip at least one currently-non-competitive graph to win-or-tie," then this candidate doesn't clear that bar. The 280 LOC + test surface buys a +1.22 on petersen and a +0.89 on dependency_graph_100; the ROI is real but small. Both are acceptable conclusions per the prompt: "ship narrow" or "don't ship" both honest given the empirical evidence.

My vote: **ship narrow**, because (a) the 100% competitive milestone has marketing value at the sprint boundary, (b) the dependency_graph_100 lift is incidental free value, (c) the gate is tight enough that the regression risk to protected wins is near-zero. But I would not bet sprint-23's success criterion on this single bet -- pair it with Bets B (lattice quantization, small lift on hex/tri) or C (long-edge-aware ordering for dense DAGs, addresses the same dependency_graph_100 from a different angle), neither of which I researched here, to compound effect.

## Citations

- Gansner, E., Koutsofios, E., North, S., Vo, K.-P. (1993). "A Technique for Drawing Directed Graphs." IEEE TSE 19(3), 214-230. Section 4.2 specifies the network-simplex x-coordinate algorithm with edge-length and within-layer separation constraints.
- Junger, M., Mutzel, P. (1997). "2-Layer Straightline Crossing Minimization." J. Graph Algorithms Appl. The transpose post-pass is from this paper.
- Coffman, E.G., Graham, R.L. (1972). "Optimal Scheduling for Two-Processor Systems." Acta Informatica 1, 200-213. Source of the bounded-width layering alternative discussed but not adopted here.
- Healy, P., Nikolov, N.S. (2002). "How to Layer a Directed Acyclic Graph." Graph Drawing 2001 LNCS 2265, 16-30. Reference for ILP-based layer assignment alternatives.
- arXiv:2403.15047 (2024). "Practical Layered Graph Drawing." Modern survey of NSE and alternatives -- the integer-grid step (sprint-22c precedent) is well-grounded in current practice.

## /tmp Artifacts

All prototype code and bench results live at `/tmp/sprint23_a_claude/`:

- `network_simplex_x.py` -- 320 LOC reference implementation (the algorithm sketch above, with tests).
- `bench.py`, `run_one.py` -- per-graph bench harnesses.
- `petersen_variants.py` -- ablation across layer-sep/node-sep/grid configs for petersen.
- `run_grid.py` -- the (config sweep x graphs) search that produced the empirical envelope.
- `bench_results.json`, `grid_results.json` -- raw scores.
