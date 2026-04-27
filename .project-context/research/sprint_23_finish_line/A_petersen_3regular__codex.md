# Sprint 23 Area A -- Petersen / 3-Regular Network-Simplex Candidate

## TL;DR

- **Single biggest call:** do **not** ship the researched Area A candidate as a sprint-23 implementation. A real `dagua.metrics.composite(dagua.metrics.full(...))` check against fresh HEAD positions makes the best prototype a regression on `petersen_10`: `74.64 -> 73.40` (`-1.24`), while `igraph_sugiyama` remains `77.36`.
- The original hypothesis was partly right but incomplete. GKNV93's coordinate network-simplex step is necessary for dot/Sugiyama quality, but Petersen's missing score is dominated by rank/order/dummy-expansion decisions that reduce crossings, not by the x-coordinate LP alone.
- A narrow connected-cubic gate hits only two locally available variant-suite graphs: `petersen_10` and `regular_3_30`. Both reject under a `+0.5` picker margin when compared to fresh HEAD layouts.
- Full-score evidence: `petersen_10` best candidate crossing rate improves vs fresh dagua (`0.1081 -> 0.0946`) but edge-length CV worsens sharply (`0.2129 -> 0.4706`), and it does not approach igraph's crossing rate (`0.0270`).
- Recommendation: abandon the isolated x-coordinate candidate; if Area A continues, implement a full Sugiyama candidate with dummy-expanded mincross ordering and rank assignment variants, then let the existing metric picker choose it. Expected LOC is closer to `500-750`, not the `250-350` initially budgeted.

## Algorithm Sketch

Source basis: Gansner, Koutsofios, North, and Vo, "A Technique for Drawing Directed Graphs," IEEE Transactions on Software Engineering 19(3):214-230, 1993, DOI `10.1109/32.221135` ([IEEE](https://ieeexplore.ieee.org/document/221135/), [DBLP](https://dblp.org/rec/journals/tse/GansnerKNV93.html)). The paper describes a four-pass layered drawing pipeline: rank assignment by network simplex, crossing reduction within ranks, coordinate assignment by constructing/ranking an auxiliary graph, then spline routing. The prototype only validates the coordinate-assignment bet plus a minimal residual-DAG setup.

Working scratch implementation: `/tmp/sprint23_a_codex/prototype.py`.

Condensed pseudocode from the prototype:

```python
from scipy.optimize import Bounds, LinearConstraint, milp


def area_a_candidate(edge_index, num_nodes, pitch_x=128.0, pitch_y=72.0):
    """Return a GKNV-like layered integer-x layout for a graph."""
    back_mask = _detect_back_edges_dfs(edge_index, num_nodes)
    dag_edges = edge_index[:, ~back_mask]

    layers = longest_path_layering(dag_edges, num_nodes)
    layers = torch.as_tensor(layers, dtype=torch.long)

    rows = [[] for _ in range(int(layers.max().item()) + 1)]
    for node in range(num_nodes):
        rows[int(layers[node].item())].append(node)

    pred, succ = adjacency_by_direction(dag_edges, num_nodes)
    order_pos = {node: i for row in rows for i, node in enumerate(row)}
    for pass_idx in range(12):
        sweep = range(1, len(rows)) if pass_idx % 2 == 0 else range(len(rows) - 2, -1, -1)
        neighbors = pred if pass_idx % 2 == 0 else succ
        for layer in sweep:
            rows[layer].sort(
                key=lambda node: (
                    median([order_pos[n] for n in neighbors[node]])
                    if neighbors[node]
                    else order_pos[node],
                    order_pos[node],
                    node,
                )
            )
        order_pos = {node: i for row in rows for i, node in enumerate(row)}

    # MILP variables are real-node x coordinates plus one nonnegative slack
    # variable per edge. x variables are integer grid coordinates.
    edge_count = edge_index.shape[1]
    var_count = num_nodes + edge_count
    objective = np.zeros(var_count)
    objective[num_nodes:] = 1.0

    constraints, lower, upper = [], [], []
    for row in rows:
        for left, right in zip(row, row[1:]):
            coeff = np.zeros(var_count)
            coeff[right] = 1.0
            coeff[left] = -1.0
            constraints.append(coeff)
            lower.append(1.0)      # adjacent same-rank separation
            upper.append(np.inf)

    for edge_id, (src, dst) in enumerate(edge_index.t().tolist()):
        slack = num_nodes + edge_id

        coeff = np.zeros(var_count)
        coeff[slack] = 1.0
        coeff[src] = -1.0
        coeff[dst] = 1.0
        constraints.append(coeff)
        lower.append(0.0)          # slack >= x_src - x_dst
        upper.append(np.inf)

        coeff = np.zeros(var_count)
        coeff[slack] = 1.0
        coeff[src] = 1.0
        coeff[dst] = -1.0
        constraints.append(coeff)
        lower.append(0.0)          # slack >= x_dst - x_src
        upper.append(np.inf)

    coeff = np.zeros(var_count)
    coeff[0] = 1.0                 # remove translation freedom
    constraints.append(coeff)
    lower.append(0.0)
    upper.append(0.0)

    result = milp(
        c=objective,
        integrality=np.ones(var_count),
        bounds=Bounds(
            np.r_[np.full(num_nodes, -20.0), np.zeros(edge_count)],
            np.r_[np.full(num_nodes, 20.0), np.full(edge_count, 40.0)],
        ),
        constraints=LinearConstraint(np.vstack(constraints), lower, upper),
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )

    x = result.x[:num_nodes] - np.mean(result.x[:num_nodes])
    y = layers.numpy().astype(float) - float(layers.float().mean().item())
    return torch.tensor(np.c_[x * pitch_x, y * pitch_y], dtype=torch.float32)
```

Important limitations:

- The prototype does not create a full dummy-node expanded graph. It orders real nodes only, then optimizes original-edge x-span slacks.
- It uses longest-path layering on the residual DAG, not network-simplex rank assignment.
- It does not run the GKNV two-start mincross procedure or local transposition on dummy-expanded ranks.
- It does use SciPy/HiGHS MILP integrality, so the failed result is not just an LP-relaxation artifact.

## Empirical Validation

Scoring command family:

- Fresh target check used `dagua.layout.engine.layout(..., LayoutConfig(seed=0, device="cpu"))`.
- Candidate and competitors used `dagua.metrics.composite(dagua.metrics.full(...))`.
- Direct-call node sizes were `torch.tensor([[40.0, 20.0]] * N)`, as requested.
- Full scorer settings in scratch matched the benchmark-style reduced full call: `stress_sources=100`, `stress_targets=250`, `crossing_samples=100_000`, `neighborhood_samples=1_000`.

The variant benchmark manifest has 105 graph names. Local generation resolved 101; the four unavailable names were TorchLens cached cases: `tl_cnn_small`, `tl_mlp_3layer`, `tl_resnet_2block`, `tl_transformer_1layer`. The connected-cubic gate accepted only two available graphs: `petersen_10` and `regular_3_30`.

### Candidate Table

`baseline` means fresh HEAD layout for the two gated graphs. For non-gated protected/sample graphs, the candidate is a no-op and the table uses the cached `variant_bench_full/positions/<graph>__dagua.pt` tensor only to show the gate envelope.

| Graph | Gate | Baseline composite | Candidate composite | Delta | Picker decision |
|---|---:|---:|---:|---:|---|
| `petersen_10` | yes | 74.64 | 73.40 | -1.24 | reject |
| `regular_3_30` | yes | 77.54 | 68.59 | -8.94 | reject |
| `complete_bipartite_8x12` | no | 57.67 | 57.67 | +0.00 | skip/no-op |
| `small_world_100` | no | 57.13 | 57.13 | +0.00 | skip/no-op |
| `small_world_500` | no | 57.26 | 57.26 | +0.00 | skip/no-op |
| `hexagonal_lattice_42` | no | 79.62 | 79.62 | +0.00 | skip/no-op |
| `triangular_lattice_36` | no | 86.78 | 86.78 | +0.00 | skip/no-op |
| `deep_chain_20` | no | 97.49 | 97.49 | +0.00 | skip/no-op |
| `grid_5x5` | no | 89.48 | 89.48 | +0.00 | skip/no-op |
| `clustered_medium_5x20` | no | 70.89 | 70.89 | +0.00 | skip/no-op |
| `outerplanar_dag_20` | no | 71.22 | 71.22 | +0.00 | skip/no-op |
| `multi_component_80` | no | 72.49 | 72.49 | +0.00 | skip/no-op |

Target competitor reference under the same full scorer:

| Layout | Composite | Crossing rate | Edge-length CV | Depth rho | Edge straightness |
|---|---:|---:|---:|---:|---:|
| fresh dagua HEAD | 74.64 | 0.1081 | 0.2129 | 0.9387 | 25.39 |
| best prototype, pitch `(128, 72)` | 73.40 | 0.0946 | 0.4706 | 1.0000 | 20.99 |
| `igraph_sugiyama` cached | 77.36 | 0.0270 | 0.4898 | 0.9813 | 29.80 |
| `graphviz_dot` cached | 72.07 | 0.0946 | 0.4565 | 0.9688 | 27.05 |

Interpretation:

- The prototype gets the expected layered/Sugiyama shape: perfect DAG consistency and depth correlation, no overlaps, and slightly fewer crossings than fresh dagua.
- The candidate gives up dagua's strongest Petersen advantage: compact, low-CV 3-regular geometry. Edge-length CV more than doubles.
- It matches Graphviz dot's crossing rate, not igraph Sugiyama's crossing rate. That is the key failure. The prompt's premise singled out x-coordinate network simplex, but the empirical gap is mostly before x assignment.
- `regular_3_30` is an even stronger warning: current dagua is already much better than this prototype and better than both `igraph_sugiyama` and `graphviz_dot` on the same full scorer.

### Pitch Search

I ran a full-score pitch grid over `pitch_x, pitch_y in {40, 50, 60, 72, 82, 96, 110, 128, 150, 180}`.

Best rows:

| Graph | Best pitch | Best candidate | Fresh baseline | Delta |
|---|---:|---:|---:|---:|
| `petersen_10` | `(128, 72)` | 73.40 | 74.64 | -1.24 |
| `regular_3_30` | `(72, 128)` | 68.59 | 77.54 | -8.94 |

The pitch search rules out an easy constant-tuning fix. The candidate's structure, not its scale, is the limiting factor.

## Risk / Regression Analysis

The minimum safe production gate is:

```python
eligible = (
    num_nodes >= 8
    and num_nodes <= 64
    and is_connected_undirected(edge_index, num_nodes)
    and torch.all(undirected_degrees(edge_index, num_nodes) == 3)
)
accept = eligible and candidate_score >= baseline_score + 0.5
```

Under this gate, the candidate would be evaluated on only `petersen_10` and `regular_3_30` in the local 101-graph manifest envelope, then rejected on both. This makes the implementation safe but useless.

Broader gates are risky:

- **Any 3-regular graph:** already fails because `regular_3_30` loses almost nine points vs fresh HEAD.
- **Any non-planar graph:** would include `small_world_*`, dense random graphs, complete bipartite graphs, and power-law graphs. The prototype is not designed for those; it forces a layered drawing and would likely destroy force-directed wins.
- **Any graph where dot/Sugiyama is competitive:** not enough. Petersen shows `graphviz_dot` is worse than fresh dagua (`72.07` vs `74.64`), and igraph's advantage is not reproduced by this simplified x step.
- **Any DAG/lattice:** this overlaps Bet B/C and risks touching protected lattice wins. The no-op gate intentionally excludes `hexagonal_lattice_42`, `triangular_lattice_36`, `grid_5x5`, `deep_chain_20`, `outerplanar_dag_20`, and `clustered_medium_5x20`.

The practical regression risk is not memory or runtime. The MILP is tiny for `N <= 64`. The risk is aesthetic: layered integer-x compaction converts compact symmetric regular layouts into tall rank drawings with worse edge-length uniformity. The current metrics penalize that enough that the picker rejects it.

## Recommended Implementation

Do not implement the isolated Area A x-coordinate candidate.

If the team still wants to chase Petersen, the implementation should be reframed as a full Sugiyama candidate, not a network-simplex x polish pass:

1. Build residual DAG variants:
   - existing `_detect_back_edges_dfs`;
   - possibly one alternative feedback-arc heuristic that minimizes long back edges for cubic graphs.

2. Add rank-assignment variants:
   - `longest_path_layering`;
   - Coffman-Graham or width-aware layering;
   - optional network-simplex rank assignment if a compact Python implementation is acceptable.

3. Dummy-expand all long edges before crossing reduction:
   - real and dummy nodes share rank order;
   - edge weights distinguish original edges, dummy chains, and back-edge dummies.

4. Run GKNV-style ordering:
   - two starts, top-down and bottom-up;
   - weighted median sweeps;
   - adjacent transpose passes, including tie flipping every other sweep as described by GKNV93.

5. Run x assignment:
   - same-rank adjacency separation constraints;
   - auxiliary edge-pair constraints for original/dummy edges;
   - integer grid via MILP or an internal network-simplex ranker.

6. Drop dummy nodes and score as a metric-picked candidate:
   - `candidate_score >= baseline_score + 0.5`;
   - hard reject on overlaps;
   - hard reject if edge-length CV increases by more than a tuned guard unless crossing rate improves enough to compensate.

Recommended gate predicate for the next attempt:

```python
def should_try_cubic_sugiyama(edge_index: torch.Tensor, num_nodes: int) -> bool:
    degree = undirected_degrees(edge_index, num_nodes)
    return (
        8 <= num_nodes <= 64
        and is_connected_undirected(edge_index, num_nodes)
        and bool(torch.all(degree == 3))
    )
```

Pipeline structure:

- New ops in `dagua/layout/ops/sugiyama_cubic.py` or integrated into existing Sugiyama ops if they already own dummy expansion/order state.
- Candidate wrapper in `dagua/layout/ops/pipelines/dagua_native.py`, next to existing polish candidates, so it can call `composite(full(...))` and be rejected without affecting the baseline.
- Tests in `tests/test_layout/` for:
  - connected-cubic gate accepts Petersen and `regular_3_30`;
  - gate rejects lattices, complete bipartite, disconnected, non-cubic;
  - picker rejects candidates below margin;
  - dummy expansion preserves original node order mapping.

LOC estimate:

- Isolated x candidate: `250-350` LOC, but not worth shipping.
- Full useful candidate: `500-750` LOC production code plus `100-180` LOC tests. The extra cost is dummy expansion, weighted mincross ordering, two-start selection, and clean candidate plumbing.

## Controversial Choices

- I used fresh HEAD positions for the two gated graphs because the cached `variant_bench_full` `petersen_10__dagua.pt` tensor is stale. It scores `71.42`; fresh HEAD scores `74.64`, matching the sprint-23 context. Using the stale tensor would falsely show a `+1.97` candidate lift.
- I did not claim a 93-graph full-score sweep. The local variant manifest has 105 names, and 101 were available through `get_test_graphs(max_nodes=10_000)`. I ran a full gate sweep over those 101 and full metrics on the selected target/protected table. Because the candidate is no-op outside the gate and rejected inside the gate, a full-score run over every no-op graph would not add decision value.
- I used SciPy `milp` rather than writing a bespoke network-simplex implementation in scratch. That is sufficient for this research question because the candidate failed despite exact integer x variables on the small target graphs.

## Concerns

- The prototype does not prove that a complete GKNV-style implementation cannot close Petersen. It proves that x-coordinate network simplex without full dummy-expanded ordering is not the missing sprint-23 drop.
- The current `petersen_10` gap vs `igraph_sugiyama` is crossing-driven. Any next attempt should first reproduce igraph's crossing count/rate on Petersen, then worry about x compactness.
- `regular_3_30` suggests dagua's existing force/hybrid layout is already strong on cubic graphs. A broad cubic override would regress real benchmark coverage unless every output is picker-scored.

## Knowledge

- Fresh HEAD deterministic `petersen_10` score is reproducible at `74.64037795612978` for seeds `0`, `42`, and `123` under the direct full scorer used here.
- Cached `igraph_sugiyama` remains `77.3636466496967` on `petersen_10`, matching the sprint context.
- Cached `graphviz_dot` scores only `72.07102883958513` on `petersen_10`; "dot-like x" is not enough to match igraph's result.
- The connected-cubic gate over the local variant manifest accepts exactly `petersen_10` and `regular_3_30`.
- The research scratch directory is `/tmp/sprint23_a_codex`; primary artifacts are `prototype.py`, `results.json`, and `fresh_summary.json`.
