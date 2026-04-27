# Sprint 23 Area C: Dense DAG Ordering Research

## TL;DR

- Dagua already has the expected Sugiyama pieces wired for the native layered-DAG route: optional dummy-node expansion, post-gradient barycenter reorder, median sweep, transpose heuristic, Brandes-Koepf horizontal refinement, final overlap projection, and dummy stripping.
- The gap is not "no median-transpose"; it is where and how the order is used. Current median/transpose defaults are only 4 median passes + 8 transpose passes after the continuous optimizer. Dot's model makes dummy-expanded ordering and coordinate assignment the core layered layout, not a late polish over an already-saturated geometry.
- A scratch original-edge median-transpose projection improved `dependency_500` from `55.284` to `57.089` (`+1.805`) under `dagua.metrics.full`, mostly by reducing edge-length CV `0.9054 -> 0.7903`. It leaves the graph close to the sprint context competitor target but does not fully close the loss alone.
- A scratch dummy-expanded post-projection did not help `dependency_500` (`55.284 -> 55.046`). My read is that dummy-expanded ordering is only useful if coordinate assignment consumes it before strip/refit; applying it after baseline coordinates mostly preserves the original-node order while worsening straightness.
- Recommendation: ship a scored polish candidate, not a forced replacement. Gate it narrowly to large connected DAGs with high CV and full composite validation. A forced replacement regresses `random_dag_200` in the empirical table.

## Audit: What Dagua Does Today Vs. What Dot Does

Current entry point:

- `dagua/layout/ops/pipelines/native_layered_dag.py` is a thin wrapper. It copies the config, defaults `insert_dummy_nodes=True`, `use_native_median_transpose=True`, and `brandes_koepf_refine=True`, then delegates to `dagua_native_legacy.build_dagua_pipeline()`.

The relevant pipeline sequence in `dagua_native_legacy.py` is:

1. `NativeEngineInit` assigns layer-y and first x ordering.
2. `Force2DInitIfFlat` handles degenerate cyclic/flat layerings.
3. `InsertDummyNodes` + `ActivateExpandedGraphState` run only if `_should_use_native_dummy_nodes()` passes. The gate requires a connected DAG with long layer-span edges, at least 20 nodes, non-singleton layers, and no `dense_dag` topology tag.
4. Gradient optimization runs through `build_gradient_core()`.
5. `BarycenterReorder(iterations=8)` reassigns x positions within layers by barycenter order.
6. If `_should_use_native_median_transpose()` passes, `MedianSweep(passes=4)` then `TransposeHeuristic(passes=8)` run. This is disabled for non-acyclic graphs and for tiny DAGs with `N <= 30`.
7. `BrandesKoepfHorizontalRefine` computes x positions from the resulting layer order when enabled.
8. `OverlapProjection`, `StripDummyNodes`, and `AspectRatioFit` finish the layout.

The ordering ops are in `dagua/layout/ops/ordering.py`:

- `MedianSweep` resolves the active graph. If `state.extras["expanded_graph"]` is active and state tensors match the expanded node count, it orders the expanded graph. Otherwise it falls back to the original edge index.
- It derives parents/children either from `state.adjacency` or directly from the active edge list, runs repeated down/up median sweeps, stores `state.ordering`, and applies the rank ordering back to x positions by permuting the existing x slots in each layer.
- `TransposeHeuristic` resolves the same active graph, builds parent/child lists, then calls `dagua.layout.init_placement._transpose_heuristic`. That helper swaps adjacent same-layer nodes when local parent/child crossing counts improve.
- Both ops preserve y positions. That protects DAG consistency and depth Spearman, but it also means they cannot repair an x-coordinate scale or gap model unless a later coordinate assignment consumes the new order.

Dot / Graphviz's layered pipeline, per Gansner-Koutsofios-North-Vo 1993, treats this sequence as the main layout:

1. Break cycles / rank nodes.
2. Expand long edges into dummy chains.
3. Run weighted median ordering sweeps and transpose passes on the dummy-expanded graph.
4. Assign coordinates with network-simplex / rank constraints and separation constraints.
5. Remove dummies and route edges.

So Dagua has the named components, but the empirical gap is consistent with a late-polish limitation: the continuous optimizer chooses a geometry first, then discrete ordering permutes existing x slots. Dot lets the dummy-expanded order drive the coordinate assignment from the start.

## Algorithm Sketch

This is the scratch prototype that produced the empirical table. The production version should keep the scoring gate and use project helpers for layers and scoring, but the core is about 120 LOC.

```python
def median_transpose_candidate(pos, edge_index, node_sizes, score_fn):
    """Return a scored same-layer ordering candidate for large DAGs.

    Parameters
    ----------
    pos : torch.Tensor
        Baseline positions with shape [N, 2].
    edge_index : torch.Tensor
        Directed edges with shape [2, E].
    node_sizes : torch.Tensor
        Node sizes with shape [N, 2].
    score_fn : Callable[[torch.Tensor], float]
        Composite scoring callback used by the polish picker.

    Returns
    -------
    torch.Tensor
        Candidate positions, or the baseline when the gate rejects it.
    """
    layers = longest_path_layering(edge_index, pos.shape[0])
    if not should_try(layers, edge_index, pos, node_sizes):
        return pos

    ordered_layers = group_by_layer_sorted_by_x(layers, pos[:, 0])
    parents, children = build_adjacent_layer_neighbors(edge_index, layers)

    for sweep in range(24):
        order = order_map(ordered_layers)
        if sweep % 2 == 0:
            layer_range = range(1, len(ordered_layers))
            reference = parents
        else:
            layer_range = range(len(ordered_layers) - 2, -1, -1)
            reference = children

        for layer in layer_range:
            nodes = ordered_layers[layer]
            scores = {}
            stable = {node: i for i, node in enumerate(nodes)}
            for node in nodes:
                neighbor_ranks = sorted(order[n] for n in reference[node] if n in order)
                if len(neighbor_ranks) == 0:
                    scores[node] = order[node]
                elif len(neighbor_ranks) % 2 == 1:
                    scores[node] = neighbor_ranks[len(neighbor_ranks) // 2]
                else:
                    mid = len(neighbor_ranks) // 2
                    scores[node] = 0.5 * (neighbor_ranks[mid - 1] + neighbor_ranks[mid])
            nodes.sort(key=lambda node: (scores[node], stable[node], node))

        changed = True
        while changed:
            changed = False
            for layer, nodes in enumerate(ordered_layers):
                for i in range(len(nodes) - 1):
                    u, v = nodes[i], nodes[i + 1]
                    before = local_crossings(u, v, layer, ordered_layers, parents, children)
                    nodes[i], nodes[i + 1] = v, u
                    after = local_crossings(v, u, layer, ordered_layers, parents, children)
                    if after < before:
                        changed = True
                    else:
                        nodes[i], nodes[i + 1] = u, v

    candidate = pos.clone()
    for layer, nodes in enumerate(ordered_layers):
        if len(nodes) < 2:
            continue
        x_slots = torch.sort(pos[nodes, 0]).values
        for rank, node in enumerate(nodes):
            candidate[node, 0] = x_slots[rank]

    base_score = score_fn(pos)
    candidate_score = score_fn(candidate)
    if candidate_score >= base_score + 0.5 and no_new_overlaps(candidate, node_sizes):
        return candidate
    return pos
```

I tested two variants:

- `post_original_mt`: run the above on the original graph, using long edges directly in the median/transpose neighborhoods.
- `post_dummy_mt`: expand long edges into dummy chains, run the same algorithm on the expanded graph, then project only original-node order back onto the original x slots.

The second variant is closer to dot structurally, but as a post-projection it is not equivalent to dot because it does not run coordinate assignment while dummies are still active.

Two details are worth preserving if this becomes production code.

First, the transpose phase should use a bounded local crossing check, not an exact all-edge crossing count. The existing `_transpose_heuristic()` in `init_placement.py` uses the same principle: compare only edges incident to the two adjacent nodes being swapped and only against adjacent layers. That is not a complete crossing objective, but it keeps the pass cheap enough for polish use. Exact crossing enumeration in every adjacent-swap trial would be the wrong complexity profile for `dependency_500` and would be especially risky if this candidate were accidentally admitted on larger DAGs.

Second, this candidate should preserve the set of x slots inside each layer. That is what keeps it compatible with the existing overlap and spacing assumptions. The projection is a permutation, not a fresh coordinate solver. If a future implementation wants to move from this polish candidate to dot-style coordinate assignment, it should become a separate coordinate op with explicit node-size separation constraints; it should not mutate this low-risk candidate into a half-coordinate solver.

## Empirical Validation

Setup:

- Scratch directory: `/tmp/sprint23_c_codex`.
- Script: `/tmp/sprint23_c_codex/sprint23_c_experiment.py`.
- Results: `/tmp/sprint23_c_codex/results.json`.
- Baseline positions: reused `/tmp/sprint22_d_cache/*.pt` or `/tmp/sprint22_d_focused/*__default_seed0.pt` where present; fresh `layout(..., LayoutConfig(device="cpu", seed=0))` only for the two tiny chain graphs without cache.
- Scoring: `dagua.metrics.full()` with `stress_sources=50`, `stress_targets=200`, `crossing_samples=50_000`, `neighborhood_samples=500`, then `dagua.metrics.composite()`. This matches the sprint-22 focused research calibration for `dependency_500` (`55.284`).

| Graph | N/E | Baseline | Original-edge MT | Delta | Dummy MT | Delta | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `dependency_500` | 500 / 1470 | 55.284 | 57.089 | +1.805 | 55.046 | -0.238 | Primary target; original-edge pass improves CV strongly. |
| `clustered_medium_5x20` | 100 / 193 | 69.784 | 69.708 | -0.076 | 69.784 | +0.000 | No useful lift. Gate should skip by size. |
| `outerplanar_dag_20` | 20 / 37 | 72.417 | 72.417 | +0.000 | 72.417 | +0.000 | Tiny graph; no same-layer work. |
| `multi_component_80` | 80 / 81 | 74.461 | 74.380 | -0.082 | 74.461 | +0.000 | Multi-component finisher needs component permutation, not ordering. |
| `random_dag_200` | 383 / 300 | 74.130 | 70.954 | -3.175 | 62.137 | -11.993 | Protected win: forced replacement is unsafe. |
| `org_chart_deep` | 79 / 78 | 92.441 | 92.441 | +0.000 | 92.441 | +0.000 | Ceiling graph unchanged. |
| `hub_fanout_label_skew` | 10 / 13 | 93.737 | 93.737 | +0.000 | 93.737 | +0.000 | Ceiling graph unchanged. |
| `linear_3layer_mlp` | 6 / 5 | 97.500 | 97.500 | +0.000 | 97.500 | +0.000 | Metric ceiling unchanged. |
| `deep_chain_20` | 22 / 21 | 97.500 | 97.500 | +0.000 | 97.500 | +0.000 | Metric ceiling unchanged. |

Important metric details for `dependency_500`:

| Variant | Score | CV | Crossing Rate | DAG | Depth Rho | Overlaps | Straightness |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 55.284 | 0.9054 | 0.1030 | 1.000 | 0.9939 | 0 | 71.81 |
| original-edge MT | 57.089 | 0.7903 | 0.1211 | 1.000 | 0.9939 | 0 | 80.41 |
| dummy MT | 55.046 | 0.9049 | 0.1040 | 1.000 | 0.9939 | 0 | 76.22 |

The original-edge variant improves the metric that sprint-22 area D identified as the residual bottleneck: edge-length CV. It pays a crossing-rate and straightness cost, but crossing is already near the composite floor and straightness is already bad enough that most of the CV gain survives.

The dummy-expanded variant accepted many swaps, but those swaps are mostly among dummy nodes. Once stripped, original-node x order barely changes, so CV does not move enough to compensate for the straightness penalty. This is the strongest empirical reason not to ship a "dummy-expanded post-projection" as the sprint-23 implementation.

Interpretation by graph:

- `dependency_500`: This is the only clear hit. The result is consistent with sprint-22 area D's finding that the residual loss is mostly CV, not DAG consistency or overlap. The candidate improves CV by 0.115 while preserving the two high-value hierarchy terms exactly.
- `clustered_medium_5x20`: The graph has high CV, but it is only 100 nodes and the original-edge pass found no accepted swaps. Its close loss is probably not the same failure mode. The dummy pass changes dummy order but produces no visible original-node improvement.
- `outerplanar_dag_20`: Every layer is effectively determined; the median/transpose algorithm has no useful freedom. This graph belongs with the small outer-face / local permutation finishers, not Area C.
- `multi_component_80`: The current close loss is component packing / component ordering. A within-layer order pass is the wrong lever, and the original-edge variant slightly worsens CV and crossings.
- `random_dag_200`: This is the regression that decides the deployment strategy. The original-edge candidate makes hundreds of swaps and lowers the score by more than three points. The dummy version is worse because the projected x slots create overlaps. The graph does not have `dependency_500`'s high-CV/high-density signature, so a conservative gate rejects it.
- `org_chart_deep`, `hub_fanout_label_skew`, `linear_3layer_mlp`, and `deep_chain_20`: These are already at or near ceiling. The algorithm finds no work and is neutral, which is good, but neutrality on small cases is not enough to justify forced use.

One measurement caveat: some baseline positions came from prior `/tmp` sprint caches. For cached payloads that included edge tensors and node sizes, I scored against the cached tensors rather than regenerated graph objects, because regenerated benchmark node ordering can differ and corrupt DAG/depth terms. `dependency_500` recalibrated to the sprint-22 focused score (`55.284`), so the primary target delta is comparable with the prior research note. I did not rerun full benchmark competitors; this report measures Dagua baseline-vs-candidate deltas only.

## Polish-Candidate Vs Forced-Replacement Decision

Ship as a polish candidate.

Do not force it into the main ordering pass. The `random_dag_200` row is decisive: the original-edge candidate loses `-3.175`, and the dummy candidate loses `-11.993` due to two overlaps and worse crossing/straightness. Even if the exact random-DAG score varies with benchmark cache/source, the direction is consistent with the prior sprint-22 note: protected wins are sensitive to forced x-order perturbations.

Recommended candidate gate:

- Graph must be directed acyclic or have a prepared acyclic edge set.
- `N >= 400`.
- `E / N >= 2.0`.
- One connected component.
- Baseline `edge_length_cv >= 0.8`.
- Baseline `dag_consistency >= 0.99` and `depth_spearman_rho >= 0.95`; the candidate is meant to preserve a good hierarchy, not repair a bad one.
- Skip when baseline has low CV (`< 0.5`), any overlap risk that cannot be re-projected cheaply, tree/chain tags, or multi-component layouts.
- Accept only through the existing picker/scoring mechanism, requiring at least `+0.5` composite and no increase in overlap count.

This gate admits `dependency_500` and rejects every protected graph in the requested table. It also rejects `clustered_medium_5x20`, so this bet should not be expected to close that graph. `clustered_medium_5x20` likely needs cluster bridge routing/spacing or component/cluster-aware coordinate assignment rather than generic median-transpose.

The picker should compare three positions, not just "baseline vs candidate":

1. The unmodified baseline coming out of current native polish.
2. Existing sprint-22 candidates, especially `gap_validated_layer_swaps`, because that candidate already improves the same CV failure mode.
3. This median-transpose candidate.

If both gap swaps and median-transpose are present, I would try median-transpose after gap swaps in the candidate list but still score each against the original baseline and against the current best. The two methods may overlap: both are same-layer discrete x-order changes. The median-transpose pass is broader and more systematic; the gap swap pass is more targeted to longest-edge endpoints. A composed candidate might add a little more, but it should be treated as a separate scored candidate only after the individual candidates are validated. Do not blindly run both in sequence as the final layout.

Forced replacement is unattractive for another reason: Dagua's current median/transpose ops are already part of the main pipeline. Replacing them with a stronger version would mostly increase the amount of late x-slot permutation for every eligible DAG. The empirical result says that extra permutation is valuable only when the graph has `dependency_500`'s dense, high-CV signature. It is not a universal crossing reducer.

## Implementation

Best production slot:

- Add a private helper in `dagua/layout/ops/pipelines/dagua_native.py` near the existing polish helpers such as `_gap_validated_layer_swaps`.
- Add the candidate to the `_best_of_polish` candidate list after the existing edge-equalize and gap-swap candidates, because it is another scored x-order projection over final positions.
- Use the same `safe_score()` callback/margin pattern as the current polish picker.

Estimated LOC:

- Gate helper: 35-50 LOC.
- Median-transpose projection helper: 120-170 LOC if implemented directly; less if it reuses `ordering.py` helpers, but those helpers are private and state/op-oriented.
- Candidate registration in `_best_of_polish`: 10-20 LOC.
- Focused tests: 80-120 LOC. Tests should assert the candidate is considered/accepted on a synthetic high-CV dense DAG and skipped on chain/tree/small/multi-component cases; exact `dependency_500` composite deltas are probably too slow/flaky for unit tests.

I would not implement this as a replacement for `MedianSweep` / `TransposeHeuristic` in `ordering.py` yet. The op stack is already present there, and the dummy-expanded post-projection result says the next structural dot-like improvement must couple dummy-expanded ordering to coordinate assignment, not just add more sweeps. If sprint-23 wants that deeper path, the right shape is a new ordering-aware coordinate op that runs before `BrandesKoepfHorizontalRefine` and preserves dummy corridor constraints until x coordinates are fixed.

Suggested test plan:

- Unit-test the gate on small synthetic graphs: chain, tree, disconnected graph, sparse random DAG, and dense high-CV DAG. The first four should skip; the dense graph should attempt.
- Unit-test the projection invariant: y coordinates unchanged, per-layer x multiset unchanged, output shape unchanged, and no node crosses layers.
- Unit-test deterministic ordering: running the helper twice on the same inputs returns identical coordinates.
- Add a slow-marked regression smoke for `dependency_500` if runtime is acceptable in the research/benchmark tier. It should assert that the candidate does not reduce composite and usually improves by at least `+0.5` under fixed metric sample sizes.
- Add a protected smoke for `random_dag_200` asserting the candidate is skipped by the gate. This is more valuable than asserting a forced-candidate regression, because the production behavior should be "do not try it."

Dead-code / cleanup note:

No production code was modified for this research task. If sprint-23 ships this candidate, the scratch-only dummy-expanded post-projection should not be ported as-is. It is empirically negative on the primary target and dangerous on `random_dag_200`. The useful part of the dummy result is diagnostic: it points toward a future coordinate-assignment bet, not a standalone polish pass.

References:

- Eades, Sugiyama, and Tamassia, "Algorithms for Drawing Graphs", 1981: original layered drawing framing.
- Gansner, Koutsofios, North, and Vo, "A Technique for Drawing Directed Graphs", IEEE TSE 19(3), 1993: dot-style ranking, median/transpose crossing reduction, and coordinate assignment.
- Junger and Mutzel, "2-Layer Straightline Crossing Minimization", Algorithmica, 1997: analysis of two-layer crossing minimization and local transpose behavior.
