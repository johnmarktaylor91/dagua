# Sprint 22 Area E: Close-Loss Tail Finishers - codex

## TL;DR

- Biggest call: implement a margin-gated motif coordinate synthesizer, not another gradient-weight tweak. Two of three targets flipped with template-level coordinate assignment.
- `clustered_medium_5x20`: staggered narrow cluster columns measured `+2.27` over cached Dagua and `+1.76` over cached `graphviz_dot` in the current HEAD scorer.
- `outerplanar_dag_20`: source-fan vertical-spine profile measured `+2.02` over cached Dagua and `+0.22` over cached `igraph_sugiyama`.
- `recurrent_feedback_cell`: current HEAD/cache already has Dagua ahead of the prompt's named competitor, but a feedback micro-polish still measured `+0.34` over cached Dagua.
- Shared pattern: accept a geometry template only after real metric scoring. The useful candidates intentionally change aspect/bbox; bbox-preserving variants were no-ops or regressions.
- Implementation order: clustered staggered columns first, outerplanar fan second, recurrent feedback third, then unify under one template-picker gate.

## Measurement Setup

Read first: `.project-context/research/sprint_22_algo_bets/CONTEXT.md`.

Empirical script: `/tmp/sprint22_e_close_loss_finishers.py`.

The script hand-implements candidate transforms in `/tmp`, loads cached tensors from `eval_output/variant_bench_full/positions`, builds only the three target graphs, and uses `dagua.eval.benchmark._metric_payload(..., "full")` for final benchmark-equivalent scoring. Search uses a cheaper deterministic proxy, but all deltas quoted below are from the full scorer.

Important caveat: the current checkout is exactly `c821eb6`, but the local cached scores do not numerically match the prompt table. For example, `recurrent_feedback_cell` scores `77.29` for cached Dagua and `76.08` for cached `igraph_sugiyama`, so it is already flipped in this cache/scorer combination. I report the actual measured values from this workspace, not the stale prompt values.

Assumption: because this is a research-only task, I did not edit `dagua/` or run project quality gates. The executable evidence is the `/tmp` script plus the full-score output quoted here.

## 1. `clustered_medium_5x20`

### Dominant Losing Metric

Current full-score comparison:

| layout | score | dag | edge CV | depth rho | straight deg | crossing rate | edge-node rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dagua cached | 73.92 | 1.000 | 1.540 | 1.000 | 9.21 | 0.0345 | 0.1255 |
| `graphviz_dot` cached | 74.43 | 0.984 | 1.363 | 0.942 | 10.51 | 0.0161 | 0.1134 |
| candidate | 76.19 | 0.995 | 1.380 | 0.941 | 3.57 | 0.0127 | 0.1007 |

The competitor wins by making clusters into narrow diagonal columns. Dagua keeps perfect DAG/depth alignment and good straightness, but its cluster columns are vertically interleaved in a way that creates worse edge-length variance, more sampled crossings, and more edge-node interactions. `graphviz_dot` is not preserving Dagua's tall narrow bbox; it widens the structure into a diagonal set of cluster spines.

### Competitor Strategy

Cached shape inspection:

- Dagua bbox: x `[-115.6, 115.6]`, y `[0.6, 3699.3]`.
- `graphviz_dot` bbox: x `[60.0, 806.0]`, y `[-4446.0, -34.0]`.
- Dot cluster widths are only `23-26` units, but cluster centers step horizontally by roughly `160-230` units and vertically overlap as staggered bands.

The structural win is not cluster separation as a metric, because benchmark scoring here does not pass `cluster_ids` into `full()`. The win is the secondary effect of cluster-aware compression: shorter cross-cluster bridges, fewer crossings, and fewer edge-node intersections.

### Fix Sketch

Detect graphs whose top-level clusters are mostly internally path-like and whose inter-cluster edges flow forward between adjacent clusters. For each cluster, sort members by local topological depth or original member order, then synthesize narrow vertical columns with a vertical stagger between adjacent clusters.

Winning measured parameters:

- `x_sep = 120`
- `y_pitch = 95`
- `y_stagger = 1140`
- `wiggle = 0`
- Crucial: do not rescale back into Dagua's original bbox.

Complete working pseudocode:

```python
def staggered_cluster_columns(graph, baseline):
    clusters = sorted(graph.clusters.items(), key=lambda kv: kv[0])
    if len(clusters) < 3:
        return baseline

    if not all(is_path_like(graph, members) for _, members in clusters):
        return baseline
    if not mostly_forward_adjacent_cluster_edges(graph, clusters):
        return baseline

    best = baseline
    best_score = full_composite(graph, baseline)
    for x_sep in [100, 120, 160, 220]:
        for y_pitch in [72, 85, 95, 110]:
            for y_stagger_mult in [9, 10, 12, 14]:
                y_stagger = y_pitch * y_stagger_mult
                pos = baseline.copy()
                for cluster_idx, (_, members) in enumerate(clusters):
                    ordered = sort_by_depth_then_member_order(graph, members)
                    for local_rank, node_idx in enumerate(ordered):
                        pos[node_idx, 0] = cluster_idx * x_sep
                        pos[node_idx, 1] = cluster_idx * y_stagger + local_rank * y_pitch

                if has_node_overlaps(graph, pos):
                    continue
                score = full_or_proxy_composite(graph, pos)
                if score > best_score + 0.25:
                    best = pos
                    best_score = score
    return best
```

### Empirical Validation

`/tmp/sprint22_e_close_loss_finishers.py` selected:

`staggered_columns/x120/p95/s1140/w0`

Measured full delta:

- Dagua `73.92` -> candidate `76.19`: `+2.27`.
- Candidate vs `graphviz_dot` `74.43`: `+1.76`.

This is the strongest Area E result. The candidate loses some `dag_consistency` and `depth_spearman`, but the straightness/crossing/edge-node gains dominate.

## 2. `outerplanar_dag_20`

### Dominant Losing Metric

Current full-score comparison:

| layout | score | edge CV | straight deg | angular deg | edge-node rate |
|---|---:|---:|---:|---:|---:|
| Dagua cached | 76.22 | 0.814 | 49.64 | 49.94 | 0.000 |
| `igraph_sugiyama` cached | 78.02 | 0.933 | 24.51 | 45.61 | 0.009 |
| candidate | 78.24 | 0.949 | 21.30 | 40.26 | 0.036 |

Dagua's outerplanar layout is a broad diagonal path: node `0` at top center, node `1` far left, then the path sweeps steadily right. That preserves edge-node cleanliness and decent edge CV, but path edges are badly non-vertical, so `edge_straightness_mean_deg` is the dominant loss. `igraph_sugiyama` sacrifices edge CV to impose a mostly vertical chain.

### Competitor Strategy

Cached `igraph_sugiyama` puts nodes `1..5` on one vertical column and then gradually drifts later path nodes right. Source node `0` sits to the side/top, so fan edges are diagonal but the backbone path is mostly vertical.

The useful template is even simpler: keep all path nodes in one vertical spine and place the fan source to the right.

### Fix Sketch

Detect an outerplanar source-fan path:

- path edges `(i, i + 1)` cover most nodes,
- one source has high out-degree to later path nodes,
- graph is acyclic and sparse,
- no clusters.

Then synthesize a vertical path spine with source offset horizontally. Accept only if the full/proxy score beats the baseline margin and overlap count remains zero.

Winning measured parameters:

- `path_x = 0`
- `tail_slope = 0`
- `bend_start = 1`
- `source_x = 360`
- `y_pitch = 50`

Complete working pseudocode:

```python
def outerplanar_source_fan_profile(graph, baseline):
    path = recover_hamiltonian_path_edges(graph)
    source = find_source_fan_node(graph, path)
    if path is None or source is None:
        return baseline

    best = baseline
    best_score = full_composite(graph, baseline)
    for source_x in [240, 300, 360, 420]:
        for path_x in [-80, 0, 80]:
            for tail_slope in [0, 15, 30]:
                for y_pitch in [45, 50, 60]:
                    pos = baseline.copy()
                    for rank, node_idx in enumerate(path):
                        pos[node_idx, 1] = rank * y_pitch
                        if node_idx == source:
                            pos[node_idx, 0] = source_x
                        else:
                            pos[node_idx, 0] = path_x + max(0, rank - 1) * tail_slope

                    if has_node_overlaps(graph, pos):
                        continue
                    score = full_or_proxy_composite(graph, pos)
                    if score > best_score + 0.25:
                        best = pos
                        best_score = score
    return best
```

### Empirical Validation

`/tmp/sprint22_e_close_loss_finishers.py` selected:

`fan_profile/path0/slope0/bend1/src360/yp50`

Measured full delta:

- Dagua `76.22` -> candidate `78.24`: `+2.02`.
- Candidate vs `igraph_sugiyama` `78.02`: `+0.22`.

The candidate is not universally prettier: it increases edge-node crossings from `0.000` to `0.036` and worsens edge CV to `0.949`. The composite still improves because straightness falls from `49.64 deg` to `21.30 deg`.

## 3. `recurrent_feedback_cell`

### Dominant Losing Metric

Under the current cache/scorer, this graph is not a loss:

| layout | score | dag | edge CV | straight deg | angular deg | edge-node rate |
|---|---:|---:|---:|---:|---:|---:|
| Dagua cached | 77.29 | 0.667 | 0.497 | 24.12 | 63.52 | 0.000 |
| `igraph_sugiyama` cached | 76.08 | 0.833 | 0.810 | 8.86 | 51.64 | 0.167 |
| candidate | 77.63 | 0.833 | 0.697 | 20.86 | 45.00 | 0.037 |

The prompt called this `-0.39`; I cannot reproduce that with the local cached tensors and current full scorer. The live issue is whether Dagua can keep its compactness while getting `5/6` DAG-consistent edges like Sugiyama.

### Competitor Strategy

`igraph_sugiyama` puts `state_prev` below `output`, so the feedback edge `output -> state_prev` becomes forward and only `state_prev -> state_update` remains backward. That improves `dag_consistency` and straightness, but it introduces an edge-node crossing and much worse edge-length CV.

Dagua keeps `state_prev` and `state_proj` side-by-side, which gives compact and uniform edges but leaves both recurrent edges backward.

### Fix Sketch

Detect the five-node recurrent motif: one self-loop, one feedback edge returning from output-like sink to previous-state node, and one previous-state edge into the update node. Move the previous-state node just below the output layer, keep it horizontally central, and place projection on one side.

Winning measured parameters:

- `state_prev x = 0`
- `state_prev y = 4 * pitch`
- `state_proj x = -55`
- `output x = 0`
- `pitch = 55`

Complete working pseudocode:

```python
def recurrent_feedback_micro_polish(graph, baseline):
    motif = detect_recurrent_cell(graph)
    if motif is None:
        return baseline

    input_node = motif.input
    update_node = motif.update
    prev_node = motif.previous_state
    proj_node = motif.projection
    output_node = motif.output

    best = baseline
    best_score = full_composite(graph, baseline)
    for pitch in [50, 55, 65, 75]:
        for proj_x in [-65, -55, 55, 65]:
            for prev_layer in [3.6, 4.0, 4.4]:
                pos = baseline.copy()
                pos[input_node] = [0, 0]
                pos[update_node] = [0, pitch]
                pos[proj_node] = [proj_x, 2 * pitch]
                pos[output_node] = [0, 3 * pitch]
                pos[prev_node] = [0, prev_layer * pitch]

                if has_node_overlaps(graph, pos):
                    continue
                score = full_or_proxy_composite(graph, pos)
                if score > best_score + 0.15:
                    best = pos
                    best_score = score
    return best
```

### Empirical Validation

`/tmp/sprint22_e_close_loss_finishers.py` selected:

`feedback/prevx0/prevl4/proj-55/out0/p55`

Measured full delta:

- Dagua `77.29` -> candidate `77.63`: `+0.34`.
- Candidate vs `igraph_sugiyama` `76.08`: `+1.55`.

This is a small but real lift in the current scorer. It is mostly a `dag_consistency` gain from `0.667` to `0.833`, partly offset by worse edge CV and edge-node crossing.

## Cluster Recommendation

Do not build three unrelated polish ops. Build one margin-gated `motif_coordinate_synth` phase with three detectors/templates:

1. `clustered_staggered_columns` for chain-like clustered DAGs.
2. `outerplanar_source_fan_profile` for source-fan path DAGs.
3. `recurrent_feedback_micro_polish` for small recurrent cells.

The common implementation pattern is:

```python
def motif_coordinate_synth(graph, baseline):
    best = baseline
    best_score = full_or_fast_candidate_score(graph, baseline)
    for template in [cluster_columns, source_fan, recurrent_feedback]:
        if not template.matches(graph):
            continue
        for candidate in template.generate_candidates(graph, baseline):
            if has_node_overlaps(graph, candidate):
                continue
            score = full_or_fast_candidate_score(graph, candidate)
            if score > best_score + template.margin:
                best = candidate
                best_score = score
    return best
```

This should live after existing polish primitives and before final routing metrics are frozen. It must be picker-safe: no unconditional template replacement.

## Risk / Regression Analysis

Primary risk: these templates can easily destroy wins if applied by broad tags like `"clustered"` or `"dag"`. They must require structural motif matches and a score margin.

Specific protected wins to verify:

- Clustered templates: `compound_dag_5x30`, `compound_10x20`, `resnet_stack_4x16`, `dependency_graph_100`, `dependency_500`, `nested_cluster_label_stack`, `interleaved_cluster_crosstalk`, `small_label_storm`, `moe_router_sparse`, `transformer_full_4h_2l`.
- Outerplanar/fan templates: `deep_chain_20`, `weighted_chain_20`, `linear_3layer_mlp`, `long_skip_only_24`, `hub_skip_superfan`, `hub_fanout_label_skew`, `wide_1_100_1`, `parallel_multiedge_bundle`.
- Recurrent feedback template: `center_port_backedge_hub`, `braided_feedback_tails`, `recurrent_feedback_cell`, `parallel_multiedge_bundle`, `shape_and_routing_matrix`.

Metric risks:

- Staggered clusters may trade away `depth_spearman_rho`; require net score margin and no more than a small `dag_consistency` drop.
- Outerplanar fan profile increases edge-node crossing rate; require edge-node rate below a cap such as `0.05`.
- Recurrent feedback polish worsens edge CV; require either a `dag_consistency` increase or direct full-score improvement.

## Implementation Order

1. Add detector and candidate generator for `clustered_staggered_columns`. This has the largest measured lift and beats the named competitor.
2. Add detector and candidate generator for `outerplanar_source_fan_profile`. This flips the close loss but needs a tight structural detector.
3. Add `recurrent_feedback_micro_polish` only after verifying the stale prompt/cache mismatch. It is useful but lower leverage.
4. Put all three under a shared template-picker wrapper with margin gates and overlap/edge-node caps.
5. Run the required targeted and final benchmark gates, plus the protected graph list above.

## Knowledge

- The close-loss tail is not saturated at the metric ceiling; at least two targets need discrete coordinate synthesis that changes bbox/aspect.
- Bbox preservation is actively harmful for `clustered_medium_5x20`; the competitor's win depends on widening the layout into staggered columns.
- `outerplanar_dag_20` is a straightness problem more than a crossing problem. A vertical path spine wins even with worse edge CV.
- `recurrent_feedback_cell` appears stale in the sprint table for this local cache/scorer, but the back-edge idea still gives a small current-score lift.
