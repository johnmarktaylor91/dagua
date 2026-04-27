# Sprint 32 Post-31a Report -- Codex

## TL;DR

**Ship, but only as a narrow structural new op:** `unet_small` remains the highest-leverage measured loss after sprint-31a, and no existing decomposed op moves the real nodes needed for this class; a class-gated nested-skip U-spine pass gives `unet_small` +9.66 fixed-seed, +8.14 jitter mean with min +5.22, and is no-op on the protected wins.

Assumption: the task says HEAD should be `16b3866`, but this checkout is `bca630a`. I did not stop because `.project-context/research/sprint_31_hierarchical_skip/SPRINT_31A_RESULT.md` is present and the live transformer result reproduces the sprint-31a improvement (`extreme_mixed_width_transformer` now scores 86.41, no longer a loss).

## Target Picked + Why

`/tmp/sprint32_h2h.csv` did not exist when I started. I computed a quick H2H estimate by regenerating Dagua layouts live with `LayoutConfig(seed=42, device="cpu")`, then rescoring saved competitor position tensors from `eval_output/benchmark_full/positions/` through the current `full()` + `composite()` metric with fixed `node_sizes=[[40, 20]] * N`. The available competitor tensor coverage in `benchmark_full` is complete for the first 23 small benchmark graphs, not all 93, so this is a target-selection estimate rather than a full benchmark rerun.

Worst available deltas:

| graph | Dagua | best saved competitor | delta | best engine |
|---|---:|---:|---:|---|
| `unet_small` | 70.785 | 84.704 | -13.919 | `cytoscape_fcose` seed48 |
| `mixed_width_labels` | 77.584 | 84.601 | -7.017 | `igraph_rt` |
| `cluster_member_style_stress` | 75.871 | 82.430 | -6.560 | `classic_rt` |
| `disconnected_encoder_residual` | 81.186 | 85.870 | -4.683 | `igraph_rt` |
| `hierarchical_residual_stage` | 82.285 | 84.706 | -2.421 | `dagre` |

I picked `unet_small` because it is the largest observed remaining loss and it is the cleanest surviving instance of the width-1 nested encoder-decoder skip class. The sprint-31a gate refinement fixed the width-3 transformer subtype but left pure width-1 real nodes on a single vertical spine. `unet_small` also has a clear out-of-suite structural synthetic: a larger U-Net-like chain with mirrored skip pairs.

Two measurement caveats matter:

1. The `cytoscape_fcose` comparison is harsher than the known sprint-31 prompt comparison. The prompt's open `unet_small` gap was `-6.25` versus ELK/Dagre-like layered engines. The saved fCoSE seed is a force-style layout that optimizes edge-length regularity much better than the layered competitors, so it exposes extra headroom not specific to the hierarchical-skip bug.
2. I did not use stored Dagua positions for the final target numbers. Dagua was regenerated live at this checkout because sprint-31a changed native gate behavior. Competitor tensors were reused only as fixed reference layouts and rescored through the current metric.

That means the proposed fix should be judged on whether it repairs the structural class regression, not on whether it completely beats every saved competitor seed. It does repair the class regression: it eliminates the collinear skip-over-spine crossings while keeping the same depth ordering.

## Metric Breakdown

For `unet_small`, Dagua is not losing DAG direction, depth order, overlap, or edge-node crossings. It loses because the straight-line center-to-center metric counts the three long skip edges as collinear overlaps with the main chain. `cytoscape_fcose` wins overall by nearly perfect edge-length CV, but it gives up DAG consistency and straightness. ELK/Dagre win the known sprint-31 class by making the encoder-decoder spine U-shaped: no crossings, still depth-monotone.

| layout | composite | dag | edge CV | depth rho | overlaps | straight deg | crossing | angular deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dagua HEAD | 70.785 | 1.000 | 0.746 | 1.000 | 0 | 0.0 | 0.250 | 25.7 |
| `cytoscape_fcose` seed48 | 84.704 | 0.909 | 0.014 | 0.983 | 0 | 59.4 | 0.000 | 76.4 |
| `elk_layered` | 77.036 | 1.000 | 0.906 | 1.000 | 0 | 6.2 | 0.000 | 32.2 |
| `dagre` | 77.036 | 1.000 | 0.874 | 1.000 | 0 | 13.2 | 0.000 | 39.6 |
| proposed U-spine candidate | 80.444 | 1.000 | 0.738 | 1.000 | 0 | 6.7 | 0.000 | 33.5 |

Interpretation:

- The proposed pass does not try to chase `fCoSE`'s edge-length CV optimum. It fixes the class-specific failure: skip edges become real lateral chords instead of lying on top of the main chain.
- It beats the known layered competitors for this graph (`80.44` vs `77.04`) while preserving Dagua's DAG/depth advantages.
- It reduces the largest measured gap from `-13.92` to `-4.26` versus `cytoscape_fcose`. That is still not a full H2H flip, but it removes the moderate-or-bigger loss against ELK/Dagre and addresses the actual hierarchical-skip defect.

## Existing-Op Recomposition vs New Code

I checked the existing decomposed ops before proposing anything new.

`InsertDummyNodes` + `ActivateExpandedGraphState` + `BrandesKoepfHorizontalRefine` now fire after sprint-31a, but they do not move the real width-1 endpoints enough on `unet_small`. The expanded dummy graph can give long edges their own dummy-node corridors, but `StripDummyNodes` truncates back to the original nodes. The benchmark metric and public `layout()` return value still score straight center-to-center original edges, so dummy corridors alone are not visible to the measured quality surface.

`ReconstructEdgeRoutes` exists in `dagua/layout/ops/edge_route.py` and is the closest missing existing op. It can rebuild per-edge polylines from `expanded_graph.edge_paths`. However, it writes `state.edge_routes`, not `state.pos`, and the current benchmark composite uses only node positions. Adding route reconstruction would be visually principled for renderers that consume routes, but it would not fix the reported 93-graph benchmark loss unless the metric and public layout pipeline were also changed to evaluate routed edges. That is a broader contract change and not the right sprint-32 fix.

This route point is important because it is the easiest way to fool ourselves into thinking an existing op solves the loss. In a full Sugiyama drawing model, long edges are not straight lines between original node centers; they are polylines through dummy nodes. Under that model, dummy insertion plus route reconstruction is exactly the right abstraction. Dagua's current benchmark surface is different: `full(pos, edge_index, node_sizes=...)` receives only `[N, 2]` original-node positions and reconstitutes every edge as a straight segment. For the sprint-32 benchmark, an edge-route-only patch would be a visual/rendering improvement with zero composite delta. I therefore do not recommend calling route reconstruction the sprint-32 quality fix unless the implementation also intentionally changes the metric/render contract, which would be outside the scoped "one remaining quality loss" task.

The pure `sugiyama` pipeline already has dummy expansion, barycenter ordering, coordinate assignment, and optional route reconstruction. It improves `unet_small` crossing rate to zero but scores only `63.83` under the current composite because it pays too much in straightness/CV. So this is not a dispatch bug where the default should simply choose `algorithm="sugiyama"`.

`MedianSweep`, `TransposeHeuristic`, and `BarycenterReorder` are no-ops in the important degree of freedom: each real layer has one node. There is no within-layer ordering to improve.

`SpreadFanoutChildren` targets high-fanout hubs with several children on the same layer. `unet_small` has degree-2/3 nested skip pairs, not a fanout layer. `ClusterGridArrange` is irrelevant because `unet_small` has no clusters. Force/stress alternatives (`force_directed`, `fr`, `kk`, `stress_majorization`) can remove crossings, but they lose the semantic DAG/depth layout and scored between `28.68` and `49.81` in my comparison sweep.

Conclusion: the gap is case (d), with an important constraint. There is no existing op that applies a topology-derived **cross-layer x drift to real nodes** for a width-1 DAG with nested mirrored long edges. Existing Sugiyama/dummy/route ops operate on dummy nodes and edge routes; the benchmark loss is in real-node coordinates.

The proposed op is not a replacement for Sugiyama. It is a small coordinate correction for a narrow case where standard Sugiyama machinery has the right conceptual diagnosis (long edges need side corridors) but the current Dagua scoring interface observes only real-node coordinates. The op should be implemented as an ordinary registered op, not an anonymous helper hidden in the pipeline.

## Class Predicate

The proposed fix applies to this structural class:

> A connected, directed, width-1 layered DAG with a contiguous one-node spine and at least two long edges whose source layers increase while target layers decrease, with source and target halves separated.

This is the encoder-decoder/U-Net skip class. It is not a graph-name or N/E predicate. It rejects:

- pure chains: no long edges;
- one-off residual chains like `mixed_width_labels`: fewer than two nested skips;
- residual stage stacks like `hierarchical_residual_stage`: skips are sequential, not mirrored/nested;
- fanout/tree/random/scale-free graphs: not width-1 nested skip spines.

The "source layers increase while target layers decrease" condition is the key anti-overfit guard. It captures the common U-Net topology where encoder feature maps skip to matching decoder depths. It does not fire merely because a graph has long edges or because a vertical-spine layout would get a crossing-rate boost from any lateral perturbation. In particular, it rejects the broader chain-with-skip family that failed sprint-31 jitter validation. This is why the fix is intentionally narrower than the full remaining width-1 residual-chain bucket.

## Pseudocode of Proposed Change

Add a registered post-strip coordinate op, for example `NestedSkipSpineBend`, with no score callback and no composite picker. It should run after `StripDummyNodes()` and before `AspectRatioFit()` in the native layered-DAG pipeline so it acts on original real nodes and preserves y/depth.

```python
def _nested_skip_spine_layers(edge_index: torch.Tensor, num_nodes: int) -> tuple[bool, torch.Tensor]:
    """Classify width-1 encoder-decoder DAGs with nested mirrored skips.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    tuple[bool, torch.Tensor]
        Whether the structural class applies and the longest-path layers
        with shape ``[N]``.
    """
    layers = longest_path_layering(edge_index.cpu(), num_nodes, device="cpu")
    counts = torch.bincount(layers - layers.min())
    if int(counts.max().item()) != 1:
        return False, layers

    spans = layers[edge_index[1].cpu()] - layers[edge_index[0].cpu()]
    long_edges = sorted(
        (int(layers[s]), int(layers[t]))
        for edge_id, (s, t) in enumerate(edge_index.t().cpu().tolist())
        if int(spans[edge_id].item()) >= 2
    )
    if len(long_edges) < 2:
        return False, layers

    source_layers = [source for source, _ in long_edges]
    target_layers = [target for _, target in long_edges]
    is_nested = source_layers == sorted(source_layers) and target_layers == sorted(
        target_layers, reverse=True
    )
    halves_are_separate = max(source_layers) < min(target_layers)
    return is_nested and halves_are_separate, layers


def apply_nested_skip_spine_bend(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Move real nodes onto a U-shaped spine for nested skip DAGs.

    Parameters
    ----------
    pos : torch.Tensor
        Original-node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Updated positions. Y coordinates are preserved exactly.
    """
    enabled, layers = _nested_skip_spine_layers(edge_index=edge_index, num_nodes=pos.shape[0])
    if not enabled:
        return pos.clone()

    long_spans = layers[edge_index[1].cpu()] - layers[edge_index[0].cpu()]
    long_pairs = [
        (int(layers[s]), int(layers[t]))
        for edge_id, (s, t) in enumerate(edge_index.t().cpu().tolist())
        if int(long_spans[edge_id].item()) >= 2
    ]
    outer_source = min(source for source, _ in long_pairs)
    outer_target = max(target for _, target in long_pairs)

    y_values = torch.sort(torch.unique(pos[:, 1].detach().cpu()))[0]
    y_gaps = torch.diff(y_values).abs()
    pitch = float(torch.median(y_gaps).item()) if y_gaps.numel() else 150.0
    lane = max(20.0, min(80.0, 0.20 * pitch))

    out = pos.detach().clone()
    for node in range(out.shape[0]):
        layer = int(layers[node].item())
        depth = max(0, min(layer - outer_source, outer_target - layer))
        out[node, 0] = float(depth) * lane
    out[:, 0] -= out[:, 0].mean()
    return out
```

The lane scale is deliberately based on observed vertical pitch, not composite feedback. The point is to produce a readable U-spine with bounded horizontal movement, not to search the metric.

The candidate's actual `unet_small` x coordinates were:

| node | Dagua x | candidate x |
|---|---:|---:|
| `input` | 0.00 | -40.87 |
| `enc1` | 0.00 | -40.87 |
| `enc2` | 0.00 | 0.00 |
| `enc3` | 0.00 | 40.87 |
| `bottleneck` | 0.00 | 81.75 |
| `dec3` | 0.00 | 40.87 |
| `dec2` | 0.00 | 0.00 |
| `dec1` | 0.00 | -40.87 |
| `output` | 0.00 | -40.87 |

This matches the qualitative ELK shape (`12, 12, 32, 52, 72, 52, 32, 12, 12`) while keeping Dagua's native y spacing. The transform is therefore a semantic U-spine, not a random de-collinearity jitter.

## Empirical Validation

Fixed-seed validation:

| graph | gate | base | candidate | delta | crossing | edge CV | angular | straight |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `unet_small` | true | 70.785 | 80.444 | +9.659 | 0.250 -> 0.000 | 0.746 -> 0.738 | 25.7 -> 33.5 | 0.0 -> 6.7 |
| `synthetic_unet_4` | true | 67.806 | 77.525 | +9.719 | 0.273 -> 0.000 | 0.860 -> 0.848 | 20.0 -> 30.9 | 0.0 -> 8.4 |
| `mixed_width_labels` | false | 77.584 | 77.584 | +0.000 | unchanged | unchanged | unchanged | unchanged |
| `hierarchical_residual_stage` | false | 82.285 | 82.285 | +0.000 | unchanged | unchanged | unchanged | unchanged |
| `cluster_member_style_stress` | false | 75.871 | 75.871 | +0.000 | unchanged | unchanged | unchanged | unchanged |

Sigma=0.5 jitter, 8 trials, same noise added to base and candidate positions:

| graph | mean delta | min | max | trial deltas |
|---|---:|---:|---:|---|
| `unet_small` | +8.141 | +5.220 | +9.681 | +9.663, +5.225, +8.001, +9.681, +9.662, +7.997, +9.680, +5.220 |
| `synthetic_unet_4` | +7.945 | +2.751 | +9.731 | +2.751, +7.299, +8.814, +9.722, +9.726, +9.731, +9.726, +5.788 |

Protected wins:

| graph | gate | base | candidate | delta | reason rejected |
|---|---|---:|---:|---:|---|
| `deep_chain_20` | false | 97.500 | 97.500 | +0.000 | no long nested skips |
| `random_dag_200` | false | 74.424 | 74.424 | +0.000 | not width-1 layered |
| `ba_500` | false | 63.138 | 63.138 | +0.000 | not width-1 layered |
| `org_chart_deep` | false | 92.441 | 92.441 | +0.000 | not width-1 layered |
| `hub_fanout_label_skew` | false | 93.737 | 93.737 | +0.000 | not width-1 layered |

This passes the requested empirical guard: target mean is positive and min is greater than -1, the out-of-suite synthetic has the same sign and rough magnitude, and protected deltas are within +/-0.5 by non-application.

I also checked the related open graphs as negative controls. `mixed_width_labels` and `hierarchical_residual_stage` remain unchanged because the gate rejects them. That leaves real quality debt, but accepting that debt is preferable to reintroducing a broad "skip corridor" polish after sprint-31 already showed that broad corridor transforms can pass fixed-seed scores and still fail jitter or synthetics.

## Concrete Edit Paths

1. Add the new op in `dagua/layout/ops/coordinate.py` near the existing layered coordinate ops, probably after `BrandesKoepfHorizontalRefine` at lines 1491-1590 and before `BrandesKopf4Pass` at line 1593. Keep it registered with `@register_op`, category `OpCategory.COORDINATE`, reads `("pos", "layers")`, writes `("pos",)`, requires `("pos", "layers")`.

2. Import it in `dagua/layout/ops/pipelines/dagua_native_legacy.py` alongside `BrandesKoepfHorizontalRefine` at lines 34-38.

3. Insert it in `dagua/layout/ops/pipelines/dagua_native_legacy.py` immediately after `StripDummyNodes()` at line 1319 and before `AspectRatioFit()` at line 1325. It must run after strip because the operation intentionally moves original real nodes, not dummy nodes.

4. Tests should cover `tests/test_layout/`:
   - `unet_small` fixed-seed score or metric regression: crossing rate drops from 0.25 to 0.0 and composite increases by at least a conservative threshold, e.g. +5.
   - synthetic nested-skip U-Net graph fires and improves crossing rate.
   - `mixed_width_labels`, `hierarchical_residual_stage`, `deep_chain_20`, and `hub_fanout_label_skew` do not fire.

Suggested verification commands for the implementation agent:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

## Risks

The predicate is intentionally narrow. It will not fix `mixed_width_labels` or `hierarchical_residual_stage`; those are related width-1 skip problems but not nested encoder-decoder skip spines. This is a feature, not an oversight: earlier broad skip-corridor ideas failed jitter or synthetic guards. A later sprint can target one-off residual chains separately if it finds a similarly structural mechanism.

The `lane = clamp(0.20 * y_pitch, 20, 80)` scale is a real design choice. It is not fitted by score search, but it should be reviewed visually. Too small leaves near-collinearity; too large hurts straightness and edge-length CV. The current prototype's straightness cost is modest (+6.7 degrees) and the crossing/angular gains dominate.

This does not exploit `cytoscape_fcose`'s very low edge-length CV. Dagua remains about 4.26 points behind the best saved fCoSE seed on this graph. I consider that acceptable for sprint-32 because the proposed change fixes the hierarchical-skip defect while preserving Dagua's DAG semantics; chasing fCoSE's CV would likely require a different force/balance objective and would risk the metric-gaming pattern.

Do not wire this through `_best_of_polish` or any composite-score picker. The validation above is external; the production mechanism should be deterministic from topology and existing coordinates.
