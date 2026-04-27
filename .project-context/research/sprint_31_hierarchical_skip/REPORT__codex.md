# Sprint 31 Hierarchical Skip Report -- Codex

## TL;DR

- **Do not ship a sprint-31 fix from this pass.** I found point-score lifts, but the broadest candidate fails the sprint jitter guard on three of the four targets.
- The live Dagua baselines reproduce the prompt exactly with `composite(full(..., node_sizes=[[40, 20]] * N))`: `mixed_width_labels` 77.58, `unet_small` 70.79, `extreme_mixed_width_transformer` 74.46, `hierarchical_residual_stage` 82.29.
- The original diagnosis is only partly right. For `mixed_width_labels`, `unet_small`, and `hierarchical_residual_stage`, dummy nodes and Brandes-Koepf are not being undone; they never run because the resolved layering has max width 1. `extreme_mixed_width_transformer` runs BK but skips dummy nodes due the small-N dummy gate.
- The loss mechanism is real: the three thin residual-chain targets lose almost entirely through collinear-overlap crossings on long skip edges. `unet_small` has 9 exact crossings, all 9 collinear and all involving skip edges.
- A class-gated "skip corridor" coordinate pass lifted all four targets in raw score and rejected all five protected wins, but Gaussian jitter at sigma=0.5 removed much of the apparent lift. This is metric-sensitive collinearity cleanup, not a robust layout improvement.

## Reproduction Setup

I used current branch code at the given worktree and scored with:

```python
node_sizes = torch.tensor([[40.0, 20.0]] * graph.num_nodes)
score = composite(full(pos, graph.edge_index, node_sizes=node_sizes))
```

Dagua positions were generated live with `LayoutConfig(seed=42, device="cpu")`. Competitor positions for the target reproduction came from `eval_output/benchmark_full/positions/*.pt` and were rescored through the current metric. That matters because the metric now counts collinear-overlap in `segments_intersect`.

## Baseline Scores

| graph | dagua live | best reproduced | best engine | prompt delta |
|---|---:|---:|---|---:|
| `mixed_width_labels` | 77.58 | 84.52 | `elk_layered` | -6.94 |
| `unet_small` | 70.79 | 77.04 | `elk_layered` / `dagre` | -6.25 |
| `extreme_mixed_width_transformer` | 74.46 | 77.99 | `graphviz_dot` | -3.53 |
| `hierarchical_residual_stage` | 82.29 | 84.71 | `dagre` | -2.42 |

Dagua per-metric breakdown:

| graph | score | DAG | edge CV | depth rho | overlaps | straight deg | crossing rate | angular deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_width_labels` | 77.58 | 1.000 | 0.496 | 1.000 | 0 | 0.0 | 0.125 | 108.0 |
| `unet_small` | 70.79 | 1.000 | 0.746 | 1.000 | 0 | 0.0 | 0.250 | 25.7 |
| `extreme_mixed_width_transformer` | 74.46 | 1.000 | 0.592 | 0.991 | 0 | 19.8 | 0.067 | 86.8 |
| `hierarchical_residual_stage` | 82.29 | 1.000 | 0.498 | 1.000 | 0 | 0.0 | 0.053 | 112.5 |

The first-order loss is not DAG direction, depth order, or overlap. Dagua gets those right. The thin targets lose because the crossing-rate term hits exact collinear overlaps, and `unet_small` also has poor angular resolution because all residual structure is collapsed into a single centerline.

## Diagnosis

### Gate behavior

Live gate checks:

| graph | N/E | layer widths | long edges | dummy nodes? | BK? |
|---|---:|---|---:|---|---|
| `mixed_width_labels` | 6/6 | `[1,1,1,1,1,1]` | 1 | false | false |
| `unet_small` | 9/11 | `[1,1,1,1,1,1,1,1,1]` | 3 | false | false |
| `extreme_mixed_width_transformer` | 10/12 | `[1,1,1,3,1,1,1,1]` | 1 | false | true |
| `hierarchical_residual_stage` | 10/11 | `[1,1,1,1,1,1,1,1,1,1]` | 2 | false | false |

This rules out "post-gradient polish is undoing dummy nodes / BK" for three targets. There is no dummy or BK state to undo. The relevant code path is:

- `_should_use_native_dummy_nodes()` rejects `layer_assignments.shape[0] < _DUMMY_NODE_MIN_NODES` with `_DUMMY_NODE_MIN_NODES = 20`.
- It also rejects `max_layer_width <= 1`.
- `_should_apply_brandes_koepf_refine()` rejects `max_layer_width <= 1`.

For `mixed_width_labels`, `unet_small`, and `hierarchical_residual_stage`, the longest-path layering is a one-node-wide chain with residual skip edges. BK's "no horizontal spreading work" interpretation is technically consistent with the current gate, but it misses exactly the residual-corridor problem: the graph has no same-layer order to refine, yet it still needs lateral skip-edge corridors.

For `extreme_mixed_width_transformer`, BK does run because the Q/K/V layer has width 3. Dummy nodes do not run only because `N=10 < 20`. A runtime monkeypatch of `_DUMMY_NODE_MIN_NODES = 0` changed only this target materially:

| graph | default | no polish | BK off | dummy off | dummy min=0 |
|---|---:|---:|---:|---:|---:|
| `mixed_width_labels` | 77.58 | 75.25 | 77.58 | 77.58 | 77.58 |
| `unet_small` | 70.79 | 67.43 | 70.79 | 70.79 | 70.79 |
| `extreme_mixed_width_transformer` | 74.46 | 72.48 | 76.63 | 74.46 | 86.41 |
| `hierarchical_residual_stage` | 82.29 | 80.37 | 82.29 | 82.29 | 82.29 |

So there is one real bug-shaped finding: small-N dummy insertion is too conservative for width>1 layered DAGs with long edges. But that is not a sprint-wide fix, because three of four targets still fail the `max_layer_width <= 1` gate.

The no-polish runs also matter. Disabling `edge_equalize_polish` lowers the scores by 1.9 to 3.4 points, almost entirely through worse edge-length CV, but it does not remove or introduce the skip-edge crossings. That means the post-gradient picker is not the immediate regression source for this class. It is improving one metric while leaving the newly fixed crossing metric exposed. This is a useful boundary: removing the picker would make these four worse, and adding another picker candidate is exactly the wrong direction after sprint-30 unless it passes independent robustness checks.

`bk_off` also separates two cases. It is a no-op on the three one-wide layerings because BK is already disabled. On `extreme_mixed_width_transformer`, turning BK off raises the point score to 76.63 by sharply reducing edge CV while keeping the same crossing rate. That is not enough evidence to disable BK. The much stronger `dummy_min0` result says the width-3 transformer wants dummy-node treatment before coordinate compaction, not a blanket BK rollback.

### Geometry

Exact edge-pair diagnostics on the reproduced Dagua layouts:

| graph | exact crossings | collinear crossings | skip-edge collinear | intermediate nodes within 20px of skip line |
|---|---:|---:|---:|---:|
| `mixed_width_labels` | 1 | 1 | 1 | 2 |
| `unet_small` | 9 | 9 | 9 | 9 |
| `extreme_mixed_width_transformer` | 3 | 0 | 0 | 4 |
| `hierarchical_residual_stage` | 2 | 2 | 2 | 4 |

For the thin residual-chain graphs, the new metric is exposing a true degenerate geometry: a long residual edge lies on the same infinite line as shorter path edges. Competitors avoid it by giving the skip endpoint or intermediate nodes lateral displacement. Note that they do not need real same-layer spread to do this; `mixed_width_labels` and `unet_small` still have one real node per topological layer, but Graphviz/Dagre/ELK vary x across depth so residual edges are not drawn on top of the main chain.

`extreme_mixed_width_transformer` is different. It has a real width-3 layer and non-collinear crossings. Its very large improvement under forced dummy insertion suggests the dummy/BK machinery can solve this subtype when allowed to run.

The one-wide cases are awkward because "horizontal coordinate refinement" has no same-rank degrees of freedom. A standard BK pass assigns x within each layer after crossing reduction; if every layer has one real vertex, the layer order is fixed and the current gate treats the graph as horizontally solved. The missing degree of freedom is cross-layer x variation: allowing a node on depth `d+1` to sit left or right of the node on depth `d` even though neither layer has internal width. That is not dummy-node insertion in the ordinary Sugiyama sense, and it is not median/transpose ordering. It is closer to routing long residual corridors through side lanes while preserving the main chain's readability. The current tensor metric, however, only sees straight center-to-center segments, not routed polylines, so a rendered edge-routing-only fix would not move the measured score.

This is why a principled implementation cannot simply be "route skip edges around nodes" unless the metric also evaluates the routed curves. Under the present metric contract, the node coordinates themselves must change. That makes the problem harder: moving nodes enough to eliminate collinear overlaps also changes edge lengths, straightness, angular resolution, and possibly the visual hierarchy. A candidate must prove it improves the geometry, not merely perturb exact equality.

## Candidate Fix Tested

I tested a topology-gated coordinate pass, not wired into `dagua/`, that bends residual corridors after the native layout. The gate is class-based:

- directed acyclic graph;
- `4 <= N <= 64`;
- at least four layers;
- at least one edge with layer span >= 2;
- max layer width <= 3;
- at least 65% of layers are one-node layers;
- edge-span variance >= 0.4;
- long-edge fraction <= 45%.

The `N <= 64` bound is a scale guard, not an exact signature. It kept the candidate away from `random_dag_200` and `ba_500`, while the layer-width and long-edge gates kept it away from the protected fanout/tree/chain wins.

The best variant was an endpoint-shift pass: for each long edge, move the less-branching endpoint sideways by a lane derived from observed layer pitch. A second bowing variant moved intermediate nodes instead. Both are algorithmic in the limited sense that offsets come from topology and pitch, not graph names or hardcoded node tables. Both still fail the jitter guard.

I intentionally did not use composite score inside the transform. The prototype computes a single candidate from topology and current coordinates, then the research script scores it afterward. That keeps the experiment separate from the existing `_best_of_polish` anti-pattern. The downside is that the candidate is crude: it cannot choose among several plausible lane assignments, because doing so with the benchmark metric would be picker-margin acceptance again. A production-quality version would need a graph-theoretic or geometric objective such as "minimize weighted segment-node corridor incidence subject to bounded edge-length CV increase," and then use the benchmark metric only for external evaluation.

Pseudocode:

```python
from typing import Dict, List, Tuple

import torch


def classify_skip_corridor(edge_index: torch.Tensor, num_nodes: int) -> Tuple[bool, torch.Tensor]:
    """Return whether a graph is a small residual-corridor DAG.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of real nodes.

    Returns
    -------
    tuple[bool, torch.Tensor]
        Whether the class gate fires, plus longest-path layers ``[N]``.
    """
    layers = longest_path_layering(edge_index, num_nodes)
    layers = layers if isinstance(layers, torch.Tensor) else torch.tensor(layers)
    if num_nodes < 4 or num_nodes > 64 or edge_index.numel() == 0:
        return False, layers
    if detect_back_edges(edge_index, num_nodes).any():
        return False, layers

    counts = torch.bincount(layers)
    spans = layers[edge_index[1]] - layers[edge_index[0]]
    long_mask = spans >= 2
    if counts.numel() < 4 or not bool(long_mask.any()):
        return False, layers
    if int(counts.max().item()) > 3:
        return False, layers
    if float((counts == 1).float().mean().item()) < 0.65:
        return False, layers
    if float(long_mask.float().mean().item()) > 0.45:
        return False, layers
    if float(spans.float().var(unbiased=False).item()) < 0.4:
        return False, layers
    return True, layers


def skip_endpoint_corridor_polish(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Bend long residual edges away from one-node-wide layer chains.

    Parameters
    ----------
    pos : torch.Tensor
        Current node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Candidate positions with x offsets applied to selected skip endpoints.
    """
    num_nodes = int(pos.shape[0])
    enabled, layers = classify_skip_corridor(edge_index, num_nodes)
    if not enabled:
        return pos.clone()

    spans = layers[edge_index[1]] - layers[edge_index[0]]
    long_edges = torch.nonzero(spans >= 2, as_tuple=False).flatten().tolist()
    long_edges.sort(key=lambda e: (-int(spans[e]), int(edge_index[0, e]), int(edge_index[1, e])))

    indegree = torch.zeros(num_nodes, dtype=torch.long)
    outdegree = torch.zeros(num_nodes, dtype=torch.long)
    for source, target in edge_index.t().tolist():
        outdegree[int(source)] += 1
        indegree[int(target)] += 1

    unique_y = torch.sort(torch.unique(pos[:, 1]))[0]
    deltas = torch.diff(unique_y).abs()
    pitch = float(torch.median(deltas).item()) if deltas.numel() else 240.0
    lane = max(40.0, min(90.0, pitch * 0.35))

    offsets: Dict[int, float] = {}
    for ordinal, edge_id in enumerate(long_edges):
        source = int(edge_index[0, edge_id])
        target = int(edge_index[1, edge_id])
        span = int(spans[edge_id])
        sign = -1.0 if ordinal % 2 else 1.0
        movable = target if indegree[target] >= outdegree[source] else source
        offsets[movable] = offsets.get(movable, 0.0) + sign * lane * max(span - 1, 1) ** 0.5

    out = pos.detach().clone()
    for node, dx in offsets.items():
        out[node, 0] = out[node, 0] + dx
    return out
```

## Empirical Validation

Raw point scores for the endpoint candidate:

| graph | gate | base | candidate | delta | crossing rate | edge CV |
|---|---|---:|---:|---:|---|---|
| `mixed_width_labels` | true | 77.58 | 85.88 | +8.30 | 0.125 -> 0.000 | 0.496 -> 0.487 |
| `unet_small` | true | 70.79 | 80.65 | +9.87 | 0.250 -> 0.000 | 0.746 -> 0.678 |
| `extreme_mixed_width_transformer` | true | 74.46 | 75.67 | +1.21 | 0.067 -> 0.044 | 0.592 -> 0.571 |
| `hierarchical_residual_stage` | true | 82.29 | 82.59 | +0.30 | 0.053 -> 0.026 | 0.498 -> 0.489 |
| `synthetic_hourglass_skip` | true | 77.68 | 82.32 | +4.64 | 0.071 -> 0.000 | 0.574 -> 0.529 |

The synthetic graph was built out of suite via `networkx.DiGraph`: thin input/stem, width-3 middle, merge/tail, plus two long skips.

Protected win gates:

| protected graph | gate result | reason |
|---|---|---|
| `random_dag_200` | false | `N=383` and broad random layering |
| `deep_chain_20` | false | no long edges |
| `org_chart_deep` | false | `N=79`, no long edges |
| `hub_fanout_label_skew` | false | no long edges, max layer width 5 |
| `ba_500` | false | cyclic/scale-free, `N=500` |

This satisfies protected-win non-regression only by non-application. That is acceptable for a gate check, but it is not enough by itself to justify shipping because the jitter validation fails.

Sigma=0.5 jitter deltas, using the same random noise on base and candidate positions:

| graph | mean delta | min delta | per-trial deltas |
|---|---:|---:|---|
| `mixed_width_labels` | +0.81 | -1.70 | `[-1.689, -1.681, +8.302, -1.697, -1.671, -1.679, +8.310, -1.683]` |
| `unet_small` | +9.31 | +5.42 | `[+5.423, +9.873, +9.865, +9.864, +9.872, +9.878, +9.863, +9.876]` |
| `extreme_mixed_width_transformer` | +0.94 | -1.01 | `[-1.008, -1.006, +3.436, +3.441, -1.003, +1.222, +1.215, +1.219]` |
| `hierarchical_residual_stage` | -1.66 | -4.95 | `[-2.325, -2.323, -2.317, -2.312, -4.946, +0.308, +0.307, +0.317]` |
| `synthetic_hourglass_skip` | +6.79 | +4.63 | `[+7.500, +7.498, +7.503, +4.631, +7.510, +4.635, +7.502, +7.505]` |

This is the decisive failure. On the collinear-overlap targets, small random jitter often makes the baseline no longer exactly collinear, so the crossing penalty disappears without any principled layout change. A candidate whose lift depends on exact collinearity is therefore not stable evidence. The bowing variant had the same problem: it lifted raw scores, but had negative jitter trials on `mixed_width_labels`, `unet_small`, `extreme_mixed_width_transformer`, and `hierarchical_residual_stage`.

The jitter result does not mean the baseline is visually acceptable. It means this metric slice is discontinuous around exact collinearity, so point-score improvement is an unreliable oracle here. The correct bar is higher: after jitter, the candidate should still improve edge CV, angular separation, or a robust corridor-distance measure enough that the lift survives when exact collinearity is broken. The endpoint candidate fails that bar on `mixed_width_labels` and `hierarchical_residual_stage`; the bow candidate improves raw hierarchy aesthetics more than endpoint shifting but still has negative jitter trials. I would treat both as useful diagnostic probes, not ship candidates.

## Recommendation

Do not ship the skip-corridor candidate. It is class-gated and avoids the protected wins, but the improvement is not jitter-stable. Shipping it would repeat the sprint-30 failure mode in a subtler form: not an exact graph fixture, but still a metric-sensitive correction whose strongest effect is removing exact collinearity.

I would split follow-up work into two principled tracks:

1. **Bug-shaped narrow fix:** re-evaluate the small-N dummy-node gate for layered DAGs with real wide layers and long edges. `extreme_mixed_width_transformer` going 74.46 -> 86.41 under `_DUMMY_NODE_MIN_NODES = 0` is too large to ignore. The gate should be based on dummy need, e.g. `has_long_edges and max_layer_width > 1 and edge_span_variance >= threshold`, not a flat `N >= 20` rule. This still needs validation on more width>1 residual graphs before shipping.
2. **Real algorithm work for one-node-wide residual chains:** the current BK/dummy machinery explicitly rejects `max_layer_width <= 1`, but residual-chain skips need x-over-depth coordinate assignment, not same-layer ordering. A principled version should optimize a geometric objective independent of the benchmark composite, such as minimizing skip-edge/node corridor incidence plus edge-length CV under overlap constraints, then pass jitter. I did not find that robust objective in this pass.

## Concerns

- The prompt class says "wide-narrow-wide," but three targets are effectively one node wide under longest-path layering. That makes Sugiyama dummy/BK less directly applicable than the prompt diagnosis suggests.
- The metric's collinear-overlap branch is correct, but it is discontinuous under tiny coordinate noise. Any fix that only converts exact overlap into near-overlap will look good on the point metric and weak under jitter.
- `extreme_mixed_width_transformer` is probably a separate small-N dummy-gate bug, not the same failure as the three thin residual-chain losses.

## Knowledge

- `edge_equalize_polish` improves CV on all four targets but leaves crossing rates unchanged; it is not the source of the skip-edge overlap.
- BK is off whenever `max_layer_width <= 1`; that is why the three thin residual targets get no horizontal coordinate refinement.
- Lowering only `_DUMMY_NODE_MIN_NODES` helps `extreme_mixed_width_transformer` but does nothing for one-wide residual chains because the max-width dummy gate still rejects them.
