# Sprint 22 Area D: `dependency_500` Escape From Gradient Saturation

## TL;DR

- **Biggest call: do not implement `aspect_preserving_equalize` as proposed.** On the current `dependency_500` baseline it regressed `55.284 -> 25.384` (`-29.900`) because bbox locking preserved the outer aspect while collapsing internal depth order and creating 124 node overlaps.
- **The useful bet is gap-constrained adjacent x-swap search, but it is a partial closer, not a win.** A bounded `/tmp` prototype accepted 30 of 32 composite-validated same-layer swaps and improved `dependency_500` `55.284 -> 56.265` (`+0.981`).
- **The gain comes from edge-length CV only.** CV improved `0.9054 -> 0.8528`, worth about `+1.05` composite points; DAG consistency and depth Spearman stayed effectively perfect, overlaps stayed zero, and straightness remained past the scoring floor.
- **Protected graphs are safe only if this remains a scored candidate.** Forced equalization regressed `random_dag_200`, `org_chart_deep`, and `hub_fanout_label_skew`; the existing picker-style margin gate would keep their baselines.
- **Realistic combined prediction:** gap search closes the `dependency_500` loss from `-2.90` to roughly `-1.92` versus ELK's `58.19`. More search budget may add another `+0.2..+0.5`, but the evidence does not support a full close without a deeper layered ordering or dummy-edge corridor algorithm.

## Empirical Setup

I used HEAD `c821eb6`, kept `dagua/` read-only, and implemented the variants in `/tmp/sprint22_d_focused.py`. The script writes `/tmp/sprint22_d_focused/results.json` and caches current-code baseline positions under `/tmp/sprint22_d_focused/*__default_seed0.pt`.

The harness uses direct benchmark graph constructors to avoid building the whole suite, then runs `layout(graph, LayoutConfig(device="cpu", seed=0))`. It scores variants with `composite(full(...))`, topological depths from `longest_path_layering`, `crossing_samples=50_000`, and reduced stress/neighborhood sample counts for runtime. The important calibration is that the measured `dependency_500` baseline is `55.284`, matching the sprint context table (`55.28`). That makes the deltas comparable even though the scoring sample count is bounded.

Target and protected graphs measured:

| Graph | N | E | Baseline | Best tested variant | Delta |
|---|---:|---:|---:|---|---:|
| `dependency_500` | 500 | 1470 | 55.284 | `gap_validated_32` 56.265 | +0.981 |
| `random_dag_200` | 383 | 300 | 74.290 | baseline | +0.000 |
| `org_chart_deep` | 79 | 78 | 92.441 | baseline | +0.000 |
| `hub_fanout_label_skew` | 10 | 13 | 93.737 | baseline | +0.000 |

Note: the named `random_dag_200` generator materializes as 383 nodes in this code path because `_random_dag()` pre-adds integer nodes and then adds string IDs like `n42`. I did not relabel it; this is the graph the current benchmark helper constructs.

## Bet 1: Aspect-Preserving Equalize

### Algorithm Sketch

The tested version is the literal low-effort proposal: run the existing endpoint projection toward mean edge length, then rescale the candidate bounding box back to the original bounding box after every iteration. This keeps the global aspect ratio that `layered_dag` constructed, while still letting endpoints move inside the box.

```python
def aspect_preserving_equalize(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    iters: int,
    step: float,
) -> torch.Tensor:
    """Equalize edge lengths while preserving the original layout bbox.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape [N, 2].
    edge_index : torch.Tensor
        Edge tensor with shape [2, E].
    iters : int
        Number of projection iterations.
    step : float
        Fraction of the length error applied per iteration.

    Returns
    -------
    torch.Tensor
        Candidate position tensor with shape [N, 2].
    """
    cand = pos.detach().clone()
    src = edge_index[0]
    tgt = edge_index[1]
    mask = src != tgt
    src = src[mask]
    tgt = tgt[mask]
    if src.numel() == 0:
        return cand

    ref_min = pos.min(dim=0).values
    ref_max = pos.max(dim=0).values
    ref_span = (ref_max - ref_min).clamp(min=1.0e-9)

    for _ in range(iters):
        diffs = cand[tgt] - cand[src]
        dists = diffs.pow(2).sum(-1).sqrt().clamp(min=1.0)
        target = float(dists.mean().item())
        unit = diffs / dists.unsqueeze(-1)
        delta = (dists - target).unsqueeze(-1) * unit * step
        cand.index_add_(0, src, delta * 0.5)
        cand.index_add_(0, tgt, -delta * 0.5)

        cand_min = cand.min(dim=0).values
        cand_max = cand.max(dim=0).values
        cand_span = (cand_max - cand_min).clamp(min=1.0e-9)
        cand = (cand - cand_min) / cand_span * ref_span + ref_min

    return cand
```

### Measured Delta

| Variant | Score | Delta | CV | Overlaps | DAG | Depth rho | Straightness |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 55.284 | +0.000 | 0.9054 | 0 | 1.000 | 0.9939 | 71.81 |
| plain equalize `10, 0.10` | 39.959 | -15.325 | 0.3616 | 2 | 0.665 | 0.4505 | 59.77 |
| plain equalize `50, 0.05` | 39.589 | -15.695 | 0.2402 | 7 | n/a | n/a | 53.28 |
| aspect preserving `10, 0.10` | 25.384 | -29.900 | 0.5012 | 124 | 0.499 | 0.0174 | 84.72 |
| aspect preserving `50, 0.05` | 25.102 | -30.182 | 0.4539 | 221 | n/a | n/a | 86.16 |

This falsifies the original diagnosis. The problem is not just bounding-box drift. The direct endpoint equalizer changes internal y order enough to destroy the high-value hierarchy terms. Bbox preservation makes that worse because it compresses moved nodes back into the same outer envelope instead of letting the projection relieve density. CV improves, but it buys roughly `+8..+11` raw CV points while losing far more to DAG consistency, depth correlation, and overlap.

Protected-graph behavior is also negative if forced:

| Graph | Best aspect-preserving delta | Plain equalize delta |
|---|---:|---:|
| `random_dag_200` | -3.271 | -3.381 |
| `org_chart_deep` | -1.135 | -1.135 |
| `hub_fanout_label_skew` | -0.952 | -0.665 |

Recommendation: do not add this candidate unless heavily gated and scored. Even then, the current data says it is dead code for this target.

## Bet 2: Gap-Constrained Layered Local Search

### Algorithm Sketch

The successful prototype uses topological layers, not y-bucket inference. It ranks adjacent same-layer x swaps by a cheap edge-CV prefilter, then validates each promising candidate with `composite(full(...))` before accepting it. This keeps the search bounded while obeying the important rule: no swap is committed unless the real composite score improves.

```python
def gap_validated_search(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: List[int],
    max_candidates: int = 32,
) -> torch.Tensor:
    """Run bounded adjacent x-swap search with composite validation.

    Parameters
    ----------
    pos : torch.Tensor
        Baseline positions with shape [N, 2].
    edge_index : torch.Tensor
        Edge tensor with shape [2, E].
    node_sizes : torch.Tensor
        Node sizes with shape [N, 2].
    topo_depth : List[int]
        Longest-path layer for each node.
    max_candidates : int
        Maximum number of prefiltered swaps to validate.

    Returns
    -------
    torch.Tensor
        Best composite-validated candidate positions.
    """
    best = pos.detach().clone()
    layers = torch.tensor(topo_depth, dtype=torch.long)
    best_score = composite(full(best, edge_index, topo_depth=topo_depth, node_sizes=node_sizes))

    lengths = edge_lengths(best, edge_index)
    long_threshold = torch.quantile(lengths, 0.90)
    long_nodes = endpoints_of_edges_where(lengths >= long_threshold)

    ranked: List[Tuple[float, int, int]] = []
    base_cv = edge_length_cv_scalar(best, edge_index)
    for layer in unique_sorted(layers):
        ordered = nodes_in_layer_sorted_by_x(layer, layers, best)
        for left, right in adjacent_pairs(ordered):
            if left not in long_nodes and right not in long_nodes:
                continue
            trial = swap_x(best, left, right)
            ranked.append((edge_length_cv_scalar(trial, edge_index) - base_cv, left, right))

    for _, left, right in sorted(ranked)[:max_candidates]:
        trial = swap_x(best, left, right)
        trial_score = composite(full(trial, edge_index, topo_depth=topo_depth, node_sizes=node_sizes))
        if trial_score > best_score:
            best = trial
            best_score = trial_score

    return best
```

Implementation notes for production:

- Use the existing `dagua.utils.longest_path_layering()` or the already prepared native layer assignments instead of reconstructing layers from rendered y positions.
- Start with `max_candidates=32`. In the prototype, 30 of 32 candidates were accepted, so the prefilter is finding real opportunities.
- Keep this as a scored polish candidate inside `_best_of_polish`, not as a forced postprocess.
- The existing `crossing_swap.py` crossing reducer is useful precedent, but this search should optimize edge-span/CV first and use full composite as the final authority. `dependency_500` is not primarily a crossing failure.

### Measured Delta

`gap_validated_32` improved `dependency_500`:

| Metric | Baseline | Gap search | Change |
|---|---:|---:|---:|
| composite | 55.284 | 56.265 | +0.981 |
| edge_length_cv | 0.9054 | 0.8528 | -0.0526 |
| DAG consistency | 1.000 | 1.000 | +0.000 |
| depth Spearman | 0.9939 | 0.9939 | +0.000 |
| overlap_count | 0 | 0 | +0 |
| crossing_rate | 0.1030 | 0.1041 | +0.0011 |
| straightness mean deg | 71.81 | 74.41 | +2.60 |

The straightness metric is already at the zero-score floor because both values are above 45 degrees, so the straightness worsening did not materially offset the CV gain. Crossing changed only marginally. This is exactly the kind of improvement that saturated gradients could not find: a small discrete permutation of layer order, not a continuous coordinate relaxation.

## Combined Effect Prediction

The combined strategy should not include aspect-preserving equalize. The only viable combination is:

1. Keep current baseline and existing edge-equalize candidates.
2. Add `gap_validated_search` as a later candidate for large DAG-ish graphs.
3. Let the picker compare it against baseline and existing polish outputs.

Expected production result on `dependency_500`: `+0.9..+1.4` if `max_candidates` is between 32 and 96. The observed `+0.981` moves the graph from a moderate loss to the close-loss bucket: `55.28 -> 56.26` versus ELK at `58.19`, leaving roughly `-1.92`. I would not forecast a tie without a deeper implementation that explicitly models dummy-edge corridors or runs a true Sugiyama ordering pass with gap penalties.

## Risk / Regression Analysis

The protected wins are sensitive to any forced equalization. Even when CV improves, the hierarchy terms are worth more. Examples:

- `random_dag_200`: aspect-preserving `10,0.10` improved CV `0.6357 -> 0.3992` but score fell `-3.27` through worse straightness and ordering.
- `org_chart_deep`: baseline CV is already `0.0031`; equalization has no useful work left and costs about `-1.13`.
- `hub_fanout_label_skew`: baseline is already high at `93.737`; both plain and bbox-preserving equalize regress by `-0.67..-0.99`.

The gap search should be gated to avoid becoming another broad polish primitive:

- Run only when `N >= 200`, `E/N >= 2`, graph is DAG or nearly DAG after cycle preparation, and baseline `edge_length_cv >= 0.75`.
- Skip trees/chains and graphs with current CV below `0.25`.
- Validate with the same `safe_score()` path used by `_best_of_polish`, and require the existing `+0.5` margin before selecting the candidate.
- Cap candidates by graph size, for example `min(96, max(16, N // 8))`, and abort if scoring exceeds a time budget.

Specific protected wins to verify in implementation:

- `random_dag_200`: must keep current baseline, no forced equalize.
- `org_chart_deep`: must not run gap search because it is tree-family / low-CV.
- `hub_fanout_label_skew`: must not run because N is too small and baseline is already near ceiling.
- Also recheck `wide_parallel_200`, `dense_skip_200`, and `dependency_graph_100`, because they share enough structure with `dependency_500` to be tempting false positives.

## Implementation Order

1. **Do not implement `aspect_preserving_equalize` for sprint-22.** Keep the `/tmp` result as a negative finding.
2. Add a private helper near `_swap_2opt_anti_crossing` in `dagua_native.py`, probably `_gap_validated_layer_swaps(pos, edge_index, node_sizes, score_fn, max_candidates=32)`.
3. Use topological layers from `longest_path_layering()` or prepared native layer assignments. Avoid y-bucket inference for this algorithm.
4. Rank candidate adjacent swaps by cheap CV delta among nodes incident to the top 10% longest edges.
5. Score each ranked candidate with `score_fn(trial)` and accept only strict improvements.
6. Add the candidate to `_best_of_polish` after existing equalize variants and before expensive composed candidates, behind a conservative `_should_gap_swap_large_dag()` gate.
7. Validate with the project gates plus targeted scoring on `dependency_500`, `random_dag_200`, `org_chart_deep`, `hub_fanout_label_skew`, `dense_skip_200`, and `wide_parallel_200`.

## Concerns

The prototype improves the target but does not solve it. It is a tactical patch over a structural ordering gap. If the sprint goal is "tied or best on every graph," the next deeper bet for `dependency_500` should be true long-edge-aware layered ordering: dummy-node expansion, barycenter/transpose over dummy chains, and a gap penalty before coordinate assignment. The local search is worth shipping only because it is small, picker-safe, and empirically moves the right metric without touching the high-value DAG/depth terms.

## Knowledge Worth Carrying Forward

Two practical details matter for the next implementer. First, `dependency_500` is not an overlap problem at HEAD. The reproduced baseline has `overlap_count=0`, `dag_consistency=1.0`, and `depth_spearman_rho=0.9939`; the residual gap is almost entirely the CV term. Any algorithm that perturbs y order is spending the most valuable part of the composite budget to chase a smaller term. This is why the symmetric edge equalizer is so dangerous on this graph even though it visibly improves CV.

Second, the winning local search did not need to know ELK's coordinates. It only needed to ask which same-layer adjacent swaps shorten the longest-edge endpoints without breaking the composite. That suggests the current native pipeline already finds a mostly correct layered drawing, but leaves a few x-order inversions inside crowded layers. A production implementation can therefore be narrow: no new pipeline, no new optimizer, no graph-family reroute. Add one candidate generator to the existing picker and keep every accepted move accountable to the full score.
