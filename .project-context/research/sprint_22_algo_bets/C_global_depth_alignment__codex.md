# Area C — Global-depth y-alignment for multi-component DAGs — codex

## TL;DR

- The bet is real for the named target. A `/tmp/` monkeypatch at the intended tiler point lifts `disconnected_encoder_residual` from `84.0135` to `88.4741`, a measured `+4.4606` composite delta. That more than closes the known `-1.62` gap to ELK in the sprint context.
- The implementation should not be unconditional. The same candidate regresses `disconnected_label_cycle_collage` by about `-4.92` because FAS/global-depth y rows turn some cyclic component edges into 45-degree straightness penalties.
- `multi_component_80` did not move in the current HEAD probe. Its baseline depth correlation is already `0.9980`, so Area C is not the lever for that graph even though it has seven weak components.
- `sparse_pair_50` and `compound_dag_5x30` were protected in the real tiler probe: both were single-component in the native path and showed exactly `0.0000` delta.
- Recommended integration: add a candidate immediately after `_tile_component_positions(...)` in `dagua/layout/ops/pipelines/dagua_native.py:1431`, but select it with the existing `composite(full(...))` style gate. Do not replace `_tile_component_positions` globally.
- Biggest call: implement this as a picker-safe component-tiling candidate for acyclic multi-component graphs, with a hard guard against cyclic/FAS-derived relayering unless the candidate wins by score.

## Context read

I read `.project-context/research/sprint_22_algo_bets/CONTEXT.md` first, then inspected `dagua/layout/ops/pipelines/dagua_native.py` around the requested call site and the shared tiler implementation in `dagua/layout/ops/pipelines/dagua_native_legacy.py`. The relevant current flow is:

1. `layout_dagua_native_pipeline(...)` decomposes a disconnected parent problem into child problems.
2. Each child is solved independently, with Sprint-21b allowing tree/chain children to re-classify instead of forcing `legacy_monolith`.
3. The child results are passed to `_tile_component_positions(component_results, node_sep=...)`.
4. The tiled positions are aspect-fitted.
5. `_best_of_polish(...)` may choose a detached polish candidate by composite score.

That means the cleanest insertion point is not inside the child solver. It is a parent-level candidate between child tiling and the outer aspect fit, where every parent node index and every child-local position is still available.

## Algorithm

The core idea is simple, but the implementation needs two details that matter:

- The y coordinate should be computed from parent-global depth, not component-local depth.
- If y rows are globally synchronized, component packing must not rely on a multi-row row-major grid where x offsets restart on later rows. Once y offsets are discarded, those later rows can collide with earlier rows. A safe candidate should horizontally strip-pack components, preserving each component's internal x coordinates while giving every component a unique x band.

The target formula is:

```text
y(node) = base_y + global_depth(node) * pitch
```

where `pitch` is estimated from the child-local layouts before the parent transform. Using the median between-depth y step makes the candidate scale-compatible with each child solver. It also avoids hardcoding `rank_sep`; the child may have been reclassified into a tree/chain/native path with its own natural scale.

### Complete working pseudocode

This is the version I would implement, adapted from the `/tmp/` monkeypatch. It assumes it is called immediately where `dagua_native.py` currently calls `_tile_component_positions`.

```python
from collections import defaultdict
from typing import Optional

import torch


def _component_global_depths(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return parent-global longest-path depth for every node.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    num_nodes : int
        Parent node count.

    Returns
    -------
    torch.Tensor
        Integer depth tensor with shape ``[N]`` on CPU.
    """
    layers = torch.as_tensor(longest_path_layering(edge_index.cpu(), num_nodes), dtype=torch.long)
    if edge_index.numel() == 0 or num_nodes <= 2:
        return layers

    # Only invoke FAS when ordinary Kahn layering is degenerate. For normal
    # DAGs, plain longest-path depth is the metric's own reference frame.
    unique_layers = torch.unique(layers)
    counts = torch.bincount(layers, minlength=int(layers.max().item()) + 1)
    heavy_skew = float(counts.max().item()) / float(num_nodes) > 0.5
    if int(unique_layers.numel()) > 1 and not heavy_skew:
        return layers

    filtered = edge_index.cpu()[:, edge_index.cpu()[0] != edge_index.cpu()[1]]
    if filtered.numel() == 0:
        return layers

    try:
        acyclic_edges, _reversed_mask = make_acyclic_robust(filtered, num_nodes)
        relayered = torch.as_tensor(
            longest_path_layering(acyclic_edges, num_nodes),
            dtype=torch.long,
        )
    except Exception:
        return layers

    if int(torch.unique(relayered).numel()) > int(unique_layers.numel()):
        return relayered
    return layers


def _median_component_depth_pitch(
    component_results: list[tuple[torch.Tensor, torch.Tensor]],
    global_depths: torch.Tensor,
    fallback_pitch: float,
) -> float:
    """Infer a y-row pitch from child-local layouts.

    Parameters
    ----------
    component_results : list[tuple[torch.Tensor, torch.Tensor]]
        Tuples of ``(parent_indices, child_pos)``. ``child_pos`` has shape
        ``[Nc, 2]`` in the child's local coordinate system.
    global_depths : torch.Tensor
        Parent-global depth tensor with shape ``[N]`` on CPU.
    fallback_pitch : float
        Pitch used when no component has at least two populated depth rows.

    Returns
    -------
    float
        Positive y step between adjacent global-depth rows.
    """
    diffs: list[float] = []
    for parent_indices, child_pos in component_results:
        if int(parent_indices.numel()) < 2:
            continue
        y_values = child_pos.detach().cpu()[:, 1]
        parent_cpu = parent_indices.detach().cpu()
        by_depth: dict[int, list[float]] = defaultdict(list)
        for local_i, parent_i in enumerate(parent_cpu.tolist()):
            depth = int(global_depths[parent_i].item())
            by_depth[depth].append(float(y_values[local_i].item()))

        medians: dict[int, float] = {}
        for depth, values in by_depth.items():
            medians[depth] = float(torch.tensor(values, dtype=torch.float32).median().item())

        ordered_depths = sorted(medians)
        for left, right in zip(ordered_depths, ordered_depths[1:]):
            depth_delta = right - left
            if depth_delta <= 0:
                continue
            y_delta = abs(medians[right] - medians[left]) / float(depth_delta)
            if y_delta > 1.0e-4:
                diffs.append(y_delta)

    if not diffs:
        return max(float(fallback_pitch), 1.0)
    return float(torch.tensor(diffs, dtype=torch.float32).median().item())


def _tile_component_positions_global_depth_candidate(
    component_results: list[tuple[torch.Tensor, torch.Tensor]],
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sep: float,
) -> Optional[torch.Tensor]:
    """Return a globally depth-aligned component tiling candidate.

    Parameters
    ----------
    component_results : list[tuple[torch.Tensor, torch.Tensor]]
        Parent node indices and independently solved child positions.
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    num_nodes : int
        Parent node count.
    node_sep : float
        Resolved node separation from the native config.

    Returns
    -------
    Optional[torch.Tensor]
        Candidate position tensor with shape ``[N, 2]``, or ``None`` when the
        graph is not a useful multi-component candidate.
    """
    if len(component_results) < 2:
        return None

    global_depths = _component_global_depths(edge_index, num_nodes)
    if int(torch.unique(global_depths).numel()) < 2:
        return None

    pitch = _median_component_depth_pitch(
        component_results,
        global_depths,
        fallback_pitch=float(node_sep),
    )
    gap = max(float(node_sep) * 2.5, 1.0)

    packed: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for parent_indices, child_pos in component_results:
        local = child_pos.clone()
        x_min = float(local[:, 0].min().item())
        x_max = float(local[:, 0].max().item())
        local[:, 0] -= x_min
        width = max(x_max - x_min, float(node_sep))
        packed.append((parent_indices, local, width))

    # Same size-first idea as current tiling, but put every component in one
    # horizontal strip because global y rows remove row-major y offsets.
    packed.sort(key=lambda item: (-int(item[0].numel()), -item[2]))

    dtype = packed[0][1].dtype
    device = packed[0][1].device
    out = torch.zeros((num_nodes, 2), dtype=dtype, device=device)
    x_cursor = 0.0
    for parent_indices, local, width in packed:
        depth_values = global_depths[parent_indices.detach().cpu()].to(device=device, dtype=dtype)
        out[parent_indices, 0] = local[:, 0] + x_cursor
        out[parent_indices, 1] = depth_values * float(pitch)
        x_cursor += width + gap

    out -= out.mean(dim=0, keepdim=True)
    return out
```

The scoring wrapper should then compare baseline tiled output and candidate after the same aspect fit stage, preferably using the existing native `_score_native_result(...)`/`_best_of_polish(...)` pattern. My recommended shape is:

```python
tiled_positions = _tile_component_positions(component_results, node_sep=node_sep)
candidate = _tile_component_positions_global_depth_candidate(
    component_results,
    edge_index=problem.edge_index,
    num_nodes=problem.num_nodes,
    node_sep=node_sep,
)
if candidate is not None:
    tiled_positions = _choose_better_after_aspect_fit(
        baseline=tiled_positions,
        candidate=candidate,
        problem=problem,
        ctx=ctx,
        node_sizes=node_sizes,
        margin=0.25,
    )
```

The margin should be modest. `0.25` composite is enough to ignore crossing-sampling noise and tiny neutral reshapes while accepting the `+4.46` target lift.

## Empirical validation

I implemented the candidate in `/tmp/dagua_sprint22_c_global_depth_probe.py`. The script monkeypatches only `dagua.layout.ops.pipelines.dagua_native._tile_component_positions`, so no project source is changed. It runs `layout(graph, LayoutConfig(seed=0))`, then scores with `dagua.metrics.full(...)` and `crossing_samples=50_000` for runtime. This still exercises the real layout pipeline and the real composite formula. The baseline for `disconnected_encoder_residual` matches the sprint context exactly at `84.0135`.

Measured results:

| Graph | Variant | Composite | Delta | dag | depth | edge_cv | straight | cross | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| disconnected_encoder_residual | baseline | 84.0135 | +0.0000 | 1.0000 | 0.6442 | 0.4074 | 0.0000 | 0.0000 | 0 |
| disconnected_encoder_residual | global-depth strip_x | 88.4741 | +4.4606 | 1.0000 | 0.9832 | 0.4387 | 0.0000 | 0.0000 | 0 |
| disconnected_encoder_residual | global-depth rowmajor_x | 88.4741 | +4.4606 | 1.0000 | 0.9832 | 0.4387 | 0.0000 | 0.0000 | 0 |
| multi_component_80 | baseline | 74.5893 | +0.0000 | 1.0000 | 0.9980 | 1.3121 | 11.3459 | 0.0036 | 0 |
| multi_component_80 | global-depth strip_x | 74.5893 | +0.0000 | 1.0000 | 0.9980 | 1.3121 | 11.3459 | 0.0036 | 0 |
| disconnected_label_cycle_collage | baseline | 80.6300 | +0.0000 | 0.8333 | 0.9449 | 0.5863 | 0.6820 | 0.0000 | 0 |
| disconnected_label_cycle_collage | global-depth strip_x | 75.7059 | -4.9241 | 1.0000 | 1.0000 | 0.5897 | 45.0000 | 0.0000 | 0 |
| sparse_pair_50 | baseline | 87.0382 | +0.0000 | 1.0000 | 1.0000 | 0.5231 | 0.0000 | 0.0000 | 0 |
| sparse_pair_50 | global-depth strip_x | 87.0382 | +0.0000 | 1.0000 | 1.0000 | 0.5231 | 0.0000 | 0.0000 | 0 |
| compound_dag_5x30 | baseline | 80.0000 | +0.0000 | 1.0000 | 1.0000 | 1.6137 | 0.0000 | 0.0000 | 0 |
| compound_dag_5x30 | global-depth strip_x | 80.0000 | +0.0000 | 1.0000 | 1.0000 | 1.6137 | 0.0000 | 0.0000 | 0 |

I also ran a final-position-only version in `/tmp/dagua_sprint22_c_cached_position_probe.py` to understand whether exact depth synchronization after all polish would do better. On the true `disconnected_encoder_residual` baseline, final-position y replacement gives depth `1.0000` but only lifts to `86.1863` (`+2.1728`) because the edge-length CV worsens from `0.4074` to `0.5657`. The tiler-point candidate is better because it lets aspect fit and polish operate on the synchronized component stack, yielding the stronger `+4.4606` result.

## Interpretation

The original prediction was "depth_spearman 0.644 to 1.000 equals about +5.4 weighted points, minus edge-CV fallout." The measured tiler-point result follows that budget closely:

- depth improves from `0.6442` to `0.9832`, worth about `+5.09`;
- edge CV worsens from `0.4074` to `0.4387`, costing about `-0.63`;
- all other measured terms stay flat;
- net result is `+4.46`.

The candidate does not reach literal `1.0000` depth at the final score point because later aspect/polish still perturbs y slightly. That is acceptable; the score lift is larger than the exact final-y rewrite because it preserves better edge-length uniformity.

`multi_component_80` is not a depth-alignment target at HEAD. Its measured baseline depth rho is already `0.9980`, so even a perfect row sync has no meaningful 15-point depth term left to harvest. The remaining gap is more consistent with edge CV, straightness, and crossing behavior.

## Risk and regression analysis

`disconnected_label_cycle_collage` is the important protected win. The candidate raises `dag_consistency` from `0.8333` to `1.0000` and depth from `0.9449` to `1.0000`, but it destroys straightness: `0.6820` degrees to the clamped `45.0000` degree default/worst value. That costs almost ten straightness points and produces the net `-4.92` regression. The likely cause is that FAS-derived global depth is metric-attractive but geometrically false for at least one cyclic component. The current layout keeps that component locally near-horizontal; global y rows stretch it into diagonals.

This means the implementation must either:

1. skip graphs where cycle handling is needed, or
2. allow them only through a composite picker.

I recommend both: use a conservative acyclic/multi-component guard to avoid extra work, and still score-gate because even DAGs can have pitch/edge-CV tradeoffs.

`sparse_pair_50` and `compound_dag_5x30` are low risk under the proposed integration because they were single-component at the native path and the candidate returns `None`. Their measured deltas are exactly zero.

A second risk is horizontal collision. If we reuse the current row-major tiler x offsets and then erase row-major y offsets, later rows can restart at `x=0` and overlap earlier rows. It did not matter for the two-component target, but it is unsafe generally. The implementation should use strip packing for the candidate, not the existing row-major offsets.

The third risk is score-time overhead. This candidate only applies when component decomposition has already happened, and the scoring path already exists for polish candidates. It should not affect connected graphs or large single-component DAGs.

## Implementation order

1. Add an internal helper near the component tiling section of `dagua/layout/ops/pipelines/dagua_native.py`, not in `dagua/` public API.
2. Compute parent-global depths with plain `longest_path_layering` for normal DAGs. Add FAS fallback only for degenerate cyclic layerings, but default to rejecting FAS-derived candidates unless score-gated.
3. Build `_tile_component_positions_global_depth_candidate(...)` from `component_results`, `problem.edge_index`, `problem.num_nodes`, and resolved `node_sep`.
4. Apply `AspectRatioFit` to both baseline tiled positions and the candidate before scoring, so the comparison matches the real downstream geometry.
5. Pick the candidate only if it wins by at least `0.25` composite. Then let the existing `_best_of_polish(...)` run as today.
6. Add targeted tests for `disconnected_encoder_residual` improvement and `disconnected_label_cycle_collage` non-regression. Also include no-op tests for `sparse_pair_50` and `compound_dag_5x30`.

## Assumptions

- I treated `dagua/layout/ops/pipelines/dagua_native.py:1431` as the requested insertion point and did not modify source code.
- I used `crossing_samples=50_000` in the `/tmp/` scoring to keep the research run practical while several other sprint processes were consuming CPU. On the main target, crossings are zero in both baseline and candidate, so this does not affect the reported lift mechanism.
- I used `seed=0` to match the sprint context.

## Files used

- `/tmp/dagua_sprint22_c_global_depth_probe.py`: true monkeypatch at the tiler point; primary measurement source.
- `/tmp/dagua_sprint22_c_cached_position_probe.py`: final-position sensitivity check; used only to understand why exact final y sync is weaker than tiler-point sync.
