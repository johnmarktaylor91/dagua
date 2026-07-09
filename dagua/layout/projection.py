"""Hard overlap resolution via projected gradient descent.

After each optimizer step, pushes overlapping node bounding boxes apart.
Runs deterministically with torch.no_grad().

Sprint 3 scaling strategy:
- N <= 500: exact O(N^2) pairwise
- N > 500 with layer_index: vectorized sweep-line (ZERO per-layer Python loops)
- N > 500 without layer_index: vectorized grid-based
- All paths use torch.where (no CPU-GPU sync from .any() checks)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from dagua.layout.layers import LayerIndex
from dagua.utils import _cuda_stage_status_message, _is_cuda_oom_error, _vram_fits

_GPU_PROJECTION_ARGSORT_BYTES_PER_NODE = 72
_GPU_PROJECTION_VRAM_FRACTION = 0.60
_EXACT_PROJECTION_DEFAULT_DAMPING = 0.7
_EXACT_PROJECTION_STAGNATION_WINDOW = 3
# Minimum fractional decrease in total overlap penetration depth for a pass
# to count as "progress" (see `_project_exact`). A single slowly-converging
# pair decays geometrically (~35% per pass at the default damping) and
# should never be flagged stagnant; a genuinely deadlocked dense clique
# decays by a fraction of a percent per pass once it plateaus, well under
# this threshold.
_EXACT_PROJECTION_STAGNATION_MIN_PROGRESS_RATIO = 0.01
# Multiplicative safety margin on top of the theoretical minimum grid spacing
# in `_grid_spread_residual_overlaps`. Without it, adjacent cells are placed
# exactly at the overlap boundary and float32 rounding can leave a few pairs
# just barely re-flagged as overlapping.
_GRID_SPREAD_SAFETY_MARGIN = 1.01


def _copy_layer_index_to_cpu(layer_index: Optional[LayerIndex]) -> Optional[LayerIndex]:
    """Return a CPU copy of ``layer_index`` when one is available.

    Parameters
    ----------
    layer_index : LayerIndex, optional
        Layer structure to copy.

    Returns
    -------
    LayerIndex | None
        CPU-resident copy preserving the original layer metadata.
    """
    if layer_index is None:
        return None
    return LayerIndex(
        node_to_layer=layer_index.node_to_layer.cpu(),
        layer_offsets=layer_index.layer_offsets.cpu(),
        sorted_nodes=layer_index.sorted_nodes.cpu(),
        num_layers=layer_index.num_layers,
    )


def _run_projection_impl(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    layer_index: Optional[LayerIndex],
    convergent: bool = False,
) -> None:
    """Dispatch to the appropriate projection kernel for the current device.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing added between projected node boxes.
    iterations : int
        Maximum number of projection passes.
    layer_index : LayerIndex, optional
        Layer information for layered sweep projection.
    convergent : bool, default=False
        Opt into the convergent exact-path projector (see
        :func:`_project_exact`). Only affects the ``N <= 500`` exact path.

    Returns
    -------
    None
    """
    n = pos.shape[0]
    if n <= 500:
        _project_exact(pos, node_sizes, padding, iterations, convergent=convergent)
    elif layer_index is not None and pos.device.type == "cuda":
        _project_sweep_cuda(pos, node_sizes, padding, iterations, layer_index)
    elif layer_index is not None and n > 100_000_000:
        _project_sweep_streaming(pos, node_sizes, padding, iterations, layer_index)
    elif layer_index is not None:
        _project_sweep(pos, node_sizes, padding, iterations, layer_index)
    else:
        _project_grid(pos, node_sizes, padding, iterations)


def project_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
    iterations: int = 10,
    layer_index: Optional[LayerIndex] = None,
    convergent: bool = False,
) -> torch.Tensor:
    """Push overlapping node bounding boxes apart in-place.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float, default=2.0
        Extra spacing enforced between boxes.
    iterations : int, default=10
        Maximum number of projection passes.
    layer_index : LayerIndex, optional
        Layer metadata used by the sweep-based projector.
    convergent : bool, default=False
        Opt into the convergent exact-path projector (accumulated damped
        pushes + deterministic deadlock re-lay; see :func:`_project_exact`).
        Default ``False`` preserves the legacy per-pass behavior
        bit-for-bit. The r80-S2 sweep showed the convergent trajectory is
        NOT universally better: wiring it into every default call site
        regressed rgg_500 (-5.4) and r79_weighted_hub_spoke_4x18 (-8.0)
        while changing nothing else, so it is exposed opt-in for callers
        with a referee (e.g. the undirected-portfolio challenger cleanup)
        or callers that genuinely face dense overlap cliques.

    Returns
    -------
    torch.Tensor
        The same ``pos`` tensor after in-place overlap resolution.
    """
    n = pos.shape[0]
    if n <= 1:
        return pos

    device = pos.device
    cuda_projection_stage = device.type == "cuda" and layer_index is not None and n > 500
    required_bytes = n * _GPU_PROJECTION_ARGSORT_BYTES_PER_NODE
    free_bytes = 0
    gpu_sweep_active = False
    if cuda_projection_stage:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            free_bytes = 0
        gpu_sweep_active = _vram_fits(required_bytes) and (
            required_bytes <= int(free_bytes * _GPU_PROJECTION_VRAM_FRACTION)
        )
    run_on_cpu = cuda_projection_stage and not gpu_sweep_active

    if run_on_cpu:
        print(
            "[dagua]   "
            + _cuda_stage_status_message(
                "Overlap projection",
                "cpu_fallback",
                required_bytes,
                free_bytes,
            ),
            flush=True,
        )
        pos_cpu = pos.cpu()
        ns_cpu = node_sizes.cpu()
        li_cpu = _copy_layer_index_to_cpu(layer_index)
    else:
        pos_cpu = pos
        ns_cpu = node_sizes
        li_cpu = layer_index

    with torch.no_grad():
        if pos_cpu.device.type == "cuda":
            try:
                if gpu_sweep_active and li_cpu is not None:
                    print(
                        "[dagua]   "
                        + _cuda_stage_status_message(
                            "Overlap projection",
                            "cuda",
                            required_bytes,
                            free_bytes,
                        ),
                        flush=True,
                    )
                _run_projection_impl(
                    pos_cpu, ns_cpu, padding, iterations, li_cpu, convergent=convergent
                )
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                print(
                    f"[dagua]   {_cuda_stage_status_message('Overlap projection', 'oom_fallback')}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pos_cpu = pos.detach().cpu()
                ns_cpu = node_sizes.detach().cpu()
                li_cpu = _copy_layer_index_to_cpu(layer_index)
                _run_projection_impl(
                    pos_cpu, ns_cpu, padding, iterations, li_cpu, convergent=convergent
                )
                pos.data.copy_(pos_cpu.to(device))
                return pos
        else:
            _run_projection_impl(
                pos_cpu, ns_cpu, padding, iterations, li_cpu, convergent=convergent
            )

        if run_on_cpu:
            pos.data.copy_(pos_cpu.to(device))

    return pos


def _grid_spread_residual_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    node_indices: torch.Tensor,
) -> None:
    """Deterministically re-lay a stuck node subset onto a regular grid.

    Damped pairwise pushes are a Jacobi-style simultaneous update: every
    overlapping pair proposes a correction in the same pass, and dense
    cliques can settle into a fixed point where those corrections cancel out
    node-by-node even though pairwise overlaps remain (a well-known failure
    mode of simultaneous, as opposed to sequential, separation forces). This
    escape valve breaks that deadlock deterministically -- no RNG -- by
    placing every node still touching an overlap onto a grid sized from the
    largest box in the set plus padding (times a small float32 safety
    margin), which is provably overlap-free among that set. Nodes outside
    ``node_indices`` are left untouched; the caller's normal iterations
    continue afterward to reconcile the moved subset with the rest of the
    graph.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``. Updated in place for the rows in
        ``node_indices``.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing enforced between boxes.
    node_indices : torch.Tensor
        Indices of the nodes to re-lay, shaped ``[K]``.

    Returns
    -------
    None
        ``pos`` is updated in place.
    """
    k = int(node_indices.shape[0])
    if k <= 1:
        return

    center = pos[node_indices].mean(dim=0)
    n_cols = max(int(math.ceil(math.sqrt(float(k)))), 1)
    n_rows = int(math.ceil(k / n_cols))
    cell_w = (
        float(node_sizes[node_indices, 0].max().item()) + padding
    ) * _GRID_SPREAD_SAFETY_MARGIN
    cell_h = (
        float(node_sizes[node_indices, 1].max().item()) + padding
    ) * _GRID_SPREAD_SAFETY_MARGIN

    rank = torch.arange(k, device=pos.device)
    col_idx = (rank % n_cols).to(pos.dtype)
    row_idx = (rank // n_cols).to(pos.dtype)
    grid_x = (col_idx - (n_cols - 1) / 2.0) * cell_w
    grid_y = (row_idx - (n_rows - 1) / 2.0) * cell_h

    pos[node_indices, 0] = center[0] + grid_x
    pos[node_indices, 1] = center[1] + grid_y


def _project_exact(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    damping: float = _EXACT_PROJECTION_DEFAULT_DAMPING,
    convergent: bool = False,
) -> None:
    """Exact O(N^2) overlap projection for small graphs (dispatcher).

    Selects between two pass implementations:

    - ``convergent=False`` (default): the legacy per-pass update
      (:func:`_project_exact_legacy`), preserved bit-for-bit. Its
      advanced-index ``+=`` drops repeated node indices (last write wins),
      so dense overlap cliques never fully resolve -- but its trajectory is
      what every default pipeline was tuned against, and the r80-S2 sweep
      proved swapping it out globally regresses real graphs (rgg_500 -5.4,
      r79_weighted_hub_spoke_4x18 -8.0).
    - ``convergent=True``: the accumulated, damped, deadlock-escaping
      projector (:func:`_project_exact_convergent`) that provably reaches
      zero overlaps on dense cliques. Opt-in for referee-protected callers
      (undirected-portfolio challenger cleanup) and dense-clique scenarios.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``. Updated in place.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing enforced between boxes.
    iterations : int
        Maximum number of projection passes.
    damping : float, default=0.7
        Convergent-path damping multiplier (ignored by the legacy path).
    convergent : bool, default=False
        Select the convergent implementation.

    Returns
    -------
    None
        ``pos`` is updated in place.
    """
    if convergent:
        _project_exact_convergent(pos, node_sizes, padding, iterations, damping=damping)
    else:
        _project_exact_legacy(pos, node_sizes, padding, iterations)


def _project_exact_legacy(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Legacy exact overlap projection (pre-r80 behavior, bit-for-bit).

    KNOWN LIMITATION (kept deliberately): the advanced-index in-place adds
    (``pos[x_r, 0] += ...``) are last-write-wins for repeated node indices,
    NOT accumulating, so on dense overlap cliques most of the intended
    displacement is dropped and the loop can stall with overlaps remaining
    (P3B2 forensics, ranked fix item 1). Every default pipeline's tuning
    and the r79 benchmark baselines assume THIS trajectory; use
    ``convergent=True`` on the dispatcher to opt into the fixed projector.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``. Updated in place.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing enforced between boxes.
    iterations : int
        Maximum number of projection passes.

    Returns
    -------
    None
        ``pos`` is updated in place.
    """
    n = pos.shape[0]

    for _ in range(iterations):
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

        min_dx = (node_sizes[:, 0].unsqueeze(1) + node_sizes[:, 0].unsqueeze(0)) / 2 + padding
        min_dy = (node_sizes[:, 1].unsqueeze(1) + node_sizes[:, 1].unsqueeze(0)) / 2 + padding

        overlap_x = min_dx - dx.abs()
        overlap_y = min_dy - dy.abs()
        overlapping = (overlap_x > 0) & (overlap_y > 0)
        overlapping.fill_diagonal_(False)

        if not overlapping.any():
            break

        rows, cols = torch.triu_indices(n, n, offset=1, device=pos.device)
        mask = overlapping[rows, cols]

        if not mask.any():
            break

        r = rows[mask]
        c = cols[mask]
        ox = overlap_x[r, c]
        oy = overlap_y[r, c]

        push_x = ox < oy

        if push_x.any():
            x_r = r[push_x]
            x_c = c[push_x]
            x_push = ox[push_x] / 2
            sign = torch.sign(dx[x_r, x_c])
            sign[sign == 0] = 1.0
            pos[x_r, 0] += sign * x_push * 0.5
            pos[x_c, 0] -= sign * x_push * 0.5

        push_y = ~push_x
        if push_y.any():
            y_r = r[push_y]
            y_c = c[push_y]
            y_push = oy[push_y] / 2
            sign = torch.sign(dy[y_r, y_c])
            sign[sign == 0] = 1.0
            pos[y_r, 1] += sign * y_push * 0.5
            pos[y_c, 1] -= sign * y_push * 0.5


def _project_exact_convergent(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    damping: float = _EXACT_PROJECTION_DEFAULT_DAMPING,
) -> None:
    """Convergent exact O(N^2) overlap projection (opt-in).

    Per-pass pushes are accumulated per node with ``index_add_`` over ALL
    overlapping pairs before being applied. Plain advanced-index ``+=``
    (``pos[x_r] += ...``) silently drops repeated indices -- last write wins
    -- so on dense overlap cliques (many pairs sharing a node) most of the
    intended displacement was discarded and the projector never converged.
    ``index_add_`` accumulates every pair's contribution, then the summed
    delta is scaled by ``damping`` to avoid overshoot from nodes touched by
    many pairs at once.

    The pass loop stops when the overlap count hits zero, when the total
    pushed penetration depth (the sum of each overlapping pair's
    to-be-resolved overlap along its push axis -- see below) fails to
    shrink by at least ``_EXACT_PROJECTION_STAGNATION_MIN_PROGRESS_RATIO``
    for ``_EXACT_PROJECTION_STAGNATION_WINDOW`` consecutive passes, or after
    ``iterations`` passes -- whichever comes first.

    Depth, not the raw overlap *count*, is the progress signal: count is a
    poor proxy because a single slowly-converging pair can hold the same
    count for many passes while its actual overlap keeps shrinking
    geometrically every pass (verified against
    ``tests/test_ops_project.py::test_overlap_projection_honors_padding_when_separating_nodes``,
    where a literal count-based check misfired on iteration 3 of a
    perfectly healthy two-node convergence and triggered the grid-spread
    fallback below early). A plain strictly-decreasing check on depth has
    the opposite problem: dense overlap cliques decay geometrically too,
    just at a rate low enough (a fraction of a percent per pass once
    plateaued, versus ~35% per pass for a healthy isolated pair at the
    default damping) that they would still technically be "decreasing" for
    hundreds of passes without ever getting usefully close to zero. The
    relative-progress-ratio threshold below distinguishes the two: a
    genuinely converging pair clears it every pass; a Jacobi-update
    deadlock on a dense clique falls under it within a handful of passes.
    On the first such stagnation (see
    :func:`_grid_spread_residual_overlaps`), the still-overlapping node
    subset is re-laid on a deterministic grid once, then ordinary damped
    passes resume for the remaining budget so any overlaps that re-lay
    introduces against untouched nodes get mopped up.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``. Updated in place.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing enforced between boxes.
    iterations : int
        Maximum number of accumulated projection passes.
    damping : float, default=0.7
        Multiplier applied to the accumulated per-node push each pass.

    Returns
    -------
    None
        ``pos`` is updated in place.
    """
    n = pos.shape[0]
    if n <= 1 or iterations <= 0:
        return

    rows, cols = torch.triu_indices(n, n, offset=1, device=pos.device)
    prev_overlap_depth: Optional[float] = None
    stagnant_passes = 0
    grid_spread_used = False

    for _ in range(iterations):
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

        min_dx = (node_sizes[:, 0].unsqueeze(1) + node_sizes[:, 0].unsqueeze(0)) / 2 + padding
        min_dy = (node_sizes[:, 1].unsqueeze(1) + node_sizes[:, 1].unsqueeze(0)) / 2 + padding

        overlap_x = min_dx - dx.abs()
        overlap_y = min_dy - dy.abs()

        pair_overlap_x = overlap_x[rows, cols]
        pair_overlap_y = overlap_y[rows, cols]
        mask = (pair_overlap_x > 0) & (pair_overlap_y > 0)

        if not bool(mask.any()):
            break

        r = rows[mask]
        c = cols[mask]
        ox = pair_overlap_x[mask]
        oy = pair_overlap_y[mask]
        pair_dx = dx[r, c]
        pair_dy = dy[r, c]

        push_x = ox < oy
        overlap_depth = float(torch.where(push_x, ox, oy).sum().item())

        if prev_overlap_depth is not None:
            progress_ratio = (prev_overlap_depth - overlap_depth) / max(prev_overlap_depth, 1.0e-9)
            stagnant_passes = (
                0
                if progress_ratio >= _EXACT_PROJECTION_STAGNATION_MIN_PROGRESS_RATIO
                else stagnant_passes + 1
            )
        prev_overlap_depth = overlap_depth

        if stagnant_passes >= _EXACT_PROJECTION_STAGNATION_WINDOW and not grid_spread_used:
            involved = torch.zeros(n, dtype=torch.bool, device=pos.device)
            involved[r] = True
            involved[c] = True
            _grid_spread_residual_overlaps(
                pos=pos,
                node_sizes=node_sizes,
                padding=padding,
                node_indices=involved.nonzero(as_tuple=True)[0],
            )
            grid_spread_used = True
            stagnant_passes = 0
            prev_overlap_depth = None
            continue

        if stagnant_passes >= _EXACT_PROJECTION_STAGNATION_WINDOW:
            break

        delta = torch.zeros_like(pos)

        if push_x.any():
            x_r = r[push_x]
            x_c = c[push_x]
            x_push = ox[push_x] / 2
            sign = torch.sign(pair_dx[push_x])
            sign[sign == 0] = 1.0
            values = sign * x_push * 0.5
            delta[:, 0].index_add_(0, x_r, values)
            delta[:, 0].index_add_(0, x_c, -values)

        push_y = ~push_x
        if push_y.any():
            y_r = r[push_y]
            y_c = c[push_y]
            y_push = oy[push_y] / 2
            sign = torch.sign(pair_dy[push_y])
            sign[sign == 0] = 1.0
            values = sign * y_push * 0.5
            delta[:, 1].index_add_(0, y_r, values)
            delta[:, 1].index_add_(0, y_c, -values)

        pos += delta * damping


def _project_sweep(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    layer_index: LayerIndex,
) -> None:
    """Sweep-line overlap projection — ZERO per-layer Python loops.

    Key insight: nodes in the same layer are contiguous in sorted_nodes.
    Sort within each layer by x-coordinate, then check consecutive pairs.
    This is O(N log N) per iteration with no per-layer Python loops.

    Uses composite sort key: layer * BIG_NUMBER + x to get within-layer
    x-ordering via a single global sort.
    """
    n = pos.shape[0]
    device = pos.device
    layers = layer_index.node_to_layer  # [N]
    half_w = node_sizes[:, 0] / 2
    for _ in range(iterations):
        sorted_indices = _layer_sorted_x_indices(pos, layers)

        # Get sorted layer assignments to find same-layer consecutive pairs
        sorted_layers = layers[sorted_indices]  # [N]

        # Consecutive pairs that are in the same layer
        same_layer = sorted_layers[:-1] == sorted_layers[1:]  # [N-1]

        if not same_layer.any():
            break

        # Get the actual node indices for consecutive pairs
        idx_a = sorted_indices[:-1]  # [N-1]
        idx_b = sorted_indices[1:]  # [N-1]

        # Compute overlap for same-layer consecutive pairs
        # (These are already sorted by x, so idx_a is always left of idx_b)
        dx = pos[idx_b, 0] - pos[idx_a, 0]  # positive (b is right of a)
        min_sep_x = half_w[idx_a] + half_w[idx_b] + padding

        overlap_x = min_sep_x - dx  # positive means overlap

        # Only process same-layer pairs with x-overlap
        needs_push = same_layer & (overlap_x > 0)

        if not needs_push.any():
            break

        # Push apart in x (conservative 1/4 factor for stability)
        push_amount = torch.where(needs_push, overlap_x * 0.25, torch.zeros_like(overlap_x))

        # Scatter push amounts to nodes
        # Node a moves left, node b moves right
        push_a = torch.zeros(n, device=device)
        push_b = torch.zeros(n, device=device)
        push_a.scatter_add_(0, idx_a, -push_amount)
        push_b.scatter_add_(0, idx_b, push_amount)

        pos[:, 0] += push_a + push_b

        # Also check near-neighbors (window=2) for wider overlaps.
        # Skip for large graphs — window-1 consecutive resolution is sufficient
        # at scale, and this doubles tensor operations per iteration.
        if n <= 100_000 and n > 2:
            same_layer_2 = sorted_layers[:-2] == sorted_layers[2:]
            idx_a2 = sorted_indices[:-2]
            idx_b2 = sorted_indices[2:]
            dx2 = pos[idx_b2, 0] - pos[idx_a2, 0]
            min_sep_x2 = half_w[idx_a2] + half_w[idx_b2] + padding
            overlap_x2 = min_sep_x2 - dx2
            needs_push2 = same_layer_2 & (overlap_x2 > 0)

            if needs_push2.any():
                push_amount2 = torch.where(
                    needs_push2, overlap_x2 * 0.125, torch.zeros_like(overlap_x2)
                )
                push_a2 = torch.zeros(n, device=device)
                push_b2 = torch.zeros(n, device=device)
                push_a2.scatter_add_(0, idx_a2, -push_amount2)
                push_b2.scatter_add_(0, idx_b2, push_amount2)
                pos[:, 0] += push_a2 + push_b2


def _layer_sorted_x_indices(pos: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
    """Return node indices sorted by ``(layer, x)`` using stable tensor sorts.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    layers : torch.Tensor
        Layer assignment tensor shaped ``[N]``.

    Returns
    -------
    torch.Tensor
        Node indices shaped ``[N]`` ordered first by layer, then by x.
    """
    x_sorted = pos[:, 0].argsort(stable=True)
    layer_sorted = layers[x_sorted].argsort(stable=True)
    return x_sorted[layer_sorted]


def _project_sweep_cuda(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    layer_index: LayerIndex,
) -> None:
    """Run the layered sweep projector with repeated CUDA segmented sorts.

    Parameters
    ----------
    pos : torch.Tensor
        CUDA position tensor shaped ``[N, 2]``.
    node_sizes : torch.Tensor
        CUDA node-size tensor shaped ``[N, 2]``.
    padding : float
        Extra spacing enforced between boxes.
    iterations : int
        Maximum number of projection passes.
    layer_index : LayerIndex
        CUDA-resident layer metadata for segmented sweep ordering.

    Returns
    -------
    None
    """
    n = pos.shape[0]
    device = pos.device
    layers = layer_index.node_to_layer
    half_w = node_sizes[:, 0] / 2

    for _ in range(iterations):
        ordered = _layer_sorted_x_indices(pos, layers)
        ordered_layers = layers[ordered]
        same_layer = ordered_layers[:-1] == ordered_layers[1:]
        if not same_layer.any():
            break

        left = ordered[:-1]
        right = ordered[1:]
        dx = pos[right, 0] - pos[left, 0]
        min_sep = half_w[left] + half_w[right] + padding
        overlap = min_sep - dx
        needs_push = same_layer & (overlap > 0)
        if not needs_push.any():
            break

        delta = torch.zeros(n, dtype=pos.dtype, device=device)
        push = torch.where(needs_push, overlap * 0.25, torch.zeros_like(overlap))
        delta.scatter_add_(0, left, -push)
        delta.scatter_add_(0, right, push)
        pos[:, 0].add_(delta)

        # Match the CPU sweep's second-neighbor pass so overlap chains settle
        # the same way on CUDA.
        if n <= 100_000 and n > 2:
            same_layer_2 = ordered_layers[:-2] == ordered_layers[2:]
            if same_layer_2.any():
                left_2 = ordered[:-2]
                right_2 = ordered[2:]
                dx_2 = pos[right_2, 0] - pos[left_2, 0]
                min_sep_2 = half_w[left_2] + half_w[right_2] + padding
                overlap_2 = min_sep_2 - dx_2
                needs_push_2 = same_layer_2 & (overlap_2 > 0)
                if needs_push_2.any():
                    delta_2 = torch.zeros(n, dtype=pos.dtype, device=device)
                    push_2 = torch.where(
                        needs_push_2,
                        overlap_2 * 0.125,
                        torch.zeros_like(overlap_2),
                    )
                    delta_2.scatter_add_(0, left_2, -push_2)
                    delta_2.scatter_add_(0, right_2, push_2)
                    pos[:, 0].add_(delta_2)


def _project_sweep_streaming(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    layer_index: LayerIndex,
) -> None:
    """Per-layer sweep projection for very large graphs (N > 100M).

    Same algorithm as _project_sweep but processes one layer at a time,
    reducing memory from O(N) to O(max_layer_width). At 1B nodes with
    666K per layer: ~5 MB instead of ~54 GB.
    """
    device = pos.device
    num_layers = layer_index.num_layers
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes
    half_w = node_sizes[:, 0] / 2

    for _ in range(iterations):
        any_push = False
        for layer in range(num_layers):
            lo = int(offsets[layer].item())
            hi = int(offsets[layer + 1].item())
            W = hi - lo
            if W <= 1:
                continue

            layer_nodes = sorted_nodes[lo:hi]  # [W] — view into sorted_nodes

            # Sort by x within this layer
            x_pos = pos[layer_nodes, 0]  # [W]
            order = x_pos.argsort()  # [W]
            ordered = layer_nodes[order]  # [W]

            # Check consecutive pairs (already x-sorted)
            a = ordered[:-1]  # [W-1]
            b = ordered[1:]  # [W-1]

            dx = pos[b, 0] - pos[a, 0]  # [W-1]
            min_sep = half_w[a] + half_w[b] + padding  # [W-1]
            overlap = min_sep - dx  # [W-1]

            mask = overlap > 0
            if not mask.any():
                continue
            any_push = True

            push = overlap.clamp(min=0) * 0.25  # [W-1]

            # Scatter push amounts to local node array, then apply to global pos
            push_neg = torch.zeros(int(W), device=device)  # a moves left
            push_pos = torch.zeros(int(W), device=device)  # b moves right
            local_a = order[:-1]  # local indices into layer_nodes
            local_b = order[1:]
            push_neg.scatter_add_(0, local_a, push)
            push_pos.scatter_add_(0, local_b, push)

            # Apply to global positions via advanced indexing
            delta = push_pos - push_neg  # [W]
            pos[layer_nodes, 0] += delta

        if not any_push:
            break


def _project_grid(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Vectorized grid-based overlap projection (fallback without layer info).

    Grid construction uses tensor sorting instead of Python dicts.
    """
    n = pos.shape[0]
    device = pos.device

    max_w = node_sizes[:, 0].max().item()
    max_h = node_sizes[:, 1].max().item()
    cell_size = max(max_w, max_h) + padding
    if cell_size < 1.0:
        cell_size = 1.0

    half_w = node_sizes[:, 0] / 2
    half_h = node_sizes[:, 1] / 2

    for _ in range(iterations):
        # Assign cells via tensor ops
        cx = torch.floor(pos[:, 0] / cell_size).long()
        cy = torch.floor(pos[:, 1] / cell_size).long()

        cx_min = cx.min()
        cy_min = cy.min()
        cx_rel = cx - cx_min
        cy_rel = cy - cy_min
        cy_range = int(max(cy_rel.max().item() + 1, 1))
        cell_keys = cx_rel * cy_range + cy_rel

        sort_idx = cell_keys.argsort()
        sorted_keys = cell_keys[sort_idx]

        # Find cell boundaries
        changes = torch.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), changes])
        ends = torch.cat([changes, torch.tensor([n], dtype=torch.long, device=device)])
        cell_sizes_arr = ends - starts

        # Process cells with 2+ nodes
        multi_mask = cell_sizes_arr >= 2
        multi_starts = starts[multi_mask]
        multi_ends = ends[multi_mask]

        any_pushed = False
        n_multi = int(multi_starts.shape[0])

        for i in range(min(n_multi, 10000)):
            s = int(multi_starts[i].item())
            e = int(multi_ends[i].item())
            cell_nodes = sort_idx[s:e]
            m = cell_nodes.shape[0]

            if m > 200:
                perm = torch.randperm(m, device=device)[:200]
                cell_nodes = cell_nodes[perm]
                m = 200

            p = pos[cell_nodes]
            hw_c = half_w[cell_nodes]
            hh_c = half_h[cell_nodes]

            dx = p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0)
            dy = p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0)
            min_dx = hw_c.unsqueeze(1) + hw_c.unsqueeze(0) + padding
            min_dy = hh_c.unsqueeze(1) + hh_c.unsqueeze(0) + padding
            ox = min_dx - dx.abs()
            oy = min_dy - dy.abs()
            overlapping = (ox > 0) & (oy > 0)
            overlapping.fill_diagonal_(False)

            if not overlapping.any():
                continue

            any_pushed = True
            rows, cols = torch.triu_indices(m, m, offset=1, device=device)
            mask = overlapping[rows, cols]
            if not mask.any():
                continue

            r = rows[mask]
            c = cols[mask]
            ox_m = ox[r, c]
            oy_m = oy[r, c]

            push_x = ox_m < oy_m
            if push_x.any():
                xr = cell_nodes[r[push_x]]
                xc = cell_nodes[c[push_x]]
                x_amt = ox_m[push_x] / 4
                x_sign = torch.sign(dx[r[push_x], c[push_x]])
                x_sign[x_sign == 0] = 1.0
                pos[:, 0].scatter_add_(0, xr, x_sign * x_amt)
                pos[:, 0].scatter_add_(0, xc, -x_sign * x_amt)

            push_y = ~push_x
            if push_y.any():
                yr = cell_nodes[r[push_y]]
                yc = cell_nodes[c[push_y]]
                y_amt = oy_m[push_y] / 4
                y_sign = torch.sign(dy[r[push_y], c[push_y]])
                y_sign[y_sign == 0] = 1.0
                pos[:, 1].scatter_add_(0, yr, y_sign * y_amt)
                pos[:, 1].scatter_add_(0, yc, -y_sign * y_amt)

        if not any_pushed:
            break
