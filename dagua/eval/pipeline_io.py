"""Internal evaluation helpers shared by fidelity and quality/runtime pipelines.

This module is internal infrastructure. It is intentionally NOT re-exported
from ``dagua.eval.__all__`` because it is meant for scripts under ``scripts/``
and tests, not public dagua users.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    import h5py


MIN_VALID_NODE_COUNT = 3


def stable_seed(*parts: str) -> int:
    """Build a process-stable 32-bit seed from SHA-256 of string parts.

    Parameters
    ----------
    *parts : str
        Components to hash. Joined with ``"::"`` before hashing.

    Returns
    -------
    int
        Unsigned 32-bit integer derived from the first 8 hex digits of
        ``hashlib.sha256(joined)``. Safe to use in
        ``torch.Generator().manual_seed()``.

    Notes
    -----
    This replaces Python's built-in ``hash()`` for seed derivation.
    Python hashes str/tuple types with a per-process random salt
    (PYTHONHASHSEED), which breaks cross-process reproducibility.
    SHA-256 is process-stable.
    """
    joined = "::".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8], 16)


def validate_positions(
    positions: torch.Tensor,
    expected_nodes: int,
    *,
    min_valid_nodes: int = MIN_VALID_NODE_COUNT,
) -> Optional[str]:
    """Validate a loaded layout tensor. Return a rejection reason or None.

    Parameters
    ----------
    positions : torch.Tensor
        Candidate position tensor.
    expected_nodes : int
        Expected node count for the graph.
    min_valid_nodes : int, optional
        Minimum number of nodes required. Defaults to ``MIN_VALID_NODE_COUNT``.

    Returns
    -------
    str | None
        Canonical rejection reason, or ``None`` when the tensor is valid.

    Rejection reasons (canonical enum, must match existing fidelity behavior):
        - ``"tensor_not_2d"``      -- positions.ndim != 2
        - ``"tensor_not_xy"``      -- positions.shape[1] != 2
        - ``"too_few_nodes"``      -- positions.shape[0] < min_valid_nodes
        - ``"node_count_mismatch"`` -- positions.shape[0] != expected_nodes
        - ``"contains_nan"``       -- torch.isnan(positions).any()
        - ``"contains_inf"``       -- torch.isinf(positions).any()
    """
    if positions.ndim != 2:
        return "tensor_not_2d"
    if positions.shape[1] != 2:
        return "tensor_not_xy"
    if positions.shape[0] < min_valid_nodes:
        return "too_few_nodes"
    if positions.shape[0] != expected_nodes:
        return "node_count_mismatch"
    if torch.isnan(positions).any().item():
        return "contains_nan"
    if torch.isinf(positions).any().item():
        return "contains_inf"
    return None


def load_position_tensor(
    *,
    record_key: Optional[str],
    positions_file: Optional[str],
    input_dir: Path,
    h5_file: Optional["h5py.File"] = None,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Raw position loader. Return ``(tensor, rejection_reason)``.

    Parameters
    ----------
    record_key : str | None
        Stable HDF5 key for this layout (e.g. "graph::engine::seed42").
        Required for HDF5 lookup; when None, the function skips HDF5
        and goes straight to the .pt fallback.
    positions_file : str | None
        Benchmark-relative path to the .pt file, already including the
        ``"positions/"`` prefix (e.g. ``"positions/graph__engine__seed42.pt"``).
        When None, the function returns early with ``"missing_positions_file"``.
    input_dir : Path
        Benchmark root directory. The .pt fallback resolves to
        ``input_dir / positions_file``.
    h5_file : h5py.File | None, optional
        Worker-local HDF5 read handle. When None, the function goes
        straight to the .pt fallback.

    Returns
    -------
    tuple[torch.Tensor | None, str | None]
        Exactly one of the two is None. On success returns
        ``(tensor, None)``. On failure returns ``(None, reason)``.

    Rejection reasons (canonical enum):
        - ``"missing_positions_file"`` -- positions_file is None
        - ``"h5_load_failure"``        -- HDF5 read raised an exception
        - ``"load_failure"``           -- torch.load raised (covers missing .pt files)
        - ``"not_tensor"``             -- loaded object is not a torch.Tensor

    Notes
    -----
    * Shape/NaN/Inf validation is NOT done here. Call
      :func:`validate_positions` separately after loading.
    * When ``record_key`` is missing from the HDF5 store, the function
      silently falls through to the .pt fallback path. This matches the
      existing fidelity loader behavior at scripts/fidelity_analysis.py:793-797.
    * When the HDF5 read raises an exception, the function returns
      ``(None, "h5_load_failure")`` WITHOUT falling back to .pt. This also
      matches existing behavior.
    """
    if positions_file is None:
        return None, "missing_positions_file"

    if h5_file is not None and record_key and record_key in h5_file:
        try:
            arr = h5_file[record_key][:]
            tensor = torch.from_numpy(arr).to(dtype=torch.float32)
            return tensor, None
        except Exception:
            return None, "h5_load_failure"

    pt_path = input_dir / positions_file
    try:
        loaded = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None, "load_failure"
    if not isinstance(loaded, torch.Tensor):
        return None, "not_tensor"
    return loaded.detach().to(dtype=torch.float32, device="cpu"), None


def open_h5_for_worker(h5_path: Path) -> Optional["h5py.File"]:
    """Open an HDF5 file for a worker process in read mode.

    Intended to be called from a ``multiprocessing.Pool`` initializer.
    The returned handle should be stored in a worker-local global and
    passed to :func:`load_position_tensor`.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file.

    Returns
    -------
    h5py.File | None
        An open read handle, or ``None`` when the file does not exist
        so that callers can fall through to the .pt loader.
    """
    if not h5_path.exists():
        return None
    import h5py

    return h5py.File(h5_path, "r")


def aspect_ratio_deviation(positions: torch.Tensor) -> float:
    """Derived metric: ``|log(max(aspect_ratio, eps))|``.

    Raw ``aspect_ratio`` (width / height) has no monotone "better"
    direction around 1.0 -- ratios of 2.0 and 0.5 are equally off.
    The log transform makes this a lower-better metric with a clear
    zero at ratio=1.0.

    Parameters
    ----------
    positions : torch.Tensor
        Shape ``[N, 2]`` position tensor.

    Returns
    -------
    float
        Absolute log-aspect deviation. ``0.0`` when the bounding box is
        square. Returns ``inf`` for degenerate layouts.
    """
    from dagua.metrics import aspect_ratio

    raw = aspect_ratio(positions)
    ratio = float(raw.get("aspect_ratio", 1.0))
    if ratio <= 1e-9:
        return float("inf")
    return abs(math.log(ratio))


def compute_quick_metrics_seeded(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    seed: int,
    metric_filter: Optional[frozenset[str]] = None,
) -> dict[str, float]:
    """Call ``dagua.metrics.quick()`` with a stable seed and filter the output.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    seed : int
        Seed for stochastic quick metrics. Callers should derive this from
        :func:`stable_seed` so the value is process-stable.
    metric_filter : frozenset[str], optional
        Optional allow-list of metric names to retain.

    Returns
    -------
    dict[str, float]
        Numeric metric values keyed by metric name. Non-numeric metadata fields
        such as ``"_compute_time_seconds"`` are dropped.
    """
    from dagua.metrics import quick

    raw = quick(positions, edge_index, node_sizes=node_sizes, seed=seed)
    result: dict[str, float] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, (int, float)):
            continue
        if metric_filter is not None and key not in metric_filter:
            continue
        result[key] = float(value)
    return result


def compute_sampled_metrics_seeded(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    seed: int,
    stress_sources: int = 32,
    stress_targets: int = 128,
    crossing_samples: int = 50_000,
) -> dict[str, float]:
    """Compute sampled Tier-2 metrics with controlled budgets and seeding.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    seed : int
        Seed for stochastic sampled metrics.
    stress_sources : int, optional
        Maximum number of BFS sources for ``sampled_stress``.
    stress_targets : int, optional
        Maximum number of reachable targets per source for ``sampled_stress``.
    crossing_samples : int, optional
        Maximum number of sampled edge pairs for ``sampled_crossing_rate``.

    Returns
    -------
    dict[str, float]
        Numeric sampled metrics keyed by metric name.

    Notes
    -----
    ``sampled_stress`` is already deterministic. The seed affects
    ``sampled_crossing_rate`` and the sampled large-graph branch inside
    ``count_crossings``.
    """
    from dagua.metrics import count_crossings, sampled_crossing_rate, sampled_stress

    result: dict[str, float] = {}
    stress_out = sampled_stress(
        positions,
        edge_index,
        num_nodes=num_nodes,
        n_sources=stress_sources,
        n_targets=stress_targets,
    )
    result.update(
        {key: float(value) for key, value in stress_out.items() if isinstance(value, (int, float))}
    )

    cross_out = sampled_crossing_rate(
        positions,
        edge_index,
        n_samples=crossing_samples,
        seed=seed,
    )
    result.update(
        {key: float(value) for key, value in cross_out.items() if isinstance(value, (int, float))}
    )

    try:
        result["edge_crossings"] = float(count_crossings(positions, edge_index, seed=seed))
    except Exception:
        result["edge_crossings"] = float("nan")

    return result
