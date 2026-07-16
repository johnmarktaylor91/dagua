"""Bounded W5 differentiable finisher for native layout candidates."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_surrogates import (
    angular_resolution_loss,
    barrier_floor_loss,
    crossing_angle_loss,
    depth_order_score_surrogate,
    edge_length_cv_loss,
    gabriel_intrusion_loss,
    overlap_hinge_loss,
    path_continuity_loss,
    signed_flow_score_surrogate,
    soft_crossing_loss,
    soft_knn_neighborhood_loss,
)

_LOGGER = logging.getLogger(__name__)
_ABSOLUTE_DEADLINE_RESERVE_S = 5.0
_FINISHER_SCORE_RESERVE_S = 2.0
_DEFAULT_FINISHER_SLICE_S = 4.0
_MIN_FINISHER_ENTRY_S = 1.0
_W5_ACCEPT_MARGIN = 0.05


@dataclass(frozen=True)
class W5Seed:
    """Warm-start position for the W5 finisher.

    Parameters
    ----------
    name : str
        Stable seed family label.
    pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    """

    name: str
    pos: torch.Tensor


@dataclass(frozen=True)
class W5Checkpoint:
    """Honest-scored W5 checkpoint telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Finisher mode.
    step : int
        Optimization step represented by the checkpoint.
    surrogate_delta : float
        Start loss minus checkpoint loss, so positive means the surrogate improved.
    honest_delta : float
        Honest score minus the incumbent/current-winner score.
    honest_score : float
        Frozen-ruler score for the checkpoint.
    accepted : bool
        Whether this checkpoint cleared the honest W5 accept margin.
    """

    seed: str
    mode: str
    step: int
    surrogate_delta: float
    honest_delta: float
    honest_score: float
    accepted: bool


@dataclass(frozen=True)
class W5Candidate:
    """Accepted W5 candidate.

    Parameters
    ----------
    name : str
        Candidate label.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    score : float
        Honest frozen-ruler score.
    mode : str
        Finisher mode that produced the candidate.
    """

    name: str
    pos: torch.Tensor
    score: float
    mode: str


@dataclass(frozen=True)
class W5FinisherResult:
    """Result of one W5 finisher invocation.

    Parameters
    ----------
    candidates : tuple[W5Candidate, ...]
        Accepted W5 candidates.
    checkpoints : tuple[W5Checkpoint, ...]
        Honest-scored checkpoints, accepted or rejected.
    mode : str
        Routed mode, or ``"skip"`` when no seed ran.
    steps : int
        Total optimizer steps completed across seeds.
    skipped_reason : str, optional
        Reason the finisher skipped all work.
    """

    candidates: tuple[W5Candidate, ...]
    checkpoints: tuple[W5Checkpoint, ...]
    mode: str
    steps: int
    skipped_reason: Optional[str] = None


def _remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return remaining benchmark seconds when a deadline is known.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Remaining seconds, or ``None`` outside benchmark deadline mode.
    """
    deadline = getattr(config, "_dagua_native_deadline_s", None) if config is not None else None
    if deadline is None:
        return None
    return float(deadline) - time.perf_counter()


def _finisher_slice_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return the bounded W5 work slice, or ``None`` when it must skip.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Available finisher seconds before the return reserve.
    """
    remaining = _remaining_s(config)
    if remaining is None:
        return _DEFAULT_FINISHER_SLICE_S
    available = remaining - _ABSOLUTE_DEADLINE_RESERVE_S
    if available < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    total_budget = float(getattr(config, "_dagua_native_total_budget_s", remaining))
    return max(0.0, min(0.35 * total_budget, available))


def _dedupe_seeds(seeds: Sequence[W5Seed], max_seeds: int = 3) -> list[W5Seed]:
    """Return finite, non-duplicate W5 seeds.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Candidate warm starts.
    max_seeds : int, default=3
        Maximum seeds to keep.

    Returns
    -------
    list[W5Seed]
        Deduplicated finite seeds.
    """
    kept: list[W5Seed] = []
    for seed in seeds:
        if len(kept) >= max_seeds or not bool(torch.isfinite(seed.pos).all().item()):
            continue
        if any(
            torch.allclose(seed.pos, existing.pos, atol=1.0e-5, rtol=1.0e-5) for existing in kept
        ):
            continue
        kept.append(seed)
    return kept


def _longest_path_depth(
    edge_index: torch.Tensor,
    node_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return longest-path depths, falling back to zeros on cyclic inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_count : int
        Number of nodes.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Depth tensor with shape ``[N]``.
    """
    try:
        from dagua.utils import longest_path_layering

        depth = longest_path_layering(edge_index.detach().to(device="cpu"), node_count)
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth, dtype=torch.long)
        return depth.to(device=device, dtype=torch.long)
    except Exception:  # noqa: BLE001 -- cyclic semantic graphs use flow-only barriers
        return torch.zeros(node_count, dtype=torch.long, device=device)


def _route_mode(
    seed_pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth: torch.Tensor,
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
) -> str:
    """Choose the W5 descent mode from structural/local incumbent signals.

    Parameters
    ----------
    seed_pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler treats the graph as hierarchical.

    Returns
    -------
    str
        One of ``"x_only"``, ``"barrier_2d"``, or ``"undirected_2d_sampled"``.
    """
    if not is_semantically_directed:
        return "undirected_2d_sampled"
    if not declared_hierarchical:
        return "barrier_2d"
    flow = float(signed_flow_score_surrogate(seed_pos, edge_index).detach().item())
    depth = float(depth_order_score_surrogate(seed_pos, topo_depth).detach().item())
    node_count = int(seed_pos.shape[0])
    if flow >= 0.95 and depth >= 0.95 and (node_count <= 64 or node_count >= 250):
        return "x_only"
    return "barrier_2d"


def _overlap_count(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Count exact overlapping boxes for W5 regression rejection.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    int
        Number of overlapping unordered node pairs.
    """
    if pos.shape[0] <= 1:
        return 0
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    work_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    dx = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (work_sizes[:, None, 0] + work_sizes[None, :, 0]) * 0.5
    min_dy = (work_sizes[:, None, 1] + work_sizes[None, :, 1]) * 0.5
    overlap = (dx < min_dx) & (dy < min_dy)
    overlap.fill_diagonal_(False)
    return int(overlap.triu(diagonal=1).sum().item())


def _is_degenerate(pos: torch.Tensor, node_sizes: torch.Tensor) -> bool:
    """Return whether positions are non-finite or collapsed.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when the candidate is not safe to score.
    """
    if not bool(torch.isfinite(pos).all().item()):
        return True
    if pos.shape[0] <= 1:
        return False
    extent = pos.detach().amax(dim=0) - pos.detach().amin(dim=0)
    min_extent = float(node_sizes.detach().to(dtype=torch.float32).mean().item()) * 0.1
    return float(extent.max().item()) <= max(1.0e-6, min_extent)


def _surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
) -> torch.Tensor:
    """Evaluate the composite-weighted W5 surrogate objective.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    floors : dict[str, float]
        Incumbent floor values for barrier terms.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize.
    """
    crossing = soft_crossing_loss(pos, edge_index)
    flow_score = signed_flow_score_surrogate(pos, edge_index)
    depth_score = depth_order_score_surrogate(pos, topo_depth)
    common_scale = 0.75 if mode in {"x_only", "barrier_2d"} else 1.0
    loss = (
        common_scale * 20.0 * crossing
        + common_scale * 13.0 * overlap_hinge_loss(pos, node_sizes)
        + common_scale * 12.0 * soft_knn_neighborhood_loss(pos, edge_index)
        + common_scale * 7.0 * edge_length_cv_loss(pos, edge_index)
        + common_scale * 5.0 * gabriel_intrusion_loss(pos, edge_index)
        + common_scale * 5.0 * crossing_angle_loss(pos, edge_index)
        + common_scale * 4.0 * angular_resolution_loss(pos, edge_index)
        + common_scale * 4.0 * path_continuity_loss(pos, edge_index)
    )
    if mode in {"x_only", "barrier_2d"}:
        loss = loss + 16.0 * (1.0 - flow_score) + 9.0 * (1.0 - depth_score)
    if mode == "barrier_2d":
        loss = (
            loss
            + 64.0 * barrier_floor_loss(flow_score, floors.get("flow"))
            + 36.0 * barrier_floor_loss(depth_score, floors.get("depth"))
            + 20.0 * torch.relu(crossing - floors.get("crossing_loss", crossing.detach())).square()
        )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _optimize_seed(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
    """Run one bounded W5 descent from ``seed``.

    Parameters
    ----------
    seed : W5Seed
        Warm start.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    deadline : float
        Absolute ``time.monotonic()`` deadline for optimizer work.

    Returns
    -------
    tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]
        Final positions, completed steps, start loss, and checkpoint
        positions/losses.
    """
    work = seed.pos.detach().clone().to(dtype=torch.float32)
    work.requires_grad_(True)
    start_y = work[:, 1].detach().clone()
    floors = {
        "flow": float(signed_flow_score_surrogate(work, edge_index).detach().item()),
        "depth": float(depth_order_score_surrogate(work, topo_depth).detach().item()),
        "crossing_loss": float(soft_crossing_loss(work, edge_index).detach().item()),
    }
    start_loss_tensor = _surrogate_loss(work, edge_index, node_sizes, topo_depth, mode, floors)
    start_loss = float(start_loss_tensor.detach().item())
    median_size = float(node_sizes.detach().to(dtype=torch.float32).mean().item())
    lr = max(0.01, min(4.0, 0.04 * median_size))
    optimizer = torch.optim.Adam([work], lr=lr)
    node_count = int(work.shape[0])
    max_steps = 24 if node_count >= 300 else 36
    checkpoints: list[tuple[int, torch.Tensor, float]] = []
    checkpoint_steps = {max(1, max_steps // 2), max_steps}
    completed_steps = 0
    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = _surrogate_loss(work, edge_index, node_sizes, topo_depth, mode, floors)
        if not bool(torch.isfinite(loss).all().item()):
            break
        loss.backward()
        optimizer.step()
        if mode == "x_only":
            with torch.no_grad():
                work[:, 1] = start_y
        completed_steps = step
        if step in checkpoint_steps:
            checkpoint_pos = work.detach().clone()
            checkpoint_loss = float(
                _surrogate_loss(checkpoint_pos, edge_index, node_sizes, topo_depth, mode, floors)
                .detach()
                .item()
            )
            checkpoints.append((step, checkpoint_pos, checkpoint_loss))
    return work.detach(), completed_steps, start_loss, checkpoints


def run_w5_finisher(
    *,
    seeds: Sequence[W5Seed],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], float],
    current_best_score: float,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    config: Optional[LayoutConfig] = None,
    accept_margin: float = _W5_ACCEPT_MARGIN,
) -> W5FinisherResult:
    """Run the W5 finisher and return honest-accepted candidates only.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Warm starts already generated by the native contest/polish path.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], float]
        Frozen honest scorer.
    current_best_score : float
        Honest score of the incumbent/current winner to beat.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler uses directed hierarchy terms.
    config : LayoutConfig, optional
        Prepared native configuration carrying optional benchmark deadline.
    accept_margin : float, default=0.05
        Required honest-score improvement over ``current_best_score``.

    Returns
    -------
    W5FinisherResult
        Accepted candidates plus telemetry for all honest-scored checkpoints.
    """
    slice_s = _finisher_slice_s(config)
    if slice_s is None:
        return W5FinisherResult((), (), "skip", 0, "no_budget")
    kept_seeds = _dedupe_seeds(seeds)
    if not kept_seeds:
        return W5FinisherResult((), (), "skip", 0, "no_finite_seed")
    deadline = time.monotonic() + slice_s
    edge_work = edge_index.detach().to(device=kept_seeds[0].pos.device, dtype=torch.long)
    size_work = node_sizes.detach().to(device=kept_seeds[0].pos.device, dtype=torch.float32)
    topo_depth = _longest_path_depth(
        edge_work,
        int(kept_seeds[0].pos.shape[0]),
        kept_seeds[0].pos.device,
    )
    accepted: list[W5Candidate] = []
    checkpoints: list[W5Checkpoint] = []
    steps_total = 0
    routed_mode = "skip"
    incumbent_overlap = _overlap_count(kept_seeds[0].pos, size_work)
    for seed in kept_seeds:
        if time.monotonic() >= deadline - _FINISHER_SCORE_RESERVE_S:
            break
        mode = _route_mode(
            seed.pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
        )
        routed_mode = mode
        try:
            final_pos, steps, start_loss, scored_points = _optimize_seed(
                seed,
                edge_work,
                size_work,
                topo_depth,
                mode,
                deadline - _FINISHER_SCORE_RESERVE_S,
            )
        except Exception:  # noqa: BLE001 -- W5 is optional candidate generation
            _LOGGER.warning("W5 finisher seed %s failed", seed.name, exc_info=True)
            continue
        steps_total += steps
        if not scored_points or scored_points[-1][0] != steps:
            final_loss = float(
                _surrogate_loss(final_pos, edge_work, size_work, topo_depth, mode, {})
                .detach()
                .item()
            )
            scored_points.append((steps, final_pos, final_loss))
        for step, checkpoint_pos, checkpoint_loss in scored_points[:2]:
            if time.monotonic() >= deadline:
                break
            viable = not _is_degenerate(checkpoint_pos, size_work) and (
                _overlap_count(checkpoint_pos, size_work) <= incumbent_overlap
            )
            if not viable:
                continue
            try:
                score_pos = checkpoint_pos.to(device=edge_index.device, dtype=torch.float32)
                honest = float(score_fn(score_pos))
            except Exception:
                continue
            if not math.isfinite(honest):
                continue
            honest_delta = honest - current_best_score
            surrogate_delta = start_loss - checkpoint_loss
            is_accepted = honest_delta > float(accept_margin)
            checkpoints.append(
                W5Checkpoint(
                    seed=seed.name,
                    mode=mode,
                    step=int(step),
                    surrogate_delta=float(surrogate_delta),
                    honest_delta=float(honest_delta),
                    honest_score=float(honest),
                    accepted=is_accepted,
                )
            )
            if is_accepted:
                name = f"w5_{mode}_{seed.name}_{step}"
                accepted.append(
                    W5Candidate(
                        name=name,
                        pos=checkpoint_pos.to(device=seed.pos.device, dtype=seed.pos.dtype),
                        score=float(honest),
                        mode=mode,
                    )
                )
    skipped = None if checkpoints or accepted else "no_checkpoint_improved"
    return W5FinisherResult(tuple(accepted), tuple(checkpoints), routed_mode, steps_total, skipped)


def log_w5_telemetry(result: W5FinisherResult, config: Optional[LayoutConfig]) -> None:
    """Emit and attach structured W5 finisher telemetry.

    Parameters
    ----------
    result : W5FinisherResult
        Finisher result to report.
    config : LayoutConfig, optional
        Config receiving ``_dagua_native_w5_telemetry`` when available.

    Returns
    -------
    None
        Telemetry is logged and optionally stored on ``config``.
    """
    payload = {
        "event": "native_w5_finisher",
        "mode": result.mode,
        "steps": result.steps,
        "skipped_reason": result.skipped_reason,
        "accepted": [candidate.name for candidate in result.candidates],
        "checkpoints": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "step": checkpoint.step,
                "surrogate_delta": checkpoint.surrogate_delta,
                "honest_delta": checkpoint.honest_delta,
                "honest_score": checkpoint.honest_score,
                "accepted": checkpoint.accepted,
            }
            for checkpoint in result.checkpoints
        ],
    }
    if config is not None:
        existing = list(getattr(config, "_dagua_native_w5_telemetry", []))
        existing.append(payload)
        setattr(config, "_dagua_native_w5_telemetry", existing)
    _LOGGER.info("Native W5 finisher telemetry %s", json.dumps(payload, sort_keys=True))


__all__ = [
    "W5Candidate",
    "W5Checkpoint",
    "W5FinisherResult",
    "W5Seed",
    "log_w5_telemetry",
    "run_w5_finisher",
]
