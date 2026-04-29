#!/usr/bin/env python3
"""Render raw side-by-side layout panels for algo fidelity inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.graphs import get_test_graphs  # noqa: E402
from dagua.eval.pipeline_io import load_position_tensor, validate_positions  # noqa: E402


def normalize_for_panel(
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Align the candidate and target layouts into the target display frame.

    Parameters
    ----------
    candidate : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    target : torch.Tensor
        Target positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, float]
        Aligned candidate positions, normalized target positions, and RMSD.
    """
    candidate_centered = candidate - candidate.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    candidate_norm = float(candidate_centered.norm().item())
    target_norm = float(target_centered.norm().item())
    if candidate_norm > 0.0:
        candidate_centered = candidate_centered / candidate_norm
    if target_norm > 0.0:
        target_centered = target_centered / target_norm

    covariance = candidate_centered.t() @ target_centered
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    det_value = torch.det(left_singular @ right_singular_t)
    correction = torch.diag(
        torch.tensor([1.0, float(torch.sign(det_value).item())], dtype=candidate_centered.dtype)
    )
    rotation = left_singular @ correction @ right_singular_t
    aligned = candidate_centered @ rotation
    rmsd = float(torch.sqrt(((aligned - target_centered).square()).sum(dim=1).mean()).item())

    reflected_rotation = left_singular @ right_singular_t
    reflected_aligned = candidate_centered @ reflected_rotation
    reflected_rmsd = float(
        torch.sqrt(((reflected_aligned - target_centered).square()).sum(dim=1).mean()).item()
    )
    if reflected_rmsd < rmsd:
        return reflected_aligned, target_centered, reflected_rmsd
    return aligned, target_centered, rmsd


def load_results(input_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Load benchmark results from disk.

    Parameters
    ----------
    input_dir : Path
        Benchmark root containing ``results.json``.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Raw result payload keyed by record key.
    """
    raw = json.loads((input_dir / "results.json").read_text())
    if not isinstance(raw, dict):
        raise ValueError("results.json must contain an object")
    return {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}


def select_record(
    results: Mapping[str, Mapping[str, Any]],
    graph: str,
    engine: str,
) -> tuple[str, Mapping[str, Any]]:
    """Select the canonical successful result for a graph/engine.

    Parameters
    ----------
    results : Mapping[str, Mapping[str, Any]]
        Raw benchmark results.
    graph : str
        Benchmark graph name.
    engine : str
        Engine name.

    Returns
    -------
    tuple[str, Mapping[str, Any]]
        Result key and payload.
    """
    matches = [
        (key, payload)
        for key, payload in results.items()
        if payload.get("graph_name") == graph
        and payload.get("engine_name") == engine
        and payload.get("status") == "ok"
    ]
    if not matches:
        raise ValueError(f"No successful result for {graph}/{engine}")
    matches.sort(
        key=lambda item: (
            -1 if item[1].get("seed") is None else int(item[1].get("seed")),
            item[0],
        )
    )
    return matches[0]


def load_positions(
    input_dir: Path,
    key: str,
    payload: Mapping[str, Any],
) -> torch.Tensor:
    """Load and validate one result tensor.

    Parameters
    ----------
    input_dir : Path
        Benchmark root directory.
    key : str
        Result key.
    payload : Mapping[str, Any]
        Raw result payload.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions_file = payload.get("positions_file")
    positions, error = load_position_tensor(
        record_key=key,
        positions_file=str(positions_file) if positions_file is not None else None,
        input_dir=input_dir,
    )
    if error is not None or positions is None:
        raise ValueError(f"Could not load positions for {key}: {error}")
    validation_error = validate_positions(positions, int(payload.get("num_nodes") or 0))
    if validation_error is not None:
        raise ValueError(f"Invalid positions for {key}: {validation_error}")
    return positions


def edge_index_for_graph(graph_name: str) -> Optional[torch.Tensor]:
    """Return edge index for a benchmark graph when available.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.

    Returns
    -------
    torch.Tensor | None
        Edge index with shape ``[2, E]``, or ``None`` when the graph is absent
        from the local evaluation registry.
    """
    for test_graph in get_test_graphs():
        if test_graph.name == graph_name:
            return test_graph.graph.edge_index.detach().cpu()
    return None


def draw_layout(
    axis: Any,
    positions: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    title: str,
) -> None:
    """Draw one raw layout panel.

    Parameters
    ----------
    axis : Any
        Matplotlib axis.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor | None
        Optional edge tensor with shape ``[2, E]``.
    title : str
        Axis title.
    """
    xy = positions.detach().cpu()
    if edge_index is not None and edge_index.numel() > 0:
        for edge_idx in range(int(edge_index.shape[1])):
            src = int(edge_index[0, edge_idx].item())
            dst = int(edge_index[1, edge_idx].item())
            if src >= xy.shape[0] or dst >= xy.shape[0]:
                continue
            axis.plot(
                [float(xy[src, 0]), float(xy[dst, 0])],
                [float(xy[src, 1]), float(xy[dst, 1])],
                color="#8a8f98",
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )
    axis.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(), s=18, color="#1f77b4", zorder=2)
    axis.set_title(title, fontsize=10)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def render_panel(
    graph: str,
    dagua_engine: str,
    target_engine: str,
    output: Path,
    input_dir: Path,
) -> None:
    """Render one side-by-side panel PNG.

    Parameters
    ----------
    graph : str
        Benchmark graph name.
    dagua_engine : str
        Dagua engine name.
    target_engine : str
        Target engine name.
    output : Path
        Destination PNG path.
    input_dir : Path
        Benchmark root directory.
    """
    results = load_results(input_dir)
    dagua_key, dagua_payload = select_record(results, graph, dagua_engine)
    target_key, target_payload = select_record(results, graph, target_engine)
    dagua_positions = load_positions(input_dir, dagua_key, dagua_payload)
    target_positions = load_positions(input_dir, target_key, target_payload)
    if dagua_positions.shape[0] != target_positions.shape[0]:
        raise ValueError("Position tensors have different node counts")

    aligned_dagua, aligned_target, rmsd = normalize_for_panel(dagua_positions, target_positions)
    edge_index = edge_index_for_graph(graph)
    all_positions = torch.cat([aligned_dagua, aligned_target], dim=0)
    min_xy = all_positions.min(dim=0).values
    max_xy = all_positions.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=1e-6)
    margin = 0.08 * float(span.max().item())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
    draw_layout(axes[0], aligned_dagua, edge_index, dagua_engine)
    draw_layout(axes[1], aligned_target, edge_index, target_engine)
    for axis in axes:
        axis.set_xlim(float(min_xy[0]) - margin, float(max_xy[0]) + margin)
        axis.set_ylim(float(min_xy[1]) - margin, float(max_xy[1]) + margin)
    fig.suptitle(f"{graph} | RMSD {rmsd:.4f}", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph")
    parser.add_argument("dagua_engine")
    parser.add_argument("target_engine")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=Path("eval_output/benchmark_full"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the panel CLI.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    output = args.output
    if output is None:
        safe_graph = args.graph.replace("/", "_")
        output = Path(f"{safe_graph}__{args.dagua_engine}__{args.target_engine}.png")
    render_panel(args.graph, args.dagua_engine, args.target_engine, output, args.input_dir)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
