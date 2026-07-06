"""Evaluate default and multilevel native layouts on deterministic scale ladders."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import psutil
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline
from dagua.layout.ops.pipelines.native_stress_ml import layout_native_stress_ml_pipeline
from dagua.metrics import composite_large, quick

_GRAPH_TYPES = ("sparse_er", "scale_free_ba", "grid_2d")
_RUNGS = (20_000, 100_000, 1_000_000)
_DEFAULT_OUTPUT = Path(".project-context/research/r79_native/r79_scale_ladder.json")


def _path_from_arg(raw: str) -> Path:
    """Convert a CLI path argument into a ``Path``.

    Parameters
    ----------
    raw : str
        Raw command-line value.

    Returns
    -------
    pathlib.Path
        Parsed path.
    """
    return Path(raw)


def _sparse_er_edges(num_nodes: int, seed: int) -> torch.Tensor:
    """Generate a deterministic sparse ER-style graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Torch RNG seed.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]`` and average degree near four.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    edge_count = max(num_nodes * 2, 1)
    sources = torch.randint(0, num_nodes, (edge_count,), generator=generator, dtype=torch.long)
    targets = torch.randint(0, num_nodes, (edge_count,), generator=generator, dtype=torch.long)
    targets = torch.where(targets == sources, (targets + 1) % num_nodes, targets)
    return torch.stack([sources, targets])


def _scale_free_edges(num_nodes: int, seed: int) -> torch.Tensor:
    """Generate a deterministic BA-style scale-free graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Torch RNG seed.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]`` and average degree near four.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nodes = torch.arange(1, num_nodes, dtype=torch.long)
    # Prefer older nodes with probability proportional to 1 / sqrt(rank), a
    # vectorized approximation to preferential attachment that avoids a Python
    # loop at million-node scale.
    first = torch.floor(torch.rand((num_nodes - 1,), generator=generator).pow(2.0) * nodes).long()
    second = torch.floor(torch.rand((num_nodes - 1,), generator=generator).pow(2.0) * nodes).long()
    second = torch.where(second == first, (second + 1).remainder(nodes), second)
    return torch.cat([torch.stack([nodes, first]), torch.stack([nodes, second])], dim=1)


def _grid_edges(num_nodes: int) -> torch.Tensor:
    """Generate a deterministic near-square 2D grid graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    width = int(math.sqrt(num_nodes))
    width = max(width, 1)
    nodes = torch.arange(num_nodes, dtype=torch.long)
    right_mask = (nodes % width != width - 1) & (nodes + 1 < num_nodes)
    down_mask = nodes + width < num_nodes
    right = torch.stack([nodes[right_mask], nodes[right_mask] + 1])
    down = torch.stack([nodes[down_mask], nodes[down_mask] + width])
    return torch.cat([right, down], dim=1)


def generate_graph(graph_type: str, num_nodes: int, seed: int) -> torch.Tensor:
    """Generate one deterministic ladder graph.

    Parameters
    ----------
    graph_type : str
        One of ``sparse_er``, ``scale_free_ba``, or ``grid_2d``.
    num_nodes : int
        Number of nodes.
    seed : int
        RNG seed.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if graph_type == "sparse_er":
        return _sparse_er_edges(num_nodes, seed)
    if graph_type == "scale_free_ba":
        return _scale_free_edges(num_nodes, seed)
    if graph_type == "grid_2d":
        return _grid_edges(num_nodes)
    raise ValueError(f"unknown graph type: {graph_type}")


def _evaluate_positions(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> dict[str, float]:
    """Compute quick metrics and the large-graph composite score.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        Quick metric dictionary plus ``composite_large``.
    """
    # The scale ladder graphs are semantically undirected. Passing a neutral
    # depth vector keeps quick-tier scoring O(N + E) instead of invoking DAG
    # layering on cyclic million-node inputs.
    topo_depth = torch.zeros((pos.shape[0],), dtype=torch.long)
    metrics = quick(pos, edge_index, topo_depth=topo_depth, node_sizes=node_sizes, seed=0)
    metrics["composite_large"] = composite_large(metrics)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _layout_worker(payload: dict[str, Any], queue: mp.Queue) -> None:
    """Run one layout in a child process and report result metadata.

    Parameters
    ----------
    payload : dict[str, Any]
        Engine, graph, and configuration payload.
    queue : multiprocessing.Queue
        Result channel.

    Returns
    -------
    None
        Results are written to ``queue``.
    """
    try:
        edge_index = generate_graph(payload["graph_type"], payload["num_nodes"], payload["seed"])
        node_sizes = torch.ones((payload["num_nodes"], 2), dtype=torch.float32)
        config = LayoutConfig(seed=payload["seed"], device="cpu", steps=payload["steps"])
        if payload["engine"] == "default":
            pos = layout_dagua_native_pipeline(
                edge_index=edge_index,
                num_nodes=payload["num_nodes"],
                node_sizes=node_sizes,
                config=config,
                seed=payload["seed"],
            )
        elif payload["engine"] == "native_stress_ml":
            ml_params = {
                "ml_min_nodes": 5_000,
                "ml_min_edges": 50_000,
                "coarsest_nodes": 1_000,
                "coarse_steps": payload["steps"],
                "refine_steps": max(1, payload["steps"] // 4),
                "overlap_max_nodes": 5_000,
            }
            config.algorithm_params.update(ml_params)
            pos = layout_native_stress_ml_pipeline(
                edge_index=edge_index,
                num_nodes=payload["num_nodes"],
                node_sizes=node_sizes,
                config=config,
                seed=payload["seed"],
            )
        else:
            raise ValueError(f"unsupported engine: {payload['engine']}")
        metrics = _evaluate_positions(pos, edge_index, node_sizes)
        queue.put({"ok": True, "num_edges": int(edge_index.shape[1]), "metrics": metrics})
    except BaseException as exc:
        queue.put({"ok": False, "error": repr(exc)})


def _monitor_child(
    process: mp.Process,
    queue: mp.Queue,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Monitor child wall time and peak RSS.

    Parameters
    ----------
    process : multiprocessing.Process
        Running child process.
    queue : multiprocessing.Queue
        Result queue.
    timeout_seconds : float
        Maximum wall time before terminating the child.

    Returns
    -------
    dict[str, Any]
        Child result augmented with wall time and peak RSS bytes.
    """
    start = time.perf_counter()
    peak_rss = 0
    ps_process: Optional[psutil.Process]
    try:
        ps_process = psutil.Process(process.pid) if process.pid is not None else None
    except psutil.Error:
        ps_process = None
    while process.is_alive():
        if time.perf_counter() - start > timeout_seconds:
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join()
            return {
                "ok": False,
                "error": f"engine timeout after {timeout_seconds:.1f}s",
                "wall_seconds": time.perf_counter() - start,
                "peak_rss_bytes": peak_rss,
                "timeout": True,
            }
        if ps_process is not None:
            try:
                rss = ps_process.memory_info().rss
                for child in ps_process.children(recursive=True):
                    rss += child.memory_info().rss
                peak_rss = max(peak_rss, int(rss))
            except psutil.Error:
                pass
        time.sleep(0.25)
    process.join()
    wall = time.perf_counter() - start
    result = queue.get() if not queue.empty() else {"ok": False, "error": "no child result"}
    result["wall_seconds"] = wall
    result["peak_rss_bytes"] = peak_rss
    return result


def run_engine(
    engine: str,
    graph_type: str,
    num_nodes: int,
    seed: int,
    steps: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one engine and return measured results.

    Parameters
    ----------
    engine : str
        Engine name.
    graph_type : str
        Ladder graph type.
    num_nodes : int
        Number of nodes.
    seed : int
        RNG seed.
    steps : int
        Layout step budget.
    timeout_seconds : float
        Per-engine wall-time timeout.

    Returns
    -------
    dict[str, Any]
        Result dictionary.
    """
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_layout_worker,
        args=(
            {
                "engine": engine,
                "graph_type": graph_type,
                "num_nodes": num_nodes,
                "seed": seed,
                "steps": steps,
            },
            queue,
        ),
    )
    process.start()
    result = _monitor_child(process, queue, timeout_seconds=timeout_seconds)
    result.update({"engine": engine, "graph_type": graph_type, "num_nodes": num_nodes})
    return result


def _write_sfdp_dot(edge_index: torch.Tensor, path: Path) -> None:
    """Write an undirected DOT graph for Graphviz SFDP.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    path : pathlib.Path
        Output DOT path.

    Returns
    -------
    None
        The DOT file is written to disk.
    """
    with path.open("w", encoding="ascii") as handle:
        handle.write("graph G {\n")
        for node in range(int(edge_index.max().item()) + 1 if edge_index.numel() else 0):
            handle.write(f"  {node};\n")
        for source, target in edge_index.t().tolist():
            handle.write(f"  {int(source)} -- {int(target)};\n")
        handle.write("}\n")


def run_sfdp_reference(graph_type: str, num_nodes: int, seed: int) -> dict[str, Any]:
    """Run Graphviz SFDP when available for smaller ladder rungs.

    Parameters
    ----------
    graph_type : str
        Ladder graph type.
    num_nodes : int
        Number of nodes.
    seed : int
        RNG seed.

    Returns
    -------
    dict[str, Any]
        Result or skip record.
    """
    if num_nodes > 100_000 or shutil.which("sfdp") is None:
        return {
            "ok": False,
            "engine": "graphviz_sfdp",
            "graph_type": graph_type,
            "num_nodes": num_nodes,
            "skipped": True,
            "error": "sfdp unavailable or rung too large",
        }
    edge_index = generate_graph(graph_type, num_nodes, seed)
    with tempfile.TemporaryDirectory(prefix="dagua_sfdp_") as tmp:
        dot_path = Path(tmp) / "graph.dot"
        _write_sfdp_dot(edge_index, dot_path)
        start = time.perf_counter()
        proc = subprocess.run(
            ["sfdp", "-Tplain", str(dot_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
            check=False,
        )
    return {
        "ok": proc.returncode == 0,
        "engine": "graphviz_sfdp",
        "graph_type": graph_type,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_bytes": None,
        "error": proc.stderr[-1000:] if proc.returncode != 0 else "",
    }


def _check_heavy_run_memory(num_nodes: int) -> None:
    """Abort 1M-node runs when less than 40 GiB RAM is available.

    Parameters
    ----------
    num_nodes : int
        Active rung node count.

    Returns
    -------
    None
        The function raises when the host is below the requested budget.
    """
    if num_nodes < 1_000_000:
        return
    available = psutil.virtual_memory().available
    minimum = 40 * 1024**3
    if available < minimum:
        raise MemoryError(f"aborting 1M rung: available RAM {available / 1024**3:.1f} GiB < 40 GiB")


def main() -> None:
    """Run the scale ladder and write JSON results.

    Returns
    -------
    None
        Results are printed and written to the output path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=_path_from_arg, default=_DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=max(_RUNGS))
    parser.add_argument("--engine-timeout", type=float, default=900.0)
    parser.add_argument("--graph-types", default=",".join(_GRAPH_TYPES))
    parser.add_argument("--engines", default="default,native_stress_ml")
    parser.add_argument("--include-sfdp", action="store_true")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    graph_types = tuple(item for item in args.graph_types.split(",") if item)
    engines = tuple(item for item in args.engines.split(",") if item)
    for num_nodes in _RUNGS:
        if num_nodes > args.max_nodes:
            continue
        _check_heavy_run_memory(num_nodes)
        for graph_type in graph_types:
            for engine in engines:
                result = run_engine(
                    engine,
                    graph_type,
                    num_nodes,
                    args.seed,
                    args.steps,
                    args.engine_timeout,
                )
                print(json.dumps(result, sort_keys=True), flush=True)
                results.append(result)
            if args.include_sfdp and num_nodes <= 100_000:
                result = run_sfdp_reference(graph_type, num_nodes, args.seed)
                print(json.dumps(result, sort_keys=True), flush=True)
                results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="ascii")


if __name__ == "__main__":
    main()
