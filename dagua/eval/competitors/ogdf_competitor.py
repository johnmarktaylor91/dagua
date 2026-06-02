"""OGDF competitor adapters backed by the standalone runner subprocess."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
OGDF_RUNNER = _REPO_ROOT / "scripts" / "ogdf_runner"


def _resolve_ogdf_runner() -> Optional[str]:
    """Resolve the OGDF runner executable path.

    Returns
    -------
    str | None
        Absolute path to the compiled runner, or ``None`` when it is not
        available either in the repository or on ``PATH``.
    """
    if OGDF_RUNNER.exists():
        return str(OGDF_RUNNER)
    return shutil.which("ogdf_runner")


def _ogdf_available() -> bool:
    """Report whether the standalone OGDF runner is available.

    Returns
    -------
    bool
        ``True`` when the compiled runner can be resolved.
    """
    return _resolve_ogdf_runner() is not None


def _graph_edges(graph: DaguaGraph) -> list[list[int]]:
    """Convert ``edge_index`` into a JSON-serializable edge list.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose edges should be exported.

    Returns
    -------
    list[list[int]]
        Edge list shaped like ``[[source, target], ...]``.
    """
    edge_index = graph.edge_index.cpu().numpy()
    return [
        [int(edge_index[0, idx]), int(edge_index[1, idx])] for idx in range(edge_index.shape[1])
    ]


def _is_connected_graph(graph: DaguaGraph) -> bool:
    """Return whether the undirected view of ``graph`` is connected.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to test.

    Returns
    -------
    bool
        ``True`` when every node is reachable from node zero.
    """
    if graph.num_nodes <= 1:
        return True
    if graph.edge_index.numel() == 0:
        return False

    adjacency: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for source, target in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()):
        adjacency[source].append(target)
        adjacency[target].append(source)

    visited = [False] * graph.num_nodes
    queue = [0]
    visited[0] = True
    head = 0
    while head < len(queue):
        node_idx = queue[head]
        head += 1
        for neighbor_idx in adjacency[node_idx]:
            if visited[neighbor_idx]:
                continue
            visited[neighbor_idx] = True
            queue.append(neighbor_idx)
    return all(visited)


def _run_ogdf(
    graph: DaguaGraph,
    algorithm: str,
    timeout: float,
    seed: Optional[int] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Run an OGDF algorithm through the standalone subprocess wrapper.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    algorithm : str
        OGDF algorithm selector understood by the C++ runner.
    timeout : float
        Wall-clock timeout in seconds for the subprocess.
    seed : int | None, optional
        Seed forwarded to the runner. When omitted, the runner preserves its
        historical default seed.
    options : Mapping[str, Any], optional
        Optional algorithm-specific JSON options forwarded to the runner.

    Returns
    -------
    torch.Tensor
        Position tensor shaped ``[N, 2]`` on CPU.

    Raises
    ------
    FileNotFoundError
        If the OGDF runner binary is not available.
    RuntimeError
        If the runner exits with a non-zero status or emits invalid JSON.
    subprocess.TimeoutExpired
        If the subprocess exceeds the timeout budget.
    """
    if graph.num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    runner = _resolve_ogdf_runner()
    if runner is None:
        raise FileNotFoundError("OGDF runner binary not found")

    payload_data: dict[str, Any] = {
        "nodes": graph.num_nodes,
        "edges": _graph_edges(graph),
        "algorithm": algorithm,
    }
    if seed is not None:
        payload_data["seed"] = int(seed)
    if options is not None:
        payload_data.update(dict(options))
    payload = json.dumps(payload_data)
    result = subprocess.run(
        [runner],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"OGDF failed: {stderr or 'unknown error'}")

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OGDF returned invalid JSON: {exc}") from exc

    positions = output.get("positions")
    if not isinstance(positions, list):
        raise RuntimeError("OGDF output missing positions")
    pos = torch.tensor(positions, dtype=torch.float32)
    if pos.shape != (graph.num_nodes, 2):
        raise RuntimeError(
            "OGDF returned invalid position shape: "
            f"expected {(graph.num_nodes, 2)}, got {tuple(pos.shape)}"
        )
    return pos


class _OGDFBase(CompetitorBase):
    """Base class for OGDF subprocess-backed adapters."""

    algorithm: str = ""

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured OGDF algorithm through the helper binary.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Wall-clock timeout budget for the subprocess.
        seed : int | None, default=None
            Optional seed forwarded to the helper binary.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if execution fails.
        """
        start = time.perf_counter()
        try:
            if self.algorithm == "pivot_mds" and not _is_connected_graph(graph):
                elapsed = time.perf_counter() - start
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error="requires connected graph",
                )
            pos = _run_ogdf(graph, self.algorithm, timeout, seed=seed)
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report whether the OGDF helper binary is available.

        Returns
        -------
        bool
            ``True`` when the compiled runner exists locally or on ``PATH``.
        """
        return _ogdf_available()

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run layout with optional OGDF runner parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Wall-clock timeout budget for the subprocess.
        seed : int | None, default=None
            Optional seed forwarded to the helper binary.
        variant_params : Mapping[str, Any] | None, default=None
            Optional runner parameters. ``ogdf_gem`` accepts ``rounds`` or
            ``max_iters``. ``ogdf_fmmm`` accepts ``fixed_iterations`` or
            ``steps``. ``ogdf_stress`` accepts ``iterations``. ``ogdf_pivot_mds``
            accepts ``n_pivots``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if execution fails.
        """
        options: dict[str, Any] = {}
        if variant_params is not None:
            params = dict(variant_params)
            if self.algorithm == "gem":
                rounds = params.get("rounds", params.get("max_iters"))
                if rounds is not None:
                    options["gemRounds"] = int(rounds)
            if self.algorithm == "fmmm":
                fixed_iterations = params.get("fixed_iterations", params.get("steps"))
                if fixed_iterations is not None:
                    options["fmmmFixedIterations"] = int(fixed_iterations)
            if self.algorithm == "stress":
                iterations = params.get("iterations", params.get("steps"))
                if iterations is not None:
                    options["iterations"] = int(iterations)
            if self.algorithm == "pivot_mds":
                n_pivots = params.get("n_pivots")
                if n_pivots is not None:
                    options["numberOfPivots"] = int(n_pivots)

        start = time.perf_counter()
        try:
            if self.algorithm == "pivot_mds" and not _is_connected_graph(graph):
                elapsed = time.perf_counter() - start
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error="requires connected graph",
                )
            pos = _run_ogdf(
                graph=graph,
                algorithm=self.algorithm,
                timeout=timeout,
                seed=seed,
                options=options or None,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class OGDFGem(_OGDFBase):
    """Competitor adapter for OGDF's GEM layout."""

    name = "ogdf_gem"
    algorithm = "gem"
    max_nodes = 20_000
    variant_param_names = frozenset({"max_iters", "rounds"})


@register
class OGDFFMMM(_OGDFBase):
    """Competitor adapter for OGDF's FM^3 layout."""

    name = "ogdf_fmmm"
    algorithm = "fmmm"
    max_nodes = 100_000
    variant_param_names = frozenset({"fixed_iterations", "steps"})


@register
class OGDFStress(_OGDFBase):
    """Competitor adapter for OGDF's stress minimization layout."""

    name = "ogdf_stress"
    algorithm = "stress"
    max_nodes = 10_000
    variant_param_names = frozenset({"iterations", "steps"})


@register
class OGDFPivotMDS(_OGDFBase):
    """Competitor adapter for OGDF's Pivot-MDS layout."""

    name = "ogdf_pivot_mds"
    algorithm = "pivot_mds"
    max_nodes = 100_000
    variant_param_names = frozenset({"n_pivots"})


@register
class OGDFDavidsonHarel(_OGDFBase):
    """Competitor adapter for OGDF's Davidson-Harel layout."""

    name = "ogdf_davidson_harel"
    algorithm = "davidson_harel"
    max_nodes = 500


@register
class OGDFSugiyama(_OGDFBase):
    """Competitor adapter for OGDF's Sugiyama layout."""

    name = "ogdf_sugiyama"
    algorithm = "sugiyama"
    max_nodes = 20_000


# OGDFLinLog removed: OGDF has no LinLog implementation. The class was a
# placeholder that threw "unsupported algorithm: linlog" on every call,
# producing 105 deterministic errors per benchmark with zero information value.
# Dagua's own LinLog pipeline (dagua/layout/ops/pipelines/linlog.py) remains
# available via the native engine.
