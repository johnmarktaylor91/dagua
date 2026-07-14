"""Native reference adapters for GRIP, Omega, and tidy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_GRIP_BINARY_ENV = "DAGUA_GRIP_BINARY"
_OMEGA_BINARY_ENV = "DAGUA_OMEGA_BINARY"
_TIDY_BINARY_ENV = "DAGUA_TIDY_BINARY"
_GRIP_BINARY = Path.home() / "tools" / "dagua-refs" / "grip" / "original" / "grip_headless_layout"
_OMEGA_BINARY = Path.home() / "tools" / "dagua-refs" / "egraph-rs" / "target" / "release" / "omega"
_TIDY_BINARY = (
    Path.home() / "tools" / "dagua-refs" / "tidy" / "rust" / "target" / "release" / "tidy_reference"
)


def _resolve_binary(env_var: str, fallback_path: Path, fallback_name: str) -> Optional[Path]:
    """Resolve a native reference executable.

    Parameters
    ----------
    env_var : str
        Environment variable that may point at an executable.
    fallback_path : pathlib.Path
        Durable reference path used by the local verification environment.
    fallback_name : str
        Executable name searched on ``PATH`` as a final fallback.

    Returns
    -------
    pathlib.Path or None
        Executable path when found.
    """
    configured = os.environ.get(env_var)
    if configured:
        path = Path(configured)
        return path if path.exists() and os.access(path, os.X_OK) else None
    if fallback_path.exists() and os.access(fallback_path, os.X_OK):
        return fallback_path
    located = shutil.which(fallback_name)
    return None if located is None else Path(located)


def _edge_pairs(graph: DaguaGraph) -> list[tuple[int, int]]:
    """Return valid graph edge pairs in tensor order.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    list[tuple[int, int]]
        Valid source-target pairs.
    """
    if graph.edge_index.numel() == 0:
        return []
    pairs: list[tuple[int, int]] = []
    for raw_source, raw_target in graph.edge_index.detach().cpu().to(torch.long).t().tolist():
        source = int(raw_source)
        target = int(raw_target)
        if source < 0 or target < 0 or source >= graph.num_nodes or target >= graph.num_nodes:
            continue
        pairs.append((source, target))
    return pairs


def _grip_components(graph: DaguaGraph) -> list[list[int]]:
    """Return undirected connected components for GRIP reference runs.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    list[list[int]]
        Sorted component node ids.
    """
    neighbors: list[set[int]] = [set() for _ in range(graph.num_nodes)]
    for source, target in _edge_pairs(graph):
        if source == target:
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(graph.num_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(neighbors[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _write_grip_component_file(path: Path, graph: DaguaGraph, nodes: list[int]) -> None:
    """Write one remapped GRIP component edge file.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    graph : DaguaGraph
        Source graph.
    nodes : list[int]
        Component node ids.

    Returns
    -------
    None
        File is written.
    """
    local_index = {node: index for index, node in enumerate(nodes)}
    pairs = [
        (local_index[source], local_index[target])
        for source, target in _edge_pairs(graph)
        if source in local_index and target in local_index and source != target
    ]
    lines = [f"{len(nodes)} {len(pairs)}"]
    lines.extend(f"{source} {target}" for source, target in pairs)
    path.write_text("\n".join(lines) + "\n")


def _parse_position_text(text: str, num_nodes: int) -> torch.Tensor:
    """Parse ``POSITIONS`` line output from native reference runners.

    Parameters
    ----------
    text : str
        Subprocess stdout.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    in_positions = False
    seen: set[int] = set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "POSITIONS":
            in_positions = True
            continue
        if not in_positions or len(parts) < 3:
            continue
        node = int(parts[0])
        if 0 <= node < num_nodes:
            positions[node, 0] = float(parts[1])
            positions[node, 1] = float(parts[2])
            seen.add(node)
    missing = sorted(set(range(num_nodes)) - seen)
    if missing:
        raise ValueError(f"reference omitted coordinates for nodes {missing[:5]}")
    return positions


def _position_extents(positions: torch.Tensor, nodes: list[int]) -> tuple[float, float]:
    """Return min/max x extents for point positions.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    nodes : list[int]
        Node ids to inspect.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum x coordinates.
    """
    values = [float(positions[node, 0].item()) for node in nodes]
    return min(values), max(values)


def _run_subprocess(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a native reference command and capture text output.

    Parameters
    ----------
    command : list[str]
        Command arguments.
    timeout : float
        Maximum runtime in seconds.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed process.
    """
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _write_omega_json(path: Path, graph: DaguaGraph) -> None:
    """Write egraph-rs ``GraphData`` JSON.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    None
        JSON file is written.
    """
    payload = {
        "nodes": [
            {"id": str(node), "x": None, "y": None, "data": None} for node in range(graph.num_nodes)
        ],
        "links": [
            {"source": str(source), "target": str(target), "data": None}
            for source, target in _edge_pairs(graph)
            if source != target
        ],
    }
    path.write_text(json.dumps(payload))


def _read_omega_positions(path: Path, num_nodes: int) -> torch.Tensor:
    """Read egraph-rs omega JSON position output.

    Parameters
    ----------
    path : pathlib.Path
        Output JSON path.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    raw = json.loads(path.read_text())
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node in range(num_nodes):
        coords = raw[str(node)]
        positions[node, 0] = float(coords[0])
        positions[node, 1] = float(coords[1])
    return positions


def _node_sizes(graph: DaguaGraph) -> torch.Tensor:
    """Return node sizes with default tidy dimensions filled in.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    torch.Tensor
        Size tensor with shape ``[N, 2]``.
    """
    if graph.node_sizes is None or graph.node_sizes.numel() == 0:
        return torch.ones((graph.num_nodes, 2), dtype=torch.float64)
    return graph.node_sizes.detach().cpu().to(torch.float64).clamp_min(1.0)


def _tidy_children_and_roots(
    graph: DaguaGraph,
) -> tuple[list[list[int]], list[int], list[Optional[int]]]:
    """Build tidy child lists using the first incoming parent per node.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph interpreted as parent-to-child edges.

    Returns
    -------
    tuple[list[list[int]], list[int], list[int | None]]
        Child lists, root ids, and parent ids.
    """
    children: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    parents: list[Optional[int]] = [None for _ in range(graph.num_nodes)]
    for source, target in _edge_pairs(graph):
        if source == target or parents[target] is not None:
            continue
        parents[target] = source
        children[source].append(target)
    roots = [node for node, parent in enumerate(parents) if parent is None]
    return children, roots, parents


def _component_nodes(root: int, children: list[list[int]]) -> list[int]:
    """Return parent-before-child nodes for one tidy component.

    Parameters
    ----------
    root : int
        Root node id.
    children : list[list[int]]
        Child lists.

    Returns
    -------
    list[int]
        Component node ids in insertion order for ``TidyTree``.
    """
    ordered: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        ordered.append(node)
        stack.extend(reversed(children[node]))
    return ordered


def _write_tidy_node_file(
    path: Path,
    nodes: list[int],
    parents: list[Optional[int]],
    sizes: torch.Tensor,
) -> None:
    """Write one tidy component input file.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    nodes : list[int]
        Parent-before-child node ids.
    parents : list[int | None]
        Parent id per graph node.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    None
        File is written.
    """
    lines = [str(len(nodes))]
    for node in nodes:
        parent = parents[node]
        parent_text = "-1" if parent is None else str(parent)
        lines.append(
            f"{node} {float(sizes[node, 0].item()):.17g} "
            f"{float(sizes[node, 1].item()):.17g} {parent_text}"
        )
    path.write_text("\n".join(lines) + "\n")


def _component_extents(
    positions: torch.Tensor,
    nodes: list[int],
    sizes: torch.Tensor,
) -> tuple[float, float]:
    """Return horizontal extents for one tidy component.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    nodes : list[int]
        Component node ids.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum x extents.
    """
    min_x = float("inf")
    max_x = float("-inf")
    for node in nodes:
        x_coord = float(positions[node, 0].item())
        half_width = float(sizes[node, 0].item()) / 2.0
        min_x = min(min_x, x_coord - half_width)
        max_x = max(max_x, x_coord + half_width)
    return min_x, max_x


@register
class GripReferenceCompetitor(CompetitorBase):
    """Subprocess adapter for the headless GRIP reference binary."""

    name = "grip_reference"
    max_nodes = 100_000
    supports_clusters = False
    variant_param_names = frozenset({"rounds", "final_rounds", "init_vertices", "dim"})

    def available(self) -> bool:
        """Return whether the GRIP reference executable is available.

        Returns
        -------
        bool
            ``True`` when the executable exists.
        """
        return _resolve_binary(_GRIP_BINARY_ENV, _GRIP_BINARY, "grip_headless_layout") is not None

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the GRIP reference.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Deterministic seed forwarded to the headless runner.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run GRIP with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Deterministic seed.
        variant_params : Mapping[str, Any] | None, default=None
            Optional GRIP runner parameters.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        binary = _resolve_binary(_GRIP_BINARY_ENV, _GRIP_BINARY, "grip_headless_layout")
        if binary is None:
            return CompetitorResult(self.name, None, 0.0, error="GRIP reference binary missing")
        params = {} if variant_params is None else dict(variant_params)
        resolved_seed = get_runtime_seed(0) if seed is None else seed
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="dagua_grip_ref_") as temp_dir:
                positions = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
                offset = 0.0
                for component_id, nodes in enumerate(_grip_components(graph)):
                    if len(nodes) == 1:
                        positions[nodes[0], 0] = offset
                        offset += 32.0
                        continue
                    input_path = Path(temp_dir) / f"component_{component_id}.edges"
                    _write_grip_component_file(input_path, graph, nodes)
                    command = [
                        str(binary),
                        "--input",
                        str(input_path),
                        "--dim",
                        str(int(params.get("dim", 2))),
                        "--init-vertices",
                        str(int(params.get("init_vertices", 4))),
                        "--rounds",
                        str(int(params.get("rounds", 20))),
                        "--final-rounds",
                        str(int(params.get("final_rounds", params.get("rounds", 30)))),
                        "--seed",
                        str(int(0 if resolved_seed is None else resolved_seed)),
                    ]
                    completed = _run_subprocess(command, timeout=timeout)
                    elapsed = time.perf_counter() - start
                    if completed.returncode != 0:
                        return CompetitorResult(
                            self.name,
                            None,
                            elapsed,
                            error=(completed.stderr or completed.stdout).strip()
                            or f"GRIP reference exited {completed.returncode}",
                        )
                    component_positions = _parse_position_text(completed.stdout, len(nodes))
                    for local_node, graph_node in enumerate(nodes):
                        positions[graph_node] = component_positions[local_node]
                    min_x, max_x = _position_extents(positions, nodes)
                    positions[nodes, 0] += offset - min_x
                    offset += (max_x - min_x) + 32.0
                return CompetitorResult(self.name, positions, time.perf_counter() - start)
        except Exception as exc:
            return CompetitorResult(self.name, None, time.perf_counter() - start, error=str(exc))


@register
class OmegaReferenceCompetitor(CompetitorBase):
    """Subprocess adapter for the egraph-rs omega reference CLI."""

    name = "omega_reference"
    max_nodes = 100_000
    supports_clusters = False
    variant_param_names = frozenset(
        {"d", "k", "min_dist", "unit_edge_length", "sgd_iterations", "sgd_eps"}
    )

    def available(self) -> bool:
        """Return whether the omega reference executable is available.

        Returns
        -------
        bool
            ``True`` when the executable exists.
        """
        return _resolve_binary(_OMEGA_BINARY_ENV, _OMEGA_BINARY, "omega") is not None

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the omega reference layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Deterministic seed for the patched reference CLI.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run omega with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Deterministic seed.
        variant_params : Mapping[str, Any] | None, default=None
            Optional omega parameters.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        binary = _resolve_binary(_OMEGA_BINARY_ENV, _OMEGA_BINARY, "omega")
        if binary is None:
            return CompetitorResult(self.name, None, 0.0, error="omega reference binary missing")
        params = {} if variant_params is None else dict(variant_params)
        resolved_seed = get_runtime_seed(42) if seed is None else seed
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="dagua_omega_ref_") as temp_dir:
                input_path = Path(temp_dir) / "graph.json"
                output_path = Path(temp_dir) / "positions.json"
                _write_omega_json(input_path, graph)
                command = [
                    str(binary),
                    str(input_path),
                    str(output_path),
                    "--d",
                    str(int(params.get("d", 2))),
                    "--k",
                    str(int(params.get("k", 30))),
                    "--min-dist",
                    str(float(params.get("min_dist", 1.0e-3))),
                    "--unit-edge-length",
                    str(float(params.get("unit_edge_length", 1.0))),
                    "--sgd-iterations",
                    str(int(params.get("sgd_iterations", 100))),
                    "--sgd-eps",
                    str(float(params.get("sgd_eps", 0.1))),
                    "--seed",
                    str(int(42 if resolved_seed is None else resolved_seed)),
                ]
                completed = _run_subprocess(command, timeout=timeout)
                elapsed = time.perf_counter() - start
                if completed.returncode != 0:
                    return CompetitorResult(
                        self.name,
                        None,
                        elapsed,
                        error=(completed.stderr or completed.stdout).strip(),
                    )
                return CompetitorResult(
                    self.name,
                    _read_omega_positions(output_path, graph.num_nodes),
                    elapsed,
                )
        except Exception as exc:
            return CompetitorResult(self.name, None, time.perf_counter() - start, error=str(exc))


@register
class TidyReferenceCompetitor(CompetitorBase):
    """Subprocess adapter for the tidy-tree reference runner."""

    name = "tidy_reference"
    max_nodes = 100_000
    supports_clusters = False
    variant_param_names = frozenset({"parent_child_margin", "peer_margin"})

    def available(self) -> bool:
        """Return whether the tidy reference executable is available.

        Returns
        -------
        bool
            ``True`` when the executable exists.
        """
        return _resolve_binary(_TIDY_BINARY_ENV, _TIDY_BINARY, "tidy_reference") is not None

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the tidy reference layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Unused deterministic seed accepted for registry compatibility.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        del seed
        return self.layout_with_variant(graph, timeout=timeout, seed=None, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run tidy with optional margin overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Unused deterministic seed.
        variant_params : Mapping[str, Any] | None, default=None
            Optional tidy margin parameters.

        Returns
        -------
        CompetitorResult
            Layout outcome.
        """
        del seed
        binary = _resolve_binary(_TIDY_BINARY_ENV, _TIDY_BINARY, "tidy_reference")
        if binary is None:
            return CompetitorResult(self.name, None, 0.0, error="tidy reference binary missing")
        params = {} if variant_params is None else dict(variant_params)
        parent_child_margin = float(params.get("parent_child_margin", 10.0))
        peer_margin = float(params.get("peer_margin", 10.0))
        sizes = _node_sizes(graph)
        children, roots, parents = _tidy_children_and_roots(graph)
        positions = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
        offset = 0.0
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="dagua_tidy_ref_") as temp_dir:
                for component_id, root in enumerate(roots):
                    nodes = _component_nodes(root, children)
                    input_path = Path(temp_dir) / f"component_{component_id}.txt"
                    _write_tidy_node_file(input_path, nodes, parents, sizes)
                    command = [
                        str(binary),
                        "--input",
                        str(input_path),
                        "--parent-child-margin",
                        str(parent_child_margin),
                        "--peer-margin",
                        str(peer_margin),
                    ]
                    completed = _run_subprocess(command, timeout=timeout)
                    if completed.returncode != 0:
                        return CompetitorResult(
                            self.name,
                            None,
                            time.perf_counter() - start,
                            error=(completed.stderr or completed.stdout).strip(),
                        )
                    component_positions = _parse_position_text(completed.stdout, graph.num_nodes)
                    for node in nodes:
                        positions[node] = component_positions[node]
                    min_x, max_x = _component_extents(positions, nodes, sizes)
                    positions[nodes, 0] += offset - min_x
                    offset += (max_x - min_x) + peer_margin
            return CompetitorResult(self.name, positions, time.perf_counter() - start)
        except Exception as exc:
            return CompetitorResult(self.name, None, time.perf_counter() - start, error=str(exc))
