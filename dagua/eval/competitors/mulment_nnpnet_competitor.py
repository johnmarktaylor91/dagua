"""Reference subprocess adapters for KaDraw MulMent and NNP-NET."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_MULMENT_BINARY_ENV = "DAGUA_MULMENT_BINARY"
_NNPNET_BINARY_ENV = "DAGUA_NNPNET_BINARY"
_MULMENT_DEFAULT_BINARY = Path.home() / "tools" / "dagua-refs" / "kadraw" / "build" / "kadraw"
_NNPNET_DEFAULT_BINARY = (
    Path.home() / "tools" / "dagua-refs" / "nnp-net" / "build-pthread" / "NNP-NET" / "NNPNET"
)
_DEFAULT_REFERENCE_SEED = 42


def _resolve_binary(env_var: str, default_path: Path, fallback_name: str) -> Optional[str]:
    """Resolve a reference binary path.

    Parameters
    ----------
    env_var : str
        Environment variable that may point at the binary.
    default_path : pathlib.Path
        Durable reference build path.
    fallback_name : str
        Executable name searched on ``PATH``.

    Returns
    -------
    str or None
        Executable path when available.
    """
    configured = os.environ.get(env_var)
    if configured:
        path = Path(configured)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    if default_path.exists() and os.access(default_path, os.X_OK):
        return str(default_path)
    return shutil.which(fallback_name)


def _undirected_edges(graph: DaguaGraph) -> list[tuple[int, int]]:
    """Return sorted undirected graph edges without self-loops.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to serialize.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected edges as zero-based node pairs.
    """
    edges: set[tuple[int, int]] = set()
    if graph.edge_index.numel() == 0:
        return []
    for source, target in graph.edge_index.detach().cpu().t().tolist():
        source_id = int(source)
        target_id = int(target)
        if source_id == target_id:
            continue
        lower = min(source_id, target_id)
        upper = max(source_id, target_id)
        edges.add((lower, upper))
    return sorted(edges)


def _write_kadraw_graph(path: Path, graph: DaguaGraph) -> None:
    """Write KaDraw's unweighted METIS-style graph format.

    Parameters
    ----------
    path : pathlib.Path
        Destination graph file.
    graph : DaguaGraph
        Graph to serialize.

    Returns
    -------
    None
        The graph file is written.
    """
    adjacency: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for source, target in _undirected_edges(graph):
        adjacency[source].append(target + 1)
        adjacency[target].append(source + 1)
    lines = [f"{graph.num_nodes} {sum(len(row) for row in adjacency) // 2}"]
    lines.extend(" ".join(str(target) for target in sorted(row)) for row in adjacency)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nnpnet_graph(path: Path, graph: DaguaGraph) -> None:
    """Write NNP-NET's plain edge-list text format.

    Parameters
    ----------
    path : pathlib.Path
        Destination graph file.
    graph : DaguaGraph
        Graph to serialize.

    Returns
    -------
    None
        The graph file is written.
    """
    lines = [str(graph.num_nodes)]
    lines.extend(f"{source} {target}" for source, target in _undirected_edges(graph))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_plain_positions(path: Path, num_nodes: int) -> torch.Tensor:
    """Read two-column coordinate output.

    Parameters
    ----------
    path : pathlib.Path
        Coordinate file path.
    num_nodes : int
        Number of expected positions.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for row_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if row_id >= num_nodes:
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        positions[row_id, 0] = float(parts[0])
        positions[row_id, 1] = float(parts[1])
    return positions


def _read_vna_positions(path: Path, num_nodes: int) -> torch.Tensor:
    """Read NNP-NET VNA node-position output.

    Parameters
    ----------
    path : pathlib.Path
        VNA output file path.
    num_nodes : int
        Number of expected positions.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    in_properties = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "*Node properties":
            in_properties = True
            continue
        if stripped.startswith("*") and in_properties:
            break
        if not in_properties or stripped == "ID x y":
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        node_id = int(parts[0])
        if 0 <= node_id < num_nodes:
            positions[node_id, 0] = float(parts[1])
            positions[node_id, 1] = float(parts[2])
    return positions


class _ReferenceBinaryCompetitor(CompetitorBase):
    """Base class for MulMent/NNP-NET reference binary adapters."""

    binary_env_var: str = ""
    default_binary: Path = Path()
    fallback_binary: str = ""

    def available(self) -> bool:
        """Check whether the configured reference binary is executable.

        Returns
        -------
        bool
            ``True`` when a usable binary is available.
        """
        return (
            _resolve_binary(
                self.binary_env_var,
                self.default_binary,
                self.fallback_binary,
            )
            is not None
        )

    def _run_reference(
        self,
        binary: str,
        graph: DaguaGraph,
        workdir: Path,
        seed: int,
        timeout: float,
    ) -> torch.Tensor:
        """Run the concrete reference binary.

        Parameters
        ----------
        binary : str
            Reference binary path.
        graph : DaguaGraph
            Graph to lay out.
        workdir : pathlib.Path
            Temporary working directory.
        seed : int
            Deterministic reference seed.
        timeout : float
            Maximum runtime in seconds.

        Returns
        -------
        torch.Tensor
            Position tensor with shape ``[N, 2]``.
        """
        raise NotImplementedError

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run a seeded reference layout binary.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int or None, default=None
            Requested reference seed. ``None`` uses a reproducible default.

        Returns
        -------
        CompetitorResult
            Layout outcome and runtime.
        """
        binary = _resolve_binary(self.binary_env_var, self.default_binary, self.fallback_binary)
        start = time.perf_counter()
        if binary is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=0.0,
                error=f"Reference binary not found via {self.binary_env_var}.",
            )

        reference_seed = _DEFAULT_REFERENCE_SEED if seed is None else int(seed)
        with tempfile.TemporaryDirectory(prefix=f"dagua_{self.name}_") as temp_dir:
            try:
                positions = self._run_reference(
                    binary=binary,
                    graph=graph,
                    workdir=Path(temp_dir),
                    seed=reference_seed,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - start
                return CompetitorResult(
                    name=self.name,
                    pos=positions,
                    runtime_seconds=elapsed,
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
class MulMentReferenceCompetitor(_ReferenceBinaryCompetitor):
    """Subprocess adapter for the upstream KaDraw MulMent executable."""

    name = "mulment_reference"
    max_nodes = 100_000
    binary_env_var = _MULMENT_BINARY_ENV
    default_binary = _MULMENT_DEFAULT_BINARY
    fallback_binary = "kadraw"

    def _run_reference(
        self,
        binary: str,
        graph: DaguaGraph,
        workdir: Path,
        seed: int,
        timeout: float,
    ) -> torch.Tensor:
        """Run KaDraw and read its coordinate output.

        Parameters
        ----------
        binary : str
            KaDraw executable path.
        graph : DaguaGraph
            Graph to lay out.
        workdir : pathlib.Path
            Temporary working directory.
        seed : int
            Deterministic KaDraw seed.
        timeout : float
            Maximum runtime in seconds.

        Returns
        -------
        torch.Tensor
            Position tensor with shape ``[N, 2]``.
        """
        input_path = workdir / "graph.graph"
        output_path = workdir / "layout.coord"
        _write_kadraw_graph(input_path, graph)
        command = [
            binary,
            str(input_path),
            "--seed",
            str(seed),
            "--num_threads",
            "1",
            "--burn_coordinates_to_disk",
            "--output_coord_filename",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        return _read_plain_positions(output_path, graph.num_nodes)


@register
class NNPNetReferenceCompetitor(_ReferenceBinaryCompetitor):
    """Subprocess adapter for the upstream NNP-NET executable."""

    name = "nnpnet_reference"
    max_nodes = 100_000
    binary_env_var = _NNPNET_BINARY_ENV
    default_binary = _NNPNET_DEFAULT_BINARY
    fallback_binary = "NNPNET"

    def _run_reference(
        self,
        binary: str,
        graph: DaguaGraph,
        workdir: Path,
        seed: int,
        timeout: float,
    ) -> torch.Tensor:
        """Run NNP-NET and read its VNA coordinate output.

        Parameters
        ----------
        binary : str
            NNP-NET executable path.
        graph : DaguaGraph
            Graph to lay out.
        workdir : pathlib.Path
            Temporary working directory.
        seed : int
            Deterministic NNP-NET seed.
        timeout : float
            Maximum runtime in seconds.

        Returns
        -------
        torch.Tensor
            Position tensor with shape ``[N, 2]``.
        """
        input_path = workdir / "graph.txt"
        output_path = workdir / "layout.vna"
        _write_nnpnet_graph(input_path, graph)
        command = [
            binary,
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(seed),
            "--training_epochs",
            "1",
            "--batch_size",
            "64",
            "--embedding_size",
            "4",
            "--pmds_pivots",
            "4",
            "--smoothing",
            "0",
            "--gpu",
            "false",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        return _read_vna_positions(output_path, graph.num_nodes)


__all__ = [
    "MulMentReferenceCompetitor",
    "NNPNetReferenceCompetitor",
]
