"""OpenOrd reference competitor adapter."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_SEED = 0
_DEFAULT_EDGE_CUT = 32.0 / 40.0
_REFERENCE_ROOT = Path.home() / "tools" / "dagua-refs" / "openord"


def _reference_binary(name: str, reference_root: Path) -> Path:
    """Return an OpenOrd reference binary path.

    Parameters
    ----------
    name : str
        Binary name under ``bin``.
    reference_root : pathlib.Path
        Reference checkout root.

    Returns
    -------
    pathlib.Path
        Absolute binary path.
    """
    return reference_root / "bin" / name


def _write_sim_file(graph: DaguaGraph, root: Path) -> None:
    """Write a temporary OpenOrd ``.sim`` input file.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    root : pathlib.Path
        Temporary root path without extension.

    Returns
    -------
    None
        The ``root.sim`` file is written.
    """
    edge_index = graph.edge_index.detach().to(device="cpu", dtype=torch.long)
    weights = graph.edge_weights
    if weights is None:
        weights_cpu = torch.ones(edge_index.shape[1], dtype=torch.float32)
    else:
        weights_cpu = weights.detach().to(device="cpu", dtype=torch.float32)

    lines = []
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if int(source) == int(target):
            continue
        weight = float(weights_cpu[edge_id].item())
        if weight > 0.0:
            lines.append(f"{int(source)}\t{int(target)}\t{weight:.9g}\n")
    if not lines and graph.num_nodes > 1:
        lines.append("0\t1\t1\n")
    (root.with_suffix(".sim")).write_text("".join(lines))


def _parse_coord_file(root: Path, num_nodes: int) -> torch.Tensor:
    """Parse OpenOrd ``.coord`` output into a tensor.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root path without extension.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If the reference omits any expected node coordinate.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    seen: set[int] = set()
    for line in root.with_suffix(".coord").read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        node = int(parts[0])
        if 0 <= node < num_nodes:
            positions[node, 0] = float(parts[1])
            positions[node, 1] = float(parts[2])
            seen.add(node)
    missing = sorted(set(range(num_nodes)) - seen)
    if missing:
        raise ValueError(f"OpenOrd reference omitted coordinates for nodes {missing[:5]}")
    return positions


def _run_command(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one OpenOrd command.

    Parameters
    ----------
    command : list[str]
        Command and arguments.
    cwd : pathlib.Path
        Working directory.
    timeout : float
        Timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed subprocess result.
    """
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@register
class OpenOrdCompetitor(CompetitorBase):
    """Run the cloned OpenOrd C++ reference implementation in a subprocess."""

    name = "openord"
    max_nodes = 5_000
    supports_clusters = False
    variant_param_names = frozenset({"edge_cut", "reference_root"})

    def available(self) -> bool:
        """Return whether the OpenOrd reference binaries are runnable.

        Returns
        -------
        bool
            ``True`` when ``truncate``, ``layout``, and ``recoord`` exist.
        """
        return all(
            _reference_binary(name, _REFERENCE_ROOT).exists()
            for name in ("truncate", "layout", "recoord")
        )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the OpenOrd reference layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int or None, default=None
            Deterministic reference seed.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run OpenOrd with optional reference parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int or None, default=None
            Deterministic reference seed.
        variant_params : Mapping[str, Any], optional
            Reference option overrides.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        params = {} if variant_params is None else dict(variant_params)
        reference_root = Path(str(params.get("reference_root", _REFERENCE_ROOT)))
        if not all(
            _reference_binary(name, reference_root).exists()
            for name in ("truncate", "layout", "recoord")
        ):
            return CompetitorResult(
                self.name,
                None,
                0.0,
                error="OpenOrd reference binaries missing",
            )

        resolved_seed = get_runtime_seed(_DEFAULT_SEED) if seed is None else seed
        edge_cut = float(params.get("edge_cut", _DEFAULT_EDGE_CUT))
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="openord-") as temp_name:
                temp_dir = Path(temp_name)
                root = temp_dir / "graph"
                _write_sim_file(graph, root)
                commands = [
                    [
                        _reference_binary("truncate", reference_root).as_posix(),
                        "-t",
                        "1000000",
                        root.as_posix(),
                    ],
                    [
                        _reference_binary("layout", reference_root).as_posix(),
                        "-s",
                        str(resolved_seed),
                        "-c",
                        f"{edge_cut:.9g}",
                        root.as_posix(),
                    ],
                    [_reference_binary("recoord", reference_root).as_posix(), root.as_posix()],
                ]
                for command in commands:
                    result = _run_command(command, cwd=temp_dir, timeout=timeout)
                    if result.returncode != 0:
                        elapsed = time.perf_counter() - start
                        error = (result.stderr or result.stdout or "unknown OpenOrd failure")[:1000]
                        return CompetitorResult(self.name, None, elapsed, error=error)
                pos = _parse_coord_file(root=root, num_nodes=graph.num_nodes)
                return CompetitorResult(
                    name=self.name,
                    pos=pos,
                    runtime_seconds=time.perf_counter() - start,
                )
        except subprocess.TimeoutExpired:
            return CompetitorResult(self.name, None, timeout, error="timeout")
        except (OSError, ValueError) as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(self.name, None, elapsed, error=str(exc))
