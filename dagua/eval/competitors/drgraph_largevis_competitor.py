"""Reference subprocess adapters for LargeVis and DRGraph."""

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

_LARGEVIS_BINARY_ENV = "DAGUA_LARGEVIS_BINARY"
_DRGRAPH_BINARY_ENV = "DAGUA_DRGRAPH_BINARY"
_REFERENCE_SEED = 314159265


def _resolve_binary(env_var: str, fallback_name: str) -> Optional[str]:
    """Resolve a reference binary path.

    Parameters
    ----------
    env_var : str
        Environment variable that may point at the binary.
    fallback_name : str
        Executable name searched on ``PATH``.

    Returns
    -------
    str or None
        Binary path when available.
    """
    configured = os.environ.get(env_var)
    if configured:
        path = Path(configured)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which(fallback_name)


def _write_reference_graph(path: Path, graph: DaguaGraph) -> None:
    """Write the C++ reference directed weighted edge format.

    Parameters
    ----------
    path : pathlib.Path
        Destination graph file.
    graph : DaguaGraph
        Graph to serialize.

    Returns
    -------
    None
        File is written to disk.
    """
    lines: list[str] = []
    if graph.edge_index.numel() > 0:
        edge_index = graph.edge_index.detach().cpu()
        weights = graph.edge_weights.detach().cpu() if graph.edge_weights is not None else None
        for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
            weight = float(weights[edge_id].item()) if weights is not None else 1.0
            lines.append(f"{int(source)} {int(target)} {weight:.9g}")
            lines.append(f"{int(target)} {int(source)} {weight:.9g}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _read_reference_positions(path: Path, num_nodes: int) -> torch.Tensor:
    """Read C++ reference coordinates.

    Parameters
    ----------
    path : pathlib.Path
        Reference output path.
    num_nodes : int
        Number of expected nodes.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for row_id, line in enumerate(path.read_text().splitlines()):
        parts = line.split()
        if len(parts) < 2 or row_id >= num_nodes:
            continue
        if len(parts) >= 3:
            positions[row_id, 0] = float(parts[-2])
            positions[row_id, 1] = float(parts[-1])
        else:
            positions[row_id, 0] = float(parts[0])
            positions[row_id, 1] = float(parts[1])
    return positions


class _ReferenceBinaryCompetitor(CompetitorBase):
    """Base class for C++ reference binary adapters."""

    binary_env_var: str = ""
    fallback_binary: str = ""
    command_kind: str = ""

    def available(self) -> bool:
        """Check whether the configured reference binary is executable.

        Returns
        -------
        bool
            ``True`` when the binary exists.
        """
        return _resolve_binary(self.binary_env_var, self.fallback_binary) is not None

    def _build_command(
        self,
        binary: str,
        input_path: Path,
        output_path: Path,
        seed: Optional[int],
    ) -> list[str]:
        """Build the reference command.

        Parameters
        ----------
        binary : str
            Reference binary path.
        input_path : pathlib.Path
            Input graph path.
        output_path : pathlib.Path
            Output coordinate path.
        seed : int or None
            Runtime seed. The upstream binaries ignore this because their
            source hard-codes ``314159265``.

        Returns
        -------
        list[str]
            Subprocess command arguments.
        """
        del seed
        if self.command_kind == "largevis":
            return [
                binary,
                "-fea",
                "0",
                "-input",
                str(input_path),
                "-output",
                str(output_path),
                "-threads",
                "1",
            ]
        return [
            binary,
            "-input",
            str(input_path),
            "-output",
            str(output_path),
            "-threads",
            "1",
            "-mode",
            "1",
        ]

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run a C++ reference layout binary.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int or None, default=None
            Requested seed. Upstream references use their hard-coded seed.

        Returns
        -------
        CompetitorResult
            Layout outcome and runtime.
        """
        binary = _resolve_binary(self.binary_env_var, self.fallback_binary)
        start = time.perf_counter()
        if binary is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=0.0,
                error=f"Reference binary not found via {self.binary_env_var}.",
            )

        with tempfile.TemporaryDirectory(prefix=f"dagua_{self.name}_") as temp_dir:
            input_path = Path(temp_dir) / "graph.txt"
            output_path = Path(temp_dir) / "layout.txt"
            _write_reference_graph(input_path, graph)
            command = self._build_command(binary, input_path, output_path, seed)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - start
                if completed.returncode != 0:
                    return CompetitorResult(
                        name=self.name,
                        pos=None,
                        runtime_seconds=elapsed,
                        error=(completed.stderr or completed.stdout).strip(),
                    )
                return CompetitorResult(
                    name=self.name,
                    pos=_read_reference_positions(output_path, graph.num_nodes),
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
class LargeVisReferenceCompetitor(_ReferenceBinaryCompetitor):
    """Subprocess adapter for the upstream LargeVis executable."""

    name = "largevis_reference"
    max_nodes = 10_000_000
    binary_env_var = _LARGEVIS_BINARY_ENV
    fallback_binary = "LargeVis"
    command_kind = "largevis"


@register
class DRGraphReferenceCompetitor(_ReferenceBinaryCompetitor):
    """Subprocess adapter for the upstream DRGraph ``Vis`` executable."""

    name = "drgraph_reference"
    max_nodes = 10_000_000
    binary_env_var = _DRGRAPH_BINARY_ENV
    fallback_binary = "Vis"
    command_kind = "drgraph"


__all__ = [
    "DRGraphReferenceCompetitor",
    "LargeVisReferenceCompetitor",
]
