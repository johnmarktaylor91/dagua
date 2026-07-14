"""Sparse-stress Java reference competitor adapter."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_REFERENCE_JAR = Path("/tmp/sparse-stress-ref/manual-build/sparse-stress.jar")
_DEFAULT_SEED = 0
_DEFAULT_PIVOTS = 8
_DEFAULT_ITERATIONS = 20
_DEFAULT_MDS_PIVOTS = 8
_DEFAULT_SAMPLER = "kmeans"


def _graph_input_text(graph: DaguaGraph, weighted: bool) -> str:
    """Serialize a graph in the sparse-stress reference edge-list format.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    weighted : bool
        Whether to emit edge weights.

    Returns
    -------
    str
        Reference input text.
    """
    lines = [str(graph.num_nodes)]
    seen: set[tuple[int, int]] = set()
    weights = getattr(graph, "edge_weights", None)
    if graph.edge_index.numel() > 0:
        for edge_pos in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_pos].item())
            target = int(graph.edge_index[1, edge_pos].item())
            if source == target:
                continue
            key = (source, target) if source < target else (target, source)
            if key in seen:
                continue
            seen.add(key)
            if weighted and weights is not None:
                weight = float(weights[edge_pos].item())
                lines.append(f"{key[0]},{key[1]},{weight}")
            else:
                lines.append(f"{key[0]},{key[1]}")
    return "\n".join(lines) + "\n"


def _parse_positions(stdout: str, num_nodes: int) -> torch.Tensor:
    """Parse sparse-stress stdout coordinates.

    Parameters
    ----------
    stdout : str
        Reference process stdout.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` and dtype ``float64``.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float64)
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < num_nodes:
        raise ValueError(f"expected {num_nodes} coordinate rows, got {len(lines)}")
    for index in range(num_nodes):
        x_text, y_text = lines[index].split(",", maxsplit=1)
        pos[index, 0] = float(x_text)
        pos[index, 1] = float(y_text)
    return pos


@register
class SparseStressCompetitor(CompetitorBase):
    """Run Mark Ortmann's sparse-stress Java reference."""

    name = "sparse_stress"
    max_nodes = 5_000
    supports_clusters = False
    variant_param_names = frozenset(
        {
            "break_condition",
            "factor",
            "jar_path",
            "kmeans_features",
            "mds_pivots",
            "pivots",
            "sampler",
            "steps",
            "weighted",
        }
    )

    def available(self) -> bool:
        """Return whether the manually built reference jar exists.

        Returns
        -------
        bool
            ``True`` when the reference jar can be executed by ``java -jar``.
        """
        return _REFERENCE_JAR.exists()

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run sparse-stress with default fidelity parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int, optional
            Sampler seed. ``None`` resolves through benchmark runtime seed.

        Returns
        -------
        CompetitorResult
            Reference layout result.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[dict[str, Any]] = None,
    ) -> CompetitorResult:
        """Run sparse-stress with optional reference parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int, optional
            Sampler seed. ``None`` resolves through benchmark runtime seed.
        variant_params : dict[str, Any], optional
            Reference CLI option overrides.

        Returns
        -------
        CompetitorResult
            Reference layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        jar_path = Path(params.get("jar_path", _REFERENCE_JAR))
        if not jar_path.exists():
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=0.0,
                error=f"sparse-stress reference jar not found: {jar_path}",
            )
        resolved_seed = get_runtime_seed(_DEFAULT_SEED) if seed is None else seed
        sampler = str(params.get("sampler", _DEFAULT_SAMPLER))
        pivots = int(params.get("pivots", _DEFAULT_PIVOTS))
        steps = int(params.get("steps", _DEFAULT_ITERATIONS))
        factor = float(params.get("factor", 1.0))
        mds_pivots = int(params.get("mds_pivots", _DEFAULT_MDS_PIVOTS))
        weighted = bool(params.get("weighted", False))
        break_condition = bool(params.get("break_condition", False))
        kmeans_features = int(params.get("kmeans_features", max(1, min(pivots, mds_pivots))))
        graph_text = _graph_input_text(graph, weighted=weighted)
        with tempfile.NamedTemporaryFile("w", suffix=".sparse_stress", delete=False) as handle:
            handle.write(graph_text)
            graph_path = Path(handle.name)
        command: List[str] = [
            "java",
            "-jar",
            str(jar_path),
            "-p",
            str(pivots),
            "-s",
            sampler,
            "-f",
            str(factor),
            "-i",
            str(steps),
            "+b" if break_condition else "-b",
            "+w" if weighted else "-w",
            "-r",
            str(_DEFAULT_SEED if resolved_seed is None else int(resolved_seed)),
            "-m",
            str(mds_pivots),
        ]
        if sampler == "kmeans":
            command.extend(["--features", str(kmeans_features)])
        command.append(str(graph_path))
        start = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=(result.stderr or result.stdout)[:500],
                )
            return CompetitorResult(
                name=self.name,
                pos=_parse_positions(result.stdout, graph.num_nodes),
                runtime_seconds=elapsed,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )
        finally:
            graph_path.unlink(missing_ok=True)
