"""R graphlayouts reference adapter for backbone layout fidelity checks."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_KEEP = 0.2
_RSCRIPT = "Rscript"


def _graph_edge_csv(graph: DaguaGraph) -> str:
    """Serialize graph edges as one-based CSV rows for R igraph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to serialize.

    Returns
    -------
    str
        CSV text with ``source,target`` rows.
    """
    lines = ["source,target"]
    seen: set[tuple[int, int]] = set()
    for edge_pos in range(int(graph.edge_index.shape[1])):
        source = int(graph.edge_index[0, edge_pos].item())
        target = int(graph.edge_index[1, edge_pos].item())
        if source == target:
            continue
        key = (source, target) if source < target else (target, source)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{key[0] + 1},{key[1] + 1}")
    return "\n".join(lines) + "\n"


def _reference_script() -> str:
    """Return the R script used to run ``layout_as_backbone``.

    Returns
    -------
    str
        R source code.
    """
    return textwrap.dedent(
        """
        args <- commandArgs(trailingOnly = TRUE)
        edge_path <- args[[1]]
        n <- as.integer(args[[2]])
        keep <- as.numeric(args[[3]])
        seed <- as.integer(args[[4]])
        suppressPackageStartupMessages(library(igraph))
        suppressPackageStartupMessages(library(graphlayouts))
        set.seed(seed)
        edges <- read.csv(edge_path)
        if (nrow(edges) == 0) {
          g <- make_empty_graph(n = n, directed = FALSE)
        } else {
          g <- graph_from_edgelist(as.matrix(edges[, c("source", "target")]), directed = FALSE)
          if (vcount(g) < n) {
            g <- add_vertices(g, n - vcount(g))
          }
        }
        result <- layout_as_backbone(g, keep = keep, backbone = TRUE)
        cat("BACKBONE", paste(result$backbone, collapse = ","), "\\n", sep = ",")
        for (i in seq_len(nrow(result$xy))) {
          cat(i, sprintf("%.17g", result$xy[i, 1]), sprintf("%.17g", result$xy[i, 2]), sep = ",")
          cat("\\n")
        }
        """
    ).strip()


def _parse_reference_output(stdout: str, num_nodes: int) -> tuple[torch.Tensor, list[int]]:
    """Parse graphlayouts adapter stdout.

    Parameters
    ----------
    stdout : str
        Process stdout.
    num_nodes : int
        Expected node count.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        Position tensor with shape ``[N, 2]`` and one-based backbone edge ids.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("BACKBONE,"):
        raise ValueError("missing BACKBONE header from graphlayouts output")
    backbone_text = lines[0].split(",", maxsplit=1)[1]
    backbone = [int(value) for value in backbone_text.split(",") if value]
    rows = lines[1:]
    if len(rows) != num_nodes:
        raise ValueError(f"expected {num_nodes} coordinate rows, got {len(rows)}")
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for row in rows:
        index_text, x_text, y_text = row.split(",", maxsplit=2)
        index = int(index_text) - 1
        positions[index, 0] = float(x_text)
        positions[index, 1] = float(y_text)
    return positions, backbone


@register
class BackboneCompetitor(CompetitorBase):
    """Run ``graphlayouts::layout_as_backbone`` through Rscript."""

    name = "backbone"
    max_nodes = 2_000
    supports_clusters = False
    variant_param_names = frozenset({"keep"})

    def available(self) -> bool:
        """Return whether required R packages are installed.

        Returns
        -------
        bool
            ``True`` when Rscript can load graphlayouts, igraph, and oaqc.
        """
        command = [
            _RSCRIPT,
            "-e",
            (
                "cat(requireNamespace('igraph', quietly=TRUE) && "
                "requireNamespace('graphlayouts', quietly=TRUE) && "
                "requireNamespace('oaqc', quietly=TRUE))"
            ),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10.0)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip().endswith("TRUE")

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run graphlayouts with default parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int, optional
            R seed. graphlayouts stress resets to 42 internally; this is kept
            for adapter parity.

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
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run graphlayouts with optional parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int, optional
            R seed.
        variant_params : Mapping[str, Any], optional
            Variant overrides. Supports ``keep``.

        Returns
        -------
        CompetitorResult
            Reference layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        keep = float(params.get("keep", _DEFAULT_KEEP))
        resolved_seed = 42 if seed is None else int(seed)

        with tempfile.TemporaryDirectory(prefix="dagua_backbone_ref_") as tmpdir:
            tmp_path = Path(tmpdir)
            edge_path = tmp_path / "edges.csv"
            script_path = tmp_path / "run_backbone.R"
            edge_path.write_text(_graph_edge_csv(graph), encoding="utf-8")
            script_path.write_text(_reference_script(), encoding="utf-8")

            command = [
                _RSCRIPT,
                str(script_path),
                str(edge_path),
                str(graph.num_nodes),
                str(keep),
                str(resolved_seed),
            ]
            start = time.perf_counter()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=time.perf_counter() - start,
                    error=str(exc),
                )
            runtime = time.perf_counter() - start
            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=runtime,
                    error=(result.stderr or result.stdout).strip(),
                )
            try:
                positions, _backbone = _parse_reference_output(result.stdout, graph.num_nodes)
            except ValueError as exc:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=runtime,
                    error=str(exc),
                )
            return CompetitorResult(name=self.name, pos=positions, runtime_seconds=runtime)
