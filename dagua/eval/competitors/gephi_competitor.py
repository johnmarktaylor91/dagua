"""Gephi YifanHu competitor adapter backed by the Gephi toolkit."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graph_to_edge_list(graph: DaguaGraph) -> List[List[int]]:
    """Convert ``edge_index`` into a JSON-serializable edge list.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose edges should be serialized.

    Returns
    -------
    list[list[int]]
        Edge list in ``[[source, target], ...]`` format.
    """
    if graph.edge_index.numel() == 0:
        return []
    edge_index = graph.edge_index.cpu()
    return [
        [int(edge_index[0, edge_idx].item()), int(edge_index[1, edge_idx].item())]
        for edge_idx in range(edge_index.shape[1])
    ]


def _positions_to_tensor(
    raw_positions: Mapping[str, Sequence[float]],
    num_nodes: int,
) -> torch.Tensor:
    """Convert Gephi JSON positions into a ``[N, 2]`` tensor.

    Parameters
    ----------
    raw_positions : Mapping[str, Sequence[float]]
        Node-position mapping emitted by the Java helper.
    num_nodes : int
        Number of nodes expected in the output tensor.

    Returns
    -------
    torch.Tensor
        CPU float tensor shaped ``[N, 2]``.

    Raises
    ------
    ValueError
        Raised when the helper emits malformed coordinate data.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        coords = raw_positions.get(str(node_idx))
        if coords is None:
            continue
        if len(coords) < 2:
            msg = f"missing coordinates for node {node_idx}"
            raise ValueError(msg)
        pos[node_idx, 0] = float(coords[0])
        pos[node_idx, 1] = float(coords[1])
    return pos


def _trim_process_error(result: subprocess.CompletedProcess[str]) -> str:
    """Return a bounded subprocess error message.

    Parameters
    ----------
    result : subprocess.CompletedProcess[str]
        Completed subprocess result.

    Returns
    -------
    str
        First 500 characters from stderr, or stdout when stderr is empty.
    """
    error_text = result.stderr or result.stdout or "unknown subprocess failure"
    return error_text[:500]


def _java_command_works(binary: str) -> bool:
    """Return whether a Java tool can be executed successfully.

    Parameters
    ----------
    binary : str
        Java executable name, typically ``java`` or ``javac``.

    Returns
    -------
    bool
        ``True`` when the executable exists and responds to ``-version``.
    """
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@register
class GephiYifanHu(CompetitorBase):
    """Gephi YifanHu layout adapter."""

    name = "gephi_yifanhu"
    max_nodes = 50_000

    _JAR_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "lib"
        / "gephi-toolkit-0.10.1-all.jar"
    )
    _JAVA_SRC = Path(__file__).resolve().parent / "gephi_layout.java"
    _JAVA_CLASS_DIR = Path(__file__).resolve().parent / "_gephi_build"

    def _ensure_compiled(self) -> None:
        """Compile the Java helper when the cached class file is stale.

        Parameters
        ----------
        None
            The helper uses class constants for its source, class directory,
            and toolkit JAR path.

        Returns
        -------
        None
            The helper compiles the Java source in place when needed.

        Raises
        ------
        RuntimeError
            Raised when the source or toolkit JAR is missing, or when
            compilation fails.
        """
        if not self._JAR_PATH.is_file():
            msg = f"missing Gephi toolkit jar: {self._JAR_PATH}"
            raise RuntimeError(msg)
        if not self._JAVA_SRC.is_file():
            msg = f"missing Gephi helper source: {self._JAVA_SRC}"
            raise RuntimeError(msg)

        class_file = self._JAVA_CLASS_DIR / "gephi_layout.class"
        if class_file.exists():
            class_mtime = class_file.stat().st_mtime
            source_mtime = self._JAVA_SRC.stat().st_mtime
            jar_mtime = self._JAR_PATH.stat().st_mtime
            if class_mtime >= max(source_mtime, jar_mtime):
                return

        self._JAVA_CLASS_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "javac",
                "-cp",
                str(self._JAR_PATH),
                "-d",
                str(self._JAVA_CLASS_DIR),
                str(self._JAVA_SRC),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            msg = _trim_process_error(result)
            raise RuntimeError(f"failed to compile Gephi helper: {msg}")

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run Gephi YifanHu with default adapter options.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed used to initialize node positions before
            the YifanHu iterations run.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run Gephi YifanHu with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed used to initialize node positions before
            layout.
        variant_params : Mapping[str, Any] | None, default=None
            Optional Gephi layout parameters forwarded to the Java helper.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        start = time.perf_counter()
        try:
            self._ensure_compiled()

            payload: Dict[str, object] = {
                "num_nodes": graph.num_nodes,
                "edges": _graph_to_edge_list(graph),
                "algorithm": "yifanhu",
                "seed": 42 if seed is None else seed,
                "params": dict(variant_params) if variant_params is not None else {},
            }
            input_data = json.dumps(payload)
            classpath = os.pathsep.join([str(self._JAR_PATH), str(self._JAVA_CLASS_DIR)])
            result = subprocess.run(
                [
                    "java",
                    "-Djava.awt.headless=true",
                    "-cp",
                    classpath,
                    "gephi_layout",
                ],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=_trim_process_error(result),
                )

            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                raise ValueError("Gephi helper returned non-object JSON")
            pos = _positions_to_tensor(data, graph.num_nodes)
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error="timeout"
            )
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )

    def available(self) -> bool:
        """Return whether Java, javac, the toolkit JAR, and the helper work.

        Returns
        -------
        bool
            ``True`` when the adapter can compile and execute its Java helper.
        """
        if not self._JAR_PATH.is_file():
            return False
        if not _java_command_works("java") or not _java_command_works("javac"):
            return False
        try:
            self._ensure_compiled()
            return True
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
