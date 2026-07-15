"""Verify the Gephi ForceAtlas1 source-port fidelity status."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.forceatlas1 import layout_forceatlas1_pipeline  # noqa: E402
from dagua.metrics import composite, quick  # noqa: E402

GEPHI_TOOLKIT_URL = (
    "https://repo1.maven.org/maven2/org/gephi/gephi-toolkit/0.10.1/gephi-toolkit-0.10.1-all.jar"
)
GEPHI_CACHE_DIR = Path.home() / "tools" / "dagua-refs" / "gephi-forceatlas1"
GEPHI_TOOLKIT_JAR = GEPHI_CACHE_DIR / "gephi-toolkit-0.10.1-all.jar"
RUNNER_SOURCE = ROOT / "scripts/gephi_forceatlas1_runner/ForceAtlas1ReferenceRunner.java"
RUNNER_CLASSES = GEPHI_CACHE_DIR / "classes"
RUNNER_CLASS = "ForceAtlas1ReferenceRunner"
VERIFY_STEPS = 25
VERIFY_SEED = 42


@dataclass(frozen=True)
class VerificationGraph:
    """Small graph used by the ForceAtlas1 verification script.

    Parameters
    ----------
    name : str
        Report name.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    params : dict[str, object]
        ForceAtlas1 variant parameters.
    """

    name: str
    edge_index: torch.Tensor
    num_nodes: int
    edge_weights: Optional[torch.Tensor] = None
    node_sizes: Optional[torch.Tensor] = None
    params: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class VerificationResult:
    """Result from one ForceAtlas1 Gephi comparison.

    Parameters
    ----------
    residual : float
        Procrustes RMSD between Gephi and Dagua layouts.
    max_abs : float
        Maximum absolute coordinate difference before alignment.
    tier : str
        Fidelity tier label.
    quality : float
        Quick composite quality score for the Gephi reference coordinates.
    """

    residual: float
    max_abs: float
    tier: str
    quality: float


class ReferenceRuntimeError(RuntimeError):
    """Raised when the Gephi toolkit reference runtime cannot be executed."""


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from a tuple list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _node_size_values(node_sizes: Optional[torch.Tensor], num_nodes: int) -> list[float]:
    """Return Gephi scalar node sizes for the Java runner.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N]`` or ``[N, 2]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[float]
        Scalar node sizes in input order.
    """
    if node_sizes is None:
        return [0.0] * num_nodes
    sizes = node_sizes.detach().cpu().to(dtype=torch.float64)
    if sizes.ndim == 1 and sizes.shape[0] == num_nodes:
        return [float(value) for value in sizes.tolist()]
    if sizes.ndim == 2 and sizes.shape[0] == num_nodes:
        return [float(value) for value in torch.amax(sizes, dim=1).tolist()]
    raise ValueError("node_sizes must have shape [N] or [N, 2].")


def _edge_weight_values(edge_weights: Optional[torch.Tensor], num_edges: int) -> list[float]:
    """Return edge weights for the Java runner.

    Parameters
    ----------
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    num_edges : int
        Number of graph edges.

    Returns
    -------
    list[float]
        Edge weights in edge-index order.
    """
    if edge_weights is None:
        return [1.0] * num_edges
    weights = edge_weights.detach().cpu().to(dtype=torch.float64)
    if weights.ndim != 1 or weights.shape[0] != num_edges:
        raise ValueError("edge_weights must have shape [E].")
    return [float(value) for value in weights.tolist()]


def _verification_graphs() -> list[VerificationGraph]:
    """Return the fixed ForceAtlas1 verification corpus.

    Returns
    -------
    list[VerificationGraph]
        Small deterministic graphs covering defaults and requested variants.
    """
    return [
        VerificationGraph(
            name="path_default",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4)]),
            num_nodes=5,
        ),
        VerificationGraph(
            name="weighted_outbound",
            edge_index=_edge_index([(0, 1), (0, 2), (2, 3), (3, 1), (3, 4)]),
            num_nodes=5,
            edge_weights=torch.tensor([1.0, 0.5, 2.0, 1.5, 0.75], dtype=torch.float64),
            params={"outbound_attraction_distribution": True},
        ),
        VerificationGraph(
            name="adjust_sizes",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 0), (2, 3)]),
            num_nodes=4,
            node_sizes=torch.full((4, 2), 18.0),
            params={"adjust_sizes": True},
        ),
        VerificationGraph(
            name="no_freeze_gravity",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
            num_nodes=4,
            params={"freeze_balance": False, "gravity": 10.0},
        ),
    ]


def _ensure_gephi_toolkit() -> Path:
    """Download the Gephi toolkit jar when it is not already cached.

    Returns
    -------
    pathlib.Path
        Path to the cached toolkit jar.

    Raises
    ------
    ReferenceRuntimeError
        If Maven Central cannot provide the toolkit artifact.
    """
    GEPHI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GEPHI_TOOLKIT_JAR.exists() and GEPHI_TOOLKIT_JAR.stat().st_size > 0:
        return GEPHI_TOOLKIT_JAR
    try:
        urllib.request.urlretrieve(GEPHI_TOOLKIT_URL, GEPHI_TOOLKIT_JAR)
    except OSError as exc:
        raise ReferenceRuntimeError(f"gephi-toolkit-download: {exc}") from exc
    return GEPHI_TOOLKIT_JAR


def _compile_gephi_runner(toolkit_jar: Path) -> Path:
    """Compile the Java ForceAtlas1 reference runner.

    Parameters
    ----------
    toolkit_jar : pathlib.Path
        Path to ``gephi-toolkit-0.10.1-all.jar``.

    Returns
    -------
    pathlib.Path
        Directory containing compiled runner classes.

    Raises
    ------
    ReferenceRuntimeError
        If ``javac`` is unavailable or compilation fails.
    """
    RUNNER_CLASSES.mkdir(parents=True, exist_ok=True)
    class_file = RUNNER_CLASSES / f"{RUNNER_CLASS}.class"
    if class_file.exists() and class_file.stat().st_mtime >= RUNNER_SOURCE.stat().st_mtime:
        return RUNNER_CLASSES
    command = [
        "javac",
        "-cp",
        str(toolkit_jar),
        "-d",
        str(RUNNER_CLASSES),
        str(RUNNER_SOURCE),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise ReferenceRuntimeError(f"javac-unavailable: {exc}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ReferenceRuntimeError(f"javac-compile-failed: {message}")
    return RUNNER_CLASSES


def _fidelity_tier(residual: float, max_abs: float) -> str:
    """Classify a ForceAtlas1 residual.

    Parameters
    ----------
    residual : float
        Procrustes RMSD between compared layouts.
    max_abs : float
        Maximum absolute coordinate difference before alignment.

    Returns
    -------
    str
        Fidelity tier label for the report.
    """
    if max_abs == 0.0:
        return "bit-exact"
    if residual <= 1.0e-6:
        return "positional"
    return "distributional"


def _format_float(value: float) -> str:
    """Format a float for round-tripping through Java ``Float.parseFloat``.

    Parameters
    ----------
    value : float
        Numeric value to serialize.

    Returns
    -------
    str
        Compact decimal representation.
    """
    return repr(float(value))


def _gephi_input(graph: VerificationGraph, initial_positions: torch.Tensor) -> str:
    """Serialize one verification graph for the Java reference runner.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and parameters.
    initial_positions : torch.Tensor
        Fixed initial coordinates with shape ``[N, 2]``.

    Returns
    -------
    str
        Tab-separated runner input.
    """
    params = dict(graph.params or {})
    edges = graph.edge_index.detach().cpu().T.tolist()
    edge_weights = _edge_weight_values(graph.edge_weights, len(edges))
    node_sizes = _node_size_values(graph.node_sizes, graph.num_nodes)
    header = [
        graph.num_nodes,
        len(edges),
        VERIFY_STEPS,
        params.get("attraction_strength", 10.0),
        params.get("repulsion_strength", 200.0),
        params.get("inertia", 0.1),
        str(bool(params.get("outbound_attraction_distribution", False))).lower(),
        str(bool(params.get("adjust_sizes", False))).lower(),
        str(bool(params.get("freeze_balance", True))).lower(),
        params.get("freeze_strength", 80.0),
        params.get("freeze_inertia", 0.2),
        params.get("gravity", 30.0),
        params.get("speed", 1.0),
        params.get("cooling", 1.0),
        params.get("max_displacement", 10.0),
    ]
    lines = ["\t".join(str(value) for value in header)]
    for node_index in range(graph.num_nodes):
        x_coord = _format_float(float(initial_positions[node_index, 0].item()))
        y_coord = _format_float(float(initial_positions[node_index, 1].item()))
        size = _format_float(node_sizes[node_index])
        lines.append(f"n{node_index}\t{x_coord}\t{y_coord}\t{size}\tfalse")
    for (source, target), weight in zip(edges, edge_weights):
        lines.append(f"{int(source)}\t{int(target)}\t{_format_float(weight)}")
    return "\n".join(lines) + "\n"


def _run_gephi_reference(graph: VerificationGraph, initial_positions: torch.Tensor) -> torch.Tensor:
    """Run Gephi ForceAtlas1 through the headless toolkit runner.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and parameters.
    initial_positions : torch.Tensor
        Fixed initial coordinates with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Gephi final coordinates with shape ``[N, 2]``.

    Raises
    ------
    ReferenceRuntimeError
        If Java, the toolkit, or the runner fails.
    """
    toolkit_jar = _ensure_gephi_toolkit()
    classes_dir = _compile_gephi_runner(toolkit_jar)
    command = [
        "java",
        "-Djava.awt.headless=true",
        "-cp",
        f"{classes_dir}:{toolkit_jar}",
        RUNNER_CLASS,
    ]
    try:
        completed = subprocess.run(
            command,
            input=_gephi_input(graph, initial_positions),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ReferenceRuntimeError(f"java-unavailable: {exc}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ReferenceRuntimeError(f"gephi-toolkit-runtime: {message}")

    rows: list[tuple[int, float, float]] = []
    for line in completed.stdout.splitlines():
        node_index, x_coord, y_coord = line.split("\t")
        rows.append((int(node_index), float(x_coord), float(y_coord)))
    rows.sort(key=lambda row: row[0])
    if len(rows) != graph.num_nodes:
        raise ReferenceRuntimeError(
            f"gephi-output-shape: expected {graph.num_nodes} rows, got {len(rows)}"
        )
    return torch.tensor([[x_coord, y_coord] for _, x_coord, y_coord in rows], dtype=torch.float64)


def verify_graph(graph: VerificationGraph) -> VerificationResult:
    """Verify one graph and return residual, tier, and quality.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and ForceAtlas1 parameters.

    Returns
    -------
    VerificationResult
        Residuals, tier, and quick composite quality.
    """
    params = dict(graph.params or {})
    initial_positions = layout_forceatlas1_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=graph.edge_weights,
        steps=0,
        seed=VERIFY_SEED,
        fidelity_dtype=torch.float64,
        **params,
    )
    common = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "node_sizes": graph.node_sizes,
        "edge_weights": graph.edge_weights,
        "steps": VERIFY_STEPS,
        "seed": VERIFY_SEED,
        "fidelity_dtype": torch.float64,
        **params,
    }
    source_port = layout_forceatlas1_pipeline(**common)
    reference = _run_gephi_reference(graph, initial_positions)
    residual = procrustes_rmsd(reference, source_port)
    max_abs = float(torch.max(torch.abs(reference - source_port)).item())
    quality_metrics = quick(
        reference.to(dtype=torch.float32),
        graph.edge_index,
        node_sizes=graph.node_sizes,
        seed=VERIFY_SEED,
    )
    quality = composite(quality_metrics)
    return VerificationResult(
        residual=residual,
        max_abs=max_abs,
        tier=_fidelity_tier(residual, max_abs),
        quality=quality,
    )


def main() -> None:
    """Print the ForceAtlas1 fidelity report.

    Returns
    -------
    None
        Writes a line-oriented report to stdout.
    """
    print("ForceAtlas1 fidelity verification")
    print("reference_runtime: Gephi toolkit 0.10.1 headless Java runner")
    print("reference_runtime_path: direct GraphModel factory + ForceAtlasLayout.goAlgo")
    print("model_status: externally verified Gephi ForceAtlasLayout.java port")
    try:
        for graph in _verification_graphs():
            result = verify_graph(graph)
            print(
                f"{graph.name}: residual={result.residual:.3e} "
                f"max_abs={result.max_abs:.3e} tier={result.tier} "
                f"quality={result.quality:.2f}"
            )
    except ReferenceRuntimeError as exc:
        print(f"reference_runtime: blocked ({exc})")
        print("first_divergent_stage: reference-runtime")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
