"""Topology sketching for scale-aware layout routing."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

SKETCH_SCHEMA_VERSION = "topology-sketch-v1"
_FINGERPRINT_SIZE_BYTES = 16
_INT32_MAX = 2_147_483_647
_BYTES_PER_INT32 = 4
_BYTES_PER_INT64 = 8
_BYTES_PER_BOOL = 1
_BYTES_PER_UINT8 = 1
_DEGREE_SAMPLE_NODE_CAP = 1_000_000
_DEFAULT_BOUNDED_SAMPLE_EDGES = 2_000_000
_DEFAULT_BOUNDED_CHUNK_EDGES = 10_000_000
_EXACT_CSR_WORKSPACE_MULTIPLIER = 2
_DEPTH_FRONTIER_MASKS = 2
_DECLARED_TOPOLOGIES = {"directed_cyclic", "directed_acyclic", "undirected"}


@dataclass(frozen=True)
class TopologySketch:
    """Immutable graph sketch used by the scale router.

    Parameters
    ----------
    schema_version : str
        Version tag for persisted sketch compatibility.
    fingerprint : str
        Stable topology fingerprint derived from graph shape and edge bytes.
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.
    edges_per_node : float
        Directed edge density as ``E / max(N, 1)``.
    degree_p50 : float
        Median total degree.
    degree_p90 : float
        90th percentile total degree.
    degree_p99 : float
        99th percentile total degree.
    max_degree : int
        Maximum total degree.
    connected_components : int
        Number of weakly connected components.
    largest_component_size : int
        Size of the largest weakly connected component.
    scc_count : int
        Number of exact directed strongly connected components.
    largest_scc_size : int
        Size of the largest exact directed SCC.
    is_directed : bool
        Whether the topology contains at least one asymmetric edge.
    is_acyclic : bool
        Whether every directed SCC is a singleton and no self-loop exists.
    depth : Optional[int]
        Longest-path depth when the capped probe finishes, otherwise ``None``.
    depth_cap : int
        Configured maximum depth accepted by the probe.
    depth_cap_tripped : bool
        Whether longest-path probing exceeded the cap or scan budget.
    depth_edges_scanned : int
        Number of directed edges touched by the capped depth probe.
    peak_bytes : int
        Conservative peak byte estimate for scale-stage routing.
    """

    schema_version: str
    fingerprint: str
    num_nodes: int
    num_edges: int
    edges_per_node: float
    degree_p50: float
    degree_p90: float
    degree_p99: float
    max_degree: int
    connected_components: int
    largest_component_size: int
    scc_count: int
    largest_scc_size: int
    is_directed: bool
    is_acyclic: bool
    depth: Optional[int]
    depth_cap: int
    depth_cap_tripped: bool
    depth_edges_scanned: int
    peak_bytes: int

    @classmethod
    def from_edge_index(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        depth_cap: int = 256,
    ) -> "TopologySketch":
        """Build an exact structural sketch from a directed edge tensor.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``. Entries are interpreted as
            directed for SCC and depth probing and as undirected for weak
            connected components.
        num_nodes : int
            Number of graph nodes.
        depth_cap : int, default=256
            Longest-path probe cap. A DAG whose depth exceeds this cap is
            considered hostile to layered routing.

        Returns
        -------
        TopologySketch
            Immutable, fingerprinted sketch for router decisions.
        """
        n = int(num_nodes)
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, E].")

        edges = _compact_cpu_edges(edge_index, n)
        e = int(edges.shape[1])
        if e == 0:
            return cls(
                schema_version=SKETCH_SCHEMA_VERSION,
                fingerprint=_fingerprint_edges(edges, n),
                num_nodes=n,
                num_edges=0,
                edges_per_node=0.0,
                degree_p50=0.0,
                degree_p90=0.0,
                degree_p99=0.0,
                max_degree=0,
                connected_components=max(n, 0),
                largest_component_size=1 if n > 0 else 0,
                scc_count=max(n, 0),
                largest_scc_size=1 if n > 0 else 0,
                is_directed=False,
                is_acyclic=True,
                depth=0 if n > 0 else 0,
                depth_cap=int(depth_cap),
                depth_cap_tripped=False,
                depth_edges_scanned=0,
                peak_bytes=_estimate_peak_bytes(n, 0),
            )

        _validate_edge_bounds(edges, n)
        total_degree = _total_degrees_from_edges(edges, n)
        components, largest_component, scc_sizes, is_directed = _component_summary_from_edges(
            edges,
            n,
        )
        largest_scc = max(scc_sizes) if scc_sizes else 0
        has_self_loop = _has_self_loop(edges, chunk_edges=_DEFAULT_BOUNDED_CHUNK_EDGES)
        is_acyclic = largest_scc <= 1 and not has_self_loop
        if not is_acyclic:
            depth = None
            depth_tripped = False
            scanned = 0
        else:
            depth, depth_tripped, scanned = _capped_longest_path_depth_from_edges(
                edges,
                n,
                depth_cap=int(depth_cap),
                edge_budget=max(int(depth_cap) * max(e, 1), e),
            )
        return cls(
            schema_version=SKETCH_SCHEMA_VERSION,
            fingerprint=_fingerprint_edges(edges, n),
            num_nodes=n,
            num_edges=e,
            edges_per_node=float(e) / float(max(n, 1)),
            degree_p50=_percentile(total_degree, 50.0),
            degree_p90=_percentile(total_degree, 90.0),
            degree_p99=_percentile(total_degree, 99.0),
            max_degree=_max_degree(total_degree),
            connected_components=components,
            largest_component_size=largest_component,
            scc_count=len(scc_sizes),
            largest_scc_size=largest_scc,
            is_directed=is_directed,
            is_acyclic=is_acyclic,
            depth=depth,
            depth_cap=int(depth_cap),
            depth_cap_tripped=depth_tripped,
            depth_edges_scanned=scanned,
            peak_bytes=_estimate_peak_bytes(n, e),
        )

    @classmethod
    def from_edge_index_bounded(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        depth_cap: int = 256,
        sample_edges: int = _DEFAULT_BOUNDED_SAMPLE_EDGES,
        chunk_edges: int = _DEFAULT_BOUNDED_CHUNK_EDGES,
    ) -> "TopologySketch":
        """Build a bounded-memory sketch for over-budget scale routing.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of graph nodes.
        depth_cap : int, default=256
            Longest-path cap recorded in the sketch. The bounded sketch does
            not run a global depth pass.
        sample_edges : int, default=2_000_000
            Maximum deterministic edge samples used for degree percentiles.
        chunk_edges : int, default=10_000_000
            Edge chunk size for forward/cyclic probing.

        Returns
        -------
        TopologySketch
            Approximate sketch with conservative route fields.

        Notes
        -----
        This path is used only when the exact sketch exceeds the declared
        memory budget. It proves LAYERS only for forward-only edge tensors
        whose capped depth passes; any self-loop or backward edge is
        conservatively routed as FIELD.
        """
        n = int(num_nodes)
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, E].")
        edges = _compact_cpu_edges(edge_index, n)
        e = int(edges.shape[1])
        if e == 0:
            return cls.from_edge_index(edges, n, depth_cap=depth_cap)

        sample = _deterministic_edge_sample(edges, min(int(sample_edges), e))
        _validate_edge_bounds_chunked(edges, n, chunk_edges=int(chunk_edges))
        sampled_degree = _bounded_total_degrees(edges, n, chunk_edges=int(chunk_edges))
        has_backward_or_self = _has_backward_or_self_edge(edges, chunk_edges=int(chunk_edges))
        is_acyclic = not has_backward_or_self
        if is_acyclic:
            depth, depth_tripped, scanned = _capped_longest_path_depth_from_edges(
                edges,
                n,
                depth_cap=int(depth_cap),
                edge_budget=max(int(depth_cap) * max(e, 1), e),
            )
        else:
            depth = None
            depth_tripped = False
            scanned = e
        largest_scc = 1 if is_acyclic else max(_nontrivial_scc_min_size_for_sketch(n), 2)
        scc_count = n if is_acyclic else max(1, n - largest_scc + 1)
        return cls(
            schema_version=SKETCH_SCHEMA_VERSION,
            fingerprint=_fingerprint_edges(edges, n),
            num_nodes=n,
            num_edges=e,
            edges_per_node=float(e) / float(max(n, 1)),
            degree_p50=_percentile(sampled_degree, 50.0),
            degree_p90=_percentile(sampled_degree, 90.0),
            degree_p99=_percentile(sampled_degree, 99.0),
            max_degree=_max_degree(sampled_degree),
            connected_components=1 if e > 0 else max(n, 0),
            largest_component_size=n if e > 0 else (1 if n > 0 else 0),
            scc_count=scc_count,
            largest_scc_size=largest_scc,
            is_directed=True,
            is_acyclic=is_acyclic,
            depth=depth,
            depth_cap=int(depth_cap),
            depth_cap_tripped=depth_tripped,
            depth_edges_scanned=scanned,
            peak_bytes=_estimate_bounded_peak_bytes(n, e, int(sample.shape[1])),
        )

    @classmethod
    def from_declared_topology(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        topology: str,
        *,
        declared_num_edges: Optional[int] = None,
        depth_cap: int = 256,
        depth: Optional[int] = None,
        depth_cap_tripped: Optional[bool] = None,
        chunk_edges: int = _DEFAULT_BOUNDED_CHUNK_EDGES,
    ) -> "TopologySketch":
        """Build a sketch from caller-declared topology plus bounded stats.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``. The declared path scans it only
            for bounds, degree percentiles, and fingerprinting; SCC and depth
            are not recomputed.
        num_nodes : int
            Number of graph nodes.
        topology : str
            One of ``"directed_cyclic"``, ``"directed_acyclic"``, or
            ``"undirected"``.
        declared_num_edges : int, optional
            Caller-declared edge count. When present it must match
            ``edge_index.shape[1]``.
        depth_cap : int, default=256
            Router depth cap recorded in the sketch.
        depth : int, optional
            Caller-declared DAG depth. Values above ``depth_cap`` trip the cap.
        depth_cap_tripped : bool, optional
            Explicit caller declaration that the DAG depth cap tripped.
        chunk_edges : int, default=10_000_000
            Edge columns processed per bounded scan chunk.

        Returns
        -------
        TopologySketch
            Sketch whose route-critical topology fields come from the caller
            declaration and whose degree fields come from bounded scans.
        """
        n = int(num_nodes)
        normalized = _normalize_declared_topology(topology)
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, E].")
        edges = _compact_cpu_edges(edge_index, n)
        e = int(edges.shape[1])
        if declared_num_edges is not None and int(declared_num_edges) != e:
            raise ValueError("declared_num_edges must match edge_index.shape[1].")
        _validate_edge_bounds_chunked(edges, n, chunk_edges=int(chunk_edges))
        total_degree = _bounded_total_degrees(edges, n, chunk_edges=int(chunk_edges))
        depth_value, tripped = _declared_depth_fields(
            normalized,
            depth=depth,
            depth_cap=int(depth_cap),
            depth_cap_tripped=depth_cap_tripped,
        )
        is_acyclic = normalized == "directed_acyclic"
        largest_scc = 1 if is_acyclic else max(_nontrivial_scc_min_size_for_sketch(n), 2)
        scc_count = n if is_acyclic else max(1, n - largest_scc + 1)
        return cls(
            schema_version=SKETCH_SCHEMA_VERSION,
            fingerprint=_fingerprint_edges(edges, n),
            num_nodes=n,
            num_edges=e,
            edges_per_node=float(e) / float(max(n, 1)),
            degree_p50=_percentile(total_degree, 50.0),
            degree_p90=_percentile(total_degree, 90.0),
            degree_p99=_percentile(total_degree, 99.0),
            max_degree=_max_degree(total_degree),
            connected_components=1 if e > 0 else max(n, 0),
            largest_component_size=n if e > 0 else (1 if n > 0 else 0),
            scc_count=scc_count,
            largest_scc_size=largest_scc,
            is_directed=normalized != "undirected",
            is_acyclic=is_acyclic,
            depth=depth_value,
            depth_cap=int(depth_cap),
            depth_cap_tripped=tripped,
            depth_edges_scanned=0,
            peak_bytes=_estimate_declared_peak_bytes(n, e),
        )

    @classmethod
    def load(cls, path: Path) -> "TopologySketch":
        """Load and validate a persisted sketch JSON file.

        Parameters
        ----------
        path : pathlib.Path
            JSON path written by :meth:`save`.

        Returns
        -------
        TopologySketch
            Persisted sketch after schema validation.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        sketch = cls.from_dict(payload)
        if sketch.schema_version != SKETCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported sketch schema {sketch.schema_version!r}; "
                f"expected {SKETCH_SCHEMA_VERSION!r}"
            )
        return sketch

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TopologySketch":
        """Rehydrate a sketch from a typed dictionary.

        Parameters
        ----------
        payload : dict[str, object]
            Serialized dataclass fields.

        Returns
        -------
        TopologySketch
            Immutable sketch instance.
        """
        return cls(**payload)

    def save(self, path: Path) -> None:
        """Persist the sketch to JSON with schema and fingerprint fields.

        Parameters
        ----------
        path : pathlib.Path
            Destination JSON path. Parent directories must already exist.

        Returns
        -------
        None
            The file is written in UTF-8 JSON format.
        """
        path.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        """Return the sketch as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, object]
            Serialized sketch fields.
        """
        return asdict(self)


def _compact_cpu_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return CPU-contiguous edges using int32 when node IDs fit.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        CPU edge tensor with shape ``[2, E]`` and compact integer dtype.
    """
    dtype = torch.int32 if int(num_nodes) <= _INT32_MAX else torch.int64
    return edge_index.detach().to(device="cpu", dtype=dtype).contiguous()


def _validate_edges(edge_index: torch.Tensor, num_nodes: int) -> Tuple[List[int], List[int]]:
    """Validate edge bounds and return source/target lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of valid node IDs.

    Returns
    -------
    tuple[list[int], list[int]]
        Source and target node ID lists.
    """
    _validate_edge_bounds(edge_index, int(num_nodes))
    sources = edge_index[0].to(dtype=torch.long).tolist()
    targets = edge_index[1].to(dtype=torch.long).tolist()
    return sources, targets


def _validate_edge_bounds(edge_index: torch.Tensor, num_nodes: int) -> None:
    """Validate all edge endpoints with vectorized min/max checks.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of valid node IDs.

    Returns
    -------
    None
        Raises ``ValueError`` if any endpoint is out of bounds.
    """
    if edge_index.numel() == 0:
        return
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= int(num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes).")


def _validate_edge_bounds_chunked(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    chunk_edges: int,
) -> None:
    """Validate edge endpoints in bounded chunks.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of valid node IDs.
    chunk_edges : int
        Edge columns processed per chunk.

    Returns
    -------
    None
        Raises ``ValueError`` if any endpoint is out of bounds.
    """
    chunk = max(1, int(chunk_edges))
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        _validate_edge_bounds(edge_index[:, start:end], int(num_nodes))


def _total_degrees_from_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return total directed degree for every node.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        ``out_degree + in_degree`` tensor with shape ``[N]``.
    """
    return _bounded_total_degrees(
        edge_index,
        int(num_nodes),
        chunk_edges=_DEFAULT_BOUNDED_CHUNK_EDGES,
    )


def _bounded_total_degrees(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    chunk_edges: int,
) -> torch.Tensor:
    """Return exact total degrees using one node-sized counter.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    chunk_edges : int
        Edge columns processed per chunk.

    Returns
    -------
    torch.Tensor
        Exact total degree tensor with shape ``[N]``.
    """
    degree_dtype = _degree_dtype_for_edges(int(edge_index.shape[1]))
    degree = torch.zeros((int(num_nodes),), dtype=degree_dtype)
    chunk = max(1, int(chunk_edges))
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        sources = edge_index[0, start:end].to(dtype=torch.long)
        targets = edge_index[1, start:end].to(dtype=torch.long)
        ones = torch.ones((end - start,), dtype=degree_dtype)
        degree.index_add_(0, sources, ones)
        degree.index_add_(0, targets, ones)
    return degree


def _degree_dtype_for_edges(num_edges: int) -> torch.dtype:
    """Return the smallest safe degree-counter dtype.

    Parameters
    ----------
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    torch.dtype
        ``torch.int32`` when the total endpoint touches fit, otherwise
        ``torch.int64``.
    """
    return torch.int32 if int(num_edges) * 2 <= _INT32_MAX else torch.int64


def _component_summary_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[int, int, List[int], bool]:
    """Return weak/SCC component summaries using CSR when available.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[int, int, list[int], bool]
        Weak component count, largest weak component size, exact SCC sizes, and
        whether the directed adjacency is asymmetric.
    """
    try:
        return _component_summary_scipy(edge_index, int(num_nodes))
    except Exception:
        sources = edge_index[0].tolist()
        targets = edge_index[1].tolist()
        out_neighbors, _in_neighbors, undirected_neighbors = _build_adjacency(
            sources,
            targets,
            int(num_nodes),
        )
        components, largest_component = _weak_components(undirected_neighbors)
        scc_sizes = _strongly_connected_component_sizes(out_neighbors)
        return components, largest_component, scc_sizes, _has_asymmetric_edge(sources, targets)


def _component_summary_scipy(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[int, int, List[int], bool]:
    """Return exact component summaries through scipy sparse graph kernels.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[int, int, list[int], bool]
        Weak component count, largest weak component size, exact SCC sizes, and
        directedness.
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse import csgraph

    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    data = np.ones((int(edge_index.shape[1]),), dtype=np.uint8)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(int(num_nodes), int(num_nodes)))
    weak_count, weak_labels = csgraph.connected_components(
        matrix,
        directed=True,
        connection="weak",
        return_labels=True,
    )
    weak_sizes = np.bincount(weak_labels, minlength=int(weak_count))
    strong_count, strong_labels = csgraph.connected_components(
        matrix,
        directed=True,
        connection="strong",
        return_labels=True,
    )
    strong_sizes = np.bincount(strong_labels, minlength=int(strong_count)).astype(np.int64)
    asymmetry_matrix: Any = matrix != matrix.transpose()
    asymmetry = asymmetry_matrix.nnz > 0
    return (
        int(weak_count),
        int(weak_sizes.max()) if weak_sizes.size else 0,
        [int(value) for value in strong_sizes.tolist()],
        bool(asymmetry),
    )


def _build_adjacency(
    sources: List[int],
    targets: List[int],
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Build directed and weak adjacency lists in one pass.

    Parameters
    ----------
    sources : list[int]
        Directed edge source nodes.
    targets : list[int]
        Directed edge target nodes.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[list[list[int]], list[list[int]], list[list[int]]]
        Out-neighbors, in-neighbors, and weak-neighbor lists.
    """
    out_neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    in_neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    weak_neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(sources, targets):
        out_neighbors[source].append(target)
        in_neighbors[target].append(source)
        weak_neighbors[source].append(target)
        if source != target:
            weak_neighbors[target].append(source)
    return out_neighbors, in_neighbors, weak_neighbors


def _weak_components(weak_neighbors: List[List[int]]) -> Tuple[int, int]:
    """Return weak connected component count and largest size.

    Parameters
    ----------
    weak_neighbors : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    tuple[int, int]
        Component count and largest component cardinality.
    """
    n = len(weak_neighbors)
    visited = [False] * n
    components = 0
    largest = 0
    for root in range(n):
        if visited[root]:
            continue
        components += 1
        size = 0
        visited[root] = True
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in weak_neighbors[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        largest = max(largest, size)
    return components, largest


def _strongly_connected_component_sizes(out_neighbors: List[List[int]]) -> List[int]:
    """Compute exact SCC sizes with iterative Tarjan traversal.

    Parameters
    ----------
    out_neighbors : list[list[int]]
        Directed adjacency list.

    Returns
    -------
    list[int]
        Sizes of all exact strongly connected components.
    """
    n = len(out_neighbors)
    index = 0
    indices = [-1] * n
    lowlinks = [0] * n
    stack: List[int] = []
    on_stack = [False] * n
    scc_sizes: List[int] = []

    for root in range(n):
        if indices[root] >= 0:
            continue
        frames: List[Tuple[int, int]] = [(root, 0)]
        parent: List[Optional[int]] = [None] * n
        while frames:
            node, next_child = frames[-1]
            if indices[node] < 0:
                indices[node] = index
                lowlinks[node] = index
                index += 1
                stack.append(node)
                on_stack[node] = True

            if next_child < len(out_neighbors[node]):
                neighbor = out_neighbors[node][next_child]
                frames[-1] = (node, next_child + 1)
                if indices[neighbor] < 0:
                    parent[neighbor] = node
                    frames.append((neighbor, 0))
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])
                continue

            frames.pop()
            node_parent = parent[node]
            if node_parent is not None:
                lowlinks[node_parent] = min(lowlinks[node_parent], lowlinks[node])
            if lowlinks[node] == indices[node]:
                size = 0
                while stack:
                    member = stack.pop()
                    on_stack[member] = False
                    size += 1
                    if member == node:
                        break
                scc_sizes.append(size)
    return scc_sizes


def _capped_longest_path_depth(
    out_neighbors: List[List[int]],
    in_neighbors: List[List[int]],
    *,
    depth_cap: int,
    edge_budget: int,
) -> Tuple[Optional[int], bool, int]:
    """Probe longest DAG depth with a cap and edge-scan budget.

    Parameters
    ----------
    out_neighbors : list[list[int]]
        Directed adjacency list.
    in_neighbors : list[list[int]]
        Reverse directed adjacency list.
    depth_cap : int
        Maximum acceptable depth.
    edge_budget : int
        Maximum edge touches allowed before aborting.

    Returns
    -------
    tuple[int or None, bool, int]
        Depth when known, whether the cap/budget tripped, and scanned edge
        count.
    """
    n = len(out_neighbors)
    indegree = [len(in_neighbors[node]) for node in range(n)]
    depth = [0] * n
    queue: deque[int] = deque(node for node, degree in enumerate(indegree) if degree == 0)
    visited = 0
    scanned = 0
    max_depth = 0
    while queue:
        node = queue.popleft()
        visited += 1
        node_depth = depth[node]
        max_depth = max(max_depth, node_depth)
        if max_depth > depth_cap or scanned > edge_budget:
            return None, True, scanned
        for neighbor in out_neighbors[node]:
            scanned += 1
            if scanned > edge_budget:
                return None, True, scanned
            next_depth = node_depth + 1
            if next_depth > depth[neighbor]:
                depth[neighbor] = next_depth
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if visited != n:
        return None, True, scanned
    if max_depth > depth_cap:
        return None, True, scanned
    return max_depth, False, scanned


def _capped_longest_path_depth_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    depth_cap: int,
    edge_budget: int,
) -> Tuple[Optional[int], bool, int]:
    """Probe DAG depth with bounded memory by scanning edge chunks per wave.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    depth_cap : int
        Maximum acceptable longest-path depth.
    edge_budget : int
        Maximum edge touches allowed before aborting.

    Returns
    -------
    tuple[int or None, bool, int]
        Depth when known, whether the cap/budget tripped, and scanned edge
        count.

    Notes
    -----
    This is an exact Kahn-style depth probe for DAGs, but it trades extra edge
    passes for bounded memory. It stops as soon as the configured cap or scan
    budget is exceeded, which is the only depth fact needed by the router.
    """
    n = int(num_nodes)
    if n == 0:
        return 0, False, 0
    indegree = torch.zeros((n,), dtype=_degree_dtype_for_edges(int(edge_index.shape[1])))
    chunk = _DEFAULT_BOUNDED_CHUNK_EDGES
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        targets = edge_index[1, start:end].to(dtype=torch.long)
        indegree.index_add_(0, targets, torch.ones((end - start,), dtype=indegree.dtype))

    active = indegree == 0
    seen = torch.zeros((n,), dtype=torch.bool)
    visited = 0
    scanned = 0
    depth = 0
    while bool(active.any().item()):
        frontier_size = int(active.sum().item())
        seen |= active
        visited += frontier_size
        if depth > int(depth_cap):
            return None, True, scanned
        next_active = torch.zeros((n,), dtype=torch.bool)
        for start in range(0, int(edge_index.shape[1]), chunk):
            end = min(int(edge_index.shape[1]), start + chunk)
            sources = edge_index[0, start:end].to(dtype=torch.long)
            targets = edge_index[1, start:end].to(dtype=torch.long)
            edge_mask = active[sources]
            scanned += end - start
            if scanned > int(edge_budget):
                return None, True, scanned
            if not bool(edge_mask.any().item()):
                continue
            touched = targets[edge_mask]
            decrement = torch.full((int(touched.shape[0]),), -1, dtype=indegree.dtype)
            indegree.index_add_(0, touched, decrement)
            maybe_ready = torch.unique(touched)
            ready = maybe_ready[indegree[maybe_ready] == 0]
            if ready.numel() > 0:
                next_active[ready] = True
        active = next_active & ~seen
        if bool(active.any().item()):
            depth += 1
    if visited != n:
        return None, True, scanned
    if depth > int(depth_cap):
        return None, True, scanned
    return depth, False, scanned


def _has_asymmetric_edge(sources: List[int], targets: List[int]) -> bool:
    """Return whether any directed edge lacks its reverse counterpart.

    Parameters
    ----------
    sources : list[int]
        Directed edge sources.
    targets : list[int]
        Directed edge targets.

    Returns
    -------
    bool
        ``True`` if the graph has an asymmetric directed edge.
    """
    pairs = set(zip(sources, targets))
    return any((target, source) not in pairs for source, target in pairs if source != target)


def _percentile(values: Union[List[int], torch.Tensor], percentile: float) -> float:
    """Compute a deterministic nearest-rank percentile.

    Parameters
    ----------
    values : list[int] or torch.Tensor
        Integer values.
    percentile : float
        Percentile in ``[0, 100]``.

    Returns
    -------
    float
        Nearest-rank percentile value.
    """
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return 0.0
        rank = int(round((float(percentile) / 100.0) * (int(values.numel()) - 1)))
        kth = max(1, min(rank + 1, int(values.numel())))
        return float(values.reshape(-1).kthvalue(kth).values.item())
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round((float(percentile) / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(rank, len(ordered) - 1))])


def _max_degree(values: Union[List[int], torch.Tensor]) -> int:
    """Return the maximum degree from a list or tensor.

    Parameters
    ----------
    values : list[int] or torch.Tensor
        Degree values.

    Returns
    -------
    int
        Maximum degree, or zero for empty values.
    """
    if isinstance(values, torch.Tensor):
        return int(values.max().item()) if values.numel() else 0
    return max(values) if values else 0


def _fingerprint_edges(edge_index: torch.Tensor, num_nodes: int) -> str:
    """Return a stable fingerprint for graph shape and edge bytes.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU contiguous long edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    str
        Hex digest suitable for persisted cache validation.
    """
    digest = hashlib.blake2b(digest_size=_FINGERPRINT_SIZE_BYTES)
    digest.update(SKETCH_SCHEMA_VERSION.encode("utf-8"))
    digest.update(int(num_nodes).to_bytes(8, "little", signed=False))
    digest.update(int(edge_index.shape[1]).to_bytes(8, "little", signed=False))
    chunk = _DEFAULT_BOUNDED_CHUNK_EDGES
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        chunk_edges = edge_index[:, start:end].to(dtype=torch.long).contiguous()
        digest.update(chunk_edges.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_edge_sample(edge_index: torch.Tensor, sample_edges: int) -> torch.Tensor:
    """Return evenly spaced edge columns without random state.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    sample_edges : int
        Maximum number of sampled columns.

    Returns
    -------
    torch.Tensor
        Sampled edge tensor with shape ``[2, K]``.
    """
    edge_count = int(edge_index.shape[1])
    count = min(max(0, int(sample_edges)), edge_count)
    if count == edge_count:
        return edge_index
    if count == 0:
        return torch.empty((2, 0), dtype=torch.long)
    sample_ids = torch.linspace(0, edge_count - 1, steps=count, dtype=torch.float64).to(
        dtype=torch.long
    )
    return edge_index[:, sample_ids].contiguous()


def _has_backward_or_self_edge(edge_index: torch.Tensor, *, chunk_edges: int) -> bool:
    """Return whether a chunked scan finds a self-loop or backward edge.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    chunk_edges : int
        Edge columns processed per chunk.

    Returns
    -------
    bool
        ``True`` when any ``source >= target`` endpoint pair is present.
    """
    chunk = max(1, int(chunk_edges))
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        if bool((edge_index[0, start:end] >= edge_index[1, start:end]).any().item()):
            return True
    return False


def _nontrivial_scc_min_size_for_sketch(num_nodes: int) -> int:
    """Return a conservative synthetic SCC size for bounded cyclic sketches.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    int
        Size large enough to trip FIELD's giant-SCC route.
    """
    return max(2, int(int(num_nodes) * 0.05))


def _normalize_declared_topology(topology: str) -> str:
    """Normalize and validate a caller-declared topology class.

    Parameters
    ----------
    topology : str
        Declared topology class.

    Returns
    -------
    str
        Normalized topology class.
    """
    normalized = str(topology).strip().lower()
    if normalized not in _DECLARED_TOPOLOGIES:
        allowed = ", ".join(sorted(_DECLARED_TOPOLOGIES))
        raise ValueError(f"scale_declared_topology must be one of: {allowed}")
    return normalized


def _declared_depth_fields(
    topology: str,
    *,
    depth: Optional[int],
    depth_cap: int,
    depth_cap_tripped: Optional[bool],
) -> Tuple[Optional[int], bool]:
    """Return router depth fields implied by a declared topology.

    Parameters
    ----------
    topology : str
        Normalized declared topology class.
    depth : int or None
        Caller-declared DAG depth.
    depth_cap : int
        Configured depth cap.
    depth_cap_tripped : bool or None
        Explicit caller depth-cap declaration.

    Returns
    -------
    tuple[int or None, bool]
        Sketch ``depth`` and ``depth_cap_tripped`` fields.
    """
    if topology != "directed_acyclic":
        return None, False
    explicit_tripped = bool(depth_cap_tripped) if depth_cap_tripped is not None else False
    if depth is None:
        return None, explicit_tripped
    depth_value = int(depth)
    tripped = explicit_tripped or depth_value > int(depth_cap)
    return (None if tripped else depth_value), tripped


def _has_self_loop(edge_index: torch.Tensor, *, chunk_edges: int) -> bool:
    """Return whether any edge chunk contains a self-loop.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    chunk_edges : int
        Edge columns processed per chunk.

    Returns
    -------
    bool
        ``True`` when ``source == target`` for any edge.
    """
    chunk = max(1, int(chunk_edges))
    for start in range(0, int(edge_index.shape[1]), chunk):
        end = min(int(edge_index.shape[1]), start + chunk)
        if bool((edge_index[0, start:end] == edge_index[1, start:end]).any().item()):
            return True
    return False


def _estimate_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate peak bytes required by scale routing machinery.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Conservative peak byte estimate.
    """
    n = int(num_nodes)
    e = int(num_edges)
    endpoint_bytes = _endpoint_index_bytes(n)
    input_edges = 2 * e * _BYTES_PER_INT64
    compact_edges = 2 * e * endpoint_bytes
    degree = n * _degree_counter_bytes(e)
    csr = _csr_bytes(n, e, endpoint_bytes)
    csr_workspace = _EXACT_CSR_WORKSPACE_MULTIPLIER * csr
    labels = 2 * n * _BYTES_PER_INT64
    depth_workspace = n * _degree_counter_bytes(e) + _DEPTH_FRONTIER_MASKS * n * _BYTES_PER_BOOL
    return int(
        input_edges + compact_edges + degree + csr + csr_workspace + labels + depth_workspace
    )


def _estimate_bounded_peak_bytes(num_nodes: int, num_edges: int, sample_edges: int) -> int:
    """Estimate peak bytes for bounded topology sketching.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    num_edges : int
        Number of directed edge entries.
    sample_edges : int
        Number of sampled edge entries.

    Returns
    -------
    int
        Conservative bounded-sketch peak estimate.
    """
    n = int(num_nodes)
    e = int(num_edges)
    endpoint_bytes = _endpoint_index_bytes(n)
    input_edges = 2 * e * _BYTES_PER_INT64
    compact_edges = 2 * e * endpoint_bytes
    degree = n * _degree_counter_bytes(e)
    chunk = min(e, _DEFAULT_BOUNDED_CHUNK_EDGES)
    chunk_workspace = 2 * chunk * _BYTES_PER_INT64
    sample = int(sample_edges) * 2 * _BYTES_PER_INT64
    node_probe = min(n, _DEGREE_SAMPLE_NODE_CAP) * _BYTES_PER_INT64
    return int(input_edges + compact_edges + degree + chunk_workspace + sample + node_probe)


def _estimate_declared_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate peak bytes for declared-topology routing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Conservative declared-bypass peak estimate.
    """
    return _estimate_bounded_peak_bytes(
        int(num_nodes),
        int(num_edges),
        min(int(num_edges), _DEFAULT_BOUNDED_SAMPLE_EDGES),
    )


def _endpoint_index_bytes(num_nodes: int) -> int:
    """Return endpoint index width for compact edge storage.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Bytes per endpoint index.
    """
    return _BYTES_PER_INT32 if int(num_nodes) <= _INT32_MAX else _BYTES_PER_INT64


def _degree_counter_bytes(num_edges: int) -> int:
    """Return degree counter width for the edge count.

    Parameters
    ----------
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Bytes per degree counter.
    """
    return _BYTES_PER_INT32 if int(num_edges) * 2 <= _INT32_MAX else _BYTES_PER_INT64


def _csr_bytes(num_nodes: int, num_edges: int, endpoint_bytes: int) -> int:
    """Return CSR storage bytes for one sparse adjacency matrix.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.
    endpoint_bytes : int
        Bytes per CSR column index.

    Returns
    -------
    int
        Modeled CSR bytes.
    """
    return int(
        int(num_edges) * (int(endpoint_bytes) + _BYTES_PER_UINT8)
        + (int(num_nodes) + 1) * _BYTES_PER_INT64
    )


def estimate_topology_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate peak bytes before constructing a topology sketch.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Conservative peak byte estimate for the sketching stage.
    """
    return _estimate_peak_bytes(num_nodes, num_edges)


def estimate_bounded_topology_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate peak bytes before constructing a bounded topology sketch.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Conservative peak byte estimate for bounded sketching.
    """
    return _estimate_bounded_peak_bytes(
        int(num_nodes),
        int(num_edges),
        min(int(num_edges), _DEFAULT_BOUNDED_SAMPLE_EDGES),
    )


def estimate_declared_topology_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate peak bytes before constructing a declared-topology sketch.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.

    Returns
    -------
    int
        Conservative peak byte estimate for the declared-topology bypass.
    """
    return _estimate_declared_peak_bytes(int(num_nodes), int(num_edges))
