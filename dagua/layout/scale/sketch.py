"""Topology sketching for scale-aware layout routing."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SKETCH_SCHEMA_VERSION = "topology-sketch-v1"
_FINGERPRINT_SIZE_BYTES = 16
_BYTES_PER_NODE_ESTIMATE = 160
_BYTES_PER_EDGE_ESTIMATE = 96
_PYTHON_WORKSPACE_FACTOR = 3.0


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

        edges = edge_index.detach().to(device="cpu", dtype=torch.long).contiguous()
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

        sources, targets = _validate_edges(edges, n)
        total_degree = _total_degrees_from_edges(edges, n)
        components, largest_component, scc_sizes, is_directed = _component_summary_from_edges(
            edges,
            n,
        )
        largest_scc = max(scc_sizes) if scc_sizes else 0
        has_self_loop = bool((edges[0] == edges[1]).any().item())
        is_acyclic = largest_scc <= 1 and not has_self_loop
        if not is_acyclic:
            depth = None
            depth_tripped = False
            scanned = 0
        else:
            out_neighbors, in_neighbors, _undirected_neighbors = _build_adjacency(
                sources,
                targets,
                n,
            )
            depth, depth_tripped, scanned = _capped_longest_path_depth(
                out_neighbors,
                in_neighbors,
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
            max_degree=max(total_degree) if total_degree else 0,
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


def _validate_edges(edge_index: torch.Tensor, num_nodes: int) -> Tuple[List[int], List[int]]:
    """Validate edge bounds and return source/target lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of valid node IDs.

    Returns
    -------
    tuple[list[int], list[int]]
        Source and target node ID lists.
    """
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    for source, target in zip(sources, targets):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes).")
    return sources, targets


def _total_degrees_from_edges(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Return total directed degree for every node.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        ``out_degree + in_degree`` for each node.
    """
    out_degree = torch.bincount(edge_index[0], minlength=int(num_nodes))
    in_degree = torch.bincount(edge_index[1], minlength=int(num_nodes))
    return (out_degree + in_degree).to(dtype=torch.long).tolist()


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
    asymmetry = (matrix != matrix.transpose()).nnz > 0
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


def _percentile(values: List[int], percentile: float) -> float:
    """Compute a deterministic nearest-rank percentile.

    Parameters
    ----------
    values : list[int]
        Integer values.
    percentile : float
        Percentile in ``[0, 100]``.

    Returns
    -------
    float
        Nearest-rank percentile value.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round((float(percentile) / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(rank, len(ordered) - 1))])


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
    digest.update(edge_index.numpy().tobytes(order="C"))
    return digest.hexdigest()


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
    base = int(num_nodes) * _BYTES_PER_NODE_ESTIMATE + int(num_edges) * _BYTES_PER_EDGE_ESTIMATE
    return int(base * _PYTHON_WORKSPACE_FACTOR)


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
