"""Registered operations implementing the OpenOrd layout schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Optional, Protocol, Union, cast

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.drl import (
    DRLDensityGridConfig,
    _build_undirected_adjacency,
    _DrlParameters,
    _PhaseParameters,
    _run_reference_drl,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


class OpenOrdOptionObject(Protocol):
    """Attribute-style OpenOrd option container."""

    def __getattr__(self, name: str) -> object:
        """Return an option value by attribute name.

        Parameters
        ----------
        name : str
            Attribute name to resolve.

        Returns
        -------
        object
            Option value.
        """


OpenOrdOptions = Union[str, Mapping[str, object], OpenOrdOptionObject]


@dataclass(frozen=True)
class OpenOrdPrepareStateConfig:
    """Configuration for :class:`OpenOrdPrepareState`.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject, default="default"
        Preset name or override provider.
    edge_cut : float, optional
        Edge-cutting ratio in ``[0, 1]``. ``None`` uses the preset default.
    multilevel : bool, optional
        Whether to use the recursive coarsen/refine path. ``None`` enables it
        for graphs with at least 20 nodes while preserving small-corpus output.
    """

    options: OpenOrdOptions = "default"
    edge_cut: Optional[float] = None
    multilevel: Optional[bool] = None


_OPENORD_PRESETS: dict[str, _DrlParameters] = {
    "default": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsen": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsest": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(200, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "refine": _DrlParameters(
        edge_cut=0.5,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 500.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(0, 250.0, 0.5, 0.0),
    ),
    "final": _DrlParameters(
        edge_cut=0.5,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 50.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(25, 250.0, 0.5, 0.0),
    ),
}


def _lookup_option(options: OpenOrdOptions, name: str) -> Optional[object]:
    """Read one OpenOrd option from a mapping or attribute object.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject
        Raw OpenOrd options.
    name : str
        Option key.

    Returns
    -------
    object or None
        Returned option value if present.
    """
    if isinstance(options, str):
        return None
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def _resolve_openord_parameters(
    options: OpenOrdOptions,
    edge_cut: Optional[float] = None,
) -> _DrlParameters:
    """Resolve OpenOrd parameters against source-preserving presets.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject
        Preset name or override container.
    edge_cut : float, optional
        Explicit edge-cutting ratio overriding the preset.

    Returns
    -------
    _DrlParameters
        Resolved phase schedule and edge-cut parameter.

    Raises
    ------
    ValueError
        If a preset name is unknown or edge cutting is outside ``[0, 1]``.
    """
    preset_name = (
        options if isinstance(options, str) else str(_lookup_option(options, "preset") or "default")
    )
    try:
        default = _OPENORD_PRESETS[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(_OPENORD_PRESETS))
        raise ValueError(
            f"unknown OpenOrd preset {preset_name!r}; expected one of {available}."
        ) from exc

    values: dict[str, float] = {
        "edge_cut": default.edge_cut,
        "init_iterations": float(default.init.iterations),
        "init_temperature": default.init.temperature,
        "init_attraction": default.init.attraction,
        "init_damping_mult": default.init.damping_mult,
        "liquid_iterations": float(default.liquid.iterations),
        "liquid_temperature": default.liquid.temperature,
        "liquid_attraction": default.liquid.attraction,
        "liquid_damping_mult": default.liquid.damping_mult,
        "expansion_iterations": float(default.expansion.iterations),
        "expansion_temperature": default.expansion.temperature,
        "expansion_attraction": default.expansion.attraction,
        "expansion_damping_mult": default.expansion.damping_mult,
        "cooldown_iterations": float(default.cooldown.iterations),
        "cooldown_temperature": default.cooldown.temperature,
        "cooldown_attraction": default.cooldown.attraction,
        "cooldown_damping_mult": default.cooldown.damping_mult,
        "crunch_iterations": float(default.crunch.iterations),
        "crunch_temperature": default.crunch.temperature,
        "crunch_attraction": default.crunch.attraction,
        "crunch_damping_mult": default.crunch.damping_mult,
        "simmer_iterations": float(default.simmer.iterations),
        "simmer_temperature": default.simmer.temperature,
        "simmer_attraction": default.simmer.attraction,
        "simmer_damping_mult": default.simmer.damping_mult,
    }
    for key in tuple(values):
        override = _lookup_option(options=options, name=key)
        if override is not None:
            values[key] = float(cast(float, override))
    if edge_cut is not None:
        values["edge_cut"] = float(edge_cut)
    if not 0.0 <= values["edge_cut"] <= 1.0:
        raise ValueError("edge_cut must be between 0 and 1.")

    return _DrlParameters(
        edge_cut=values["edge_cut"],
        init=_PhaseParameters(
            int(values["init_iterations"]),
            values["init_temperature"],
            values["init_attraction"],
            values["init_damping_mult"],
        ),
        liquid=_PhaseParameters(
            int(values["liquid_iterations"]),
            values["liquid_temperature"],
            values["liquid_attraction"],
            values["liquid_damping_mult"],
        ),
        expansion=_PhaseParameters(
            int(values["expansion_iterations"]),
            values["expansion_temperature"],
            values["expansion_attraction"],
            values["expansion_damping_mult"],
        ),
        cooldown=_PhaseParameters(
            int(values["cooldown_iterations"]),
            values["cooldown_temperature"],
            values["cooldown_attraction"],
            values["cooldown_damping_mult"],
        ),
        crunch=_PhaseParameters(
            int(values["crunch_iterations"]),
            values["crunch_temperature"],
            values["crunch_attraction"],
            values["crunch_damping_mult"],
        ),
        simmer=_PhaseParameters(
            int(values["simmer_iterations"]),
            values["simmer_temperature"],
            values["simmer_attraction"],
            values["simmer_damping_mult"],
        ),
    )


def _initialize_openord_positions(num_nodes: int) -> torch.Tensor:
    """Create OpenOrd's default initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``. The
        reference ``Node`` constructor sets both coordinates to zero unless a
        ``.real`` file provides fixed coordinates.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    return torch.zeros((num_nodes, 2), dtype=torch.float64)


def _openord_full_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> list[tuple[int, int, float]]:
    """Convert tensor edges to positive OpenOrd ``.full`` rows.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    list[tuple[int, int, float]]
        Positive non-self-loop rows in input order.
    """
    if edge_index.numel() == 0:
        return []
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_weights is None:
        weights = torch.ones(edge_index_cpu.shape[1], dtype=torch.float64)
    else:
        weights = edge_weights.to(device="cpu", dtype=torch.float64)
    rows: list[tuple[int, int, float]] = []
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        weight = float(weights[edge_id].item())
        if int(source) != int(target) and weight > 0.0:
            rows.append((int(source), int(target), weight))
    return rows


def _openord_truncate_edges(
    full_edges: list[tuple[int, int, float]],
    num_nodes: int,
    topn: int,
    normalize: bool,
) -> list[tuple[int, int, float]]:
    """Create OpenOrd ``.int`` rows from ``.full`` rows.

    Parameters
    ----------
    full_edges : list[tuple[int, int, float]]
        Full positive similarity rows.
    num_nodes : int
        Number of nodes in the current level.
    topn : int
        Number of strongest links emitted for each row.
    normalize : bool
        Whether to apply OpenOrd row-sum normalization.

    Returns
    -------
    list[tuple[int, int, float]]
        Directed ``.int`` rows ordered like ``truncate.cpp``.
    """
    rows: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    for source, target, weight in full_edges:
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            rows[source][target] = float(weight)
            rows[target][source] = float(weight)
    denoms = [sum(row.values()) for row in rows] if normalize else [1.0 for _ in range(num_nodes)]
    int_edges: list[tuple[int, int, float]] = []
    for source, row in enumerate(rows):
        sortable: list[tuple[float, int]] = []
        for target, weight in row.items():
            denom = math.sqrt(denoms[source] * denoms[target])
            normalized = 0.0 if denom == 0.0 else weight / denom
            sortable.append((normalized, target))
        sortable.sort()
        for weight, target in reversed(sortable[-topn:]):
            int_edges.append((source, target, float(weight)))
    return int_edges


def _openord_edges_to_tensor(
    edges: list[tuple[int, int, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert OpenOrd edge rows to tensor inputs.

    Parameters
    ----------
    edges : list[tuple[int, int, float]]
        Edge rows.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index with shape ``[2, E]`` and weights with shape ``[E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float64)
    edge_index = torch.tensor([[source, target] for source, target, _ in edges], dtype=torch.long)
    weights = torch.tensor([weight for _, _, weight in edges], dtype=torch.float64)
    return edge_index.t().contiguous(), weights


def _openord_adjacency_to_iedges(
    adjacency: list[dict[int, float]],
) -> list[tuple[int, int, float]]:
    """Serialize mutable neighbor maps like OpenOrd ``write_sim``.

    Parameters
    ----------
    adjacency : list[dict[int, float]]
        Post-layout mutable adjacency.

    Returns
    -------
    list[tuple[int, int, float]]
        Directed remaining edge rows.
    """
    rows: list[tuple[int, int, float]] = []
    for source, neighbors in enumerate(adjacency):
        for target in sorted(neighbors):
            rows.append((source, target, float(neighbors[target])))
    return rows


def _openord_layout_rows(
    int_edges: list[tuple[int, int, float]],
    num_nodes: int,
    initial_positions: torch.Tensor,
    params: _DrlParameters,
    seed: int,
    density_config: DRLDensityGridConfig,
) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    """Run one OpenOrd layout level and return cut-edge state.

    Parameters
    ----------
    int_edges : list[tuple[int, int, float]]
        Current ``.int`` rows.
    num_nodes : int
        Number of current-level nodes.
    initial_positions : torch.Tensor
        Starting coordinates with shape ``[N, 2]``.
    params : _DrlParameters
        Phase schedule for this level.
    seed : int
        Seed for OpenOrd's libc RNG stream.
    density_config : DRLDensityGridConfig
        Density-grid constants.

    Returns
    -------
    tuple[torch.Tensor, list[tuple[int, int, float]]]
        Final positions and directed remaining edge rows.
    """
    edge_index, edge_weights = _openord_edges_to_tensor(int_edges)
    adjacency = _build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    positions = _run_reference_drl(
        initial_positions=initial_positions,
        adjacency=adjacency,
        params=params,
        seed=seed,
        density_config=density_config,
        rng_kind="libc",
    )
    return positions, _openord_adjacency_to_iedges(adjacency)


def _openord_average_link_clusters(
    positions: torch.Tensor,
    full_edges: list[tuple[int, int, float]],
    iedges: list[tuple[int, int, float]],
) -> list[int]:
    """Cluster one level using OpenOrd average-link rules.

    Parameters
    ----------
    positions : torch.Tensor
        Layout coordinates with shape ``[N, 2]``.
    full_edges : list[tuple[int, int, float]]
        Current full similarity rows.
    iedges : list[tuple[int, int, float]]
        Directed remaining edge rows after layout.

    Returns
    -------
    list[int]
        Fine-to-coarse assignment with shape ``[N]``.
    """
    num_nodes = int(positions.shape[0])
    pair_distances: dict[tuple[int, int], float] = {}
    for source, target, _ in iedges:
        if source != target:
            pair = (source, target) if source < target else (target, source)
            pair_distances[pair] = float(
                torch.linalg.vector_norm(positions[source] - positions[target]).item()
            )
    min_sim = [0.0 for _ in range(num_nodes)]
    short_rows: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    for source, target, weight in full_edges:
        if weight <= 0.0 or source == target:
            continue
        dist = float(torch.linalg.vector_norm(positions[source] - positions[target]).item())
        if min_sim[source] == 0.0 or min_sim[source] > dist:
            min_sim[source] = dist
        if min_sim[target] == 0.0 or min_sim[target] > dist:
            min_sim[target] = dist
        for row_source, row_target in ((source, target), (target, source)):
            row = short_rows[row_source]
            if len(row) < 1:
                row[row_target] = dist
            else:
                farthest = max(row, key=lambda node: (row[node], node))
                if dist < row[farthest]:
                    row[row_target] = dist
                    del row[farthest]
    for source, row in enumerate(short_rows):
        for target, dist in row.items():
            pair = (source, target) if source < target else (target, source)
            pair_distances[pair] = dist
    if not pair_distances:
        return list(range(num_nodes))
    positive = sorted(value for value in min_sim if value > 0.0)
    threshold = positive[len(positive) // 2] if positive else min(pair_distances.values())

    cluster = [0 for _ in range(num_nodes)]
    newcluster = [0 for _ in range(2 * num_nodes + 2)]
    sumdist = [0.0 for _ in range(2 * num_nodes + 2)]
    sum_x = [0.0 for _ in range(2 * num_nodes + 2)]
    sum_y = [0.0 for _ in range(2 * num_nodes + 2)]
    n_papers = [0 for _ in range(2 * num_nodes + 2)]
    n_cords = [0 for _ in range(2 * num_nodes + 2)]
    joinable = [0 for _ in range(2 * num_nodes + 2)]
    n_clusters = 0

    def root(cluster_id: int) -> int:
        """Return the live cluster id after average-link joins.

        Parameters
        ----------
        cluster_id : int
            Possibly stale cluster id.

        Returns
        -------
        int
            Current representative id.
        """
        while cluster_id != 0 and newcluster[cluster_id] != 0:
            cluster_id = newcluster[cluster_id]
        return cluster_id

    sorted_pairs = sorted((dist, pair[0], pair[1]) for pair, dist in pair_distances.items())
    for dist, pid1, pid2 in sorted_pairs:
        cluster1 = root(cluster[pid1])
        cluster2 = root(cluster[pid2])
        cluster[pid1] = cluster1
        cluster[pid2] = cluster2
        x1 = float(positions[pid1, 0].item())
        y1 = float(positions[pid1, 1].item())
        x2 = float(positions[pid2, 0].item())
        y2 = float(positions[pid2, 1].item())
        if cluster1 == 0 and cluster2 == 0:
            n_clusters += 1
            cluster[pid1] = n_clusters
            cluster[pid2] = n_clusters
            sumdist[n_clusters] = 2.0 * dist
            sum_x[n_clusters] = x1 + x2
            sum_y[n_clusters] = y1 + y2
            n_papers[n_clusters] = 2
            n_cords[n_clusters] = 2
            if dist > threshold:
                joinable[n_clusters] = 1
        elif cluster1 == 0:
            cluster[pid1] = cluster2
            sumdist[cluster2] += dist
            sum_x[cluster2] += x1
            sum_y[cluster2] += y1
            n_papers[cluster2] += 1
            n_cords[cluster2] += 1
        elif cluster2 == 0:
            cluster[pid2] = cluster1
            sumdist[cluster1] += dist
            sum_x[cluster1] += x2
            sum_y[cluster1] += y2
            n_papers[cluster1] += 1
            n_cords[cluster1] += 1
        elif cluster1 != cluster2 and (
            dist <= threshold or joinable[cluster1] == 1 or joinable[cluster2] == 1
        ):
            dx1 = sum_x[cluster1] / n_papers[cluster1]
            dx2 = sum_x[cluster2] / n_papers[cluster2]
            dy1 = sum_y[cluster1] / n_papers[cluster1]
            dy2 = sum_y[cluster2] / n_papers[cluster2]
            distclusters = math.sqrt((dx1 - dx2) ** 2 + (dy1 - dy2) ** 2)
            avedist1 = sumdist[cluster1] / n_cords[cluster1]
            avedist2 = sumdist[cluster2] / n_cords[cluster2]
            distedge1 = 0.564 * avedist1 * math.sqrt(float(n_papers[cluster1]))
            distedge2 = 0.564 * avedist2 * math.sqrt(float(n_papers[cluster2]))
            expected = distedge1 + distedge2 + ((dist - avedist1) / 2.0) + ((dist - avedist2) / 2.0)
            z_score = (distclusters - expected) / dist if dist != 0.0 else -1.0
            if z_score < 0.0 or dist > threshold:
                n_clusters += 1
                newcluster[cluster1] = n_clusters
                newcluster[cluster2] = n_clusters
                sumdist[n_clusters] = sumdist[cluster1] + sumdist[cluster2] + dist
                sum_x[n_clusters] = sum_x[cluster1] + sum_x[cluster2]
                sum_y[n_clusters] = sum_y[cluster1] + sum_y[cluster2]
                n_papers[n_clusters] = n_papers[cluster1] + n_papers[cluster2]
                n_cords[n_clusters] = n_cords[cluster1] + n_cords[cluster2] + 1
                if joinable[cluster1] == 1 and joinable[cluster2] == 1:
                    joinable[n_clusters] = 1
    renumber: dict[int, int] = {}
    assignment = [0 for _ in range(num_nodes)]
    for node in range(num_nodes):
        cluster_id = root(cluster[node])
        if cluster_id == 0:
            cluster_id = node + n_clusters + 1
        if cluster_id not in renumber:
            renumber[cluster_id] = len(renumber)
        assignment[node] = renumber[cluster_id]
    return assignment


def _openord_coarsen_full(
    full_edges: list[tuple[int, int, float]],
    assignment: list[int],
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]], int]:
    """Aggregate one OpenOrd coarse graph.

    Parameters
    ----------
    full_edges : list[tuple[int, int, float]]
        Fine-level full rows.
    assignment : list[int]
        Fine-to-coarse assignment.

    Returns
    -------
    tuple[list[tuple[int, int, float]], list[tuple[int, int, float]], int]
        Coarse full rows, coarse int rows, and coarse node count.
    """
    coarse_nodes = max(assignment) + 1 if assignment else 0
    coarse_rows: list[dict[int, float]] = [dict() for _ in range(coarse_nodes)]
    sizes = [0 for _ in range(coarse_nodes)]
    for coarse in assignment:
        sizes[coarse] += 1
    for source, target, weight in full_edges:
        coarse_source = assignment[source]
        coarse_target = assignment[target]
        coarse_rows[coarse_source][coarse_target] = (
            coarse_rows[coarse_source].get(coarse_target, 0.0) + weight
        )
        coarse_rows[coarse_target][coarse_source] = (
            coarse_rows[coarse_target].get(coarse_source, 0.0) + weight
        )
    coarse_full = [
        (source, target, float(weight))
        for source, row in enumerate(coarse_rows)
        for target, weight in sorted(row.items())
        if source != target
    ]
    denoms = [sum(row.values()) for row in coarse_rows]
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    coarse_int: list[tuple[int, int, float]] = []
    for source, row in enumerate(coarse_rows):
        sortable: list[tuple[float, int]] = []
        for target, weight in row.items():
            if source == target:
                continue
            denom = math.sqrt(denoms[source] * denoms[target])
            sortable.append((0.0 if denom == 0.0 else weight / denom, target))
        topn = 5
        if min_size != max_size:
            numerator = math.log(float(sizes[source])) - math.log(float(min_size))
            denominator = math.log(float(max_size)) - math.log(float(min_size))
            topn = int(5.0 + 10.0 * (numerator / denominator))
        sortable.sort()
        for weight, target in reversed(sortable[-topn:]):
            coarse_int.append((source, target, float(weight)))
    return coarse_full, coarse_int, coarse_nodes


def _openord_project_coarse_positions(
    coarse_positions: torch.Tensor,
    assignment: list[int],
    scale: float,
) -> torch.Tensor:
    """Project coarse coordinates to a finer level like ``refine``.

    Parameters
    ----------
    coarse_positions : torch.Tensor
        Coarse coordinates with shape ``[N_c, 2]``.
    assignment : list[int]
        Fine-to-coarse assignment.
    scale : float
        Positive OpenOrd scaling target.

    Returns
    -------
    torch.Tensor
        Fine initial coordinates with shape ``[N_f, 2]``.
    """
    projected = coarse_positions[torch.tensor(assignment, dtype=torch.long)].clone()
    max_abs = (
        float(torch.max(torch.abs(coarse_positions)).item()) if coarse_positions.numel() else 0.0
    )
    if scale > 0.0 and max_abs > 0.0:
        projected = projected * (scale / max_abs)
    return projected.to(dtype=torch.float64)


def _run_openord_recursive_multilevel(
    problem: LayoutProblem,
    params: _DrlParameters,
    density_config: DRLDensityGridConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run OpenOrd's level-2 recursive coarsen/refine path.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    params : _DrlParameters
        Resolved default parameters.
    density_config : DRLDensityGridConfig
        Density-grid constants.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Final coordinates and stage metadata.
    """
    full_edges = _openord_full_edges(problem.edge_index, problem.edge_weights)
    if not full_edges:
        return _initialize_openord_positions(problem.num_nodes), {"levels": 1}
    level0_int = _openord_truncate_edges(full_edges, problem.num_nodes, topn=10, normalize=False)
    coarsen_params = _DrlParameters(
        edge_cut=1.0,
        init=params.init,
        liquid=params.liquid,
        expansion=params.expansion,
        cooldown=params.cooldown,
        crunch=params.crunch,
        simmer=params.simmer,
    )
    level0_pos, level0_iedges = _openord_layout_rows(
        int_edges=level0_int,
        num_nodes=problem.num_nodes,
        initial_positions=_initialize_openord_positions(problem.num_nodes),
        params=coarsen_params,
        seed=problem.seed,
        density_config=density_config,
    )
    assignment = _openord_average_link_clusters(level0_pos, full_edges, level0_iedges)
    coarse_full, coarse_int, coarse_nodes = _openord_coarsen_full(full_edges, assignment)
    if coarse_nodes <= 0 or coarse_nodes >= problem.num_nodes:
        return level0_pos, {"levels": 1, "coarse_nodes": coarse_nodes}
    del coarse_full
    coarse_pos, _ = _openord_layout_rows(
        int_edges=coarse_int,
        num_nodes=coarse_nodes,
        initial_positions=_initialize_openord_positions(coarse_nodes),
        params=_resolve_openord_parameters("coarsest", edge_cut=0.8),
        seed=problem.seed,
        density_config=density_config,
    )
    refine_initial = _openord_project_coarse_positions(coarse_pos, assignment, scale=450.0)
    final_pos, _ = _openord_layout_rows(
        int_edges=level0_int,
        num_nodes=problem.num_nodes,
        initial_positions=refine_initial,
        params=_resolve_openord_parameters("final", edge_cut=0.5),
        seed=problem.seed,
        density_config=density_config,
    )
    return final_pos, {"levels": 2, "coarse_nodes": coarse_nodes, "assignment": assignment}


@register_op
@dataclass(frozen=True)
class OpenOrdPrepareState(Op):
    """Resolve OpenOrd parameters and build mutable weighted adjacency."""

    name: ClassVar[str] = "openord_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("extras",)
    requires: ClassVar[tuple[str, ...]] = ()
    config: OpenOrdPrepareStateConfig = field(default_factory=OpenOrdPrepareStateConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate OpenOrd phase parameters and prunable adjacency.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state receiving OpenOrd extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with resolved OpenOrd parameters and adjacency.
        """
        del ctx
        state.extras["openord_params"] = _resolve_openord_parameters(
            options=self.config.options,
            edge_cut=self.config.edge_cut,
        )
        state.extras["openord_multilevel"] = self.config.multilevel
        state.extras["openord_adjacency"] = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdInitializePositions(Op):
    """Seed the serial OpenOrd starting coordinates."""

    name: ClassVar[str] = "openord_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create deterministic OpenOrd initial coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs carrying node count and seed.
        state : SolveState
            Mutable solve state receiving coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial ``float64`` positions.
        """
        del ctx
        state.pos = _initialize_openord_positions(num_nodes=problem.num_nodes)
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdPhaseSolve(Op):
    """Run OpenOrd's five-phase annealing and edge-cut loop."""

    name: ClassVar[str] = "openord_phase_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[tuple[str, ...]] = ("pos", "extras")
    density_grid: DRLDensityGridConfig = field(default_factory=DRLDensityGridConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute OpenOrd's source-matched serial state machine.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing initialized positions and extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final unscaled OpenOrd coordinates.
        """
        del ctx
        if state.pos is None:
            raise ValueError("OpenOrdPhaseSolve requires state.pos to be set.")
        multilevel = state.extras.get("openord_multilevel")
        if multilevel is None:
            multilevel = problem.num_nodes >= 20
        if bool(multilevel):
            state.pos, metadata = _run_openord_recursive_multilevel(
                problem=problem,
                params=state.extras["openord_params"],
                density_config=self.density_grid,
            )
            state.extras["openord_multilevel_metadata"] = metadata
            return state
        state.pos = _run_reference_drl(
            initial_positions=state.pos,
            adjacency=state.extras["openord_adjacency"],
            params=state.extras["openord_params"],
            seed=problem.seed,
            density_config=self.density_grid,
            rng_kind="libc",
        )
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdFinalizePositions(Op):
    """Cast final OpenOrd coordinates to Dagua's output dtype and device."""

    name: ClassVar[str] = "openord_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move final coordinates to the requested output device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used to resolve the output device.
        state : SolveState
            Mutable solve state containing final coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final ``float32`` coordinates on the input device.
        """
        del ctx
        if state.pos is None:
            raise ValueError("OpenOrdFinalizePositions requires state.pos to be set.")
        state.pos = state.pos.to(device=problem.edge_index.device, dtype=torch.float32)
        return state


__all__ = [
    "OpenOrdFinalizePositions",
    "OpenOrdInitializePositions",
    "OpenOrdOptions",
    "OpenOrdPhaseSolve",
    "OpenOrdPrepareState",
    "OpenOrdPrepareStateConfig",
    "_OPENORD_PRESETS",
    "_initialize_openord_positions",
    "_resolve_openord_parameters",
]
