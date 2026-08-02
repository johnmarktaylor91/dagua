"""Equivalence gates for the native runtime V3 referee."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pytest
import torch

from dagua.eval.graphs import get_test_graphs
from dagua.eval.ruler_v3 import (
    _smooth_clearance_occlusion_score,
    referee_eligibility_key,
    score_core_v3,
)
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.native_v3_referee import (
    _runtime_v3_graph_meta,
    fast_smooth_clearance_occlusion_score,
    score_v3_runtime,
    score_v3_runtime_result,
)
from dagua.layout.ops.state import LayoutProblem
from dagua.metrics import _all_pairs_unweighted, _build_csr


def _case_geometries() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield C4 geometries covering random and boundary contacts.

    Returns
    -------
    Iterable[tuple[torch.Tensor, torch.Tensor]]
        Visual box centers and sizes, each with shape ``[N, 2]``.
    """
    rng = np.random.default_rng(123)
    for count in (2, 5, 20, 60):
        centers = torch.tensor(rng.normal(size=(count, 2)), dtype=torch.float64) * 20.0
        sizes = torch.tensor(rng.uniform(0.5, 4.0, size=(count, 2)), dtype=torch.float64)
        yield centers, sizes
    yield (
        torch.tensor([[0.0, 0.0], [0.5, 0.25], [0.8, 0.4]], dtype=torch.float64),
        torch.tensor([[2.0, 2.0], [2.0, 2.0], [1.5, 1.0]], dtype=torch.float64),
    )
    yield (
        torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=torch.float64),
        torch.full((3, 2), 2.0, dtype=torch.float64),
    )
    yield (
        torch.tensor([[0.0, 0.0], [2.5, 0.0], [5.25, 0.0]], dtype=torch.float64),
        torch.full((3, 2), 2.0, dtype=torch.float64),
    )


@pytest.mark.parametrize("label_inclusive", [False, True])
def test_fast_c4_is_bit_exact_against_frozen_loop(label_inclusive: bool) -> None:
    """Assert fast C4 returns bit-identical diagnostics to frozen V3 C4.

    Parameters
    ----------
    label_inclusive : bool
        Label-inclusion flag forwarded to both C4 implementations.
    """
    keys = (
        "node_occlusion_score",
        "clearance_penalty",
        "overlap_area_severity",
        "packed_seam_severity",
        "overlap_count",
        "clearance_contact_pairs",
        "clearance_abut_count",
        "visual_packing_fill",
        "legacy_node_occlusion_score",
    )
    for centers, sizes in _case_geometries():
        frozen = _smooth_clearance_occlusion_score(
            centers,
            sizes,
            label_inclusive=label_inclusive,
            seed=0,
        )
        fast = fast_smooth_clearance_occlusion_score(
            centers,
            sizes,
            label_inclusive=label_inclusive,
            seed=0,
        )
        for key in keys:
            assert fast[key] == frozen[key], key


def _problem(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> LayoutProblem:
    """Return a runtime problem for V3 equivalence checks.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    LayoutProblem
        Runtime-visible problem metadata.
    """
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=int(pos.shape[0]),
        node_sizes=node_sizes,
        direction="TB",
        clusters={"a": [0, 1], "b": [2, 3]} if int(pos.shape[0]) >= 4 else None,
        cluster_parents={"a": None, "b": None} if int(pos.shape[0]) >= 4 else None,
        structure=classify_graph(edge_index, int(pos.shape[0])),
        edge_weights=edge_weights,
    )


def _all_pairs(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return unweighted all-pairs graph distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``[N, N]``.
    """
    offsets, targets = _build_csr(edge_index, num_nodes)
    return _all_pairs_unweighted(offsets, targets, num_nodes, max_dist=num_nodes)


def _assert_runtime_matches_frozen(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    sizes: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> None:
    """Assert runtime V3 mirrors the restricted frozen oracle.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    weights : torch.Tensor, optional
        Optional runtime-visible edge weights with shape ``[E]``.

    Returns
    -------
    None
        Assertions validate bit-exact score, key, and facet equality.
    """
    problem = _problem(pos, edge_index, sizes, weights)
    distances = _all_pairs(edge_index, int(pos.shape[0]))
    runtime_key, runtime_tiered, runtime_facets = score_v3_runtime(
        pos,
        problem,
        all_pairs_dist=distances,
    )
    runtime_result = score_v3_runtime_result(pos, problem, all_pairs_dist=distances)
    frozen = score_core_v3(
        pos,
        edge_index,
        sizes,
        all_pairs_dist=distances,
        graph_meta=_runtime_v3_graph_meta(problem),
    )
    assert runtime_result.scores == frozen.scores
    assert runtime_key == referee_eligibility_key(frozen)
    assert runtime_tiered == frozen.scores["tiered"]
    assert runtime_facets == frozen.facets


@pytest.mark.parametrize("weighted", [False, True])
def test_score_v3_runtime_matches_frozen_restricted_oracle(weighted: bool) -> None:
    """Assert runtime V3 mirrors frozen V3 with restricted metadata.

    Parameters
    ----------
    weighted : bool
        Whether to include runtime-visible edge weights.
    """
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 4, 2, 4]],
        dtype=torch.long,
    )
    pos = torch.tensor(
        [[0.0, 0.0], [1.5, 0.2], [3.0, -0.1], [4.2, 0.4], [5.1, -0.2]],
        dtype=torch.float64,
    )
    sizes = torch.tensor(
        [[1.0, 0.8], [1.1, 0.7], [0.9, 1.0], [1.2, 0.8], [1.0, 0.9]],
        dtype=torch.float64,
    )
    weights = (
        torch.tensor([1.0, 2.0, 4.0, 8.0, 3.0, 6.0], dtype=torch.float64) if weighted else None
    )
    _assert_runtime_matches_frozen(pos, edge_index, sizes, weights)
    if weighted:
        return

    graph = next(tg.graph for tg in get_test_graphs() if tg.name == "hub_spoke_10x20")
    large_edge_index = graph.edge_index.detach().to(dtype=torch.long)
    large_count = int(graph.num_nodes)
    generator = torch.Generator(device="cpu").manual_seed(42)
    large_pos = torch.randn((large_count, 2), generator=generator, dtype=torch.float64)
    large_sizes = torch.full((large_count, 2), 1.0, dtype=torch.float64)
    assert large_count >= 200
    _assert_runtime_matches_frozen(large_pos, large_edge_index, large_sizes, None)
