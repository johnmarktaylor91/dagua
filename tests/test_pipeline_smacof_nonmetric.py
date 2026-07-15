"""Pipeline pins for sklearn-compatible nonmetric SMACOF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import smacof

import dagua
from dagua.config import LayoutConfig
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.layout.ops.graph_utils import shortest_path_distances
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.smacof_nonmetric import (
    _sklearn_isotonic_fit_transform,
    layout_smacof_nonmetric_pipeline,
    smacof_nonmetric_positions,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_smacof_nonmetric_pipeline_is_registered() -> None:
    """Register the public nonmetric SMACOF algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the nonmetric SMACOF entrypoint.
    """
    assert PIPELINE_REGISTRY["smacof_nonmetric"] == (
        "dagua.layout.ops.pipelines.smacof_nonmetric",
        "layout_smacof_nonmetric_pipeline",
    )
    assert get_pipeline_function("SMACOF_NONMETRIC") is layout_smacof_nonmetric_pipeline


def test_isotonic_disparities_match_sklearn() -> None:
    """Pin the nonmetric disparity isotonic fit against sklearn.

    Returns
    -------
    None
        The local fit-transform must match sklearn's wrapper behavior.
    """
    dissimilarities = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0], dtype=np.float64)
    distances = np.array([2.0, 1.0, 4.0, 3.0, 2.5, 5.0, 4.0], dtype=np.float64)

    expected = IsotonicRegression(out_of_bounds="clip").fit_transform(
        dissimilarities,
        distances,
    )
    observed = _sklearn_isotonic_fit_transform(dissimilarities, distances)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)


def test_smacof_nonmetric_single_run_matches_sklearn() -> None:
    """Match sklearn ``smacof(metric=False, n_init=1)`` on geodesics.

    Returns
    -------
    None
        Coordinates, stress, and iteration count must match the reference.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (1, 3), (3, 4), (2, 5)])
    distances = shortest_path_distances(edge_index=edge_index, num_nodes=6)
    observed_pos, observed_stress, observed_iter = smacof_nonmetric_positions(
        distances,
        seed=13,
        max_iter=25,
        eps=1.0e-6,
    )
    expected_pos, expected_stress, expected_iter = smacof(
        distances,
        metric=False,
        n_components=2,
        init=None,
        n_init=1,
        max_iter=25,
        eps=1.0e-6,
        random_state=13,
        return_n_iter=True,
        normalized_stress=False,
    )

    np.testing.assert_allclose(observed_pos, expected_pos, rtol=0.0, atol=1.0e-10)
    assert observed_stress == np.float64(expected_stress) or (
        abs(observed_stress - expected_stress) < 1.0e-14
    )
    assert observed_iter == expected_iter


def test_smacof_nonmetric_pipeline_matches_reference_up_to_similarity() -> None:
    """Run the public pipeline and compare with sklearn via Procrustes residual.

    Returns
    -------
    None
        The residual must be numerical noise only.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (3, 6)])
    distances = shortest_path_distances(edge_index=edge_index, num_nodes=7)
    reference, _stress, _n_iter = smacof(
        distances,
        metric=False,
        n_components=2,
        init=None,
        n_init=1,
        max_iter=30,
        eps=1.0e-6,
        random_state=7,
        return_n_iter=True,
        normalized_stress=False,
    )
    observed = layout_smacof_nonmetric_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        seed=7,
        max_iter=30,
    )

    assert procrustes_rmsd(observed.numpy(), reference) < 1.0e-10


def test_layout_config_algorithm_smacof_nonmetric_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='smacof_nonmetric'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("b", "d")])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="smacof_nonmetric",
            seed=5,
            algorithm_params={"max_iter": 5},
        ),
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_smacof_nonmetric_pipeline_has_no_runtime_sklearn_delegation() -> None:
    """Guard the production pipeline against sklearn runtime delegation.

    Returns
    -------
    None
        The pipeline source must not import or call sklearn.
    """
    source = (
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "smacof_nonmetric.py"
    ).read_text()
    assert "import sklearn" not in source
    assert "from sklearn" not in source
    assert "IsotonicRegression" not in source
    assert "smacof(" not in source
