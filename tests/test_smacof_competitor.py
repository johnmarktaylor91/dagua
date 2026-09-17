"""Regression tests for the sklearn SMACOF competitor adapter.

Pins the fix for the availability contract: ``available()`` previously
inherited the base-class unconditional ``True`` while ``layout()`` imported
scikit-learn OUTSIDE its try block, so a missing dependency crashed the
caller instead of producing a recorded skip / error row.
"""

from __future__ import annotations

import sys

import pytest
import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.smacof_competitor import SklearnSmacofNonmetric
from dagua.graph import DaguaGraph


def _path_graph(num_nodes: int) -> DaguaGraph:
    """Build a small connected path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    DaguaGraph
        Path graph.
    """
    edges = [(node, node + 1) for node in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return DaguaGraph.from_edge_index(edge_index, num_nodes)


def test_smacof_available_probes_import() -> None:
    """available() must reflect the actual sklearn import probe."""
    competitor = get_competitor("sklearn_smacof_nonmetric")
    assert competitor is not None

    try:
        from sklearn.manifold import smacof  # noqa: F401

        expected = True
    except ImportError:
        expected = False

    assert competitor.available() is expected


def test_smacof_missing_sklearn_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing scikit-learn must yield available() False, not a lie."""
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)

    assert SklearnSmacofNonmetric().available() is False


def test_smacof_missing_sklearn_layout_is_error_row_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """layout() must return an error row when sklearn cannot import."""
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)

    result = SklearnSmacofNonmetric().layout(_path_graph(4), seed=7)

    assert result.pos is None
    assert result.error is not None


@pytest.mark.skipif(
    not SklearnSmacofNonmetric().available(),
    reason="scikit-learn is not installed",
)
def test_smacof_layout_smoke() -> None:
    """With sklearn present the adapter should return finite positions."""
    result = SklearnSmacofNonmetric().layout(_path_graph(6), seed=7)

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert torch.isfinite(result.pos).all()
