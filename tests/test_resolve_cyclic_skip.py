"""Sprint 17: build_loss_ops should skip DagOrderingLoss for cyclic graphs."""

from __future__ import annotations

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure
from dagua.layout.ops.loss_engine import DagOrderingLoss
from dagua.layout.resolve import build_loss_ops


def _structure(is_acyclic: bool) -> GraphStructure:
    return GraphStructure(
        family=GraphFamily.GENERAL,
        num_components=1,
        max_degree=4,
        num_layers=1 if not is_acyclic else 5,
        avg_layer_width=10.0,
        is_planar_hint=True,
        is_acyclic=is_acyclic,
    )


def _config_with_structure(is_acyclic: bool) -> LayoutConfig:
    config = LayoutConfig(seed=42)
    setattr(config, "_dagua_native_structure", _structure(is_acyclic))
    setattr(config, "structure", _structure(is_acyclic))
    return config


def test_dag_loss_present_on_acyclic_graph() -> None:
    """When structure flags is_acyclic=True, DagOrderingLoss should fire."""
    config = _config_with_structure(is_acyclic=True)

    losses = build_loss_ops(config=config, node_sep=70.0, rank_sep=140.0)

    assert any(isinstance(loss, DagOrderingLoss) for loss in losses)


def test_dag_loss_skipped_on_cyclic_graph() -> None:
    """When structure flags is_acyclic=False, DagOrderingLoss should be omitted."""
    config = _config_with_structure(is_acyclic=False)

    losses = build_loss_ops(config=config, node_sep=70.0, rank_sep=140.0)

    assert not any(isinstance(loss, DagOrderingLoss) for loss in losses)


def test_dag_loss_skipped_when_w_dag_zero() -> None:
    """w_dag=0 also skips the loss, regardless of acyclicity."""
    config = _config_with_structure(is_acyclic=True)
    config.w_dag = 0.0

    losses = build_loss_ops(config=config, node_sep=70.0, rank_sep=140.0)

    assert not any(isinstance(loss, DagOrderingLoss) for loss in losses)


def test_dag_loss_present_when_no_structure_provided() -> None:
    """Backward compat: no structure -> default-true is_acyclic -> loss fires."""
    config = LayoutConfig(seed=42)
    # Don't attach structure -- mimics legacy callers
    losses = build_loss_ops(config=config, node_sep=70.0, rank_sep=140.0)

    assert any(isinstance(loss, DagOrderingLoss) for loss in losses)
