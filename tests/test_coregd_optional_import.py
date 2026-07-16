"""Regression: ``import dagua`` must not require the optional PyTorch Geometric stack.

coregd previously hard-imported ``torch_cluster`` / ``torch_geometric`` at module
scope, which broke ``import dagua`` on any environment without the (heavy, optional)
PyG stack -- violating design principle #1 (PyTorch is the only required dependency).
"""

import pytest


def test_import_dagua_succeeds_without_optional_pyg():
    import dagua  # noqa: F401  # must not raise even when torch-geometric is absent


def test_coregd_module_imports_without_pyg():
    from dagua.layout.ops.pipelines import coregd  # noqa: F401  # module import must succeed


def test_coregd_pipeline_raises_clear_error_when_pyg_missing():
    import torch

    from dagua.layout.ops.pipelines import coregd

    if coregd._PYG_AVAILABLE:
        pytest.skip("PyTorch Geometric installed; optional-dep guard path not exercised")

    with pytest.raises(ImportError, match="torch-geometric"):
        coregd.layout_coregd_pipeline(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )
