"""Regression tests for layout configuration registries."""

from __future__ import annotations

from dagua.config import PARAM_REGISTRY, LayoutConfig


def test_param_registry_defaults_match_layout_config() -> None:
    """PARAM_REGISTRY defaults should mirror LayoutConfig dataclass defaults.

    Returns
    -------
    None
        The assertion covers every registered tunable parameter.
    """
    config = LayoutConfig()

    for param in PARAM_REGISTRY:
        assert param.default == getattr(config, param.name)
