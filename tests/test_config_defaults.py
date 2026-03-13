"""Tests for authoritative LayoutConfig defaults."""

from dagua.config import LayoutConfig


def test_layout_config_default_aesthetic_values():
    config = LayoutConfig()

    assert config.node_sep == 28.0
    assert config.rank_sep == 50.0
    assert config.w_attract_x_bias == 2.4
    assert config.w_crossing == 1.8
    assert config.w_straightness == 2.2
    assert config.w_length_variance == 0.7
