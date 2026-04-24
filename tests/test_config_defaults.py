"""Tests for authoritative LayoutConfig defaults."""

from dagua.config import LayoutConfig


def test_layout_config_default_aesthetic_values() -> None:
    config = LayoutConfig()

    # Sprint 18a/b spacing defaults (28 -> 60, 50 -> 120) eliminate
    # node overlaps on dense families (bipartite, erdos_renyi, etc.)
    # and lift edge_straightness via taller layouts. adaptive_spacing
    # still scales these down for n >= 1000.
    assert config.node_sep == 70.0
    assert config.rank_sep == 200.0
    assert config.w_attract_x_bias == 2.4
    assert config.w_crossing == 1.8
    assert config.w_straightness == 2.2
    # Sprint 11: CV^2 reformulation of edge length variance loss
    # (var/mean^2) -- bumped from 0.7 to make uniformity an active
    # constraint vs background noise.
    assert config.w_length_variance == 8.0
    assert config.optimizer_fallback == "auto"
