"""Row-forensics logic tests: facet swap gains and degenerate-tie detection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import roundloop_common as rl  # noqa: E402


def _metrics(**overrides: float) -> dict:
    """Build a full common-facet metrics payload.

    Parameters
    ----------
    **overrides : float
        Facet values overriding the 0.9 default.

    Returns
    -------
    dict
        Metrics payload with hierarchy declared off.
    """
    payload: dict = {name: 0.9 for name in rl.FACET_KEYS}
    for name in ("directed_flow_score", "depth_order_score"):
        payload[name] = None
    for name in rl.FACET_KEYS:
        if name.startswith("cluster_") and name != "cluster_silhouette_score":
            payload[name] = None
    payload["declared_hierarchical"] = False
    payload.update(overrides)
    return payload


@pytest.mark.smoke
def test_facet_swap_gain_identifies_the_deficit_facet() -> None:
    """The facet where native trails the field must dominate the swap ranking."""
    native = _metrics(edge_crossing_score=0.5)
    field = _metrics(edge_crossing_score=0.95)
    gains = rl.facet_swap_gains(native, field, semantically_directed=False)
    assert gains[0][0] == "edge_crossing_score"
    assert gains[0][1] > 0.0
    # Every other facet is identical, so its swap gain is ~zero.
    others = dict(gains)
    del others["edge_crossing_score"]
    assert all(abs(gain) < 1e-9 for gain in others.values())


@pytest.mark.smoke
def test_facet_swap_gain_uses_real_ruler_weighting() -> None:
    """Equal raw deficits must rank by ruler weight (ksm 25 > angular 4)."""
    native = _metrics(ksm_score=0.6, angular_resolution_score=0.6)
    field = _metrics(ksm_score=0.9, angular_resolution_score=0.9)
    gains = dict(rl.facet_swap_gains(native, field, semantically_directed=False))
    assert gains["ksm_score"] > gains["angular_resolution_score"] > 0.0


@pytest.mark.smoke
def test_facet_table_skips_facets_absent_on_both_sides() -> None:
    """Facets inapplicable on both sides are omitted from the table."""
    native = _metrics()
    field = _metrics()
    table = rl.facet_table(native, field)
    assert "directed_flow_score" not in table
    assert table["ksm_score"] == {"native": 0.9, "field": 0.9}


def _cloud(seed: int, n: int = 24) -> torch.Tensor:
    """Build a deterministic random 2D point cloud.

    Parameters
    ----------
    seed : int
        Generator seed.
    n : int, optional
        Point count.

    Returns
    -------
    torch.Tensor
        ``[n, 2]`` float32 positions.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((n, 2), generator=generator, dtype=torch.float32)


@pytest.mark.smoke
def test_degenerate_tie_flags_rotated_scaled_copy() -> None:
    """A rotated+scaled+translated copy of a field layout is degenerate."""
    base = _cloud(7)
    angle = math.radians(30.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float32,
    )
    native = (base @ rotation.T) * 3.5 + torch.tensor([10.0, -4.0])
    store = {"native.pt": native, "field.pt": base, "other.pt": _cloud(99)}
    rows = [
        {"engine": "dagre", "position_path": "field.pt", "position_sha256": "f" * 64},
        {"engine": "elk_layered", "position_path": "other.pt", "position_sha256": "0" * 64},
    ]
    match = rl.closest_field_layout(
        "native.pt",
        "n" * 64,
        rows,
        load_positions=lambda path: store[Path(path).name],
    )
    assert match is not None
    assert match.engine == "dagre"
    assert match.rmsd < rl.DEGENERATE_RMSD
    assert match.degenerate


@pytest.mark.smoke
def test_sha_identity_short_circuits_procrustes() -> None:
    """A sha match is degenerate without loading any tensors."""

    def _must_not_load(path: str) -> torch.Tensor:
        raise AssertionError("loader must not be called on sha identity")

    rows = [{"engine": "dagre", "position_path": "x.pt", "position_sha256": "s" * 64}]
    match = rl.closest_field_layout("native.pt", "s" * 64, rows, load_positions=_must_not_load)
    assert match is not None
    assert match.sha_match and match.degenerate and match.rmsd == 0.0


@pytest.mark.smoke
def test_distinct_layout_is_not_flagged() -> None:
    """Genuinely different layouts stay unflagged (no false degenerate ties)."""
    store = {"native.pt": _cloud(1), "field.pt": _cloud(2)}
    rows = [{"engine": "dagre", "position_path": "field.pt", "position_sha256": "f" * 64}]
    match = rl.closest_field_layout(
        "native.pt",
        "n" * 64,
        rows,
        load_positions=lambda path: store[Path(path).name],
    )
    assert match is not None
    assert not match.degenerate
    assert match.rmsd > rl.NEAR_RMSD


@pytest.mark.smoke
def test_classify_matches_frozen_tie_band() -> None:
    """Round-packet statuses reuse the frozen scorer classification."""
    assert rl.classify(rl.TIE_BAND + 0.01) == "strictly_best"
    assert rl.classify(0.0) == "tied"
    assert rl.classify(-rl.TIE_BAND) == "tied"
    assert rl.classify(-rl.TIE_BAND - 0.01) == "behind"
