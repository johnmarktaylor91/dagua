"""Tests for the candidate-only radial sprawl repair op."""

from __future__ import annotations

import torch

from dagua.layout.ops.sprawl_repair import (
    RadialWinsorize,
    radial_winsorize_positions,
    robust_full_extent_ratio,
    sprawl_repair_gate,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _outlier_layout() -> torch.Tensor:
    """Return a compact core with one radial outlier.

    Returns
    -------
    torch.Tensor
        Synthetic position tensor with shape ``[10, 2]``.
    """
    core = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return torch.cat((core, torch.tensor([[100.0, 0.0]])), dim=0)


def test_radial_winsorize_matches_field_two_pass_recipe() -> None:
    """The public-safe wrapper mirrors FIELD without importing its private op."""
    pos = _outlier_layout()

    expected = pos.detach()
    for _pass_index in range(2):
        center = expected.mean(dim=0, keepdim=True)
        delta = expected - center
        radius_squared = (delta * delta).sum(dim=1)
        rms = torch.sqrt(radius_squared.mean()).clamp_min(1.0e-9)
        radius = torch.sqrt(radius_squared).clamp_min(1.0e-9)
        expected = center + delta * torch.clamp(1.6 * rms / radius, max=1.0).unsqueeze(1)

    actual = radial_winsorize_positions(pos, cap_multiple=1.6)

    assert torch.equal(actual, expected)
    assert torch.linalg.norm(actual[-1] - actual.mean(dim=0)) < torch.linalg.norm(
        pos[-1] - pos.mean(dim=0)
    )


def test_sprawl_gate_is_closed_for_in_band_geometry() -> None:
    """In-band C5 and robust extent preserve the exact input tensor path."""
    pos = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )

    assert robust_full_extent_ratio(pos) == 1.0
    assert not sprawl_repair_gate(pos, c5_whitespace_ratio=16.0)
    assert radial_winsorize_positions(pos, cap_multiple=10.0) is not pos
    assert torch.equal(radial_winsorize_positions(pos, cap_multiple=10.0), pos)


def test_sprawl_gate_accepts_runtime_c5_or_robust_extent() -> None:
    """Either declared input-only signal admits the candidate."""
    compact = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    outlier = _outlier_layout()

    assert sprawl_repair_gate(compact, c5_whitespace_ratio=16.01)
    assert sprawl_repair_gate(outlier, c5_whitespace_ratio=4.0)


def test_radial_winsorize_op_updates_only_positions() -> None:
    """The registered op applies the wrapper through the standard state seam."""
    pos = _outlier_layout()
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=int(pos.shape[0]))
    state = SolveState(pos=pos.clone())
    context = RuntimeContext()

    result = RadialWinsorize(cap_multiple=1.6).apply(problem, state, context)

    assert result is state
    assert result.pos is not None
    assert not torch.equal(result.pos, pos)
