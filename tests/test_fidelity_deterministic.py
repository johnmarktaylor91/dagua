"""Tests for Group C deterministic verdict helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fidelity_analysis import procrustes_align_rigid


class TestRigidAlignment:
    """Coverage for rigid deterministic alignment semantics."""

    def test_identity_alignment_returns_same(self) -> None:
        """Verify identity inputs are preserved by rigid alignment.

        Returns
        -------
        None
            This test asserts that identical tensors remain unchanged.
        """
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        aligned, reflected = procrustes_align_rigid(positions, positions)
        assert torch.allclose(aligned, positions, atol=1e-6)
        assert reflected is False

    def test_rotated_reimpl_aligns_back(self) -> None:
        """Verify pure rotations are removed by rigid alignment.

        Returns
        -------
        None
            This test asserts that a rotated layout aligns back to the
            reference coordinates.
        """
        pos_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        theta = torch.tensor(math.pi / 2)
        rotation = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )
        pos_b = pos_a @ rotation
        aligned, _ = procrustes_align_rigid(pos_a, pos_b)
        assert torch.allclose(aligned, pos_a, atol=1e-5)

    def test_translated_reimpl_aligns_back(self) -> None:
        """Verify translations are removed by rigid alignment.

        Returns
        -------
        None
            This test asserts that centering restores the original layout.
        """
        pos_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pos_b = pos_a + torch.tensor([[5.0, 3.0]])
        aligned, _ = procrustes_align_rigid(pos_a, pos_b)
        assert torch.allclose(aligned, pos_a, atol=1e-5)

    def test_scaled_reimpl_does_not_align_back(self) -> None:
        """Verify rigid alignment preserves scale differences.

        Returns
        -------
        None
            This test asserts that scale-normalized matches do not pass the
            rigid deterministic comparator.
        """
        pos_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pos_b = pos_a * 2.0
        aligned, _ = procrustes_align_rigid(pos_a, pos_b)
        assert not torch.allclose(aligned, pos_a, atol=1e-3)
