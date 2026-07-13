"""Regression pins for the reusable Dagre-compatible Brandes-Koepf op."""

from __future__ import annotations

import pytest

from dagua.layout.ops.brandes_koepf import brandes_koepf_x_assignment


def test_brandes_koepf_balances_diamond_exactly() -> None:
    """Pin the four-alignment balance for a symmetric diamond.

    Returns
    -------
    None
        The coordinate mapping is compared with the dagre.js-derived pin.
    """
    positions = brandes_koepf_x_assignment(
        layering=[[0], [1, 2], [3]],
        predecessors={0: [], 1: [0], 2: [0], 3: [1, 2]},
        successors={0: [1, 2], 1: [3], 2: [3], 3: []},
        widths={0: 40.0, 1: 40.0, 2: 40.0, 3: 40.0},
        dummy_nodes=set(),
        node_sep=50.0,
        edge_sep=20.0,
    )

    assert positions == {0: 45.0, 1: 0.0, 2: 90.0, 3: 45.0}


@pytest.mark.parametrize("align", ["UL", "UR", "DL", "DR"])
def test_brandes_koepf_accepts_every_dagre_alignment(align: str) -> None:
    """Exercise all four public Dagre alignment selectors.

    Parameters
    ----------
    align : str
        Parametrized UL/UR/DL/DR selector.

    Returns
    -------
    None
        Every alignment must cover every input node.
    """
    positions = brandes_koepf_x_assignment(
        layering=[[0], [1, 2]],
        predecessors={0: [], 1: [0], 2: [0]},
        successors={0: [1, 2], 1: [], 2: []},
        widths={0: 30.0, 1: 20.0, 2: 40.0},
        dummy_nodes=set(),
        align=align,
    )

    assert set(positions) == {0, 1, 2}


def test_brandes_koepf_rejects_unknown_alignment() -> None:
    """Reject alignment names that dagre.js does not expose.

    Returns
    -------
    None
        Invalid input must raise ``ValueError``.
    """
    with pytest.raises(ValueError, match="align"):
        brandes_koepf_x_assignment(
            layering=[[0]],
            predecessors={0: []},
            successors={0: []},
            widths={0: 20.0},
            dummy_nodes=set(),
            align="CENTER",
        )
