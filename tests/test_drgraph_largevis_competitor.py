"""Regression tests for the LargeVis/DRGraph reference output parser.

Both upstream binaries write a ``n_vertices out_dim`` header line before the
coordinate rows (LargeVis.cpp:134, DRGraph visualizemod.cpp:793). The adapter
parser previously consumed that header as node 0's coordinates, shifting every
node's position by one row and dropping the last node, while the fidelity
verify script (``scripts/verify_drgraph_largevis_fidelity.py``) skipped it
correctly. These tests pin the corrected parse.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dagua.eval.competitors.drgraph_largevis_competitor import _read_reference_positions


def _write_largevis_output(path: Path, rows: list[tuple[int, float, float]]) -> None:
    """Write a synthetic LargeVis-style output file (``name x y`` rows).

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    rows : list[tuple[int, float, float]]
        Node id and coordinates per output row, in file order.

    Returns
    -------
    None
        File is written.
    """
    lines = [f"{len(rows)} 2"]
    lines.extend(f"{node} {x:.6f} {y:.6f}" for node, x, y in rows)
    path.write_text("\n".join(lines) + "\n")


def test_largevis_header_line_is_skipped_no_off_by_one(tmp_path: Path) -> None:
    """The header must not be parsed as node 0 and rows must not shift."""
    num_nodes = 4
    rows = [(0, 1.5, -2.5), (1, 3.0, 4.0), (2, -5.5, 6.25), (3, 7.0, -8.0)]
    output = tmp_path / "layout.txt"
    _write_largevis_output(output, rows)

    positions = _read_reference_positions(output, num_nodes)

    assert positions.shape == (num_nodes, 2)
    # No header-as-coordinates row: node 0 must NOT be (N, out_dim) = (4, 2).
    assert not torch.equal(positions[0], torch.tensor([4.0, 2.0]))
    for node, x, y in rows:
        assert positions[node, 0].item() == pytest.approx(x)
        assert positions[node, 1].item() == pytest.approx(y)


def test_largevis_rows_map_by_id_column_not_row_order(tmp_path: Path) -> None:
    """LargeVis orders rows by first appearance in the edge file, not by id."""
    num_nodes = 3
    # Shuffled file order: the name column carries the true node identity.
    rows = [(2, 10.0, 20.0), (0, 30.0, 40.0), (1, 50.0, 60.0)]
    output = tmp_path / "layout.txt"
    _write_largevis_output(output, rows)

    positions = _read_reference_positions(output, num_nodes)

    assert positions[2].tolist() == pytest.approx([10.0, 20.0])
    assert positions[0].tolist() == pytest.approx([30.0, 40.0])
    assert positions[1].tolist() == pytest.approx([50.0, 60.0])


def test_drgraph_two_column_rows_map_by_row_order(tmp_path: Path) -> None:
    """DRGraph output has no id column; rows are in node-id order."""
    num_nodes = 3
    output = tmp_path / "layout.txt"
    output.write_text("3 2\n1.0 2.0\n3.0 4.0\n5.0 6.0\n")

    positions = _read_reference_positions(output, num_nodes)

    assert positions.shape == (num_nodes, 2)
    assert positions[0].tolist() == pytest.approx([1.0, 2.0])
    assert positions[1].tolist() == pytest.approx([3.0, 4.0])
    assert positions[2].tolist() == pytest.approx([5.0, 6.0])


def test_missing_node_rows_raise_instead_of_silent_zeros(tmp_path: Path) -> None:
    """Isolated nodes never enter the edge file; that must be an error row."""
    output = tmp_path / "layout.txt"
    # Header claims 3 nodes but only two coordinate rows follow.
    output.write_text("3 2\n0 1.0 2.0\n1 3.0 4.0\n")

    with pytest.raises(ValueError, match="omitted coordinates"):
        _read_reference_positions(output, 3)


def test_header_node_count_mismatch_raises(tmp_path: Path) -> None:
    """A header disagreeing with the expected node count must raise."""
    output = tmp_path / "layout.txt"
    output.write_text("2 2\n0 1.0 2.0\n1 3.0 4.0\n")

    with pytest.raises(ValueError, match="header reports 2 nodes"):
        _read_reference_positions(output, 3)


def test_empty_output_raises(tmp_path: Path) -> None:
    """An empty output file must raise instead of returning zeros."""
    output = tmp_path / "layout.txt"
    output.write_text("")

    with pytest.raises(ValueError, match="empty"):
        _read_reference_positions(output, 2)


def test_duplicate_node_id_raises(tmp_path: Path) -> None:
    """A repeated node id signals corrupt output and must raise."""
    output = tmp_path / "layout.txt"
    output.write_text("2 2\n0 1.0 2.0\n0 3.0 4.0\n")

    with pytest.raises(ValueError, match="repeats node id"):
        _read_reference_positions(output, 2)
