"""Tests for shared evaluation pipeline IO helpers."""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any
from unittest import mock

import h5py
import pytest
import torch

from dagua.eval.pipeline_io import (
    aspect_ratio_deviation,
    compute_quick_metrics_seeded,
    compute_sampled_metrics_seeded,
    load_position_tensor,
    open_h5_for_worker,
    stable_seed,
    validate_positions,
)


def _worker_stable_seed(args: tuple[str, ...]) -> int:
    """Compute a stable seed in a worker process.

    Parameters
    ----------
    args : tuple[str, ...]
        String parts passed through to ``stable_seed``.

    Returns
    -------
    int
        Stable seed computed in the worker.
    """
    from dagua.eval.pipeline_io import stable_seed as worker_stable_seed

    return worker_stable_seed(*args)


def _process_send_stable_seed(args: tuple[str, ...], conn: Connection) -> None:
    """Send a stable seed result through a multiprocessing pipe.

    Parameters
    ----------
    args : tuple[str, ...]
        String parts passed through to ``stable_seed``.
    conn : Connection
        Pipe endpoint used to return the computed seed.

    Returns
    -------
    None
        This helper sends one integer result then closes the pipe.
    """
    try:
        conn.send(_worker_stable_seed(args))
    finally:
        conn.close()


def _run_stable_seed_process(args: tuple[str, ...]) -> int:
    """Compute a stable seed in a dedicated child process.

    Parameters
    ----------
    args : tuple[str, ...]
        String parts passed through to ``stable_seed``.

    Returns
    -------
    int
        Stable seed returned from the child process.
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    process = mp.Process(target=_process_send_stable_seed, args=(args, child_conn))
    process.start()
    child_conn.close()
    try:
        result = int(parent_conn.recv())
    finally:
        parent_conn.close()
        process.join(timeout=5.0)
    assert process.exitcode == 0
    return result


def _build_position_artifacts(
    tmp_path: Path,
) -> tuple[str, str, torch.Tensor, torch.Tensor, Path]:
    """Create minimal .pt and HDF5 position artifacts for loader tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    tuple[str, str, torch.Tensor, torch.Tensor, Path]
        Relative ``positions_file`` path, HDF5 ``record_key``, .pt tensor,
        HDF5 tensor, and the created HDF5 file path.
    """
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()
    positions_file = "positions/graph_a__engine_a.pt"
    record_key = "graph_a::engine_a::seed0"
    pt_tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    h5_tensor = torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=torch.float32)
    torch.save(pt_tensor, tmp_path / positions_file)
    h5_path = tmp_path / "positions.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset(record_key, data=h5_tensor.numpy())
    return positions_file, record_key, pt_tensor, h5_tensor, h5_path


def test_stable_seed_reproducible() -> None:
    """Return the same seed for identical inputs.

    Returns
    -------
    None
        This test asserts deterministic seed generation.
    """
    seed_a = stable_seed("graph_a", "engine_b", "42")
    seed_b = stable_seed("graph_a", "engine_b", "42")

    assert isinstance(seed_a, int)
    assert seed_a == seed_b


def test_stable_seed_cross_process_identical() -> None:
    """Match stable seeds across worker processes.

    Returns
    -------
    None
        This test asserts cross-process reproducibility.
    """
    args = ("graph_a", "engine_b", "42")
    try:
        with mp.Pool(2) as pool:
            results = pool.map(_worker_stable_seed, [args, args])
    except PermissionError:
        results = [_run_stable_seed_process(args), _run_stable_seed_process(args)]

    assert results[0] == results[1]
    assert results[0] == stable_seed(*args)


def test_stable_seed_distinguishes_inputs() -> None:
    """Produce different seeds for different inputs.

    Returns
    -------
    None
        This test asserts input sensitivity.
    """
    assert stable_seed("graph_a", "engine_b", "42") != stable_seed("graph_a", "engine_b", "43")


def test_stable_seed_handles_empty_and_unicode_parts() -> None:
    """Accept empty strings and escaped unicode content.

    Returns
    -------
    None
        This test asserts robust string handling.
    """
    seed = stable_seed("", "caf\u00e9", "")

    assert isinstance(seed, int)
    assert seed == stable_seed("", "caf\u00e9", "")


def test_validate_positions_valid_tensor_returns_none() -> None:
    """Accept a valid ``[N, 2]`` tensor.

    Returns
    -------
    None
        This test asserts the valid path.
    """
    positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=3) is None


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        torch.zeros((3, 2, 1), dtype=torch.float32),
    ],
)
def test_validate_positions_rejects_non_2d_tensor(positions: torch.Tensor) -> None:
    """Reject tensors that are not rank 2.

    Parameters
    ----------
    positions : torch.Tensor
        Invalid test tensor.

    Returns
    -------
    None
        This test asserts the ``tensor_not_2d`` rejection.
    """
    assert validate_positions(positions, expected_nodes=3) == "tensor_not_2d"


def test_validate_positions_rejects_non_xy_tensor() -> None:
    """Reject tensors whose second dimension is not 2.

    Returns
    -------
    None
        This test asserts the ``tensor_not_xy`` rejection.
    """
    positions = torch.zeros((3, 3), dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=3) == "tensor_not_xy"


def test_validate_positions_rejects_too_few_nodes() -> None:
    """Reject tensors below the minimum node count.

    Returns
    -------
    None
        This test asserts the ``too_few_nodes`` rejection.
    """
    positions = torch.zeros((2, 2), dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=2) == "too_few_nodes"


def test_validate_positions_rejects_node_count_mismatch() -> None:
    """Reject tensors whose row count mismatches the graph.

    Returns
    -------
    None
        This test asserts the ``node_count_mismatch`` rejection.
    """
    positions = torch.zeros((3, 2), dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=4) == "node_count_mismatch"


def test_validate_positions_rejects_nan_values() -> None:
    """Reject tensors containing NaN coordinates.

    Returns
    -------
    None
        This test asserts the ``contains_nan`` rejection.
    """
    positions = torch.tensor([[0.0, 0.0], [math.nan, 1.0], [2.0, 2.0]], dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=3) == "contains_nan"


def test_validate_positions_rejects_inf_values() -> None:
    """Reject tensors containing infinite coordinates.

    Returns
    -------
    None
        This test asserts the ``contains_inf`` rejection.
    """
    positions = torch.tensor([[0.0, 0.0], [math.inf, 1.0], [2.0, 2.0]], dtype=torch.float32)

    assert validate_positions(positions, expected_nodes=3) == "contains_inf"


def test_load_position_tensor_rejects_missing_positions_file(tmp_path: Path) -> None:
    """Reject records with no positions file.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts the early missing-file rejection.
    """
    positions, reason = load_position_tensor(
        record_key="graph_a::engine_a::seed0",
        positions_file=None,
        input_dir=tmp_path,
    )

    assert positions is None
    assert reason == "missing_positions_file"


def test_load_position_tensor_uses_pt_fallback_without_h5(tmp_path: Path) -> None:
    """Load the .pt tensor when no HDF5 handle is provided.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts the .pt fallback path.
    """
    positions_file, record_key, pt_tensor, _, _ = _build_position_artifacts(tmp_path)

    positions, reason = load_position_tensor(
        record_key=record_key,
        positions_file=positions_file,
        input_dir=tmp_path,
    )

    assert reason is None
    assert positions is not None
    assert torch.equal(positions, pt_tensor)


def test_load_position_tensor_prefers_h5_when_key_exists(tmp_path: Path) -> None:
    """Prefer the HDF5 tensor when the requested key exists.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts HDF5 precedence.
    """
    positions_file, record_key, _, h5_tensor, h5_path = _build_position_artifacts(tmp_path)

    with h5py.File(h5_path, "r") as h5_file:
        positions, reason = load_position_tensor(
            record_key=record_key,
            positions_file=positions_file,
            input_dir=tmp_path,
            h5_file=h5_file,
        )

    assert reason is None
    assert positions is not None
    assert torch.equal(positions, h5_tensor)


def test_load_position_tensor_falls_back_when_h5_key_missing(tmp_path: Path) -> None:
    """Fall back to the .pt tensor when the HDF5 key is absent.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts missing-key fallback behavior.
    """
    positions_file, _, pt_tensor, _, h5_path = _build_position_artifacts(tmp_path)

    with h5py.File(h5_path, "r") as h5_file:
        positions, reason = load_position_tensor(
            record_key="graph_a::engine_a::missing",
            positions_file=positions_file,
            input_dir=tmp_path,
            h5_file=h5_file,
        )

    assert reason is None
    assert positions is not None
    assert torch.equal(positions, pt_tensor)


def test_load_position_tensor_returns_h5_failure_without_pt_fallback(tmp_path: Path) -> None:
    """Return ``h5_load_failure`` when HDF5 reads raise.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts the no-fallback integrity check.
    """
    positions_file, record_key, _, _, _ = _build_position_artifacts(tmp_path)
    broken_dataset = mock.MagicMock()
    broken_dataset.__getitem__.side_effect = RuntimeError("broken dataset")
    fake_h5: dict[str, Any] = {record_key: broken_dataset}

    positions, reason = load_position_tensor(
        record_key=record_key,
        positions_file=positions_file,
        input_dir=tmp_path,
        h5_file=fake_h5,  # type: ignore[arg-type]
    )

    assert positions is None
    assert reason == "h5_load_failure"


def test_load_position_tensor_returns_load_failure_for_missing_pt(tmp_path: Path) -> None:
    """Return ``load_failure`` when the .pt file is missing.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts missing-file load failure handling.
    """
    positions, reason = load_position_tensor(
        record_key="graph_a::engine_a::seed0",
        positions_file="positions/missing.pt",
        input_dir=tmp_path,
    )

    assert positions is None
    assert reason == "load_failure"


def test_load_position_tensor_rejects_non_tensor_pt_payload(tmp_path: Path) -> None:
    """Return ``not_tensor`` for non-tensor .pt payloads.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts payload type validation.
    """
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()
    positions_file = "positions/not_tensor.pt"
    torch.save({"positions": [1.0, 2.0]}, tmp_path / positions_file)

    positions, reason = load_position_tensor(
        record_key="graph_a::engine_a::seed0",
        positions_file=positions_file,
        input_dir=tmp_path,
    )

    assert positions is None
    assert reason == "not_tensor"


def test_load_position_tensor_returns_load_failure_for_corrupt_pt(tmp_path: Path) -> None:
    """Return ``load_failure`` for corrupt .pt files.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts corrupt-file handling.
    """
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()
    positions_file = "positions/corrupt.pt"
    (tmp_path / positions_file).write_bytes(b"not a torch pickle")

    positions, reason = load_position_tensor(
        record_key="graph_a::engine_a::seed0",
        positions_file=positions_file,
        input_dir=tmp_path,
    )

    assert positions is None
    assert reason == "load_failure"


def test_open_h5_for_worker_returns_none_when_missing(tmp_path: Path) -> None:
    """Return ``None`` for a missing HDF5 file.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts the missing-file path.
    """
    assert open_h5_for_worker(tmp_path / "missing.h5") is None


def test_open_h5_for_worker_opens_existing_file(tmp_path: Path) -> None:
    """Open an existing HDF5 file in read mode.

    Parameters
    ----------
    tmp_path : Path
        Temporary benchmark root.

    Returns
    -------
    None
        This test asserts readable worker handles.
    """
    _, record_key, _, h5_tensor, h5_path = _build_position_artifacts(tmp_path)

    h5_file = open_h5_for_worker(h5_path)
    try:
        assert h5_file is not None
        assert record_key in h5_file
        loaded = torch.from_numpy(h5_file[record_key][:]).to(dtype=torch.float32)
        assert torch.equal(loaded, h5_tensor)
    finally:
        if h5_file is not None:
            h5_file.close()


def test_compute_quick_metrics_seeded_reproducibility() -> None:
    """Return identical quick metrics for the same seeded inputs.

    Returns
    -------
    None
        This test asserts deterministic seeded quick-metric output.
    """
    pos = torch.randn(20, 2)
    edge_index = torch.stack([torch.arange(19), torch.arange(1, 20)])
    node_sizes = torch.ones(20, 2) * 0.2

    result_a = compute_quick_metrics_seeded(pos, edge_index, node_sizes, seed=42)
    result_b = compute_quick_metrics_seeded(pos, edge_index, node_sizes, seed=42)

    assert result_a == result_b


def test_compute_quick_metrics_filter() -> None:
    """Respect the metric allow-list for seeded quick metrics.

    Returns
    -------
    None
        This test asserts filtering behavior.
    """
    pos = torch.randn(20, 2)
    edge_index = torch.stack([torch.arange(19), torch.arange(1, 20)])
    node_sizes = torch.ones(20, 2) * 0.2

    result = compute_quick_metrics_seeded(
        pos,
        edge_index,
        node_sizes,
        seed=42,
        metric_filter=frozenset({"edge_length_cv"}),
    )

    assert set(result.keys()) == {"edge_length_cv"}


def test_compute_sampled_metrics_seeded_reproducibility() -> None:
    """Return identical sampled metrics for the same seeded inputs.

    Returns
    -------
    None
        This test asserts deterministic sampled-metric output.
    """
    pos = torch.randn(30, 2)
    edge_index = torch.randint(0, 30, (2, 100))

    result_a = compute_sampled_metrics_seeded(pos, edge_index, num_nodes=30, seed=42)
    result_b = compute_sampled_metrics_seeded(pos, edge_index, num_nodes=30, seed=42)

    assert result_a == result_b


def test_aspect_ratio_deviation_is_zero_for_square_bbox() -> None:
    """Return zero for square layouts.

    Returns
    -------
    None
        This test asserts the ideal square case.
    """
    positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)

    assert aspect_ratio_deviation(positions) == pytest.approx(0.0)


def test_aspect_ratio_deviation_matches_log_for_wide_bbox() -> None:
    """Return ``|log(2.0)|`` for a 2x1 bounding box.

    Returns
    -------
    None
        This test asserts the wide-layout transform.
    """
    positions = torch.tensor([[0.0, 0.0], [2.0, 1.0], [1.0, 0.5]], dtype=torch.float32)

    assert aspect_ratio_deviation(positions) == pytest.approx(abs(math.log(2.0)))


def test_aspect_ratio_deviation_matches_log_for_tall_bbox() -> None:
    """Return ``|log(0.5)|`` for a 1x2 bounding box.

    Returns
    -------
    None
        This test asserts the tall-layout transform.
    """
    positions = torch.tensor([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]], dtype=torch.float32)

    assert aspect_ratio_deviation(positions) == pytest.approx(abs(math.log(0.5)))


def test_aspect_ratio_deviation_is_inf_for_degenerate_bbox() -> None:
    """Return infinity for degenerate layouts.

    Returns
    -------
    None
        This test asserts the degenerate-layout sentinel.
    """
    positions = torch.zeros((3, 2), dtype=torch.float32)

    assert math.isinf(aspect_ratio_deviation(positions))
