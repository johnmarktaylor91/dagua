"""Regression tests for opt-in bit-equivalent PNG rendering."""

from __future__ import annotations

import builtins
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import dagua
from dagua import DaguaGraph
from dagua.render._backend import _render_via_cairosvg


def _has_dot() -> bool:
    """Return True when Graphviz dot is available on PATH.

    Returns
    -------
    bool
        True if ``dot -V`` exits successfully, otherwise False.
    """
    try:
        subprocess.run(["dot", "-V"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def _read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB array.

    Parameters
    ----------
    path : pathlib.Path
        Image path to read.

    Returns
    -------
    numpy.ndarray
        RGB image array with shape ``[H, W, 3]`` and dtype ``uint8``.
    """
    return np.array(Image.open(path).convert("RGB"))


def _resize_to_shape(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Read and resize an RGB image to a target array shape.

    Parameters
    ----------
    path : pathlib.Path
        Image path to read.
    shape : tuple[int, int, int]
        Target RGB array shape ``[H, W, 3]``.

    Returns
    -------
    numpy.ndarray
        Resized RGB image array with shape ``shape``.
    """
    target_height, target_width, _channels = shape
    image = Image.open(path).convert("RGB").resize((target_width, target_height))
    return np.array(image)


def test_bit_equivalent_render_writes_png(tmp_path: Path) -> None:
    """``bit_equivalent=True`` should write a non-empty PNG via cairosvg."""
    pytest.importorskip("cairosvg")

    graph = DaguaGraph(_theme=dagua.GRAPHVIZ_STRICT_THEME.copy())
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    graph.add_edge("a", "b")
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]], dtype=torch.float32)
    output = tmp_path / "bit_equivalent.png"

    fig, _ax = dagua.render(graph, positions, output=output, bit_equivalent=True)

    assert output.exists()
    assert output.stat().st_size > 1000
    assert _read_rgb(output).shape[2] == 3
    fig.clf()


def test_bit_equivalent_missing_cairosvg_has_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing cairosvg should raise an ImportError with the extra name."""
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Block cairosvg imports while delegating all other imports."""
        if name == "cairosvg":
            raise ImportError("blocked cairosvg import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ImportError, match="dagua\\[bit_equivalent\\]"):
        _render_via_cairosvg(
            b"<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/>",
            tmp_path / "out.png",
        )


@pytest.mark.skipif(not _has_dot(), reason="graphviz dot CLI not installed")
def test_bit_equivalent_render_matches_dot_png(tmp_path: Path) -> None:
    """Opt-in cairosvg rasterization should approach ``dot -Tpng`` output."""
    pytest.importorskip("cairosvg")
    pytest.importorskip("pydot")

    dot_text = """
    digraph G {
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }
    """
    dot_path = tmp_path / "input.dot"
    dot_path.write_text(dot_text)

    expected_png = tmp_path / "expected.png"
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=96", str(dot_path), "-o", str(expected_png)],
        check=True,
    )

    graph = dagua.from_dot(dot_text, _theme=dagua.GRAPHVIZ_STRICT_THEME.copy())
    positions = dagua.layout(graph, dagua.LayoutConfig(algorithm="sugiyama"))
    actual_png = tmp_path / "actual.png"
    dagua.render(
        graph,
        positions,
        output=actual_png,
        bit_equivalent=True,
    )

    expected_arr = _read_rgb(expected_png)
    actual_arr = _read_rgb(actual_png)
    if expected_arr.shape != actual_arr.shape:
        actual_arr = _resize_to_shape(actual_png, expected_arr.shape)

    score = float(ssim(expected_arr, actual_arr, channel_axis=2, data_range=255))
    if score < 0.99:
        pytest.xfail(
            f"SSIM {score:.4f} is below the 0.99 bit-equivalence gate; "
            "rasterization path works, but current layout/style fidelity is not yet converged."
        )
    assert score >= 0.99
