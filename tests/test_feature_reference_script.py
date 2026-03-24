"""Tests for the user-facing feature reference gallery builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    """Load the gallery builder script as an importable module.

    Returns
    -------
    ModuleType
        Imported script module.
    """

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_feature_reference.py"
    spec = importlib.util.spec_from_file_location("build_feature_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_counts_match_requested_gallery() -> None:
    """The script should expose the requested top-level catalog sizes.

    Returns
    -------
    None
        The default gallery counts are asserted in place.
    """

    module = _load_module()

    assert len(module.SHAPE_NAMES) == 20
    assert len(module.FEATURE_NAMES) == 7
    assert len(module.THEME_REGISTRY) == 44


def test_build_gallery_writes_expected_structure(tmp_path: Path) -> None:
    """Build a reduced gallery and verify the new output tree.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest directory.

    Returns
    -------
    None
        The rendered subset is asserted in place.
    """

    module = _load_module()
    output_dir = tmp_path / "gallery"

    result = module.build_feature_reference(
        output_dir=output_dir,
        shapes=("rect", "box3d"),
        arrowheads=("normal", "circle"),
        theme_names=("default", "dark"),
        feature_names=("borders", "effects"),
    )

    assert Path(result.index_path).exists()
    assert [section.title for section in result.sections] == [
        "Shapes",
        "Arrowheads",
        "Themes",
        "Feature Examples",
    ]

    shape_images = sorted((output_dir / "shapes").glob("*.png"))
    arrow_images = sorted((output_dir / "arrows").glob("*.png"))
    theme_images = sorted((output_dir / "themes").glob("*.png"))
    feature_images = sorted((output_dir / "features").glob("*.png"))

    assert [path.name for path in shape_images] == ["box3d.png", "rectangle.png"]
    assert [path.name for path in arrow_images] == ["circle.png", "normal.png"]
    assert [path.name for path in theme_images] == ["dark.png", "default.png"]
    assert [path.name for path in feature_images] == ["borders.png", "effects.png"]
    assert all(path.stat().st_size > 0 for path in shape_images + arrow_images + theme_images)
    assert all(path.stat().st_size > 0 for path in feature_images)

    html_text = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Dagua Feature Reference" in html_text
    assert "Shapes" in html_text
    assert "Arrowheads" in html_text
    assert "Themes" in html_text
    assert "Feature Examples" in html_text
    assert "shapes/rectangle.png" in html_text
    assert "themes/dark.png" in html_text
    assert not (output_dir / "competitors").exists()
