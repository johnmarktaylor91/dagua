from __future__ import annotations

import json
from pathlib import Path

from dagua.showcase_gallery import build_showcase_gallery


def test_showcase_gallery_rebuilds(tmp_path: Path):
    result = build_showcase_gallery(
        output_dir=str(tmp_path / "gallery"),
        include_animations=False,
        limit=2,
        sample_steps=14,
    )

    readme_path = Path(result.readme_path)
    manifest_path = Path(result.manifest_path)
    stills_dir = Path(result.output_dir) / "stills"

    assert readme_path.exists()
    assert manifest_path.exists()
    assert stills_dir.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["stills"]) == 2
    assert manifest["animations"] == []

    readme = readme_path.read_text(encoding="utf-8")
    assert "Dagua Showcase Gallery" in readme
    assert "Industry:" in readme
    assert len(list(stills_dir.glob("*.png"))) == 2
