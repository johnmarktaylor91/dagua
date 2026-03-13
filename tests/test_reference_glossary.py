from __future__ import annotations

import json
from pathlib import Path

from dagua.reference_glossary import build_glossary


def test_reference_glossary_rebuilds(tmp_path: Path):
    result = build_glossary(
        output_dir=str(tmp_path / "glossary"),
        compile_pdf=False,
        sample_steps=8,
    )

    tex_path = Path(result.tex_path)
    manifest_path = Path(result.manifest_path)
    figures_dir = Path(result.figures_dir)

    assert tex_path.exists()
    assert manifest_path.exists()
    assert figures_dir.exists()

    tex = tex_path.read_text(encoding="utf-8")
    assert "Dagua Exhaustive Glossary and Reference" in tex
    assert "Graph Construction and Orchestration Methods" in tex
    assert "Optimization Hyperparameters" in tex

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["public_api_count"] > 0
    assert manifest["graph_method_count"] > 0
    assert manifest["loss_count"] > 0
    assert manifest["hyperparameter_count"] > 0

    expected_figures = {
        "pipeline_stages.png",
        "direction_modes.png",
        "spacing_sweep.png",
        "crossing_sweep.png",
        "routing_styles.png",
        "flex_constraints.png",
    }
    assert expected_figures.issubset({path.name for path in figures_dir.iterdir()})
