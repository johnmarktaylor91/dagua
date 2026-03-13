"""Smoke coverage for the offline aesthetic iteration workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dagua.eval.aesthetic import run_aesthetic_iteration


@pytest.mark.smoke
def test_run_aesthetic_iteration_smoke(tmp_path):
    plt.close("all")
    result = run_aesthetic_iteration(
        rounds=1,
        output_dir=str(tmp_path / "aesthetic_review"),
        max_nodes=40,
        steps=20,
        seed=42,
    )

    out = Path(result["output_dir"])
    assert out.exists()
    assert (out / "round_01" / "round.json").exists()
    assert (out / "run_summary.json").exists()
    assert (out / "final" / "default_aesthetic_theme.yaml").exists()
    assert (out / "final" / "default_layout_config.json").exists()
    assert (out / "final" / "summary.md").exists()
    assert not plt.get_fignums()
