from pathlib import Path

import pytest

from dagua import DaguaGraph
from dagua.cli import main


@pytest.mark.smoke
def test_benchmark_status_cli_prints_json(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.partial.json").write_text(
        '{"run_id":"2026-03-12T00:00:00+00:00","suite":"standard","graphs":{"g":{"competitors":{"dagua":{"status":"RUNNING"}}}}}',
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text('{"graphs":["g"]}', encoding="utf-8")

    rc = main(["benchmark-status", "--output-dir", str(output_dir), "--suite", "standard"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"suite": "standard"' in captured.out
    assert '"is_partial": true' in captured.out


@pytest.mark.slow
def test_poster_cli_exports_png(tmp_path):
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    graph_path = tmp_path / "graph.json"
    graph.save(graph_path)
    out = tmp_path / "poster.png"

    rc = main(
        [
            "poster",
            str(graph_path),
            str(out),
            "--steps",
            "10",
            "--edge-opt-steps",
            "-1",
            "--scene",
            "powers_of_ten",
        ]
    )

    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0
