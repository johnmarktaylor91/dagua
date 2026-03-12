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


@pytest.mark.smoke
def test_benchmark_watch_cli_one_shot(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "rare" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.partial.json").write_text(
        '{"run_id":"2026-03-12T00:00:00+00:00","suite":"rare","graphs":{"g":{"competitors":{"dagua":{"status":"RUNNING"}}}}}',
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text('{"graphs":["g"]}', encoding="utf-8")

    rc = main(["benchmark-watch", "--output-dir", str(output_dir), "--suite", "rare"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"suite": "rare"' in captured.out


@pytest.mark.smoke
def test_benchmark_list_cli_lists_runs(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_a = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_b = output_dir / "benchmark_db" / "standard" / "2026-03-12T01:00:00+00:00"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "results.json").write_text('{"graphs":{"g":{}}}', encoding="utf-8")
    (run_b / "results.partial.json").write_text('{"graphs":{"h":{}}}', encoding="utf-8")

    rc = main(["benchmark-list", "--output-dir", str(output_dir), "--suite", "standard"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"run_id": "2026-03-12T00:00:00+00:00"' in captured.out
    assert '"state": "partial"' in captured.out


@pytest.mark.smoke
def test_benchmark_show_cli_prints_graph_or_competitor(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "graphs": {
            "g": {
                "n_nodes": 3,
                "competitors": {
                    "dagua": {"status": "OK", "runtime_seconds": 0.2},
                    "graphviz_dot": {"status": "SKIPPED"},
                },
            }
        }
    }
    (run_dir / "results.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    (run_dir.parent / "latest").symlink_to(run_dir.name)

    rc = main(["benchmark-show", "g", "--output-dir", str(output_dir), "--suite", "standard", "--competitor", "dagua"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"competitor": "dagua"' in captured.out
    assert '"runtime_seconds": 0.2' in captured.out


@pytest.mark.smoke
def test_benchmark_freeze_cli_copies_run(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text('{"run_id":"2026-03-12T00:00:00+00:00","graphs":{}}', encoding="utf-8")
    (run_dir.parent / "latest").symlink_to(run_dir.name)

    rc = main(["benchmark-freeze", "baseline-a", "--output-dir", str(output_dir), "--suite", "standard"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"label": "baseline-a"' in captured.out
    assert (output_dir / "benchmark_db" / "standard" / "frozen" / "baseline-a" / "results.json").exists()
    assert (output_dir / "benchmark_db" / "standard" / "frozen" / "baseline-a" / "freeze_metadata.json").exists()


@pytest.mark.smoke
def test_benchmark_compare_runs_cli_prints_deltas(tmp_path, capsys):
    output_dir = tmp_path / "eval_output"
    run_a = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_b = output_dir / "benchmark_db" / "standard" / "2026-03-12T01:00:00+00:00"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    payload_a = {
        "run_id": run_a.name,
        "graphs": {"g": {"competitors": {"dagua": {"status": "OK", "runtime_seconds": 2.0, "composite_score": 70.0}}}},
    }
    payload_b = {
        "run_id": run_b.name,
        "graphs": {"g": {"competitors": {"dagua": {"status": "OK", "runtime_seconds": 1.0, "composite_score": 72.5}}}},
    }
    (run_a / "results.json").write_text(__import__("json").dumps(payload_a), encoding="utf-8")
    (run_b / "results.json").write_text(__import__("json").dumps(payload_b), encoding="utf-8")

    rc = main(
        [
            "benchmark-compare-runs",
            "2026-03-12T00:00:00+00:00",
            "2026-03-12T01:00:00+00:00",
            "--output-dir",
            str(output_dir),
            "--suite",
            "standard",
            "--competitor",
            "dagua",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert '"mean_score_delta": 2.5' in captured.out
    assert '"runtime_delta_seconds": -1.0' in captured.out


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


@pytest.mark.smoke
def test_benchmark_deltas_cli_writes_artifacts(tmp_path):
    output_dir = tmp_path / "eval_output"
    run_a = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_b = output_dir / "benchmark_db" / "standard" / "2026-03-12T01:00:00+00:00"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    payload_a = {
        "run_id": run_a.name,
        "suite": "standard",
        "graphs": {"g": {"competitors": {"dagua": {"status": "OK", "runtime_seconds": 2.0, "metrics": {"dag_consistency": 1.0}, "composite_score": 70.0}}}},
    }
    payload_b = {
        "run_id": run_b.name,
        "suite": "standard",
        "graphs": {"g": {"competitors": {"dagua": {"status": "OK", "runtime_seconds": 1.5, "metrics": {"dag_consistency": 1.0}, "composite_score": 72.0}}}},
    }
    (run_a / "results.json").write_text(__import__("json").dumps(payload_a), encoding="utf-8")
    (run_b / "results.json").write_text(__import__("json").dumps(payload_b), encoding="utf-8")
    (run_b.parent / "latest").symlink_to(run_b.name)

    rc = main(["benchmark-deltas", "--output-dir", str(output_dir)])

    assert rc == 0
    assert (output_dir / "report" / "benchmark_deltas.json").exists()
    assert (output_dir / "report" / "benchmark_deltas.md").exists()


@pytest.mark.slow
def test_poster_cli_uses_saved_benchmark_positions(tmp_path):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    positions_dir = run_dir / "positions"
    positions_dir.mkdir(parents=True, exist_ok=True)
    pos = __import__("torch").tensor(
        [
            [0.0, 0.0],
            [0.0, 60.0],
            [0.0, 120.0],
            [0.0, 180.0],
            [0.0, 240.0],
            [-80.0, 240.0],
            [-80.0, 300.0],
            [0.0, 300.0],
            [0.0, 360.0],
            [0.0, 420.0],
        ],
        dtype=__import__("torch").float32,
    )
    __import__("torch").save(pos, positions_dir / "residual_block__dagua.pt")
    payload = {
        "run_id": run_dir.name,
        "suite": "standard",
        "graphs": {
            "residual_block": {
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "positions_path": "positions/residual_block__dagua.pt",
                    }
                }
            }
        },
    }
    (run_dir / "results.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    (run_dir.parent / "latest").symlink_to(run_dir.name)
    out = tmp_path / "benchmark-poster.png"

    rc = main(
        [
                "poster",
                "unused.json",
                str(out),
                "--benchmark-graph",
                "residual_block",
                "--benchmark-suite",
                "standard",
            "--output-dir",
            str(output_dir),
            "--scene",
            "auto",
            "--steps",
            "1",
            "--edge-opt-steps",
            "-1",
        ]
    )

    assert rc == 0
    assert out.exists()
