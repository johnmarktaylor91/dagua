import pytest


@pytest.mark.slow
@pytest.mark.benchmark
def test_standard_benchmark_suite():
    """Run the persistent standard benchmark suite and regenerate the report."""
    from dagua.eval.benchmark import run_standard_suite
    from dagua.eval.report import generate_report

    results = run_standard_suite(output_dir="eval_output")
    assert results is not None
    for graph_name, graph_results in results["graphs"].items():
        dagua_result = graph_results["competitors"]["dagua"]
        assert dagua_result["status"] == "OK", f"Dagua failed on {graph_name}"

    artifacts = generate_report(output_dir="eval_output")
    assert artifacts["tex"]

