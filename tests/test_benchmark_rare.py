import pytest


@pytest.mark.rare
@pytest.mark.benchmark
def test_rare_benchmark_suite():
    """Run the rare benchmark ladder and merge its results into the latest view."""
    from dagua.eval.benchmark import merge_latest_results, run_rare_suite

    results = run_rare_suite(output_dir="eval_output")
    assert results is not None

    merged = merge_latest_results(output_dir="eval_output")
    assert merged is not None

