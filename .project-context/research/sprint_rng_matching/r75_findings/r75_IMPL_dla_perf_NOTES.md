# r75 Classical MDS DLA Performance Notes

## Profile Findings

Before changing code, I profiled `random_dag_200` seed 100 with `cProfile` and a 35s alarm around
`layout_classical_mds_pipeline()`.

```text
PROFILE_BEFORE random_dag_200 nodes=383 edges=300 seed=100 timeout=35s
PROFILE_BEFORE_RESULT random_dag_200 elapsed=35.000296s status=timeout_after_35s
524755 calls to get_sphere(): 32.645s cumulative
1041747 calls to ndarray.astype(): 7.270s cumulative
520874 calls to ndarray.any(): 2.120s cumulative
1049510 calls to _rng_unif(): 0.791s cumulative
```

The timeout was not in shortest paths or component MDS. It was the DLA walk repeatedly scanning
and reallocating coordinate arrays for all occupied raster cells inside `get_sphere()`.

## What Changed

- Cached fixed grid x/y cell coordinates once in `_IgraphMergeGrid.__init__()`.
- Changed `get_sphere()` to inspect only the candidate sphere's strict grid-coordinate bounding
  box before applying the unchanged squared-distance predicate.
- Kept the guardrail behavior unchanged: restart and step caps still raise `RuntimeError`.
- Kept connected-path dispatch untouched.
- Bound hot walk-loop call sites locally and inlined `_rng_unif()`'s exact scalar expression:
  `low + (high - low) * rng.random()`. This preserves RNG draw count, order, and values while
  avoiding millions of helper calls.
- Added `test_igraph_merge_grid_lookup_matches_full_occupied_scan()` to compare optimized lookup
  against a brute-force occupied-cell scan.

## Timing Evidence

Direct seed-100 timings after the final fast path:

```text
WALL_AFTER5 random_dag_50 seed=100 elapsed=2.867695s target=5.0s shape=(97, 2)
WALL_AFTER5 random_dag_200 seed=100 elapsed=8.866784s target=10.0s shape=(383, 2)
```

Benchmark-path probe:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl PYTHONDONTWRITEBYTECODE=1 \
python scripts/run_benchmark.py --workers 2 --timeout 300 --seeds 5 --seed-start 100 \
  --variants --graphs random_dag_50,random_dag_200 \
  --engines classic_classical_mds_default,classic_classical_mds_igraph_fidelity \
  --output-dir /tmp/r75_dla_perf_probe
```

Result:

```text
[benchmark] Done: 20 total, 20 ok, 0 skipped, 0 errors, 0 timeouts
```

Summary excerpt:

```text
classic_classical_mds_default: 10 ok, mean 5.56s, median 4.78s, max 11.27s
classic_classical_mds_igraph_fidelity: 10 ok, mean 5.98s, median 5.27s, max 13.38s
random_dag_50: 10 ok, 0 timeouts
random_dag_200: 10 ok, 0 timeouts
```

## Bit-Identity Evidence

Before editing, I saved pre-optimization outputs to
`/tmp/r75_dla_pre_optimization_outputs.pt` for seeds 100, 101, and 102 on:

- `multi_component_80`
- `parallel_cycles_4x5`
- `disconnected_encoder_residual`

After optimizing, I rebuilt the same graphs and compared with `torch.equal`.

```text
BIT_IDENTITY multi_component_80 seed=100 torch_equal=True max_abs=0
BIT_IDENTITY multi_component_80 seed=101 torch_equal=True max_abs=0
BIT_IDENTITY multi_component_80 seed=102 torch_equal=True max_abs=0
BIT_IDENTITY parallel_cycles_4x5 seed=100 torch_equal=True max_abs=0
BIT_IDENTITY parallel_cycles_4x5 seed=101 torch_equal=True max_abs=0
BIT_IDENTITY parallel_cycles_4x5 seed=102 torch_equal=True max_abs=0
BIT_IDENTITY disconnected_encoder_residual seed=100 torch_equal=True max_abs=0
BIT_IDENTITY disconnected_encoder_residual seed=101 torch_equal=True max_abs=0
BIT_IDENTITY disconnected_encoder_residual seed=102 torch_equal=True max_abs=0
```

## Test Results

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

pytest tests/test_pipeline_classical_mds.py -x -q
13 passed, 3 warnings in 0.79s
```

Attempted but interrupted for duration:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

This broad project gate was still CPU-bound after roughly an hour and had reached 31% with only
passing progress shown. I stopped it to complete the user-specified benchmark verification.

## Assumptions

- Returning any colliding sphere id remains sufficient because `_igraph_layout_merge_dla_walk()`
  only branches on collision/non-collision; the returned id is not otherwise consumed.
- The local grid-coordinate bounding box is conservative for the unchanged strict predicate
  `delta_x^2 + delta_y^2 < radius^2`.
- The final commit SHA is reported in the task summary; embedding a self-referential final SHA in
  this committed file is not mechanically stable because amending the note changes the SHA.

## Concerns

- Benchmark worker overhead and normal runtime variance can put individual `random_dag_200` runs
  slightly above 10s, but the requested benchmark probe completed 20/20 with no timeouts.
- The broad layout/graph pytest gate appears too slow for this scoped change in the current
  environment; the classical MDS-specific test and benchmark-path probe are green.

## Knowledge

- The many-component DLA tail is dominated by collision lookup inside the random walk, not by
  component MDS or shortest paths.
- Avoiding all-occupied-cell scans matters more than optimizing sphere rasterization; `place_sphere`
  was negligible in the before profile.
