# Benchmarks

## Purpose

Performance tracking for layout and rendering at various scales. Used to validate
scaling strategy and catch performance regressions.

## Files

- **bench_layout.py** — Layout scaling: 100, 1K, 10K, 100K nodes. Measures wall-clock
  time and peak memory for the optimization loop. Tests both CPU and GPU (if available).
- **bench_render.py** — Rendering performance for each backend (mpl, svg, graphviz).
- **graphs/** — Reference graphs for reproducible benchmarks. Serialized edge lists
  and node metadata that produce deterministic inputs.

## Running

```bash
python benchmarks/bench_layout.py
python benchmarks/bench_render.py
```

## Key Metrics

- Wall-clock time per layout call (median of 5 runs)
- Peak memory (via `torch.cuda.max_memory_allocated` for GPU, tracemalloc for CPU)
- Scaling exponent: how time grows as N doubles
