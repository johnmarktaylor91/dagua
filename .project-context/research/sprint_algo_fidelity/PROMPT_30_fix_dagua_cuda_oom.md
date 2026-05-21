<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Fix the dagua native engine CUDA OOM that fires at initialization on EVERY graph
size (14 nodes to 5000 nodes), regardless of GPU memory available.

## Evidence

In `eval_output/benchmark_100seed_final/results.json`, all 95 "CUDA driver error:
out of memory" errors are on the `dagua` (native) engine. Error fires at
0.05-0.4 seconds runtime, meaning it's at initialization/preallocation, NOT
during a real layout computation. Examples:

- asymmetric_hourglass_hub: 14 nodes, 15 edges, runtime=0.056s → CUDA OOM
- binary_tree: 11 nodes, 10 edges, runtime=0.119s → CUDA OOM
- ba_500: 500 nodes, 1494 edges, runtime=0.168s → CUDA OOM
- ba_5000: 5000 nodes, 19990 edges, runtime=0.357s → CUDA OOM

A 14-node graph cannot possibly need significant GPU memory for the actual
layout. The OOM must come from a fixed preallocation (e.g., always allocating
N×N or 1024×1024 grid tensors regardless of graph size).

## Your job

1. Reproduce: build a tiny graph (e.g., 14 nodes) and call
   `dagua.layout(g, dagua.LayoutConfig(...))` for the native pipeline on CUDA.
   Confirm OOM. If you can't reproduce on local CUDA, look at the dagua native
   pipeline source for fixed-size allocations.

2. Find the preallocation: trace through `dagua.eval.competitors.dagua_competitor.py`
   (line 19, `name = "dagua"`) to find which layout function runs. From there
   into `dagua/layout/engine.py` or `dagua/layout/ops/pipelines/dagua_native.py`
   (3213 lines). Look for any unconditional large tensor allocations on the
   target device (e.g., `torch.zeros(K, K, device='cuda')` with K hardcoded).

3. Fix: make the offending allocation either:
   - Scale with `num_nodes` or `num_edges` instead of a constant
   - Allocate lazily right before use, with a try/except OOM that falls back to
     a smaller/CPU path

4. Add a regression test in `tests/test_layout/` that runs dagua native on a
   14-node graph on CPU (the bug manifests on CUDA but the size-mismatch logic
   should be visible on CPU too via tensor-shape introspection).

## Scope

- DO NOT TOUCH: render/styles, cluster sprint files, ANY benchmark
  scripts/output, ogdf_runner.cpp (built last week).
- Stage commits with explicit `git add <paths>`. NO `git add -A`.
- Commit format: `fix(layout): dagua native -- avoid fixed-size GPU preallocation on small graphs`

## Verification

After fix:
- `pytest tests/test_layout/ -x --tb=short -q -k "dagua_native or your_new_test"`
- Run the dagua native engine through the competitor adapter on the bounded
  5-graph subset (linear_3layer_mlp etc) on whatever device is available.
  Confirm no OOM, positions tensor returns with correct shape.

## Output

`eval_output/algo_fidelity/round_30/dagua_cuda_oom/SUMMARY.md` with the root
cause line:column and the fix.

</task>

<completeness_contract>
Either fix the OOM with a regression test, OR document why the OOM is
unreproducible / wontfix (e.g., specific CUDA driver version).
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
