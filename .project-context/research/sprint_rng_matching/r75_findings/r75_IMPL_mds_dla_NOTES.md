# r75 Classical MDS DLA Implementation Notes

## What Changed

- Implemented the disconnected-graph path in `dagua/layout/ops/pipelines/classical_mds.py`.
- Gate: the new DLA merge path is used only when weak components are discovered with
  `len(components) > 1`.
- Connected `igraph_fidelity=True` graphs continue through the original single-MDS kernel and were
  verified byte-identical before/after.
- Default unweighted connected graphs continue through the existing classic pipeline to preserve the
  existing test contract; default unweighted disconnected graphs use the new igraph-compatible DLA
  component merge.
- Added regression coverage in `tests/test_pipeline_classical_mds.py` for:
  - seeded deterministic DLA on disconnected graphs;
  - seed-dependent disconnected placement;
  - disconnected output no longer matching the legacy global-distance layout;
  - frozen connected igraph-fidelity output.

## Implementation Details

- Component discovery follows first unseen vertex order, with sorted component vertex lists matching
  the observed python-igraph weak subcomponent order on benchmark-style fixtures.
- Each component is laid out with the igraph single-component classical MDS semantics, including the
  two-node `[[0, 0], [1, 1]]` raw layout used by `igraph_i_layout_mds_single`.
- DLA merge uses `random.Random(seed)` through `_rng_unif()` so `RNG_UNIF` calls consume Python RNG
  draws in scalar walk order, matching the benchmark adapter's `igraph.set_random_number_generator`.
- `place_sphere()` preserves the four quadrant rasterization loops from `merge_grid.c`, including
  the asymmetric boundary checks.
- `get_sphere()` checks collision against the rasterized occupied grid cells rather than re-scanning
  every candidate quadrant in Python. This preserves the placed occupancy and collision predicate
  while avoiding per-step Python loops that made tiny disconnected fixtures take ~10 seconds.
- DLA guardrails raise `RuntimeError` rather than falling back:
  - `_IGRAPH_DLA_MAX_TOTAL_STEPS = 10_000_000`
  - `_IGRAPH_DLA_MAX_RESTARTS = 1_000_000`

## Commits

- No commit created. Project instructions say the orchestrator handles git operations.

## Probe Numbers

Command:

```bash
MPLCONFIGDIR=/tmp/mpl python3 scripts/run_benchmark.py --workers 2 --timeout 120 \
  --seeds 5 --seed-start 42 --variants --output-dir /tmp/r75_mds_probe \
  --graphs multi_component_80,parallel_cycles_4x5,random_bipartite_60 \
  --engines classic_classical_mds_default,classic_classical_mds_igraph_fidelity
```

Benchmark result: `30 total, 30 ok, 0 skipped, 0 errors, 0 timeouts`.

Stress comparison used `dagua.metrics.sampled_stress` on saved probe positions versus read-only
saved igraph reference tensors in
`/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_seeded_refs/positions`.

| Graph | Engine | Prior gap | New gap | New D | Ref R |
| --- | --- | ---: | ---: | ---: | ---: |
| multi_component_80 | default | +0.477318 | +0.071929 | 0.983037 | 0.911109 |
| multi_component_80 | igraph_fidelity | +0.477318 | +0.071929 | 0.983037 | 0.911109 |
| parallel_cycles_4x5 | default | +0.988977 | +0.331200 | 1.000000 | 0.668800 |
| parallel_cycles_4x5 | igraph_fidelity | +0.988977 | +0.331200 | 1.000000 | 0.668800 |
| random_bipartite_60 | default | +0.102470 | +0.022624 | 0.862651 | 0.840027 |
| random_bipartite_60 | igraph_fidelity | +0.102470 | +0.022624 | 0.862651 | 0.840027 |

The stress gap shrank on all 3 probe graphs for both variants.

## Connected Regression

Script: `/tmp/r75_connected_compare.py`.

Procedure:

1. Ran the script post-change and saved `/tmp/r75_connected_post.json`.
2. Ran `git stash push -u -m r75-mds-dla-verify`.
3. Ran the same script against `HEAD` and saved `/tmp/r75_connected_pre.json`.
4. Compared SHA-256 hashes of raw contiguous float64 tensor bytes.
5. Ran `git stash pop` and restored generated `__pycache__` artifacts.

Output:

```text
connected byte-identical: True
binary_tree seed=42 OK f892a5cb760d3b0187f9bf1e6f299f197b460571b3a16d8efa12dcca6518aafc
binary_tree seed=43 OK f892a5cb760d3b0187f9bf1e6f299f197b460571b3a16d8efa12dcca6518aafc
binary_tree seed=44 OK f892a5cb760d3b0187f9bf1e6f299f197b460571b3a16d8efa12dcca6518aafc
densenet_block seed=42 OK 83262c7ec4b71994960e40e46fff9e64f84b4adbda616d6a0d1c43daf0dacb5c
densenet_block seed=43 OK 83262c7ec4b71994960e40e46fff9e64f84b4adbda616d6a0d1c43daf0dacb5c
densenet_block seed=44 OK 83262c7ec4b71994960e40e46fff9e64f84b4adbda616d6a0d1c43daf0dacb5c
petersen_10 seed=42 OK 4022804648745fad7468fbdfef62c8a7bc6bc05eaff1f8f489a77e5101a51b07
petersen_10 seed=43 OK 4022804648745fad7468fbdfef62c8a7bc6bc05eaff1f8f489a77e5101a51b07
petersen_10 seed=44 OK 4022804648745fad7468fbdfef62c8a7bc6bc05eaff1f8f489a77e5101a51b07
```

## Open Residuals

- This targets rung-3 distributional equivalence, not bit-exact igraph layout parity.
- `get_sphere()` is optimized around occupied raster cells. If future work demands bit-exact
  walk traces against igraph C, port the `merge_grid.c` candidate quadrant scan literally in Cython
  or another low-overhead scalar implementation.
- Connected classical MDS degenerate-eigenspace residuals are intentionally untouched.

## Assumptions

- python-igraph weak subcomponent order for these fixtures is first unseen component order with
  ascending vertex order within each component.
- The benchmark reference's custom igraph RNG means Python `random.Random(seed)` is the correct
  DLA RNG source for distributional parity.
- Default unweighted connected graph behavior should preserve the existing test contract; default
  disconnected graph behavior should follow the approved igraph DLA path.
