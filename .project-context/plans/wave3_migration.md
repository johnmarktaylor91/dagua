# Wave 3: Op Alignment + Pipeline Migration

## Goal

Make pipelines the primary code path. Remove all classic/ imports from pipelines.
Archive classic/ as reference-only. Clean dependency graph:

```
dagua/layout/ops/pipelines/*.py  -->  dagua/layout/ops/ (ops + shared utils)
dagua/layout/_archive/classic/*.py  -->  self-contained (reference only)
dagua/eval/ (benchmark)  -->  pipelines (not classic)
dagua/ (public API)  -->  pipelines (not classic)
```

No code reaches from pipelines into archive. No zombie imports.

## Phase 1: Extract Shared Utilities

Create `dagua/layout/ops/graph_utils.py` with the ~10 functions used by 3+ pipelines:

| Function | Currently in | Used by | What it does |
|----------|-------------|---------|-------------|
| `layout_device` | each classic/*.py | 20+ | Resolve output device from input tensors |
| `normalize_positions` | each classic/*.py | 8+ | Center + scale to extent |
| `layout_extent` | each classic/*.py | 8+ | Compute bounding box scale from node count/sizes |
| `build_undirected_adjacency` | 6+ classic/*.py | 6+ | edge_index -> undirected adjacency list/dense |
| `all_pairs_shortest_paths` | tsnet, umap, maxent, classical_mds | 4 | BFS/Dijkstra APSP |
| `shortest_path_distances` | classical_mds, stress_maj | 2 | APSP + unreachable fill |
| `rescale_layout` | fr, kk, spectral | 3 | Center + normalize to max_abs |
| `build_directed_adjacency` | _graph_distances, kk | 2 | Directed adjacency for shortest paths |
| `initialize_numpy_positions` | fr, drl, gem, graphopt, lgl | 5 | np.random.RandomState unit-square init |
| `initialize_torch_positions` | linlog, neulay | 2 | torch.randn seeded init |

These are NOT ops (they don't take problem/state/ctx). They're pure utility
functions that ops and pipelines both call. Single implementation, no duplication.

## Phase 2: Inline Algorithm-Specific Logic

For each of the 23 pipelines, move algorithm-specific helpers from classic/
INTO the pipeline file itself. This is ~180 functions. Examples:

- `fr.py`: `_adjacency_matrix`, `_rescale_layout` (FR-specific variant), constants
- `davidson_harel.py`: `_energy`, `_unique_edges`, `_COOLING_FACTOR`, etc.
- `gem.py`: `_compute_impulse_sequential`, `_update_node_sequential`, etc.
- `sugiyama.py`: `_barycenter_ordering`, `_coordinate_assignment`, etc.
- `sgd2_multi.py`: `_CrossingDetector`, `_CyclicSampler`, `_criterion_loss`, etc.

After this, each pipeline file is self-contained: it imports from ops/ and
graph_utils.py, never from classic/.

## Phase 3: Archive classic/

Move `dagua/layout/classic/` -> `dagua/layout/_archive/classic/`.

Make the archive self-contained:
- Copy shared utility functions INTO _archive/classic/ files (or a local utils)
- Each file still works standalone for reference reading
- No other code imports from _archive/

Add `dagua/layout/_archive/__init__.py` with docstring:
"Archival monolithic reimplementations. Reference only. Use dagua.layout.ops.pipelines."

Add `dagua/layout/_archive/classic/__init__.py` preserving the original exports
so old imports raise a clear ImportError with migration guidance.

## Phase 4: Update Benchmark Adapter

`dagua/eval/benchmark.py` (and related) resolves engine names like "classic_fr"
to layout callables. Update the resolution to call pipeline functions:

- "classic_fr" -> `layout_fr_pipeline` (from dagua.layout.ops.pipelines.fr)
- "classic_kk" -> `layout_kk_pipeline` (from dagua.layout.ops.pipelines.kk)
- etc. for all 23 algorithms

Keep the engine NAMES unchanged ("classic_fr_steps50" etc.) so cached benchmark
results stay valid. Only the underlying callable changes.

Read `dagua/eval/benchmark.py` and `dagua/eval/compare.py` to find where
the mapping lives. Also check `dagua/eval/variants.py` for the `base_engine`
field and how it resolves to a callable.

The in-progress benchmark run in `eval_output/variant_bench_full/` must remain
valid. Since pipelines produce bit-identical output, cached results are correct.

## Phase 5: Update Public API

`dagua/__init__.py` and `dagua/graph.py` may import from classic/. Update
to import from pipelines.

Check:
- `dagua/__init__.py` -- public API surface
- `dagua/graph.py` -- DaguaGraph.layout() method
- `dagua/layout/__init__.py` -- layout package exports
- `dagua/layout/engine.py` -- the optimization engine

## Phase 6: Update Ops to Use Shared Utilities

Where Wave 1 ops duplicate logic that now lives in graph_utils.py, update them
to call the shared implementation. This ensures ONE implementation exists:

- `BuildAdjacency` op -> calls `graph_utils.build_undirected_adjacency`
- `AllPairsShortestPaths` op -> calls `graph_utils.all_pairs_shortest_paths`
- `RandomUniformInit` op (numpy backend) -> calls `graph_utils.initialize_numpy_positions`
- `CenterPositions` + `ScalePositions` ops -> calls `graph_utils.normalize_positions`
- etc.

After this, the ops ARE the shared implementation, wrapped in the Op interface.
Pipelines can compose ops OR call graph_utils directly where the Op overhead
doesn't make sense.

NOTE: This phase may change op output slightly (aligning to the canonical
shared implementation). Re-run all 570 op tests after this phase. Fix any
that break due to the alignment.

## Phase 7: Verify Everything

1. `pytest tests/test_pipeline_*.py -x --tb=short -q` -- all 367 fidelity tests
2. `pytest tests/test_ops_*.py -x --tb=short -q` -- all 570 op tests
3. `python scripts/validate_pipeline_fidelity.py --max-nodes 50` -- variant validation
4. `pytest tests/ -x --tb=short -q --ignore=tests/test_animation.py -k "not test_hierarchy_checkpoint"` -- full suite

## File Inventory

CREATE:
- `dagua/layout/ops/graph_utils.py` -- shared utility functions
- `dagua/layout/_archive/__init__.py`
- `dagua/layout/_archive/classic/__init__.py` (migration notice)
- `dagua/layout/_archive/classic/*.py` (moved from classic/)

MODIFY:
- All 23 `dagua/layout/ops/pipelines/*.py` (replace classic imports with graph_utils + inline)
- `dagua/layout/ops/*.py` (ops that duplicate graph_utils logic)
- `dagua/eval/benchmark.py` (adapter resolution)
- `dagua/eval/compare.py` (if it imports classic directly)
- `dagua/__init__.py`, `dagua/graph.py`, `dagua/layout/__init__.py` (API routing)
- `dagua/layout/ops/__init__.py` (export graph_utils)
- `dagua/layout/ops/pipelines/__init__.py` (export all pipeline entry points)

DELETE:
- `dagua/layout/classic/` (after moving to _archive/)

## Execution Strategy

This is too large for one agent. Split into batches:

**Batch A**: Create graph_utils.py + update ops to use it (Phase 1 + 6)
**Batch B**: Inline algorithm logic into pipelines (Phase 2) -- 4-6 parallel agents
**Batch C**: Archive classic/ (Phase 3)
**Batch D**: Update benchmark + API routing (Phase 4 + 5)
**Batch E**: Full verification (Phase 7)

Sequential: A -> B -> C -> D -> E. Batches within B can be parallel.

## Constraints

- Bit-identical output MUST be preserved (torch.equal)
- Benchmark engine names MUST NOT change
- Cached benchmark data MUST remain valid
- No new dependencies
- All existing tests must pass
