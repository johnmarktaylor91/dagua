# Round 27 Diff: dagua neato-mode vs Graphviz neato

Diagnostic-only line-by-line diff against `/home/jtaylor/projects/_references/graphviz/lib/neatogen/`.

## Scope finding

Dagua does not currently register an in-process `algorithm="neato"` name. `layout()` resolves
explicit algorithms through `get_pipeline_function(config.algorithm)` at
`dagua/layout/engine.py:1071-1139`, and `PIPELINE_REGISTRY` has
`"stress_majorization"` at `dagua/layout/ops/pipelines/__init__.py:77-80` and
`"classical_mds"` at `dagua/layout/ops/pipelines/__init__.py:13-16`, but no
`"neato"` alias at `dagua/layout/ops/pipelines/__init__.py:12-88`.

The benchmark "neato mode" therefore means `classic_stress_maj` or
`classic_classical_mds` compared against external `graphviz_neato`:
`dagua/eval/competitors/classic_competitor.py:189-198`,
`dagua/eval/competitors/classic_competitor.py:1074-1103`, and
`dagua/eval/competitors/graphviz_competitor.py:531-535`.

## Ranked divergences

### 1. P0: No Graphviz-neato dispatch contract in dagua

- Dagua: no `neato` entry in `PIPELINE_REGISTRY` at
  `dagua/layout/ops/pipelines/__init__.py:12-88`; explicit `algorithm="neato"`
  would fall through `get_pipeline_function()` at `dagua/layout/engine.py:1075-1078`
  and raise `KeyError`.
- Graphviz: `neato_layout()` always resolves graph attributes into `layoutMode`,
  `model`, pack mode, overlap mode, and then runs `neatoLayout()` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1344-1440`.
- Label: `dispatch_contract_gap`.
- Estimated fix size: M. Add a `neato` adapter/alias and decide whether it means
  Graphviz default `mode=major,model=shortpath` only or also exposes `mode/model/start`.

### 2. P0: Default model selection is not equivalent to Graphviz `mode=major`

- Graphviz: default mode is `MODE_MAJOR` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:634-661`;
  default model is `MODEL_SHORTPATH` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:606-632`;
  `neatoLayout()` sends non-KK/non-SGD modes to `majorization()` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1295-1300`.
- Dagua: the benchmark proxy is selected out-of-band as either dense SMACOF
  `classic_stress_maj` or pure classical MDS at
  `dagua/eval/competitors/classic_competitor.py:189-198`; the public engine
  has no Graphviz `mode/model` parser.
- Label: `model_selection_gap`.
- Estimated fix size: M-L. Implement Graphviz-style `mode` and `model` surface,
  with default `mode=major, model=shortpath`.

### 3. P0: Initialization is Graphviz random-HDE, not dagua MDS+jitter

- Graphviz: `majorization()` calls `checkStart(..., INIT_RANDOM)` for default
  major mode at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1092`;
  `setSeed()` parses `start` and seeds `srand48()` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:920-953`
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:980-992`.
  In dense stress, non-smart init calls `initLayout()` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:914-916`.
- Dagua stress proxy: always computes a rank-2 classical-MDS warm start at
  `dagua/layout/ops/stress.py:383-395`, then adds seeded Gaussian jitter at
  `dagua/layout/ops/stress.py:396-399`.
- Dagua classical-MDS proxy: is deterministic and ignores seed for geometry at
  `dagua/layout/ops/pipelines/classical_mds.py:55-96`.
- Label: `init_basin_gap`.
- Estimated fix size: M. Add a Graphviz-neato init mode using `start` semantics
  and random initial coordinates instead of MDS+jitter.

### 4. P1: Iteration count matches nominal default but convergence semantics differ

- Graphviz: default major `MaxIter = DFLT_ITERATIONS` when no `maxiter` attr is
  set at `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1283-1290`;
  dense majorization stops early on relative stress change or tiny stress at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:994-1074`.
- Dagua stress proxy: `classic_stress_maj` defaults to exactly 200 iterations at
  `dagua/eval/competitors/classic_competitor.py:194-198`; the pipeline uses
  `FixedSteps` and `Repeat(n=iterations)` at
  `dagua/layout/ops/pipelines/stress_majorization.py:80-92`, with no Graphviz
  early stop.
- Label: `convergence_policy_gap`.
- Estimated fix size: S-M. Add Graphviz-style `Epsilon` early termination to the
  stress pipeline adapter.

### 5. P1: Linear solve/update path is not Graphviz conjugate-gradient mkernel

- Graphviz: builds packed Laplacian weights from APSP at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:947-970`,
  constructs a distance-dependent Laplacian each iteration at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:997-1045`,
  then solves per axis using `conjugate_gradient_mkernel()` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:1076-1096`.
- Dagua stress proxy: precomputes a dense Laplacian pseudoinverse at
  `dagua/layout/ops/stress.py:156-173`, forms dense `B(X)` at
  `dagua/layout/ops/stress.py:496-518`, and applies the dense candidate directly.
- Label: `solver_kernel_gap`.
- Estimated fix size: L if exact; S if accepted as numerical residual because
  TOST already passes.

### 6. P1: Dagua has a monotonicity safeguard Graphviz does not have

- Graphviz: computes stress before solving and uses it only for convergence at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:1054-1074`;
  it does not halve/reject a completed candidate.
- Dagua: if candidate stress increases, blends toward the previous coordinates
  up to eight times or rejects at `dagua/layout/ops/stress.py:531-554`.
- Label: `extra_safeguard_gap`.
- Estimated fix size: S. Disable halving in Graphviz-neato fidelity mode.

### 7. P1: Disconnected-distance fill differs

- Graphviz BFS: unreachable nodes are assigned `closestDist + 10` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/bfs.c:50-53`.
- Dagua default stress/MDS: unreachable pairs use global `max_distance + 1` at
  `dagua/layout/ops/graph_utils.py:319-352`; the OGDF mode has a separate
  `sqrt(N)` fill at `dagua/layout/ops/stress.py:203-214`.
- Label: `disconnected_distance_gap`.
- Estimated fix size: S. Add a Graphviz-neato distance fill policy.

### 8. P1: Multi-edge and `len` handling differ

- Graphviz `makeGraphData()`: ignores loops, collapses multiedges by undirected
  endpoint, adds edge weights, and takes max `len` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:740-747`
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:809-828`.
- Dagua shared graph distances keep the minimum duplicate edge weight in
  `_build_min_weight_undirected_adjacency()` at
  `dagua/layout/ops/graph_utils.py:26-49`; other generic adjacency builders sum
  duplicates at `dagua/layout/ops/graph_utils.py:226-268`.
- Graphviz MDS model requires edge `len` and uses weighted APSP plus direct
  replacement of edge entries at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:618-626`
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:664-692`.
- Label: `edge_length_multiedge_gap`.
- Estimated fix size: M.

### 9. P2: `stresswt` exponent is not exposed

- Graphviz: `stresswt` defaults to 2, validates only 1 or 2 at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:956-969`;
  stress uses either `1/d^2` or `1/d` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:84-120`
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:947-953`.
- Dagua: stress weights are always inverse-square at
  `dagua/layout/ops/stress.py:156-165`.
- Label: `stress_weight_option_gap`.
- Estimated fix size: S.

### 10. P2: Graphviz `subset`/`circuit`/`mds` model branches are missing

- Graphviz: `neatoModel()` recognizes `circuit`, `subset`, `shortpath`, and
  `mds` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:606-632`;
  dense stress branches at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:846-872`.
- Dagua: classical MDS and stress proxies have no Graphviz `model` parameter at
  `dagua/layout/ops/pipelines/stress_majorization.py:98-107` and
  `dagua/layout/ops/pipelines/classical_mds.py:55-62`.
- Label: `model_branch_gap`.
- Estimated fix size: M-L.

### 11. P2: Smart/PCA init is not Graphviz-compatible when requested

- Graphviz smart init performs high-dimensional embedding, PCA, optional
  iterative PCA for 2-D, and sparse subspace majorization at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:296-305`,
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:503-545`,
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/pca.c:24-97`.
- Dagua MDS init uses direct eigendecomposition of a double-centered distance
  matrix at `dagua/layout/ops/stress.py:270-320` or
  `dagua/layout/ops/embed.py:324-386`.
- Label: `pca_smart_init_gap`.
- Estimated fix size: L.

### 12. P2: Graphviz post-processing is much broader

- Graphviz: component packing, overlap removal, edge spline routing, BB compute,
  z propagation, and final postprocess are wired at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1371-1440`;
  overlap removal includes Prism/VPSC-style paths in
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/adjust.c:984-1001`
  and `/home/jtaylor/projects/_references/graphviz/lib/neatogen/overlap.c:486-589`;
  multispline routing uses shortest paths and spline routing at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/multispline.c:684-790`.
- Dagua stress/classical-MDS proxies return point coordinates only at
  `dagua/layout/ops/stress.py:667-722` and
  `dagua/layout/ops/postprocess.py:947-1008`.
- Label: `postprocess_surface_gap`.
- Estimated fix size: L if attempting full Graphviz binary parity; out of scope
  for position-only fidelity.

## Baseline command

Requested command:

```bash
python scripts/algo_fidelity_live_compare.py classic_stress_maj graphviz_neato \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/neato/baseline
```

Results will be recorded after execution.

## Cache availability note

Execution succeeded and wrote:

- `eval_output/algo_fidelity/round_27/neato/baseline/multi_seed_rmsd.csv`
- `eval_output/algo_fidelity/round_27/neato/baseline/multi_seed_summary.json`

Aggregate stdout:

```text
Wrote 2325 rows to eval_output/algo_fidelity/round_27/neato/baseline/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_27/neato/baseline/multi_seed_summary.json
graphs: 5
median: 0.035264
p25: 0.002167
p75: 0.035275
p95: 0.037999
worst: tl_mlp_3layer 0.038680
```

Cache availability: only one `graphviz_neato` target seed was available per
requested graph (`n_graphviz_seeds: 1` for all five graphs), while dagua ran 30
seeds (`n_dagua_seeds: 30`). Therefore every graph-level TOST verdict is
`not_tested`; within-Graphviz stochastic-floor rows are unavailable for this
exact command/output because there are no Graphviz seed pairs.

Per-graph dagua-vs-Graphviz medians:

| graph | median RMSD | graphviz seeds | TOST |
| --- | ---: | ---: | --- |
| `linear_3layer_mlp` | 0.035264 | 1 | `not_tested` |
| `parallel_multiedge_bundle` | 0.001711 | 1 | `not_tested` |
| `nested_shallow_enc_dec` | 0.035275 | 1 | `not_tested` |
| `tl_mlp_3layer` | 0.038680 | 1 | `not_tested` |
| `mixed_width_labels` | 0.002167 | 1 | `not_tested` |
