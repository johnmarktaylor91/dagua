# r76-C3 UMAP scalar-faithful SGD port notes

## Phase 1 first-divergence trace

Environment:

- Worktree: `/home/jtaylor/.claude/worktrees/dagua-umap-port`
- Branch: `r76/umap-port`
- Reference package: `umap-learn 0.5.11` at
  `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/__init__.py`
- Reference source cites:
  - `umap/umap_.py:565-586`: dense/precomputed kNN distances are cast to float32 and membership strengths are built.
  - `umap/umap_.py:588-603`: fuzzy simplicial set COO construction and fuzzy union.
  - `umap/umap_.py:906-925`: `make_epochs_per_sample`.
  - `umap/umap_.py:1095-1098`: random initialization path.
  - `umap/umap_.py:1146-1152`: positive-edge schedule and optimizer RNG-state draw.
  - `umap/umap_.py:1188-1215`: embedding rescale and euclidean optimizer call with `move_other=True`.
  - `umap/layouts.py:63-187`: single-epoch SGD loop, gradient clipping, negative sample count, and tau-rand consumption.
  - `umap/layouts.py:323-325`: negative-sample and positive-sample epoch counters.
  - `umap/layouts.py:367-371`: per-sample tau-rand state derivation.
  - `umap/utils.py:40-63`: `tau_rand_int` xorshift generator.

Trace graph and parameters:

- Graph: four-node path `0-1-2-3`, represented as dagua `edge_index=[[0,1,2],[1,2,3]]`.
- Seed: `7`
- `n_neighbors=3`, `n_epochs=10`, `negative_sample_rate=5`, `min_dist=0.1`, `spread=1.0`
- Init path: `random`, matching benchmark adapter behavior for `4 <= N < 10`.

Stage table:

| Stage | Reference | Dagua | Verdict |
| --- | --- | --- | --- |
| Distance matrix | `[[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]]` | Same | Match |
| kNN indices | `[[0,1,2],[1,0,2],[2,1,3],[3,2,1]]` | Same | Match |
| kNN distances | `[[0,1,2],[0,1,1],[0,1,1],[0,1,2]]` | Same | Match |
| Sigmas | `[1.8649902, 0.00066666666, 0.00066666666, 1.8649902]` | `[1.8649902, 0.00066666672, 0.00066666672, 1.8649902]` | Numerically equivalent at float32 print precision; no downstream fuzzy weight divergence observed |
| Rhos | `[1,1,1,1]` | Same | Match |
| Fuzzy COO rows/cols/vals | `(0,1,1.0), (0,2,0.58496934), (1,0,1.0), (1,2,1.0), (1,3,0.58496934), (2,0,0.58496934), (2,1,1.0), (2,3,1.0), (3,1,0.58496934), (3,2,1.0)` | Same sorted COO triples | Match |
| Random init before rescale | `[[-8.473834,5.598376],[-1.2318153,4.4693036],[9.559791,0.7699174],[0.02240927,-8.558977]]` | Same | Match |
| Init after `[0,10]` rescale | `[[0,9.999999],[4.015842,9.202484],[10,6.5894346],[4.711334,0]]` | Same | Match |
| Optimizer base RNG state | `[-994546981,-2079149175,-504664]` | Same | Match |
| Curve `(a,b)` | `(1.5769434602912036, 0.8950608779947933)` | Same | Match |
| Positive edge order | `head=[0,0,1,1,1,2,2,2,3,3]`, `tail=[1,2,0,2,3,0,1,3,1,2]` | Same | Match |
| `epochs_per_sample` | `[1.0,1.70949133,1.0,1.0,1.70949133,1.70949133,1.0,1.0,1.70949133,1.0]` | `[1.0,1.7094913,1.0,1.0,1.7094913,1.7094913,1.0,1.0,1.7094913,1.0]` | **First divergence: max abs diff `3.4837252638197924e-08` on weak fuzzy edges.** |
| Epoch 0 events | None: every `epoch_of_next_sample >= 1.0` | None | Match |
| Epoch 1 edge visits | Edge ids `0,2,3,6,7,9` | Same | Match |
| Epoch 1 first edge `0->1` positive gradient | `dist_sq=16.7630176544`, `grad_coeff=-0.101622182743`, `grad=[0.408098625619,-0.081045206480]` | Same | Match |
| Epoch 1 first edge negative draws | `[-1591305933, 968218987, -21823166, -1923310480]`, targets `[3,3,2,0]`, last skipped as self | Same | Match |
| Epoch 1 final embedding | `[[0.90480214,9.822596],[3.662226,9.183711],[9.312577,6.427497],[4.9644647,0.35099682]]` | Same | Match |

Named first divergence:

The first divergent quantity is `epochs_per_sample` for weak fuzzy edges, before any SGD update.
Reference `make_epochs_per_sample` computes `n_samples = n_epochs * (weights / weights.max())`
with `weights` as float32 sparse graph data, then converts the already-rounded float32 `n_samples`
to float64 for division (`umap/umap_.py:922-925`). Dagua computed `max_weight / kept_weight` after
promoting `kept_weight` to torch float64 (`dagua/layout/ops/umap.py:958-966` before this patch),
which preserves extra ratio precision and shifts the sampling schedule by about `3.48e-08`.

This tiny trace did not show tau-rand divergence: epoch-1 edge schedule, draw order, draw values,
negative targets, gradient coefficients, clipped gradients, and final epoch-1 embedding matched
draw-for-draw after the schedule values were close enough not to alter the first visit boundary.

## Target combos

The requested `eval_output/fidelity_definitive/r75_final.jsonl` is absent from this worktree.
I used the existing r75 target list in
`.project-context/research/sprint_rng_matching/r75_findings/r75_targets_small_tails.json`, which
lists these seven UMAP divergent combos:

- `parallel_multiedge_bundle::classic_umap_default`
- `parallel_multiedge_bundle::classic_umap_mindist001`
- `parallel_multiedge_bundle::classic_umap_nn5`
- `parallel_multiedge_bundle::classic_umap_spread2`
- `parallel_multiedge_bundle::classic_umap_nn30`
- `random_dag_50::classic_umap_nn5`
- `random_dag_200::classic_umap_nn5`

## Implementation notes

Implemented in `dagua/layout/ops/umap.py`:

- Ported `epochs_per_sample` construction to match reference `make_epochs_per_sample`
  (`umap/umap_.py:906-925`): compute `n_samples = n_epochs * (weights / weights.max())`
  in float32, then divide in float64. This removes the first traced pre-SGD schedule
  divergence.
- Removed the optional runtime numba wrappers from the UMAP op module so dagua's UMAP
  runtime no longer imports or invokes numba. The remaining scalar helpers preserve the
  tau-rand xorshift state update (`umap/utils.py:40-63`) and the optimizer loop order
  from `umap/layouts.py:63-187`.

The scalar port passes the mandatory tiny trace through epoch 1 draw-for-draw. On larger
graphs, however, the no-numba requirement means dagua now executes Python/NumPy scalar
arithmetic where the reference still executes numba kernels with `fastmath=True`
(`umap/layouts.py:31-41`, `umap/layouts.py:222-228`). The first probe below shows that this
does not materially improve the divergent benchmark row.

## Probe evidence

Probe method:

- Graph: `random_dag_50`, built as `_random_dag(50, 70, seed=42)` from
  `dagua/eval/graphs.py:870-879` and `dagua/eval/graphs.py:2186-2201`.
- Combo: `random_dag_50::classic_umap_nn5`.
- Seeds: `0..4`.
- Distance: orthogonal Procrustes RMSD vs `UMAPGraph.layout_with_variant(...,
  variant_params={"n_neighbors": 5, "min_dist": 0.1, "spread": 1.0})`.
- Baseline: old `dagua/layout/ops/umap.py` loaded from `git show HEAD:...` in a scratch
  process and executed through the same internal stages.

| Seed | Old RMSD | New RMSD | Improvement |
| --- | ---: | ---: | ---: |
| 0 | 0.016003294 | 0.024253350 | -0.008250056 |
| 1 | 0.006955644 | 0.007833333 | -0.000877690 |
| 2 | 0.020666926 | 0.015223136 | 0.005443791 |
| 3 | 0.017046078 | 0.015522676 | 0.001523402 |
| 4 | 0.014512850 | 0.010014311 | 0.004498539 |
| Mean | 0.015036958 | 0.014569361 | 0.000467597 |

Verdict: **probe gate failed**. Mean RMSD improved only `0.000467597`, and 2 of 5 seeds
regressed. I did not run the remaining required divergent/previously-identical combo probes
because this representative divergent row is already insufficient for gate 2.

## Gate evidence

- Gate 1: PASS. Phase-1 trace table exists above and names the first divergence.
- Gate 2: FAIL. `random_dag_50::classic_umap_nn5` five-seed probe does not materially
  improve and has 2/5 regressions.
- Gate 3: PARTIAL.
  - `ruff check . --fix`: PASS, `All checks passed!`
  - `pytest tests/test_pipeline_umap_layout.py -x -q`: PASS, `14 passed, 3 warnings`
  - `pytest tests/test_ops_optimize.py -k "umap" -x -q`: PASS, `1 passed, 19 deselected, 3 warnings`
  - `pytest tests/test_ops_embed.py -k "umap" -x -q`: no selected tests, pytest exit 5,
    `42 deselected, 3 warnings`
  - `pytest tests/ -k "umap" -x -q`: FAIL/blocked, exited `-1` twice with no pytest output.
  - `mypy --follow-imports=silent dagua/cli.py`: PASS, `Success: no issues found in 1 source file`
  - `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: interrupted after
    several minutes at about 15% progress because gate 2 was already failed.
- Gate 4: PASS for touched UMAP runtime files. AST import check found no `umap` or `numba`
  imports in `dagua/layout/ops/umap.py` or `dagua/layout/ops/pipelines/umap_layout.py`.
- Gate 5: PASS for code scope. Only UMAP op code changed; no shared op, other engine,
  eval scoring, or reference runner code was modified.

Documented blocker:

The portable first divergence (`epochs_per_sample` float32 schedule rounding) is fixed, and
the tiny trace proves the epoch-1 schedule/tau-rand/gradient stream can match draw-for-draw.
The required no-numba runtime gate prevents dagua from using the same compiled numba kernels
as `umap-learn`; replacing those wrappers with scalar Python/NumPy changes larger-graph
floating arithmetic enough that the representative divergent probe does not materially
improve. Exact non-portable constructs:

- Reference reduced distance kernel is numba-compiled float32 with `fastmath=True`:
  `umap/layouts.py:31-41`.
- Reference single-epoch optimizer is numba-compiled with `fastmath=True`:
  `umap/layouts.py:222-228`, executing the update body at `umap/layouts.py:92-187`.
- Reference tau-rand itself is a numba `i4(i8[:])` function:
  `umap/utils.py:40-63`.

No commit should be made for this attempt because gate 2 failed. Additionally, the
repository-level AGENTS instruction says not to commit or push, which conflicts with the task
deliverable asking for conventional commits.

## Commit

No commit yet. The repository-level AGENTS instruction says not to commit or push; this conflicts
with the task deliverable asking for conventional commits. I will leave changes uncommitted unless
the higher-level instruction is changed.

## Attempt 2

Corrected gate interpretation:

- Numba is allowed for dagua's own UMAP kernels, but dagua runtime modules must not import
  or invoke the reference `umap-learn` package.
- Optional fallback remains available with `DAGUA_DISABLE_NUMBA=1`.

Implementation in `dagua/layout/ops/umap.py`:

- Restored optional numba import/wrapper pattern at `dagua/layout/ops/umap.py:18-30`.
- Restored membership-strength JIT wrapper at `dagua/layout/ops/umap.py:449-459`.
- Kept attempt-1's float32 `epochs_per_sample` schedule fix at
  `dagua/layout/ops/umap.py:966-983`.
- Added local BSD-3-cited copies of:
  - `umap.utils.tau_rand_int` as `_umap_tau_rand_int_kernel` at
    `dagua/layout/ops/umap.py:1115-1145`, compiled with `i4(i8[:])`.
  - `umap.layouts.rdist` as `_umap_rdist` at `dagua/layout/ops/umap.py:1149-1187`,
    compiled with signature `f4(f4[::1],f4[::1])`, `fastmath=True`, `cache=True`,
    and matching locals.
  - `umap.layouts.clip` as `_umap_clip` at `dagua/layout/ops/umap.py:1190-1211`.
  - the serial single-epoch Euclidean optimizer at
    `dagua/layout/ops/umap.py:1214-1395`, compiled with `fastmath=True,
    parallel=False`.
- During diagnosis, aligned portions of disconnected spectral init with `umap.spectral`:
  connected ARPACK branch at `dagua/layout/ops/umap.py:575-625` and component
  meta-layout at `dagua/layout/ops/umap.py:628-687`.

Kernel parity evidence:

- Direct tiny-trace call, same inputs as attempt 1, comparing dagua `_UMAP_SINGLE_EPOCH`
  to installed `umap.layouts._get_optimize_layout_euclidean_single_epoch_fn(False)`:
  - embedding after epoch 1: exact match, max abs diff `0.0`
  - per-source tau RNG state: exact match
  - `epoch_of_next_sample`: exact match
  - `epoch_of_next_negative_sample`: exact match

Probe evidence:

`random_dag_50::classic_umap_nn5`, public `ClassicUMAP` vs `UMAPGraph`, seeds `0..4`,
orthogonal Procrustes RMSD:

| Seed | Attempt 1 scalar RMSD | Attempt 2 RMSD |
| --- | ---: | ---: |
| 0 | 0.024253350 | 0.127729159 |
| 1 | 0.007833333 | 0.125753437 |
| 2 | 0.015223136 | 0.131219499 |
| 3 | 0.015522676 | 0.150322713 |
| 4 | 0.010014311 | 0.128612262 |
| Mean | 0.014569361 | 0.132727414 |

Result: **Gate 1 failed**. The corrected numba kernel did not collapse the public
adapter RMSD; it regressed this representative target versus attempt 1's scalar result.
I did not run the remaining divergent-combo and previously-identical-combo tables because
this required representative row already fails decisively.

Bisection / blocker evidence:

- Replacing dagua's local `_UMAP_SINGLE_EPOCH` with the installed reference
  `_get_optimize_layout_euclidean_single_epoch_fn(False)` inside a diagnostic process
  still produced high RMSD on `random_dag_50::classic_umap_nn5`: per-seed RMSDs
  `0.124201919`, `0.171023185`, `0.109754488`, `0.194910546`, `0.114682141`,
  mean `0.142914456`. This proves the remaining public-adapter divergence is not in
  the epoch body.
- Pre-optimizer comparison for `random_dag_50`, seed `0`, `n_neighbors=5`:
  - all-pairs graph distances: max diff `0.0`
  - fuzzy graph COO: `628` entries on both sides, max weight diff `0.0`
  - selected positive `head`: exact match
  - selected positive `tail`: exact match
  - `epochs_per_sample`: max diff `0.0`
  - curve `(a, b)`: `(1.5769434602912036, 0.8950608779947933)` on both sides
  - optimizer base RNG state: `[1172884934, -1376923273, 1970277268]` on both sides
  - spectral init before epoch 0: max abs diff remained `5.6579790115356445`
- The fuzzy graph has 2 connected components with sizes `[52, 45]`. On the largest
  connected component, direct comparison of installed `umap.spectral._spectral_layout`
  and dagua's structurally equivalent connected ARPACK branch consumed the same RNG stream
  (`[-808330553, 1636402120, 843288851]` next draw) but returned a different second
  eigenvector basis; observed max abs diff `0.3758753105490008`. This is a degenerate
  or near-degenerate eigenspace basis selection issue before SGD, not a tau-rand or
  single-epoch optimizer issue.

Fallback evidence:

- `DAGUA_DISABLE_NUMBA=1 PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest
  tests/test_pipeline_umap_layout.py -x --tb=short -q`: PASS, `14 passed, 3 warnings
  in 0.85s`.
- A fallback public-adapter fidelity probe that also invoked `UMAPGraph` segfaulted before
  producing rows. The fallback correctness gate above avoids the reference package and
  passed.

Gate evidence:

- Gate 1: FAIL. Representative target `random_dag_50::classic_umap_nn5` regressed to
  mean RMSD `0.132727414`; no order-of-magnitude improvement.
- Gate 2: NOT RUN because Gate 1 already failed.
- Gate 3: PASS for correctness. `DAGUA_DISABLE_NUMBA=1` pipeline UMAP tests passed.
- Gate 4: PARTIAL PASS.
  - `ruff check . --fix`: PASS, `All checks passed!`
  - `pytest tests/test_pipeline_umap_layout.py -x --tb=short -q`: PASS,
    `14 passed, 3 warnings in 1.62s`
  - `pytest tests/test_ops_optimize.py -k umap -x --tb=short -q`: PASS,
    `1 passed, 19 deselected, 3 warnings in 0.01s`
- Gate 5: PASS. AST check found no `umap` imports in
  `dagua/layout/ops/umap.py` or `dagua/layout/ops/pipelines/umap_layout.py`.

Commit:

- No commit. The fidelity gate failed, and the task says to leave changes uncommitted
  when gates do not pass.

## Attempt 2b: spectral revert

Scope:

- Surgically reverted the Attempt 2 diagnostic spectral-init alignment changes in
  `dagua/layout/ops/umap.py`.
- Restored `_connected_spectral_embedding` and `_component_meta_embedding` to the exact
  `git show HEAD:dagua/layout/ops/umap.py` code paths.
- Kept the float32 `epochs_per_sample` schedule fix and the local reference-parity numba kernels
  byte-for-byte apart from the `ruff-format` blank-line normalization accepted by pre-commit.

Verification evidence:

- Tiny trace, four-node path `0-1-2-3`, seed `7`, direct dagua `_UMAP_SINGLE_EPOCH` vs installed
  `umap.layouts._get_optimize_layout_euclidean_single_epoch_fn(False)`:
  - epoch-1 embedding exact match, max abs diff `0.0`
  - per-source tau RNG state exact match
  - `epoch_of_next_sample` exact match
  - `epoch_of_next_negative_sample` exact match
- Spectral restoration proof, current worktree vs injected `git show HEAD:dagua/layout/ops/umap.py`
  UMAP module:
  - `citation_dag_300`, seeds `100`, `101`, `102`, all six classic UMAP variants:
    `torch.equal=True`, max abs diff `0.0`
  - `clustered_longlabel_handoffs`, seed `100`, all six classic UMAP variants:
    `torch.equal=True`, max abs diff `0.0`
- `pytest tests/test_pipeline_umap_layout.py -x -q`: PASS, `14 passed, 3 warnings in 1.62s`
- `DAGUA_DISABLE_NUMBA=1 pytest tests/test_pipeline_umap_layout.py -x -q`: PASS,
  `14 passed, 3 warnings in 0.97s`
- `pytest tests/test_ops_optimize.py -k umap -x -q`: PASS,
  `1 passed, 19 deselected, 3 warnings in 0.01s`
- `ruff check . --fix`: PASS, `All checks passed!`
- AST runtime import check: PASS, no external `umap` imports in dagua runtime modules
  excluding eval/adapters and archive code.
- Additional project checks:
  - `mypy --follow-imports=silent dagua/cli.py`: PASS,
    `Success: no issues found in 1 source file`
  - `pytest tests/test_layout/test_umap_fidelity.py -x --tb=short -q`: PASS,
    `13 passed, 3 warnings in 1.40s`
  - `pytest tests/test_graph.py -x --tb=short -q`: PASS,
    `37 passed, 3 warnings in 0.54s`
  - `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
    stopped at known pre-existing failure
    `tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest`
    after `63 passed, 88 deselected, 34 warnings in 19.67s`.

Commit shas:

- Code commit: `795ccbd` (`fix(layout): port umap schedule and kernels`)
- Notes commit: this follow-up notes-only commit; see `git log -2 --oneline` after commit.
