# r75 DLA Quality Probe Results

Date: 2026-07-02
Repo: `/home/jtaylor/projects/dagua`
Git: `288d0ca79dbb0dc05fb8ae5a2dd4bf4fac329855`
Installed reference: `python-igraph 1.0.0`

## Commands

- Inspected current branch and code:
  - `git status --short && git log --oneline -5`
  - `sed -n '1,620p' dagua/layout/ops/pipelines/classical_mds.py`
  - `sed -n '170,270p' dagua/eval/distributional_fidelity.py`
  - `sed -n '440,535p' dagua/eval/equivalence_metrics.py`
  - `sed -n '1035,1065p' dagua/eval/equivalence_metrics.py`
  - `sed -n '1120,1410p' scripts/definitive_fidelity_analysis.py`
- Verified reference version:
  - `python - <<'PY' ... import igraph as ig; print(ig.__version__) ... PY`
- Ran focused probes with direct graph factory equivalent to registry entry:
  - `make_parallel_cycles(4, 5, seed=42)`
  - Dagua current path: `get_competitor('classic_classical_mds').layout_with_variant(graph, timeout=120, seed=100, variant_params={'igraph_fidelity': True})`
  - Installed igraph path: `ig.Graph(n=20, edges=edges, directed=False); ig.set_random_number_generator(random.Random(100)); graph.layout('mds')`
- Ran a 60-seed sweep over seeds 42..101 for current Dagua internal branch to compare finite-filled versus `inf` disconnected distances.

Note: a first attempt using full `get_test_graphs()` stalled during registry construction before layout. I used the exact factory named in `dagua/eval/graphs.py` for `parallel_cycles_4x5`, which is `make_parallel_cycles(4, 5, seed=42)`.

## Numbers

Graph:

- N=20, E=20
- components: `[[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19]]`
- each component is an independent 5-cycle.

Seed 100, current Dagua adapter path:

- adapter error: `None`
- per-component normalized stress: `[0.0110227625, 0.0110227625, 0.0110227625, 0.0110227625]`
- distributional finite-pair stress using `df.prepare_graph_distances` + `df.sample_pairs`: `0.00806504495` over 40 finite pairs
- `normalized_stress(..., all_pairs_distances=inf_dists, fit_scale=True)`: `0.0110227625`
- `normalized_stress(..., no supplied dists, fit_scale=True)`: `0.1746800963`
- first component coordinates:

```text
[[-121.042183   26.714808]
 [  -0.484573  115.324059]
 [ 121.042183   28.048723]
 [  75.592239 -114.499657]
 [ -74.024132 -115.324059]]
```

Seed 100, installed igraph 1.0.0:

- per-component normalized stress: `[0.0110227625, 0.0110227625, 0.0110227625, 0.0110227625]`
- distributional finite-pair stress: `0.00806504495` over 40 finite pairs
- `normalized_stress(..., all_pairs_distances=inf_dists, fit_scale=True)`: `0.0110227625`
- `normalized_stress(..., no supplied dists, fit_scale=True)`: `0.1231123737`
- first component coordinates:

```text
[[ 0.558705 -0.503844]
 [ 3.525979 -0.503844]
 [ 4.442917  2.318202]
 [ 2.042342  4.062321]
 [-0.358233  2.318202]]
```

Current Dagua 60-seed sweep, seeds 42..101:

- `normalized_stress` with no supplied distances: mean `0.1216437185`, min `0.0841049440`, max `0.1753767641`
- `normalized_stress` with `df.prepare_graph_distances` supplied: mean `0.0110227625`, min `0.0110227625`, max `0.0110227625`

Current code submatrix check:

- `_shortest_path_distances` fills cross-component distance `d(0,5)=3.0`; all entries finite.
- Component submatrix is correct 5-cycle distance matrix:

```text
[[0. 1. 2. 2. 1.]
 [1. 0. 1. 2. 2.]
 [2. 1. 0. 1. 2.]
 [2. 2. 1. 0. 1.]
 [1. 2. 2. 1. 0.]]
```

- sub-MDS stress for each component: `0.0110227625`

Toy mapping check, non-contiguous components triangle + path + singleton:

- components: `[[0,2,4], [1,3,5,7], [6]]`
- triangle component stress: `4.93e-32`
- path component stress: `4.88e-32`
- singleton stress: `0.0`
- no evidence of component-local row scrambling when mapping merged rows back to original vertex ids.

Artifact check:

- `eval_output/fidelity_definitive/r75_mds_rescore.jsonl` row for `parallel_cycles_4x5::classic_classical_mds_igraph_fidelity` has:
  - `git_sha=288d0ca79dbb0dc05fb8ae5a2dd4bf4fac329855`
  - `battery_stress_D_mean=0.5549602431`
  - `battery_stress_R_mean=0.0110227625`
  - `stress_D_mean=0.5833873189`
  - `stress_R_mean=0.00806504495`

That artifact value is not reproducible from the specified current benchmark-path call on this checkout.

## Verdict

Defect confirmed: not an indexing bug, not a wrong submatrix bug, and not bad per-component MDS geometry in the current checkout.

The current Dagua disconnected classical-MDS path lays out each 5-cycle well. Dagua and installed igraph have identical per-component stress and identical finite-pair distributional stress on `parallel_cycles_4x5`.

The high stress appears when disconnected pairs are treated as finite by `dagua.layout.ops.graph_utils.shortest_path_distances`, which fills unreachable distances with `max_distance + 1`. Under that metric, DLA packing geometry enters the objective. Dagua's DLA port places/rescales components much larger and farther apart than igraph:

- Dagua component radius after merge: about `167.185`
- igraph component radius after merge: about `3.313`
- Dagua seed-100 finite-filled full stress: `0.174680`
- igraph seed-100 finite-filled full stress: `0.123112`

However, `scripts/definitive_fidelity_analysis.py` should not use that finite-fill path for the battery: `stress_pairs()` builds `dists` with `df.prepare_graph_distances`, which leaves cross-component distances as `inf`, and `quality_metric_samples()` passes those `dists` into `normalized_stress`. With that supplied matrix, current Dagua seed-100 stress is `0.0110227625`, matching igraph.

Therefore the remaining anomaly in `r75_mds_rescore.jsonl` is most consistent with stale or differently generated Dagua layout payloads / rescore input contamination, not a current algorithmic row-order or submatrix defect. If the artifact really used current layouts, then the rescore path being run is not the checked-in `stress_pairs -> quality_metric_samples(..., dists)` path.

## Minimal Gated Fix Sketch

1. Add a regression gate for `parallel_cycles_4x5::classic_classical_mds_igraph_fidelity` on the actual analysis path:
   - build `dists = df.prepare_graph_distances(edges, n)`
   - assert `np.isinf(dists).any()`
   - run Dagua seed 100 through the competitor adapter
   - assert `normalized_stress(pos, edge_index, all_pairs_distances=dists, fit_scale=True) < 0.02`
   - assert the same stress without supplied distances is higher, documenting the finite-fill trap.

2. Add a rescore input guard:
   - for any row with `disconnected=True`, assert the `dists` object passed to `quality_metric_samples()` contains at least one `inf`.
   - fail fast if a finite-filled graph-utils matrix reaches battery stress.

3. Invalidate/rebuild the MDS top-up Dagua positions before trusting `r75_mds_rescore.jsonl`.
   - The row claims commit `288d0ca`, but the current commit and exact adapter call do not reproduce the bad finite-pair stress.
   - Re-run a one-graph one-variant rescore from freshly generated current layouts before making code changes to classical MDS.

4. Separately, if the project wants finite-filled disconnected stress to be meaningful, then Dagua's DLA scale still mismatches igraph and needs a source-level scale fix. That is a placement-fidelity issue, not the reported finite-pair stress issue.

## Concerns

- The current code path and the saved `r75_mds_rescore.jsonl` disagree. Treat that artifact as suspect until regenerated from fresh layouts.
- Full `get_test_graphs()` construction was too slow for this probe; I used the exact factory registered for the target graph.
- Existing working tree had unrelated dirty files before this probe; I did not modify them.
