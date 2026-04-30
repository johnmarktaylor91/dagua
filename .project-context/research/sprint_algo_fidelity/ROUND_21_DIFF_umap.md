# Round 21 adversarial diff: `classic_umap` vs `umap_graph`

Date: 2026-04-30
Scope: diagnosis only; no source changes.
Pairing: dagua `classic_umap` (`dagua.layout.ops.pipelines.umap_layout`) vs reference `umap_graph` (`umap.UMAP(metric="precomputed")` via dagua competitor adapter).

Current mega-run status: the fidelity report marks all six UMAP variants as not converged: `umap_default`, `umap_mindist001`, `umap_mindist05`, `umap_nn5`, and `umap_spread2` are `partial_match`, while `umap_nn30` is `divergent`, with RMSD `0.162-0.255` (`eval_output/fidelity_report/report.md:102-107`). This family is therefore not merely a sub-percent stochastic residual; there are hard semantic divergences.

## 1. Files read

### Dagua side

- `dagua/layout/ops/umap.py`
  - Constants, configs, adjacency, APSP, kNN, smooth-kNN, fuzzy graph, spectral init, curve fit, positive-edge selection, SGD, final normalization (`dagua/layout/ops/umap.py:24-51`, `dagua/layout/ops/umap.py:107-165`, `dagua/layout/ops/umap.py:211-236`, `dagua/layout/ops/umap.py:239-254`, `dagua/layout/ops/umap.py:257-312`, `dagua/layout/ops/umap.py:315-364`, `dagua/layout/ops/umap.py:367-386`, `dagua/layout/ops/umap.py:389-446`, `dagua/layout/ops/umap.py:449-468`, `dagua/layout/ops/umap.py:471-561`, `dagua/layout/ops/umap.py:564-580`, `dagua/layout/ops/umap.py:654-1164`).
- `dagua/layout/ops/pipelines/umap_layout.py`
  - Active pipeline wiring and callable entry point (`dagua/layout/ops/pipelines/umap_layout.py:32-94`, `dagua/layout/ops/pipelines/umap_layout.py:97-171`).
- `dagua/layout/ops/pipelines/__init__.py`
  - Registry maps short name `umap` to the active `umap_layout` pipeline (`dagua/layout/ops/pipelines/__init__.py:85-88`).
- `dagua/eval/variants.py`
  - UMAP fidelity variants and heavy/stochastic classification (`dagua/eval/variants.py:1474-1539`, `dagua/eval/variants.py:1835-1864`, `dagua/eval/variants.py:1870-1882`).
- `dagua/eval/competitors/classic_competitor.py`
  - `classic_umap` adapter spec and class (`dagua/eval/competitors/classic_competitor.py:254-258`, `dagua/eval/competitors/classic_competitor.py:1702-1718`).
- `dagua/eval/competitors/umap_competitor.py`
  - Reference adapter, graph-distance construction, tiny-graph fallback, reducer kwargs (`dagua/eval/competitors/umap_competitor.py:23-91`, `dagua/eval/competitors/umap_competitor.py:94-220`).
- `dagua/layout/ops/optimize.py`
  - Generic dormant/shared UMAP pair-SGD op, not wired into `classic_umap` but relevant for naming/semantics (`dagua/layout/ops/optimize.py:358-365`, `dagua/layout/ops/optimize.py:368-398`, `dagua/layout/ops/optimize.py:1200-1365`).
- `dagua/layout/ops/loss_classic.py`
  - Generic dormant/shared UMAP cross-entropy loss, not wired into `classic_umap` (`dagua/layout/ops/loss_classic.py:481-507`, `dagua/layout/ops/loss_classic.py:1762-1816`, `dagua/layout/ops/loss_classic.py:2441-2526`).
- `eval_output/fidelity_report/report.md`
  - Current UMAP verdict and RMSD rows (`eval_output/fidelity_report/report.md:102-107`).
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`
  - Sprint context and stochastic-floor methodology (`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:17-45`, `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:111-130`).

### Reference side

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py`
  - `smooth_knn_dist`, `nearest_neighbors`, `compute_membership_strengths`, `fuzzy_simplicial_set`, `reset_local_connectivity`, `make_epochs_per_sample`, coordinate scaling/noise, `simplicial_set_embedding`, `find_ab_params`, `UMAP.__init__`, `UMAP.fit` graph-building and embedding construction (`umap_.py:152-252`, `umap_.py:256-348`, `umap_.py:351-439`, `umap_.py:442-617`, `umap_.py:749-777`, `umap_.py:906-935`, `umap_.py:938-1275`, `umap_.py:1393-1408`, `umap_.py:1670-1725`, `umap_.py:2339-2835`).
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/layouts.py`
  - Gradient clamp and Euclidean SGD epoch implementation (`layouts.py:9-25`, `layouts.py:63-188`, `layouts.py:323-435`).
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/spectral.py`
  - Component layout and spectral layout/eigensolver initialization (`spectral.py:18-142`, `spectral.py:151-260`, `spectral.py:263-312`, `spectral.py:399-554`).
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/utils.py`
  - `tau_rand_int` RNG used by reference negative sampling (`utils.py:40-63`).

## 2. Overall pipeline structure

### Dagua `classic_umap`

The active dagua pipeline is explicit op composition:

1. Validate inputs (`dagua/layout/ops/pipelines/umap_layout.py:70-73`).
2. Store hyperparameters (`dagua/layout/ops/pipelines/umap_layout.py:73-81`).
3. Build undirected adjacency from `edge_index`, with edge weights converted to costs `1 / weight` (`dagua/layout/ops/umap.py:107-144`, `dagua/layout/ops/pipelines/umap_layout.py:82`).
4. Compute dense all-pairs shortest-path distances via Python BFS/Dijkstra, replacing infinities by `max_finite * 2` (`dagua/layout/ops/umap.py:211-236`, `dagua/layout/ops/pipelines/umap_layout.py:83`).
5. Extract dense kNN with `torch.topk` after masking the diagonal to `inf` (`dagua/layout/ops/umap.py:239-254`, `dagua/layout/ops/pipelines/umap_layout.py:84`).
6. Solve smooth-kNN sigmas/rhos (`dagua/layout/ops/umap.py:257-312`, `dagua/layout/ops/pipelines/umap_layout.py:85`).
7. Build a symmetrized fuzzy set, optionally rescaling fuzzy memberships by original edge weights after symmetrization (`dagua/layout/ops/umap.py:315-364`, `dagua/layout/ops/umap.py:923-962`, `dagua/layout/ops/pipelines/umap_layout.py:86`).
8. Spectral initialize directly from the fuzzy graph (`dagua/layout/ops/umap.py:389-446`, `dagua/layout/ops/pipelines/umap_layout.py:87`).
9. Fit curve parameters (`dagua/layout/ops/umap.py:367-386`, `dagua/layout/ops/pipelines/umap_layout.py:88`).
10. Prune weak positive edges and compute epochs-per-sample (`dagua/layout/ops/umap.py:449-468`, `dagua/layout/ops/pipelines/umap_layout.py:89`).
11. Run a Python/Torch negative-sampling SGD loop (`dagua/layout/ops/umap.py:498-561`, `dagua/layout/ops/pipelines/umap_layout.py:90`).
12. Recenter and rescale to dagua layout extent (`dagua/layout/ops/umap.py:564-580`, `dagua/layout/ops/umap.py:1139-1164`, `dagua/layout/ops/pipelines/umap_layout.py:91`).

The `classic_umap` competitor invokes `layout_umap_layout_pipeline` with graph tensors and the orchestrator seed (`dagua/eval/competitors/classic_competitor.py:1702-1718`). The short registry also points `umap` to that pipeline (`dagua/layout/ops/pipelines/__init__.py:85-88`).

### Reference `umap_graph`

The reference adapter is not a raw graph UMAP implementation; it creates an all-pairs graph-distance matrix and feeds it to `umap.UMAP(metric="precomputed")` (`dagua/eval/competitors/umap_competitor.py:54-91`, `dagua/eval/competitors/umap_competitor.py:185-198`).

Reference adapter flow:

1. Build scipy CSR adjacency with data equal to graph edge weights if present, else `1.0` (`dagua/eval/competitors/umap_competitor.py:76-84`).
2. Run `scipy.sparse.csgraph.shortest_path(adjacency, directed=False)` (`dagua/eval/competitors/umap_competitor.py:85`).
3. Replace `inf` by `max_finite * 2` and cast to `float32` (`dagua/eval/competitors/umap_competitor.py:87-91`).
4. Instantiate `umap.UMAP(n_components=2, metric="precomputed", random_state=seed or 42, n_neighbors=min(15, N-1), init="random" if N < 10 else "spectral")`, then merge variant params and cap `n_neighbors` again (`dagua/eval/competitors/umap_competitor.py:185-196`).
5. Call `fit_transform(distances)` and cast output to torch float32 (`dagua/eval/competitors/umap_competitor.py:196-199`).

Inside `umap-learn`, dense precomputed input with `N < 4096` goes through the small-data path: `check_array(... dtype=np.float32)` (`umap_.py:2372-2379`), `random_state = check_random_state(self.random_state)` (`umap_.py:2477`), `pairwise_distances(X[index], metric=_m, **kwds)` (`umap_.py:2550-2556`), `fuzzy_simplicial_set(... metric="precomputed")` (`umap_.py:2588-2607`), and `_fit_embed_data(...)` (`umap_.py:2813-2823`). For precomputed metric, `nearest_neighbors` includes the diagonal self distance because it takes the first `n_neighbors` argsorted entries (`umap_.py:312-319`).

### Most important structural divergence

Dagua's graph input to smooth-kNN excludes self from kNN (`dagua/layout/ops/umap.py:249-254`), while reference UMAP's dense precomputed kNN includes self as the zero-distance first neighbor and later assigns it membership value zero (`umap_.py:312-319`, `umap_.py:420-431`). This shifts every non-self neighborhood by one. Dagua's `n_neighbors=15` means 15 non-self neighbors; reference `n_neighbors=15` means self plus 14 non-self neighbors. This is likely the largest single divergence, especially for `nn5` and `nn30`.

## 3. Energy / loss / objective

### Fuzzy high-dimensional graph

Dagua uses shortest-path distances on an undirected graph, then dense kNN. For unweighted edges it deduplicates neighbors in sets and uses unit adjacency (`dagua/layout/ops/umap.py:113-125`). For weighted edges, it converts edge weights to shortest-path costs with `cost = 1.0 / max(weight, epsilon)` and stores the minimum parallel-edge cost (`dagua/layout/ops/umap.py:127-144`).

The reference adapter also uses shortest-path distances, but it writes edge weights directly as adjacency distances, not inverse strengths (`dagua/eval/competitors/umap_competitor.py:79-85`). This is a semantic mismatch for weighted graphs:

- Dagua: high weight means shorter distance (`dagua/layout/ops/umap.py:138-142`).
- Reference: high weight means longer distance (`dagua/eval/competitors/umap_competitor.py:79-85`).

For unweighted graphs, both ultimately use unit edge lengths, but the kNN self-inclusion mismatch remains.

### Smooth-kNN sigma/rho

Reference `smooth_knn_dist`:

- Target is `np.log2(k) * bandwidth` (`umap_.py:191`).
- `rho` uses `local_connectivity=1.0`; with default local connectivity it selects the first nonzero distance (`umap_.py:202-217`).
- Membership sum loops from `j = 1`, skipping the first kNN entry, which is normally self for dense precomputed input (`umap_.py:219-228`).
- Minimum sigma scale depends on whether `rho > 0`: per-row mean when rho positive, global mean otherwise (`umap_.py:242-251`).

Dagua:

- Target is `log2(max(n_neighbors, 2))` (`dagua/layout/ops/umap.py:267-270`).
- `rho` is the minimum positive finite distance (`dagua/layout/ops/umap.py:279-281`).
- Membership sum also uses `finite[1:]` (`dagua/layout/ops/umap.py:287-292`), but because dagua already removed self from kNN, this skips the nearest real neighbor rather than self.
- Minimum sigma is always `mean(finite) * 1e-3` (`dagua/layout/ops/umap.py:282-310`), whereas reference switches between per-row and global mean based on `rho` (`umap_.py:242-251`).

The formula is nominally the same exponential kernel `exp(-(d - rho) / sigma)`, but dagua applies it to a different neighbor set and uses a subtly different sigma floor.

### Fuzzy simplicial set union

Reference builds directed membership strengths with self memberships set to zero (`umap_.py:420-431`), creates a COO matrix (`umap_.py:584-590`), eliminates zeros (`umap_.py:591`), and applies fuzzy union `A + A.T - A*A.T` when `set_op_mix_ratio=1.0` (`umap_.py:593-603`).

Dagua builds a Python dict of directed weights (`dagua/layout/ops/umap.py:322-337`) and then computes undirected weights as `forward + backward - forward * backward` (`dagua/layout/ops/umap.py:339-350`). This matches the reference union formula for default `set_op_mix_ratio=1.0`, but dagua does not expose or implement `set_op_mix_ratio` or intersection mixing. The active variants use only default `set_op_mix_ratio`, so this is not causing current variant divergence.

Dagua then optionally rescales fuzzy weights by original edge weights after symmetrization (`dagua/layout/ops/umap.py:943-957`). Reference `umap_graph` does not post-multiply fuzzy membership by original graph weights; its weights only influence shortest-path distances (`dagua/eval/competitors/umap_competitor.py:79-85`, `umap_.py:2588-2607`). This is another weighted-graph mismatch.

### Low-dimensional objective

Both implementations optimize UMAP's sampled cross-entropy implied by low-dimensional membership

`q_ij = 1 / (1 + a * ||y_i - y_j||^(2b))`.

Reference positive update coefficient:

- `grad_coeff = -2.0 * a * b * dist_squared^(b - 1.0) / (a * dist_squared^b + 1.0)` (`layouts.py:136-143`).

Dagua positive update coefficient:

- `grad_coeff = -2.0 * a * b * distance_sq^(b - 1.0) / ((a * distance_sq^b) + 1.0)` (`dagua/layout/ops/umap.py:471-481`).

Reference negative update coefficient:

- `2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_squared^b + 1))`, with self negative samples skipped when `j == k` and `dist_squared == 0` (`layouts.py:160-181`).

Dagua negative update coefficient:

- `2.0 * gamma * b / ((0.001 + distance_sq) * ((a * distance_sq^b) + 1.0))`, but returns zero for any `distance_sq <= 0.0` (`dagua/layout/ops/umap.py:484-495`). This has the same effect for exact self samples, but also zeroes coincident non-self points; reference sets `grad_coeff=0.0` for non-self zero distance too (`layouts.py:172-181`), so it is aligned.

The dormant generic loss op computes the equivalent cross-entropy explicitly: positive `-(weight * log(q)).sum()` and negative `-gamma * log(1 - q_negative).sum()` (`dagua/layout/ops/loss_classic.py:2501-2526`). It is not used by `classic_umap` because the active pipeline wires `OptimizeUMAPEmbedding`, not an autograd loss (`dagua/layout/ops/pipelines/umap_layout.py:87-91`).

### Curve parameter fitting

Reference fits `a, b` with `curve_fit(curve, xv, yv)` and no explicit initial `p0` (`umap_.py:1393-1408`). Dagua fits the same curve grid, but supplies `p0=(1.93, 0.79)`, `maxfev=10000`, and falls back to `(1.93, 0.79)` on `RuntimeError` or `ValueError` (`dagua/layout/ops/umap.py:367-386`). The formulas match, but the optimizer start point and failure behavior differ. This likely creates small but measurable differences for `min_dist`/`spread` variants.

## 4. Force / gradient computation

The attraction and repulsion formulas match the reference's Euclidean optimizer at the scalar coefficient level:

- Reference attraction: `layouts.py:136-152`.
- Dagua attraction: `dagua/layout/ops/umap.py:471-481`.
- Reference repulsion: `layouts.py:160-182`.
- Dagua repulsion: `dagua/layout/ops/umap.py:484-495`.

The update loop differs in four important ways.

First, reference initializes epoch counters to the first sample interval, not zero:

- `epoch_of_next_sample = epochs_per_sample.copy()` and `epoch_of_next_negative_sample = epochs_per_negative_sample.copy()` (`layouts.py:323-325`).
- Dagua initializes both to zeros (`dagua/layout/ops/umap.py:518-520`).

This makes dagua sample every kept edge at epoch `0`, whereas reference waits until `n >= epochs_per_sample[i]` (`layouts.py:92-94`). This is a large schedule mismatch.

Second, reference computes `n_neg_samples = int((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])` (`layouts.py:156-158`) and then runs exactly that many samples (`layouts.py:160-186`). Dagua runs negative samples while `next_negative_epoch <= epoch`, increments by `epochs_per_negative_sample`, and caps at `negative_sample_rate` per positive edge update (`dagua/layout/ops/umap.py:541-559`). Reference has no per-update cap; the number is schedule-derived.

Third, reference RNG state is per source vertex:

- `rng_state_per_sample = full(...) + head_embedding[:,0].view(int64)` (`layouts.py:367-369`).
- Negative sample `k = tau_rand_int(rng_state_per_sample[j]) % n_vertices` where `j = head[i]` (`layouts.py:160-162`).

Dagua uses one global `torch.Generator` and draws `torch.randint` in loop order (`dagua/layout/ops/umap.py:515-516`, `dagua/layout/ops/umap.py:541-544`). This cannot produce the same sample stream.

Fourth, reference starts `alpha` as the initial alpha passed to `optimize_layout_euclidean`, then updates it after each epoch (`layouts.py:401-431`). Dagua computes the same linear formula at the top of each epoch (`dagua/layout/ops/umap.py:523-524`). These are effectively aligned for epoch `n`, assuming the same epoch schedule.

## 5. Initialization

### Tiny graphs

Reference adapter special-cases `num_nodes <= 3` and returns `torch.randn((N, 2))` using torch seed `42` or supplied seed, without calling `umap-learn` (`dagua/eval/competitors/umap_competitor.py:170-183`). Dagua has no matching adapter-level tiny fallback. Its internal spectral init returns:

- `[]` for `N=0` (`dagua/layout/ops/umap.py:397-398`).
- `[[0,0]]` for `N=1` (`dagua/layout/ops/umap.py:399-400`).
- `[[-10,0], [10,0]]` for `N=2` (`dagua/layout/ops/umap.py:401-402`).
- spectral/eigen init for `N=3` when fuzzy edges exist (`dagua/layout/ops/umap.py:404-446`).

This guarantees divergence for graphs with 2-3 nodes in any test set.

### Random vs spectral threshold

Reference adapter sets `init="random"` for `num_nodes < 10`, else `init="spectral"` (`dagua/eval/competitors/umap_competitor.py:186-192`). Dagua always runs `SpectralInitialization()` (`dagua/layout/ops/pipelines/umap_layout.py:87`), except its internal no-edge fallback uses torch uniform in `[-10, 10]` (`dagua/layout/ops/umap.py:404-407`).

For `4 <= N <= 9`, reference uses NumPy random uniform `[-10, 10]` inside UMAP (`umap_.py:1095-1098`), while dagua uses spectral initialization. This is a major mismatch for small benchmark graphs.

### Spectral details

Reference spectral path:

- Computes connected components and uses `multi_component_layout` when the fuzzy graph has multiple components (`spectral.py:463-476`).
- For connected graphs, builds normalized Laplacian `I - D * graph * D` (`spectral.py:478-485`).
- Uses `k = dim + 1`, chooses `eigsh` unless graph is huge, with `tol=1e-4`, `v0=np.ones`, and `maxiter=5*N` (`spectral.py:489-527`).
- Returns eigenvectors sorted by eigenvalues excluding the first vector (`spectral.py:545-546`).
- Then `noisy_scale_coords(... max_coord=10, noise=0.0001)` scales by global max absolute coordinate and adds NumPy normal noise (`umap_.py:1117-1120`, `umap_.py:930-935`).
- Then the embedding is rescaled per coordinate to `[0, 10]` before optimization (`umap_.py:1188-1192`).

Dagua spectral path:

- Does not check connected components (`dagua/layout/ops/umap.py:404-446`).
- Builds the same normalized Laplacian for one graph (`dagua/layout/ops/umap.py:409-419`).
- Uses dense `np.linalg.eigh` for `N < 512`; otherwise `eigsh(k=3, which="SM")` without reference's `ncv`, `v0`, tolerance, or `maxiter` (`dagua/layout/ops/umap.py:421-425`).
- Sorts eigenvectors and takes columns `1:3` (`dagua/layout/ops/umap.py:427-434`).
- Scales with global min/max to `[-10, 10]` (`dagua/layout/ops/umap.py:436-441`), not by max absolute coordinate.
- Adds torch normal noise with seed (`dagua/layout/ops/umap.py:443-446`).
- Does not apply reference's second per-axis `[0, 10]` normalization before SGD (`umap_.py:1188-1192`).

These initialization differences are high-impact because UMAP is nonconvex.

## 6. Iteration / convergence

Neither side has a convergence test; both use fixed epochs.

Epoch defaults align at the coarse heuristic:

- Dagua stores 500 epochs for `N <= 10000`, else 200 (`dagua/layout/ops/umap.py:753-759`).
- Reference `simplicial_set_embedding` uses 500 for `graph.shape[0] <= 10000`, else 200 (`umap_.py:1072-1083`).

Weak-edge pruning aligns in formula but not counter initialization:

- Reference zeroes `graph.data < graph.data.max() / n_epochs_max` when `n_epochs_max > 10` (`umap_.py:1088-1093`) and then `make_epochs_per_sample` computes `n_epochs * weight / max_weight` and returns reciprocal interval (`umap_.py:906-925`, `umap_.py:1146`).
- Dagua keeps `weight >= max_weight / n_epochs` and computes `max_weight / kept_weight` (`dagua/layout/ops/umap.py:461-468`). This equals reference `epochs_per_sample` for kept edges, but reference uses strict `<` removal while dagua uses `>=` keep; boundary behavior is aligned for equality because reference keeps equality.

The large schedule divergence is that reference begins next-sample counters at `epochs_per_sample` (`layouts.py:323-325`), while dagua begins at zero (`dagua/layout/ops/umap.py:518-520`). This changes both the number and timing of positive and negative updates.

## 7. Hyperparameter alignment table

| Parameter | Dagua default / behavior | Reference adapter + UMAP default / behavior | Match? | Notes |
|---|---:|---:|:---:|---|
| `n_components` | Fixed output `[N,2]` (`dagua/layout/ops/pipelines/umap_layout.py:97-110`) | `n_components=2` (`dagua/eval/competitors/umap_competitor.py:186-188`) | Y | Same output dimension. |
| `n_neighbors` default | `15` (`dagua/layout/ops/pipelines/umap_layout.py:32-40`) | `min(15, N-1)`, then variant override, then cap to `N-1` (`dagua/eval/competitors/umap_competitor.py:186-196`) | Partial | Dagua validates positive but does not cap before kNN; `_knn_from_distances` effectively caps to `N-1` (`dagua/layout/ops/umap.py:249-254`). Main mismatch is self exclusion. |
| kNN self inclusion | Excludes diagonal by masking to `inf` (`dagua/layout/ops/umap.py:249-254`) | Includes self in dense precomputed `fast_knn_indices` result (`umap_.py:312-319`), then self membership becomes 0 (`umap_.py:424-431`) | N | Likely largest graph-construction mismatch. |
| `min_dist` | `0.1` (`dagua/layout/ops/umap.py:95-99`) | `0.1` (`umap_.py:1673-1677`) | Y | Variant values aligned in `variants.py:1474-1539`. |
| `spread` | `1.0` (`dagua/layout/ops/umap.py:95-99`) | `1.0` (`umap_.py:1673-1677`) | Y | Variant value `2.0` aligned (`dagua/eval/variants.py:1529-1539`). |
| `learning_rate` | `1.0` (`dagua/layout/ops/umap.py:98-100`) | `1.0` (`umap_.py:1673-1675`) | Y | Not varied in current UMAP variants. |
| `negative_sample_rate` | `5` (`dagua/layout/ops/umap.py:31`, `dagua/layout/ops/umap.py:99-100`) | `5` (`umap_.py:1682-1684`) | Y value / N semantics | Value matches; schedule/RNG do not. |
| `repulsion_strength` / `gamma` | `1.0` (`dagua/layout/ops/umap.py:100-101`) | `1.0` (`umap_.py:1680-1683`) | Y | Coefficient formula matches. |
| `n_epochs=None` | 500 if `N <= 10000`, else 200 (`dagua/layout/ops/umap.py:753-759`) | 500 if `N <= 10000`, else 200 (`umap_.py:1072-1083`) | Y | DensMAP adjustment not relevant. |
| `init` | Always spectral op (`dagua/layout/ops/pipelines/umap_layout.py:87`) | Adapter uses random for `N < 10`, spectral otherwise (`dagua/eval/competitors/umap_competitor.py:186-192`) | N | Major small-graph divergence. |
| tiny graph fallback | Internal hardcoded 0/line/spectral behavior (`dagua/layout/ops/umap.py:397-407`) | Adapter returns torch normal for `N <= 3` (`dagua/eval/competitors/umap_competitor.py:170-183`) | N | Guaranteed mismatch on tiny graphs. |
| `set_op_mix_ratio` | Not exposed; effectively union only (`dagua/layout/ops/umap.py:339-350`) | Default `1.0` union (`umap_.py:1678-1681`, `umap_.py:593-601`) | Y for default | No issue for current variants. |
| `local_connectivity` | Not exposed; effectively first positive neighbor via `rho = min(positive)` (`dagua/layout/ops/umap.py:279-281`) | Default `1.0` (`umap_.py:1680-1682`, `umap_.py:202-217`) | Partial | Same intent; self-exclusion changes row contents. |
| `a`, `b` | Fit with p0 and fallback (`dagua/layout/ops/umap.py:372-386`) | Fit without explicit p0/fallback (`umap_.py:1393-1408`) | Partial | Usually close, not bit-identical. |
| `metric` | Graph shortest-path metric built internally (`dagua/layout/ops/umap.py:211-236`) | UMAP metric is `"precomputed"` over adapter distances (`dagua/eval/competitors/umap_competitor.py:186-189`) | Partial | Distance matrix mostly aligned for unweighted graphs, not weighted graphs. |
| weighted edges | Dagua uses inverse weight as shortest-path cost and post-scales fuzzy membership (`dagua/layout/ops/umap.py:138-142`, `dagua/layout/ops/umap.py:943-957`) | Reference uses weight as shortest-path distance only (`dagua/eval/competitors/umap_competitor.py:79-85`) | N | High-impact on weighted tests. |
| disconnected distances | Fill `inf` with `max_finite * 2` (`dagua/layout/ops/umap.py:231-236`) | Adapter same fill (`dagua/eval/competitors/umap_competitor.py:87-91`) | Y | But reference spectral handles components specially; dagua does not. |
| dtype | Torch float32 for distances/weights/embedding, NumPy float64 in spectral eigensolve (`dagua/layout/ops/umap.py:174-195`, `dagua/layout/ops/umap.py:411-446`) | `check_array` coerces input to `float32`; spectral Laplacian float64; embedding float32 (`umap_.py:2372-2379`, `spectral.py:478-485`, `umap_.py:1188-1192`) | Partial | Similar boundaries; different summation/order and eigensolvers. |
| RNG | Torch generator for init noise/negative samples (`dagua/layout/ops/umap.py:443-446`, `dagua/layout/ops/umap.py:515-516`, `dagua/layout/ops/umap.py:543`) | sklearn/NumPy `RandomState`, then numba tau RNG state (`umap_.py:2477`, `umap_.py:1152`, `layouts.py:367-369`, `utils.py:40-63`) | N | Same seed cannot produce same sequence. |
| output finalization | Dagua centers and scales to `layout_extent` (`dagua/layout/ops/umap.py:564-580`, `dagua/layout/ops/umap.py:1150-1164`) | Reference adapter returns raw UMAP coordinates (`dagua/eval/competitors/umap_competitor.py:196-199`) | N/low | Procrustes removes translation/scale, but not anisotropic effects from pre-SGD scaling. |

## 8. Edge cases

### Empty graph

Reference adapter returns an empty torch float32 position tensor before calling UMAP (`dagua/eval/competitors/umap_competitor.py:170-174`). Dagua validates and proceeds; internal spectral init returns empty for `N=0` (`dagua/layout/ops/umap.py:397-398`), optimization sees no edges and returns unchanged (`dagua/layout/ops/umap.py:512-513`), finalization returns zeros-like empty (`dagua/layout/ops/umap.py:566-567`). Empty behavior is effectively aligned.

### One node

Reference UMAP proper returns zeros for one sample (`umap_.py:2454-2460`), but the adapter's `num_nodes <= 3` fallback returns `torch.randn((1,2))` (`dagua/eval/competitors/umap_competitor.py:176-183`). Dagua returns zero for one node (`dagua/layout/ops/umap.py:399-400`) and finalization returns zero (`dagua/layout/ops/umap.py:566-567`). Adapter fallback causes mismatch for `N=1`.

### Two or three nodes

Reference adapter returns seeded torch normal for `N <= 3` (`dagua/eval/competitors/umap_competitor.py:176-183`). Dagua has deterministic line placement for `N=2` (`dagua/layout/ops/umap.py:401-402`) and spectral/random-internal behavior for `N=3` (`dagua/layout/ops/umap.py:404-446`). Mismatch.

### Self-loops

Dagua ignores self-loops when building adjacency and edge-weight lookup (`dagua/layout/ops/umap.py:119-123`, `dagua/layout/ops/umap.py:136-138`, `dagua/layout/ops/umap.py:161-164`). Reference adapter includes self-loop entries in the CSR construction if present (`dagua/eval/competitors/umap_competitor.py:76-84`), but shortest-path distance from a node to itself remains zero unless negative weights are involved; UMAP dense precomputed kNN then includes self as first neighbor (`umap_.py:312-319`). For normal positive self-loops, likely low impact; for weighted self-loops, semantics are not guaranteed aligned.

### Multi-edges

Dagua unweighted adjacency collapses multi-edges with a set (`dagua/layout/ops/umap.py:113-125`). Weighted adjacency keeps the minimum inverse-weight cost per undirected pair (`dagua/layout/ops/umap.py:127-144`), while `_undirected_edge_weight_lookup` sums weights for later fuzzy post-scaling (`dagua/layout/ops/umap.py:147-165`). Reference CSR construction passes duplicate entries to `csr_matrix`, which sums duplicate data entries by scipy semantics when compressed; shortest path then sees summed weights as distances (`dagua/eval/competitors/umap_competitor.py:76-85`). Therefore weighted multi-edge behavior is strongly divergent.

### Disconnected components

Both dagua and the adapter fill disconnected shortest-path distances with `max_finite * 2` (`dagua/layout/ops/umap.py:231-236`, `dagua/eval/competitors/umap_competitor.py:87-91`), so the dense precomputed graph may become fully finite. However, UMAP's spectral init can still detect multiple components in the fuzzy graph and route to `multi_component_layout` (`spectral.py:463-476`). Dagua spectral init never calls `connected_components`; it directly eigensolves the whole normalized Laplacian (`dagua/layout/ops/umap.py:409-446`). If the fuzzy graph remains disconnected after kNN/union, initialization diverges.

### Weighted edges

This is a confirmed semantic mismatch:

- Dagua interprets edge weights as strengths, converting to distance cost `1 / weight` (`dagua/layout/ops/umap.py:138-142`), then additionally multiplies fuzzy memberships by summed original weights (`dagua/layout/ops/umap.py:943-957`).
- Reference adapter interprets edge weights as distances in scipy shortest path (`dagua/eval/competitors/umap_competitor.py:79-85`) and does no fuzzy post-scaling (`umap_.py:2588-2607`).

### Larger-than-node-count `n_neighbors`

Reference UMAP warns and sets `_n_neighbors = N - 1` when `N <= n_neighbors`, except `N=1` early returns (`umap_.py:2454-2469`). The adapter also caps `n_neighbors` to `N - 1` (`dagua/eval/competitors/umap_competitor.py:195`). Dagua's `_knn_from_distances` caps to `min(n_neighbors, max(N-1, 1))` (`dagua/layout/ops/umap.py:249-254`). For `N=0`, it returns empty (`dagua/layout/ops/umap.py:245-247`). Broadly aligned except reference dense precomputed includes self in the requested count.

## 9. Numerical precision

Dagua uses torch float32 for BFS/Dijkstra distances (`dagua/layout/ops/umap.py:174-195`), kNN distances (`dagua/layout/ops/umap.py:253-254`), sigmas/rhos (`dagua/layout/ops/umap.py:267-268`), fuzzy weights (`dagua/layout/ops/umap.py:361-364`), and embedding (`dagua/layout/ops/umap.py:443-446`). Its spectral matrix data are converted to NumPy float64 for scipy sparse/dense eigen calculations (`dagua/layout/ops/umap.py:409-419`).

Reference adapter casts shortest-path distances to `float32` (`dagua/eval/competitors/umap_competitor.py:87-91`), and `UMAP.fit` also checks/coerces input to `float32` (`umap_.py:2372-2379`). Reference smooth-kNN arrays are `float32` (`umap_.py:191-193`) and membership strengths are `float32` (`umap_.py:351-359`, `umap_.py:412-435`). Spectral layout uses a float64 identity and normalized Laplacian (`spectral.py:478-485`), then embedding becomes float32 after scaling (`umap_.py:1188-1192`).

Precision boundaries are therefore similar, but summation and ordering differ:

- Dagua graph distances are computed with Python loops and torch tensor assignment (`dagua/layout/ops/umap.py:168-208`); reference uses scipy shortest_path (`dagua/eval/competitors/umap_competitor.py:69-85`).
- Dagua kNN uses `torch.topk`, which is not guaranteed to match NumPy stable mergesort tie order (`dagua/layout/ops/umap.py:249-254` vs `utils.py:31-36`).
- Reference precomputed nearest neighbors use `argsort(kind="mergesort")`, stable on ties (`utils.py:31-36`, `umap_.py:312-319`). Tied shortest-path distances are common in unweighted graphs, so tie ordering can affect the fuzzy graph.
- Dagua dense spectral path uses exact dense `np.linalg.eigh` for `N < 512` (`dagua/layout/ops/umap.py:421-423`); reference uses sparse `eigsh` for connected graphs under its heuristic (`spectral.py:489-527`). Eigenvector signs and bases in repeated/near-repeated eigenspaces can differ substantially.

## 10. RNG semantics

Dagua's `seed` is a torch seed:

- Spectral no-edge fallback: `torch.Generator.manual_seed(seed)` then `torch.rand` (`dagua/layout/ops/umap.py:404-407`).
- Spectral noise: `torch.randn(... generator=generator)` (`dagua/layout/ops/umap.py:443-446`).
- Negative sampling: global `torch.randint(... generator=generator)` (`dagua/layout/ops/umap.py:515-516`, `dagua/layout/ops/umap.py:541-544`).

Reference's same integer seed goes through sklearn/NumPy and numba:

- Adapter passes `random_state=seed or 42` into `umap.UMAP` (`dagua/eval/competitors/umap_competitor.py:186-190`).
- UMAP converts it with `check_random_state` (`umap_.py:2477`).
- Random init uses `random_state.uniform` (`umap_.py:1095-1098`).
- Spectral noise uses `random_state.normal` (`umap_.py:1117-1120`, `umap_.py:930-935`).
- SGD seeds numba RNG state with `random_state.randint(INT32_MIN, INT32_MAX, 3)` (`umap_.py:1152`).
- Per-sample RNG uses `tau_rand_int`, a custom Tausworthe-like generator (`layouts.py:160-162`, `utils.py:40-63`).

Conclusion: dagua's torch seed cannot produce the same random sequence as reference's NumPy/numba sequence. Even if all deterministic math were aligned, seeded stochastic paths would remain non-identical unless dagua ports the reference RNG state logic.

## 11. Edge-case bugs / suspicious divergences

1. **Self-neighbor off-by-one in kNN.** Dagua removes self before top-k (`dagua/layout/ops/umap.py:249-254`), but reference dense precomputed UMAP includes self in `n_neighbors` and then zeroes self membership (`umap_.py:312-319`, `umap_.py:424-431`). Because both smooth-kNN implementations skip column/index `1..` in the membership sum (`dagua/layout/ops/umap.py:287-292`, `umap_.py:219-228`), dagua skips the closest real neighbor while reference skips self. This is the clearest off-by-one bug.

2. **Epoch counters start at zero instead of first interval.** Dagua initializes `next_sample_epoch` and `next_negative_epoch` to zero (`dagua/layout/ops/umap.py:518-520`), but reference initializes them to `epochs_per_sample` and `epochs_per_negative_sample` (`layouts.py:323-325`). Dagua performs immediate epoch-0 updates that reference does not.

3. **Reference adapter random-init threshold not mirrored.** Reference adapter uses `init="random"` for `N < 10` (`dagua/eval/competitors/umap_competitor.py:186-192`); dagua always uses spectral initialization (`dagua/layout/ops/pipelines/umap_layout.py:87`). This is not a UMAP algorithm bug by itself, but it is a pairing mismatch.

4. **Tiny graph fallback mismatch.** Reference adapter bypasses UMAP and returns torch normal for `N <= 3` (`dagua/eval/competitors/umap_competitor.py:176-183`); dagua has deterministic zero/line/spectral behavior (`dagua/layout/ops/umap.py:397-407`). This is a harness-level mismatch.

5. **Weighted edge sign/meaning mismatch.** Dagua treats weights as strengths and reference treats weights as distances (`dagua/layout/ops/umap.py:138-142`, `dagua/eval/competitors/umap_competitor.py:79-85`). Dagua also post-multiplies fuzzy membership by edge weights (`dagua/layout/ops/umap.py:943-957`), which reference does not.

6. **Negative-sample count mismatch.** Reference derives `n_neg_samples` from elapsed epochs and does not cap to `negative_sample_rate` per positive update (`layouts.py:156-186`), while dagua loops until next epoch but caps at `negative_sample_rate` (`dagua/layout/ops/umap.py:541-559`). With the zero-initialized negative counter, this changes early repulsion most.

7. **KNN tie ordering mismatch.** Reference precomputed kNN uses stable mergesort (`utils.py:31-36`); dagua uses `torch.topk` (`dagua/layout/ops/umap.py:249-254`). Unweighted graph distances produce many ties, so this can alter fuzzy edges even after the self-neighbor fix.

8. **Spectral connected-component handling missing.** Reference routes disconnected fuzzy graphs to `multi_component_layout` (`spectral.py:463-476`); dagua directly eigensolves the whole graph (`dagua/layout/ops/umap.py:409-446`).

9. **Spectral scaling mismatch before optimization.** Reference `noisy_scale_coords` scales by max absolute coordinate and later rescales each dimension to `[0,10]` (`umap_.py:930-935`, `umap_.py:1188-1192`); dagua scales global min/max to `[-10,10]` and does not do the per-axis `[0,10]` pass (`dagua/layout/ops/umap.py:436-446`).

10. **Curve fitting not bit-aligned.** Dagua uses explicit `p0` and fallback (`dagua/layout/ops/umap.py:372-386`), reference does not (`umap_.py:1393-1408`). Probably lower impact than graph/init/schedule mismatches.

## 12. Ranked fix list

1. **Fix kNN self-neighbor semantics.**
   - Evidence: dagua masks diagonal before top-k (`dagua/layout/ops/umap.py:249-254`); reference includes self then zeroes it (`umap_.py:312-319`, `umap_.py:424-431`).
   - Proposed fix: for fidelity mode or default `classic_umap`, include self in kNN rows exactly as reference dense precomputed UMAP does. Use stable argsort-equivalent behavior if possible. Adjust effective `k` so `n_neighbors` counts self.
   - Expected RMSD impact: very high, especially `nn5` and `nn30`.
   - Size estimate: S/M, localized to `_knn_from_distances` and tests.

2. **Align SGD epoch counter initialization and negative-sample schedule.**
   - Evidence: reference counters start at interval copies (`layouts.py:323-325`), dagua counters start at zeros (`dagua/layout/ops/umap.py:518-520`); reference negative count formula differs (`layouts.py:156-186`) from dagua cap loop (`dagua/layout/ops/umap.py:541-559`).
   - Proposed fix: initialize `next_sample_epoch = epochs_per_sample.clone()` and `next_negative_epoch = epochs_per_sample / negative_sample_rate`; compute `n_neg_samples = int((epoch - next_negative_epoch[i]) / epochs_per_negative_sample[i])` and increment by `n_neg_samples * epochs_per_negative_sample[i]`.
   - Expected RMSD impact: very high.
   - Size estimate: M, localized to `_optimize_embedding` plus dormant `UMAPPairSGD` if consistency is desired.

3. **Mirror adapter init policy (`random` for `N < 10`, tiny fallback for `N <= 3`) or move the policy into both sides consistently.**
   - Evidence: reference adapter random/tiny policy (`dagua/eval/competitors/umap_competitor.py:176-192`) vs dagua always spectral (`dagua/layout/ops/pipelines/umap_layout.py:87`, `dagua/layout/ops/umap.py:389-446`).
   - Proposed fix: add init mode to dagua UMAP pipeline and have `classic_umap` default to the same adapter policy for fidelity comparisons. For `N <= 3`, match the adapter's torch normal fallback if the goal is exact pair fidelity.
   - Expected RMSD impact: high on small benchmark graphs.
   - Size estimate: M, config + pipeline branch + tests.

4. **Port reference RNG semantics for negative sampling.**
   - Evidence: reference creates NumPy `rng_state` (`umap_.py:1152`), per-sample state (`layouts.py:367-369`), and `tau_rand_int` draws (`layouts.py:160-162`, `utils.py:40-63`); dagua uses one torch `Generator` (`dagua/layout/ops/umap.py:515-516`, `dagua/layout/ops/umap.py:543`).
   - Proposed fix: implement NumPy-compatible initial `rng_state` generation and `tau_rand_int` in Python/Torch or NumPy, including per-source state offset by embedding first coordinate bits. This is needed after schedule alignment.
   - Expected RMSD impact: medium to high after graph/init fixes; alone may be noisy.
   - Size estimate: M/L, careful bit-level tests required.

5. **Align spectral scaling and connected-component handling.**
   - Evidence: reference component path (`spectral.py:463-476`), eigsh settings (`spectral.py:489-527`), noisy scale (`umap_.py:930-935`, `umap_.py:1117-1120`), per-axis `[0,10]` pre-SGD scaling (`umap_.py:1188-1192`); dagua direct eigensolve/scaling (`dagua/layout/ops/umap.py:409-446`).
   - Proposed fix: replace dagua spectral init with a closer port of `umap.spectral.spectral_layout` for fidelity mode, including multi-component layout and scaling/noise.
   - Expected RMSD impact: medium to high, graph-dependent.
   - Size estimate: L if fully ported; M for scaling-only subset.

6. **Use stable kNN tie ordering matching `fast_knn_indices`.**
   - Evidence: reference uses `argsort(kind="mergesort")` (`utils.py:31-36`), dagua uses `torch.topk` (`dagua/layout/ops/umap.py:249-254`).
   - Proposed fix: for CPU fidelity mode, use NumPy stable mergesort on the dense distance matrix, or deterministic lexicographic tie-break by `(distance, index)`.
   - Expected RMSD impact: medium on unweighted graphs with many equal shortest-path distances.
   - Size estimate: S/M.

7. **Fix weighted-edge semantics for reference parity.**
   - Evidence: dagua inverse-cost and fuzzy post-scale (`dagua/layout/ops/umap.py:138-142`, `dagua/layout/ops/umap.py:943-957`) vs reference direct cost and no post-scale (`dagua/eval/competitors/umap_competitor.py:79-85`, `umap_.py:2588-2607`).
   - Proposed fix: decide whether graph weights are distances or strengths in fidelity variants. For `umap_graph` parity, use weights directly as shortest-path distances and remove fuzzy post-scaling under fidelity mode.
   - Expected RMSD impact: high only on weighted/multiedge graphs; low on purely unweighted graphs.
   - Size estimate: M.

8. **Align curve-fit call exactly.**
   - Evidence: dagua p0/fallback (`dagua/layout/ops/umap.py:372-386`) vs reference default curve_fit (`umap_.py:1393-1408`).
   - Proposed fix: use reference `find_ab_params` call semantics for fidelity mode; optionally preserve fallback for robustness outside fidelity mode.
   - Expected RMSD impact: low to medium, more relevant for `mindist001`, `mindist05`, `spread2`.
   - Size estimate: S.

9. **Remove dagua final normalization for fidelity mode.**
   - Evidence: dagua normalizes to layout extent (`dagua/layout/ops/umap.py:564-580`, `dagua/layout/ops/umap.py:1150-1164`); reference returns raw UMAP coordinates (`dagua/eval/competitors/umap_competitor.py:196-199`).
   - Proposed fix: add `normalize_output`/`fidelity_mode` flag. Procrustes makes global scale/translation low impact, but degenerate one-dimensional fallback could alter orientation/relative shape.
   - Expected RMSD impact: low for Procrustes RMSD, possibly medium for degenerate tiny cases.
   - Size estimate: S.

10. **Audit dormant UMAP ops after active fixes.**
    - Evidence: generic `UMAPPairSGD` duplicates the same zero-counter and torch RNG behavior (`dagua/layout/ops/optimize.py:1304-1365`); generic UMAP loss uses torch negative sampling (`dagua/layout/ops/loss_classic.py:2496-2526`).
    - Proposed fix: if the project wants one UMAP semantics, update these too. If they are truly unused, leave them and document as non-fidelity ops.
    - Expected RMSD impact: none for active `classic_umap` unless future pipelines use them.
    - Size estimate: S/M.

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle: fix the deterministic graph/schedule mismatches before touching RNG or spectral rewrites.

Top-K scope:

1. **KNN self semantics + stable tie order.**
   - Modify `_knn_from_distances` so `n_neighbors` follows reference dense precomputed semantics: include self, stable sort by distance/index, return exactly `n_neighbors` entries where possible (`dagua/layout/ops/umap.py:239-254`, reference `umap_.py:312-319`, `utils.py:31-36`).

2. **Reference epoch schedule.**
   - Change `_optimize_embedding` counters and negative-sample count to match `optimize_layout_euclidean` (`dagua/layout/ops/umap.py:518-559`, reference `layouts.py:323-325`, `layouts.py:156-186`).

3. **Init policy parity for small graphs.**
   - Add a small config branch matching the adapter: `N <= 3` torch normal fallback and `4 <= N < 10` random uniform `[-10,10]`, seeded the same way the adapter/reference does (`dagua/eval/competitors/umap_competitor.py:176-192`, `umap_.py:1095-1098`).

4. **Run targeted UMAP fidelity variants.**
   - Rerun all six UMAP variants (`dagua/eval/variants.py:1474-1539`) and inspect per-graph deltas. Expect `nn5`/`nn30` to move most from the self-neighbor fix; small graphs should move most from init parity.

Deferred to Round 23+:

- Full reference RNG port (`umap_.py:1152`, `layouts.py:367-369`, `utils.py:40-63`), because it is harder to validate and should be done after deterministic schedule alignment.
- Full `umap.spectral` port including multi-component layout (`spectral.py:18-142`, `spectral.py:463-554`), because it is larger and may be unnecessary if kNN/schedule/init already drops RMSD enough.
- Weighted-edge parity decision, because it may affect dagua public semantics beyond UMAP fidelity (`dagua/layout/ops/umap.py:138-142`, `dagua/layout/ops/umap.py:943-957`, `dagua/eval/competitors/umap_competitor.py:79-85`).

## Final diagnosis

The biggest confirmed divergences are not subtle numeric drift. They are algorithmic/harness semantic mismatches:

1. Dagua's kNN excludes self; reference dense precomputed UMAP includes self and counts it against `n_neighbors`.
2. Dagua starts UMAP sampling counters at zero; reference starts at the per-edge interval.
3. Dagua always spectral-initializes; the reference adapter uses random init for `N < 10` and a torch-normal bypass for `N <= 3`.
4. Dagua's negative sampling uses a global torch RNG and a capped loop; reference uses per-source numba tau RNG and schedule-derived counts.
5. Weighted-edge meaning is inverted and then additionally post-scaled on the dagua side.

Assumption used in this diagnosis: the fidelity target is exactly the current `umap_graph` adapter behavior, including its tiny-graph bypass and `init="random" if N < 10 else "spectral"` policy (`dagua/eval/competitors/umap_competitor.py:176-192`), not just upstream `umap-learn` defaults in isolation.
