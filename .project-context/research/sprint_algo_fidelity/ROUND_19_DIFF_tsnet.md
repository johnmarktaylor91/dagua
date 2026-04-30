# Round 19 Adversarial Diff: tsNET vs sklearn TSNE

Status: diagnosis-only
Family: tsnet
Date: 2026-04-30

## Scope

Compared Dagua `classic_tsnet`/ops pipeline against the actual `tsne_graph`
competitor, which invokes sklearn `TSNE` on a graph shortest-path distance
matrix. Round 18 already showed that NumPy-vs-torch initialization RNG alignment
alone did not improve the aggregate floor result: median changed
`0.337116 -> 0.343569`, and the residual was classified as
`stochastic_floor_match_with_low_floor_exception`
(`eval_output/algo_fidelity/round_18/SUMMARY.md:34-70`).

## Reference Invocation

The competitor builds `TSNE` with:

- `n_components=2`
- `metric="precomputed"`
- `init="random"`
- `random_state=seed if seed is not None else 42`
- `perplexity=min(30.0, num_nodes - 1)`
- optional variant overrides for `learning_rate`, `max_iter`, and `perplexity`
- default sklearn `method="barnes_hut"`, `learning_rate="auto"`,
  `early_exaggeration=12.0`, `max_iter=1000`, `angle=0.5`,
  `n_iter_without_progress=300`, `min_grad_norm=1e-7`

Refs:

- `dagua/eval/competitors/tsne_competitor.py:138-155`
- `dagua/eval/competitors/tsne_competitor.py:62-68`
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:810-843`

## 1. Initialization

sklearn supports PCA and random initialization, but the Dagua competitor uses
`init="random"`, not the sklearn default PCA path
(`dagua/eval/competitors/tsne_competitor.py:138-145`). sklearn rejects
`init="pca"` when `metric="precomputed"` anyway
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:877-881`).

For random init, sklearn obtains `random_state = check_random_state(...)`, then
creates a float32 array with shape `[n_samples, n_components]` using
`1e-4 * random_state.standard_normal(...)`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:906-918`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1013-1018`).

Dagua initializes with `torch.Generator(device="cpu")`, seeds it with
`problem.seed`, samples `torch.randn(num_nodes, 2, dtype=torch.float32)`, scales
by `noise_scale=1e-4`, moves to the layout device, clones, and sets
`requires_grad=True` (`dagua/layout/ops/tsnet.py:33-44`,
`dagua/layout/ops/tsnet.py:126-160`). Shape and scale match sklearn for the
actual competitor path; RNG stream does not. This confirms Round 18's note:
same integer seed, different NumPy vs torch random sequence
(`eval_output/algo_fidelity/round_18/SUMMARY.md:39-46`).

PCA note: sklearn PCA init would produce `[N, 2]` float32 coordinates rescaled so
PC1 has standard deviation `1e-4`, but that branch is not used here and cannot
be used with `metric="precomputed"`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1002-1012`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:877-881`).

## 2. Distance Matrix

The reference adapter computes an undirected SciPy CSR adjacency, mirrors every
edge, uses edge weights when present or ones otherwise, runs
`scipy.sparse.csgraph.shortest_path(..., directed=False)`, then replaces
infinite disconnected entries with a global finite fill value:
`max(max_finite * 2.0, 1.0)` (`dagua/eval/competitors/tsne_competitor.py:22-59`).
The resulting dense `float32` matrix is passed to sklearn as `X` with
`metric="precomputed"` (`dagua/eval/competitors/tsne_competitor.py:138-155`).

Dagua builds an undirected adjacency through `build_undirected_adjacency`, which
accumulates duplicate-edge weights additively (`dagua/layout/ops/graph_utils.py:226-268`),
then calls `all_pairs_shortest_paths` with weighted mode only when
`problem.edge_weights is not None` (`dagua/layout/ops/tsnet.py:226-234`).
Unweighted paths use BFS; weighted paths use Dijkstra through the shared APSP
helper (`dagua/layout/ops/graph_utils.py:101-115`,
`dagua/layout/ops/graph_utils.py:279-308`). Dagua fills disconnected entries
per row with `row max + 1.0`, not the competitor's global `max_finite * 2.0`
fill (`dagua/layout/ops/graph_utils.py:301-308`).

Important distance mismatches:

- Duplicate weighted edges: sklearn sums duplicates in CSR construction only
  when duplicate coordinate entries are collapsed by SciPy; Dagua explicitly
  sums duplicate weights into adjacency maps (`dagua/layout/ops/graph_utils.py:260-268`).
  For unweighted duplicate edges, both effectively preserve unit shortest-path
  distance.
- Disconnected graphs: sklearn uses a global `2 * max_finite` fill, while Dagua
  uses per-row `max + 1`. This can materially alter high-dimensional affinities
  on disconnected or partially disconnected graphs.
- Weighted duplicates: Dagua's additive duplicate policy can inflate an edge's
  shortest-path cost relative to a minimum-edge or single-edge interpretation.

## 3. Perplexity Calibration

sklearn computes conditional probabilities by calling the compiled
`_utils._binary_search_perplexity` helper. In exact mode it passes the dense
distance matrix after casting to float32; in the default Barnes-Hut path it
passes only a nearest-neighbor distance matrix with shape `[N, n_neighbors]`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:38-68`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:71-125`).

Because the competitor does not override `method`, sklearn uses Barnes-Hut by
default (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:673-683`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:810-843`).
That path sets `n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))`
and computes a sparse kneighbors graph before binary search
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:949-999`).

Dagua binary-searches every full row of the dense APSP matrix. It clamps
perplexity to `max(num_nodes - 1, 1)`, removes self-distance by masking
`argmin(row)`, uses `target_entropy = log(perplexity)`, iterates up to 100
times with tolerance `1e-5`, and adjusts `beta` by doubling/halving or midpoint
rules (`dagua/layout/ops/tsnet.py:72-80`, `dagua/layout/ops/tsnet.py:221-275`).

The calibration target is conceptually aligned, but the support set is not:
sklearn Barnes-Hut calibrates over a truncated nearest-neighbor support; Dagua
calibrates over all off-diagonal nodes. That is a larger mismatch than the
NumPy-vs-torch RNG difference for most graphs.

## 4. High-D Affinities P

sklearn exact-mode `_joint_probabilities` casts distances to float32, calls
binary search, symmetrizes `conditional_P + conditional_P.T`, normalizes by the
sum of all symmetrized mass, and floors with machine epsilon
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:38-68`).

sklearn Barnes-Hut `_joint_probabilities_nn` sorts sparse indices, reshapes
distance data to `[N, n_neighbors]`, calls the same binary search helper,
constructs a CSR `P`, symmetrizes `P + P.T`, and normalizes by `P.sum()`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:71-125`).

For precomputed exact mode, sklearn squares distances unless metric is
`"euclidean"`; since the competitor passes `metric="precomputed"`, exact mode
would square the provided graph shortest-path distances before computing `P`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:911-942`).
In the default Barnes-Hut path, kneighbors distances are squared unconditionally
before `_joint_probabilities_nn` (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:990-999`).

Dagua computes `squared = row.square()` once, uses `weights = exp(-squared *
beta)` with a self mask, normalizes each row, then symmetrizes as
`(conditional + conditional.T) / (2*N)` and clamps to `min_distance=1e-12`
(`dagua/layout/ops/tsnet.py:236-279`). For dense all-pairs conditional
probabilities whose rows sum to 1, `/ (2*N)` is equivalent to sklearn exact
normalization by total symmetrized mass. It is not equivalent to sklearn
Barnes-Hut sparse support because absent non-neighbor probabilities remain zero
in sklearn's sparse `P`.

Potential self-mask issue: Dagua excludes `argmin(row)` rather than the diagonal
index (`dagua/layout/ops/tsnet.py:244-245`, `dagua/layout/ops/tsnet.py:274`).
This is usually the current node because diagonal distance is zero, but any
off-diagonal zero-distance weighted edge or malformed matrix tie could mask the
wrong entry.

## 5. Low-D Affinities Q

sklearn exact KL computes pairwise squared Euclidean distances in the embedding,
divides by `degrees_of_freedom`, adds 1, raises by
`(degrees_of_freedom + 1) / -2`, then normalizes condensed off-diagonal distances
as `Q = dist / (2 * sum(dist))`, floored at machine epsilon
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:174-182`).
With `n_components=2`, sklearn sets `degrees_of_freedom=max(2-1, 1)=1`, so this
is the standard Student-t kernel `1 / (1 + d^2)`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1020-1024`).

sklearn Barnes-Hut computes the analogous Student-t objective and gradient in
compiled code using `P.data`, CSR neighbors, the embedding, `angle`, and `dof`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:205-298`).

Dagua computes dense `delta = y_i - y_j`, `squared_distances`, `numerators =
(1 + squared_distances).reciprocal()`, zeros the diagonal, and normalizes by the
full off-diagonal sum (`dagua/layout/ops/tsnet.py:432-443`). For `dof=1`, this
matches sklearn exact Q semantics in full-matrix form. It does not match the
Barnes-Hut approximation path exactly.

## 6. KL Gradient Formula

sklearn exact KL uses condensed `P` and `Q`, computes the KL error as
`2 * dot(P, log(P / Q))`, then computes
`PQd = squareform((P - Q) * dist)` and for each point:
`grad[i] = dot(PQd[i], X_embedded[i] - X_embedded)`. Finally it multiplies by
`c = 2 * (dof + 1) / dof` (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:186-200`).
For two output dimensions, `dof=1`, so `c=4`.

sklearn Barnes-Hut passes sparse `P` and the embedding to
`_barnes_hut_tsne.gradient`, then applies the same `c` factor to the flattened
gradient (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:273-298`).

Dagua constructs the dense KL loss
`sum(P_eff * (log(P_eff) - log(Q)))` and relies on PyTorch autograd
(`dagua/layout/ops/tsnet.py:429-451`). For a dense symmetric P/Q setup, autograd
should produce the exact full-matrix t-SNE gradient up to numerical details and
the representation's pair-counting convention. The current implementation
computes the loss over all ordered pairs, while sklearn exact stores condensed
unordered pairs and uses explicit factors of two. These are expected to be
mathematically equivalent if P/Q are normalized consistently, but not
bit-equivalent.

More importantly, the actual reference uses Barnes-Hut by default, so Dagua's
dense exact autograd gradient differs from sklearn's approximate tree gradient
and sparse-P support.

## 7. Early Exaggeration

sklearn uses a two-stage schedule. It multiplies `P` in place by
`early_exaggeration`, runs 250 exploration iterations with momentum `0.5`, then
divides `P` back down and continues with momentum `0.8`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:804-808`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1035-1095`).

Dagua stores `early_exaggeration=12.0` and `early_exaggeration_steps=250`
(`dagua/layout/ops/tsnet.py:72-80`, `dagua/layout/ops/tsnet.py:281-286`), then
uses `effective_probabilities = probabilities * exaggeration` while
`state.step < early_steps` (`dagua/layout/ops/tsnet.py:429-431`). This aligns
with sklearn's default factor and phase length, assuming `state.step` is
incremented once per repeat iteration by the pipeline machinery.

Subtle objective mismatch: Dagua's early-exaggeration loss includes
`P_eff * log(P_eff / Q)`, while sklearn's objective receives already-exaggerated
`P` and computes the same expression (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:186-188`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1075-1087`,
`dagua/layout/ops/tsnet.py:444-450`). The constant term changes the reported KL
but should not change gradients with respect to positions.

## 8. Optimizer

sklearn uses `_gradient_descent`: flatten params, zero update, unit gains,
compute objective/grad each iteration, update gains by sign agreement with the
previous update, clip gains to `min_gain`, apply momentum update
`update = momentum * update - learning_rate * grad`, and add update to params
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:301-444`).

Dagua initializes `tsnet_update` to zeros and `tsnet_gains` to ones
(`dagua/layout/ops/tsnet.py:296-342`). Its per-step rule matches sklearn's
gain/momentum equations: sign disagreement increases gains by `0.2`, agreement
multiplies by `0.8`, gains clamp to `0.01`, then
`update = momentum * update - learning_rate * grad`, and `pos += update`
(`dagua/layout/ops/tsnet.py:83-103`, `dagua/layout/ops/tsnet.py:453-471`).

Remaining optimizer differences:

- sklearn flattens parameters; Dagua uses `[N, 2]` tensors. The elementwise rule
  is equivalent.
- sklearn computes cost only every 50 iterations by setting `compute_error`;
  gradient is still computed every iteration
  (`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:395-400`).
  Dagua computes full KL loss every iteration because autograd needs a scalar
  objective (`dagua/layout/ops/tsnet.py:444-451`).
- sklearn can stop early based on progress or gradient norm; Dagua's pipeline
  always repeats fixed `steps` (`dagua/layout/ops/pipelines/tsnet.py:51-64`).

## 9. Learning Rate

sklearn default `learning_rate="auto"` sets
`learning_rate_ = max(N / early_exaggeration / 4, 50)`, so with default
`early_exaggeration=12`, this is `max(N / 48, 50)`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:601-613`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:855-860`).

Dagua's `TsnetPrepareStateConfig` uses `lr_divisor=48.0` and `lr_floor=50.0`,
then stores both early and late learning rates as
`max(num_nodes / 48, 50)` (`dagua/layout/ops/tsnet.py:58-80`,
`dagua/layout/ops/tsnet.py:287-292`). This matches sklearn default `"auto"` for
the actual reference invocation.

Variant caveat: the reference adapter accepts `learning_rate` variants and
passes them into sklearn (`dagua/eval/competitors/tsne_competitor.py:62-68`,
`dagua/eval/competitors/tsne_competitor.py:146-154`). Dagua
`layout_tsnet_pipeline` exposes only `perplexity`, `steps`, `seed`, and
`edge_weights`; it has no public learning-rate override
(`dagua/layout/ops/pipelines/tsnet.py:69-148`).

## 10. Convergence / Iteration Count

sklearn default `max_iter=1000`, minimum allowed `250`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:618-629`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:780-789`).
It runs a 250-iteration exploration phase, then a final phase up to `max_iter`,
with progress checks every 50 iterations, early stopping after
`n_iter_without_progress=300`, and gradient norm stop at `1e-7`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:301-444`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1035-1097`).

Dagua pipeline defaults to `steps=1000`, validates non-negative steps, runs a
fixed `Repeat(n=steps)` of `TsnetGradientStep`, and finalizes positions
(`dagua/layout/ops/pipelines/tsnet.py:27-66`,
`dagua/layout/ops/pipelines/tsnet.py:69-148`). It does not implement sklearn's
gradient-norm stop or no-progress stop.

The competitor clamps variant `max_iter` to at least 250 for sklearn 1.5+
compatibility (`dagua/eval/competitors/tsne_competitor.py:150-154`). Dagua
accepts `steps=0` and any non-negative value (`dagua/layout/ops/pipelines/tsnet.py:113-118`).

## 11. RNG Semantics

sklearn uses `check_random_state(self.random_state)` and, for the actual
reference path, NumPy `RandomState.standard_normal` scaled by `1e-4`
(`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:906-918`,
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1013-1018`).
Dagua uses a torch CPU generator with `manual_seed(problem.seed)` and
`torch.randn` (`dagua/layout/ops/tsnet.py:149-160`).

Same seed does not imply same sample sequence. Round 18 tested a temporary
NumPy-aligned initialization and found no aggregate improvement, so RNG
alignment is real but not the dominant residual driver
(`eval_output/algo_fidelity/round_18/SUMMARY.md:39-59`).

## 12. Hyperparameter Alignment Table

| Topic | sklearn reference | Dagua tsnet | Alignment |
| --- | --- | --- | --- |
| Components | `n_components=2` (`tsne_competitor.py:138-145`) | fixed `[N, 2]` init (`tsnet.py:152-155`) | aligned |
| Metric input | `metric="precomputed"` (`tsne_competitor.py:138-145`) | graph APSP internally (`tsnet.py:226-234`) | conceptually aligned |
| Distance graph | SciPy mirrored CSR, shortest_path directed=False (`tsne_competitor.py:44-59`) | custom adjacency + BFS/Dijkstra (`graph_utils.py:226-308`) | partially aligned |
| Disconnected fill | global `max(max_finite * 2, 1)` (`tsne_competitor.py:55-59`) | per-row `max + 1` (`graph_utils.py:301-308`) | mismatch |
| Duplicate weighted edges | CSR duplicate behavior from mirrored edge list (`tsne_competitor.py:44-52`) | explicit additive accumulation (`graph_utils.py:260-268`) | likely aligned for sums, but verify |
| Method | default `barnes_hut` (`_t_sne.py:673-683`, `_t_sne.py:810-843`) | dense exact autograd (`tsnet.py:429-451`) | major mismatch |
| Neighbor support | `min(N-1, int(3*perplexity+1))` (`_t_sne.py:949-955`) | all off-diagonal nodes (`tsnet.py:236-279`) | major mismatch |
| Perplexity | adapter clamp to `N-1` (`tsne_competitor.py:143-149`) | clamp to `max(N-1, 1)` (`tsnet.py:221-225`) | aligned for `N>=2` |
| Binary search | compiled `_binary_search_perplexity` (`_t_sne.py:61-64`, `_t_sne.py:103-107`) | torch loop, 100 iters, tol `1e-5` (`tsnet.py:72-80`, `tsnet.py:253-272`) | close, not exact |
| P symmetrization | sparse/dense `P + P.T`, normalize by sum (`_t_sne.py:65-68`, `_t_sne.py:110-120`) | `(P + P.T)/(2*N)` (`tsnet.py:277-279`) | exact-mode aligned; BH mismatch |
| P floor | `MACHINE_EPSILON` double eps (`_t_sne.py:35`, `_t_sne.py:67`) | `1e-12` (`tsnet.py:68-80`, `tsnet.py:279`) | minor mismatch |
| Q kernel | Student-t, `dof=1`, off-diagonal normalized (`_t_sne.py:174-182`, `_t_sne.py:1020-1024`) | dense `1/(1+d^2)`, diagonal zero, normalized (`tsnet.py:432-443`) | exact-mode aligned |
| Gradient | sklearn exact formula or Barnes-Hut Cython gradient (`_t_sne.py:192-200`, `_t_sne.py:273-298`) | PyTorch autograd dense KL (`tsnet.py:444-451`) | major mismatch vs actual BH |
| Early exaggeration | `12.0`, first 250 iters (`_t_sne.py:804-808`, `_t_sne.py:1075-1087`) | `12.0`, first 250 steps (`tsnet.py:72-80`, `tsnet.py:429-431`) | aligned |
| Momentum | `0.5` then `0.8` (`_t_sne.py:1060-1094`) | `0.5` then `0.8` (`tsnet.py:83-103`, `tsnet.py:453-455`) | aligned |
| Gains | `+0.2`, `*0.8`, min `0.01` (`_t_sne.py:402-408`) | same (`tsnet.py:457-465`) | aligned |
| Learning rate | auto `max(N/12/4, 50)` (`_t_sne.py:855-860`) | `max(N/48, 50)` (`tsnet.py:287-292`) | aligned for default |
| Iterations | default 1000, early stop possible (`_t_sne.py:301-444`, `_t_sne.py:1035-1097`) | fixed `steps=1000` default (`pipelines/tsnet.py:27-66`) | partial mismatch |
| RNG | NumPy RandomState (`_t_sne.py:906-918`, `_t_sne.py:1013-1018`) | torch Generator (`tsnet.py:149-160`) | mismatch |
| Final output scale | raw sklearn embedding returned (`tsne_competitor.py:154-156`) | normalized to layout extent (`tsnet.py:517-522`) | possible downstream metric mismatch |

## 13. Ranked Fix List

1. Match the actual sklearn method support first: either make the reference use
   `method="exact"` for dense all-pairs parity, or implement sklearn-style
   Barnes-Hut sparse nearest-neighbor `P` support in Dagua. The current
   comparison is sklearn Barnes-Hut sparse affinities/gradient vs Dagua dense
   exact autograd (`_t_sne.py:949-999`, `_t_sne.py:1035-1095`,
   `tsnet.py:236-279`, `tsnet.py:429-451`).
2. Align distance-matrix disconnected fills with the competitor: global
   `max(max_finite * 2, 1)` instead of per-row `max + 1`
   (`tsne_competitor.py:55-59`, `graph_utils.py:301-308`).
3. Add a NumPy-compatible initialization option only after method/support
   alignment. Round 18 showed RNG alignment alone was insufficient
   (`SUMMARY.md:43-59`).
4. Mirror sklearn convergence controls: progress checks every 50 iterations,
   `min_grad_norm=1e-7`, and `n_iter_without_progress=300` after exploration
   (`_t_sne.py:301-444`, `_t_sne.py:1088-1095`).
5. Expose/pass `learning_rate` variants on the Dagua side if fidelity runs
   compare `classic_tsnet` variants against sklearn variants
   (`tsne_competitor.py:62-68`, `pipelines/tsnet.py:69-148`).
6. Replace `argmin(row)` self masking with explicit diagonal masking to avoid
   zero-distance tie bugs (`tsnet.py:244-245`, `tsnet.py:274`).
7. Revisit final normalization. sklearn returns raw coordinates; Dagua always
   normalizes to layout extent (`tsne_competitor.py:154-156`,
   `tsnet.py:517-522`). If the fidelity metric is scale-invariant this is low
   priority; if not, it can dominate numeric comparisons.

## 14. Recommended Round 20 Fix Scope

Recommended scope: do not spend Round 20 on RNG. The best conservative Round 20
scope is to align the reference objective family before changing stochastic
details:

1. Decide whether fidelity target is sklearn default Barnes-Hut or sklearn exact.
2. If target remains the actual adapter (`method="barnes_hut"`), implement
   sparse nearest-neighbor high-dimensional affinities in Dagua:
   `n_neighbors=min(N-1, int(3*perplexity+1))`, square neighbor distances before
   binary search, symmetrize sparse `P`, normalize by total sparse mass, and use
   dense Q/autograd as an approximation only if Barnes-Hut gradient parity is
   explicitly out of scope.
3. If target is exact t-SNE, change the competitor invocation for the fidelity
   reference to `method="exact"` and then compare Dagua's dense formulation
   against sklearn exact. In that path, prioritize disconnected-fill parity and
   NumPy RNG parity next.
4. Add focused tests for the chosen scope: distance-fill parity, P row/support
   parity, and one tiny graph deterministic initialization fixture.

Most likely root cause ranking for current residual:

1. sklearn default Barnes-Hut sparse support/gradient vs Dagua dense exact
   autograd.
2. disconnected distance fill differences.
3. NumPy vs torch initialization sequence.
4. convergence/early-stop behavior.
5. final output normalization.

## Verification

This document is diagnosis-only. No code or tests were changed. The required
output file exists at:

`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_tsnet.md`
